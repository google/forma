// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The path division algorithm is mostly based on Raph Levien's curvature approximation[1] for
// quadratic Bézier division and Adrian Colomitchi's cubic-to-quadratic approximation[2] with a few
// additions to improve robustness.
//
// The algorithm converts all possible types of curves into primitives (lines and quadratic
// Béziers) sequentially, while these are pushed onto the `PathBuilder`. Afterwards, each `Path`
// is converted into segments in parallel. This part of the algorithms tries its best not to create
// new segments if two neighboring segments are close enough to forming a line together according
// to some threshold.
//
// [1]: https://raphlinus.github.io/graphics/curves/2019/12/23/flatten-quadbez.html
// [2]: http://www.caffeineowl.com/graphics/2d/vectorial/bezierintro.html

use std::{cell::RefCell, f32, rc::Rc};

use rayon::prelude::*;

use crate::{
    consts,
    math::{GeomPresTransform, Point},
    utils::{ExtendTuple3, ExtendVec},
    GeomId,
};

// Pixel accuracy should be within 0.5 of a sub-pixel.
pub const MAX_ERROR: f32 = 1.0 / consts::PIXEL_WIDTH as f32;
const MAX_ANGLE_ERROR: f32 = 0.001;
const MIN_LEN: usize = 256;

fn lerp(t: f32, a: f32, b: f32) -> f32 {
    t.mul_add(b, (-t).mul_add(a, a))
}

fn curvature(x: f32) -> f32 {
    const C: f32 = 0.67;
    x / (1.0 - C + ((x * x).mul_add(0.25, C * C * C * C)).sqrt().sqrt())
}

fn inv_curvature(k: f32) -> f32 {
    const C: f32 = 0.39;
    k * (1.0 - C + ((k * k).mul_add(0.25, C * C)).sqrt())
}

#[derive(Clone, Copy, Debug)]
pub struct WeightedPoint {
    pub point: Point,
    pub weight: f32,
}

impl WeightedPoint {
    pub fn applied(self) -> Point {
        let w_recip = self.weight.recip();

        Point {
            x: self.point.x * w_recip,
            y: self.point.y * w_recip,
        }
    }
}

fn eval_cubic(t: f32, points: &[WeightedPoint; 4]) -> WeightedPoint {
    let x = lerp(
        t,
        lerp(
            t,
            lerp(t, points[0].point.x, points[1].point.x),
            lerp(t, points[1].point.x, points[2].point.x),
        ),
        lerp(
            t,
            lerp(t, points[1].point.x, points[2].point.x),
            lerp(t, points[2].point.x, points[3].point.x),
        ),
    );
    let y = lerp(
        t,
        lerp(
            t,
            lerp(t, points[0].point.y, points[1].point.y),
            lerp(t, points[1].point.y, points[2].point.y),
        ),
        lerp(
            t,
            lerp(t, points[1].point.y, points[2].point.y),
            lerp(t, points[2].point.y, points[3].point.y),
        ),
    );
    let weight = lerp(
        t,
        lerp(
            t,
            lerp(t, points[0].weight, points[1].weight),
            lerp(t, points[1].weight, points[2].weight),
        ),
        lerp(
            t,
            lerp(t, points[1].weight, points[2].weight),
            lerp(t, points[2].weight, points[3].weight),
        ),
    );

    WeightedPoint {
        point: Point { x, y },
        weight,
    }
}

#[derive(Debug, Default)]
pub struct ScratchBuffers {
    point_indices: Vec<usize>,
    quad_indices: Vec<usize>,
    point_commands: Vec<u32>,
}

impl ScratchBuffers {
    pub fn clear(&mut self) {
        self.point_indices.clear();
        self.quad_indices.clear();
        self.point_commands.clear();
    }
}

#[derive(Clone, Copy, Debug)]
enum PointCommand {
    Start(usize),
    Incr(f32),
    End(usize, bool),
}

impl From<u32> for PointCommand {
    fn from(val: u32) -> Self {
        if val & 0x7F80_0000 == 0x7F80_0000 {
            if val & 0x8000_0000 == 0 {
                Self::Start((val & 0x3F_FFFF) as usize)
            } else {
                Self::End((val & 0x3F_FFFF) as usize, val & 0x40_0000 != 0)
            }
        } else {
            Self::Incr(f32::from_bits(val))
        }
    }
}

impl From<PointCommand> for u32 {
    fn from(command: PointCommand) -> Self {
        match command {
            PointCommand::Start(i) => 0x7F80_0000 | (i as u32 & 0x3F_FFFF),
            PointCommand::Incr(point_command) => point_command.to_bits(),
            PointCommand::End(i, new_contour) => {
                0xFF80_0000 | (i as u32 & 0x3F_FFFF) | (u32::from(new_contour) << 22)
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Contour;

#[derive(Clone, Debug)]
struct Spline {
    curvature: f32,
    p0: Point,
    p2: Point,
    contour: Option<Contour>,
}

impl Spline {
    #[inline]
    pub fn new_spline_needed(&mut self, angle_changed: bool, point: Point) -> Option<Contour> {
        let needed = angle_changed || (point - self.p2).len() >= MAX_ERROR;

        needed.then(|| self.contour.take()).flatten()
    }
}

#[derive(Clone, Debug)]
struct Primitives {
    last_angle: Option<f32>,
    contour: Option<Contour>,
    splines: Vec<Spline>,
    x: Vec<f32>,
    y: Vec<f32>,
    weight: Vec<f32>,
    x0: Vec<f32>,
    dx_recip: Vec<f32>,
    k0: Vec<f32>,
    dk: Vec<f32>,
    curvatures_recip: Vec<f32>,
    partial_curvatures: Vec<(u32, f32)>,
}

impl Primitives {
    #[inline]
    fn last_spline_or_insert_with<F>(
        &mut self,
        angle: Option<f32>,
        point: Point,
        f: F,
    ) -> &mut Spline
    where
        F: FnOnce(Contour) -> Spline,
    {
        if let Some(contour) = self.contour.take().or_else(|| {
            fn diff(a0: f32, a1: f32) -> f32 {
                let mut diff = (a1 - a0).abs();

                if diff > f32::consts::PI {
                    diff -= f32::consts::PI;
                }

                if diff > f32::consts::FRAC_PI_2 {
                    diff = f32::consts::PI - diff;
                }

                diff
            }

            let angle_changed = if let (Some(a0), Some(a1)) = (self.last_angle, angle) {
                diff(a0, a1) > MAX_ANGLE_ERROR
            } else {
                false
            };

            self.splines
                .last_mut()
                .and_then(|spline| spline.new_spline_needed(angle_changed, point))
        }) {
            self.splines.push(f(contour));
        }

        self.splines.last_mut().unwrap()
    }

    pub fn push_contour(&mut self, contour: Contour) {
        self.contour = Some(contour);
    }

    pub fn push_line(&mut self, points: [WeightedPoint; 2]) {
        let p0 = points[0].applied();
        let p1 = points[1].applied();

        let d = p1 - p0;
        let angle = d.angle();

        let spline = self.last_spline_or_insert_with(angle, p0, |contour| Spline {
            curvature: 0.0,
            p0,
            p2: p1,
            contour: Some(contour),
        });

        spline.p2 = p1;

        self.last_angle = angle;
    }

    pub fn push_quad(&mut self, points: [WeightedPoint; 3]) {
        const PIXEL_ACCURACY_RECIP: f32 = 1.0 / MAX_ERROR;

        let p0 = points[0].applied();
        let p1 = points[1].applied();
        let p2 = points[2].applied();

        let a = p1 - p0;
        let b = p2 - p1;

        let in_angle = a.angle();
        let out_angle = b.angle();

        if in_angle.is_none() && out_angle.is_none() {
            return;
        }

        if in_angle.is_none() || out_angle.is_none() {
            return self.push_line([points[0], points[2]]);
        }

        self.x.extend(points.iter().map(|p| p.point.x));
        self.y.extend(points.iter().map(|p| p.point.y));
        self.weight.extend(points.iter().map(|p| p.weight));

        let spline = self.last_spline_or_insert_with(in_angle, p0, |contour| Spline {
            curvature: 0.0,
            p0,
            p2,
            contour: Some(contour),
        });

        spline.p2 = p2;

        let h = a - b;

        let cross = (p2.x - p0.x).mul_add(h.y, -(p2.y - p0.y) * h.x);
        let cross_recip = cross.recip();

        let mut x0 = a.x.mul_add(h.x, a.y * h.y) * cross_recip;
        let x2 = b.x.mul_add(h.x, b.y * h.y) * cross_recip;
        let mut dx_recip = (x2 - x0).recip();

        let scale = (cross / (h.len() * (x2 - x0))).abs();

        let mut k0 = curvature(x0);
        let k2 = curvature(x2);

        let mut dk = k2 - k0;
        let mut current_curvature = 0.5 * dk.abs() * (scale * PIXEL_ACCURACY_RECIP).sqrt();

        // Points are collinear.
        if !current_curvature.is_finite() || current_curvature <= 1.0 {
            // These values are chosen such that the resulting points will be found at t = 0.5 and
            // t = 1.0.
            x0 = 0.036_624_67;
            dx_recip = 1.0;
            k0 = 0.0;
            dk = 1.0;

            current_curvature = 2.0;
        }

        let total_curvature = spline.curvature + current_curvature;

        spline.curvature = total_curvature;

        self.last_angle = out_angle;

        self.x0.push(x0);
        self.dx_recip.push(dx_recip);
        self.k0.push(k0);
        self.dk.push(dk);
        self.curvatures_recip.push(current_curvature.recip());
        self.partial_curvatures
            .push((self.splines.len() as u32 - 1, total_curvature));
    }

    pub fn push_cubic(&mut self, points: [WeightedPoint; 4]) {
        const MAX_CUBIC_ERROR_SQUARED: f32 = (36.0 * 36.0 / 3.0) * MAX_ERROR * MAX_ERROR;

        let p0 = points[0].applied();
        let p1 = points[1].applied();
        let p2 = points[2].applied();

        let dx = p2.x.mul_add(3.0, -p0.x) - p1.x.mul_add(3.0, -p1.x);
        let dy = p2.y.mul_add(3.0, -p0.y) - p1.y.mul_add(3.0, -p1.y);

        let err = dx.mul_add(dx, dy * dy);

        let mult = points[1].weight.max(points[2].weight).max(1.0);

        let subdivisions = (((err * MAX_CUBIC_ERROR_SQUARED.recip()).powf(1.0 / 6.0) * mult).ceil()
            as usize)
            .max(1);
        let incr = (subdivisions as f32).recip();

        let mut quad_p0 = p0;
        for i in 1..=subdivisions {
            let t = i as f32 * incr;

            let quad_p2 = eval_cubic(t, &points).applied();

            let mid_point = eval_cubic(t - 0.5 * incr, &points).applied();

            let quad_p1 = Point {
                x: mid_point.x.mul_add(2.0, -0.5 * (quad_p0.x + quad_p2.x)),
                y: mid_point.y.mul_add(2.0, -0.5 * (quad_p0.y + quad_p2.y)),
            };

            self.push_quad([
                WeightedPoint {
                    point: quad_p0,
                    weight: 1.0,
                },
                WeightedPoint {
                    point: quad_p1,
                    weight: 1.0,
                },
                WeightedPoint {
                    point: quad_p2,
                    weight: 1.0,
                },
            ]);

            quad_p0 = quad_p2;
        }
    }

    pub fn populate_buffers(&self, buffers: &mut ScratchBuffers) {
        buffers.clear();

        let mut i = 0;
        let mut last_spline = None;
        for (spline_i, spline) in self.splines.iter().enumerate() {
            let subdivisions = spline.curvature.ceil() as usize;
            let point_command = spline.curvature / subdivisions as f32;

            let needs_start_point = last_spline.map_or(true, |last_spline: &Spline| {
                last_spline.contour.is_some() || (last_spline.p2 - spline.p0).len() > MAX_ERROR
            });

            if needs_start_point {
                buffers.point_indices.push(Default::default());
                buffers.quad_indices.push(Default::default());
                buffers
                    .point_commands
                    .push(PointCommand::Start(spline_i).into());
            }

            for pi in 1..subdivisions {
                if pi as f32 > self.partial_curvatures[i].1 {
                    i += 1;
                }

                buffers.point_indices.push(pi);
                buffers.quad_indices.push(i);
                buffers
                    .point_commands
                    .push(PointCommand::Incr(point_command).into());
            }

            buffers.point_indices.push(Default::default());
            buffers.quad_indices.push(Default::default());
            buffers
                .point_commands
                .push(PointCommand::End(spline_i, spline.contour.is_some()).into());

            last_spline = Some(spline);

            if subdivisions > 0 {
                i += 1;
            }
        }
    }

    pub fn eval_quad(&self, quad_index: usize, t: f32) -> Point {
        let i0 = 3 * quad_index;
        let i1 = i0 + 1;
        let i2 = i0 + 2;

        let weight = lerp(
            t,
            lerp(t, self.weight[i0], self.weight[i1]),
            lerp(t, self.weight[i1], self.weight[i2]),
        );
        let w_recip = weight.recip();

        let x = lerp(
            t,
            lerp(t, self.x[i0], self.x[i1]),
            lerp(t, self.x[i1], self.x[i2]),
        ) * w_recip;
        let y = lerp(
            t,
            lerp(t, self.y[i0], self.y[i1]),
            lerp(t, self.y[i1], self.y[i2]),
        ) * w_recip;

        Point::new(x, y)
    }

    pub fn into_segments(self) -> Segments {
        let mut segments = Segments::default();

        thread_local!(static BUFFERS: RefCell<ScratchBuffers> = RefCell::new(ScratchBuffers {
            point_indices: Vec::new(),
            quad_indices: Vec::new(),
            point_commands: Vec::new(),
        }));

        BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();

            self.populate_buffers(&mut buffers);

            let points = buffers
                .point_indices
                .par_iter()
                .with_min_len(MIN_LEN)
                .zip(buffers.quad_indices.par_iter().with_min_len(MIN_LEN))
                .zip(buffers.point_commands.par_iter().with_min_len(MIN_LEN))
                .map(|((&pi, &qi), &point_command)| {
                    let incr = match PointCommand::from(point_command) {
                        PointCommand::Start(spline_i) => {
                            let point = self.splines[spline_i].p0;
                            return ((point.x, point.y), false);
                        }
                        PointCommand::End(spline_i, new_contour) => {
                            let point = self.splines[spline_i].p2;
                            return ((point.x, point.y), new_contour);
                        }
                        PointCommand::Incr(incr) => incr,
                    };

                    let spline_i = self.partial_curvatures[qi].0;

                    let previous_curvature = qi
                        .checked_sub(1)
                        .and_then(|i| {
                            let partial_curvature = self.partial_curvatures[i];
                            (partial_curvature.0 == spline_i).then_some(partial_curvature.1)
                        })
                        .unwrap_or_default();
                    let ratio =
                        incr.mul_add(pi as f32, -previous_curvature) * self.curvatures_recip[qi];

                    let x = inv_curvature(ratio.mul_add(self.dk[qi], self.k0[qi]));

                    let t = ((x - self.x0[qi]) * self.dx_recip[qi]).clamp(0.0, 1.0);

                    let point = self.eval_quad(qi, t);

                    ((point.x, point.y), false)
                });

            (
                (
                    ExtendVec::new(&mut segments.x),
                    ExtendVec::new(&mut segments.y),
                ),
                ExtendVec::new(&mut segments.start_new_contour),
            )
                .par_extend(points);
        });

        segments
    }
}

impl Default for Primitives {
    fn default() -> Self {
        Self {
            last_angle: None,
            contour: Some(Contour),
            splines: Default::default(),
            x: Default::default(),
            y: Default::default(),
            weight: Default::default(),
            x0: Default::default(),
            dx_recip: Default::default(),
            k0: Default::default(),
            dk: Default::default(),
            curvatures_recip: Default::default(),
            partial_curvatures: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PathCommand {
    Move,
    Line,
    Quad,
    Cubic,
}
#[derive(Debug, Default)]
struct Segments {
    x: Vec<f32>,
    y: Vec<f32>,
    start_new_contour: Vec<bool>,
}

#[derive(Debug)]
struct PathData {
    x: Vec<f32>,
    y: Vec<f32>,
    weight: Vec<f32>,
    commands: Vec<PathCommand>,
    open_point_index: usize,
    segments: Option<Segments>,
}

macro_rules! points {
    ( $inner:expr , $i:expr , $( $d:expr ),+ $( , )? ) => {[
        $(
            WeightedPoint {
                point: Point::new($inner.x[$i - $d], $inner.y[$i - $d]),
                weight: $inner.weight[$i - $d],
            },
        )*
    ]};
}

impl PathData {
    pub fn close(&mut self) {
        let len = self.x.len();

        let last_point = WeightedPoint {
            point: Point::new(self.x[len - 1], self.y[len - 1]),
            weight: self.weight[len - 1],
        };
        let open_point = WeightedPoint {
            point: Point::new(self.x[self.open_point_index], self.y[self.open_point_index]),
            weight: self.weight[self.open_point_index],
        };

        if last_point.applied() != open_point.applied() {
            self.x.push(open_point.point.x);
            self.y.push(open_point.point.y);
            self.weight.push(open_point.weight);

            self.commands.push(PathCommand::Line);
        }
    }

    pub fn segments(&mut self) -> &Segments {
        if self.segments.is_none() {
            let mut primitives = Primitives::default();

            let mut i = 0;
            for command in &self.commands {
                match command {
                    PathCommand::Move => {
                        i += 1;

                        primitives.push_contour(Contour);
                    }
                    PathCommand::Line => {
                        i += 1;

                        let points = points!(self, i, 2, 1);
                        primitives.push_line(points);
                    }
                    PathCommand::Quad => {
                        i += 2;

                        let points = points!(self, i, 3, 2, 1);
                        primitives.push_quad(points);
                    }
                    PathCommand::Cubic => {
                        i += 3;

                        let points = points!(self, i, 4, 3, 2, 1);
                        primitives.push_cubic(points);
                    }
                }
            }

            self.segments = Some(primitives.into_segments());
        }

        self.segments.as_ref().unwrap()
    }
}

impl Default for PathData {
    fn default() -> Self {
        Self {
            x: vec![0.0],
            y: vec![0.0],
            weight: vec![1.0],
            commands: vec![PathCommand::Move],
            open_point_index: 0,
            segments: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Path {
    inner: Rc<RefCell<PathData>>,
    transform: Option<GeomPresTransform>,
}

impl Path {
    pub(crate) fn push_segments_to(
        &self,
        x: &mut Vec<f32>,
        y: &mut Vec<f32>,
        id: GeomId,
        ids: &mut Vec<Option<GeomId>>,
    ) {
        let mut inner = self.inner.borrow_mut();

        let segments = inner.segments();
        let transform = &self.transform;

        if let Some(transform) = &transform {
            let iter = segments
                .x
                .par_iter()
                .with_min_len(MIN_LEN)
                .zip(
                    segments
                        .y
                        .par_iter()
                        .with_min_len(MIN_LEN)
                        .zip(segments.start_new_contour.par_iter().with_min_len(MIN_LEN)),
                )
                .map(|(&x, (&y, start_new_contour))| {
                    let point = transform.transform(Point::new(x, y));
                    (point.x, point.y, (!start_new_contour).then_some(id))
                });

            ExtendTuple3::new((x, y, ids)).par_extend(iter);
        } else {
            let iter = segments
                .x
                .par_iter()
                .with_min_len(MIN_LEN)
                .zip(
                    segments
                        .y
                        .par_iter()
                        .with_min_len(MIN_LEN)
                        .zip(segments.start_new_contour.par_iter().with_min_len(MIN_LEN)),
                )
                .map(|(&x, (&y, start_new_contour))| (x, y, (!start_new_contour).then_some(id)));

            ExtendTuple3::new((x, y, ids)).par_extend(iter);
        }
    }

    #[inline]
    pub fn transform(&self, transform: &[f32; 9]) -> Self {
        if let Some(transform) = GeomPresTransform::new(*transform) {
            return Self {
                inner: Rc::clone(&self.inner),
                transform: Some(transform),
            };
        }

        let inner = self.inner.borrow();
        let mut data = PathData {
            x: inner.x.clone(),
            y: inner.y.clone(),
            weight: inner.weight.clone(),
            commands: inner.commands.clone(),
            open_point_index: inner.open_point_index,
            segments: None,
        };

        data.x
            .par_iter_mut()
            .with_min_len(MIN_LEN)
            .zip(
                data.y
                    .par_iter_mut()
                    .with_min_len(MIN_LEN)
                    .zip(data.weight.par_iter_mut().with_min_len(MIN_LEN)),
            )
            .for_each(|(x, (y, weight))| {
                (*x, *y, *weight) = (
                    transform[0].mul_add(*x, transform[1].mul_add(*y, transform[2] * *weight)),
                    transform[3].mul_add(*x, transform[4].mul_add(*y, transform[5] * *weight)),
                    transform[6].mul_add(*x, transform[7].mul_add(*y, transform[8] * *weight)),
                );
            });

        Self {
            inner: Rc::new(RefCell::new(data)),
            transform: None,
        }
    }
}

impl Eq for Path {}

impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

#[derive(Clone, Debug, Default)]
pub struct PathBuilder {
    inner: Rc<RefCell<PathData>>,
}

impl PathBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn move_to(&mut self, p: Point) -> &mut Self {
        {
            let mut inner = self.inner.borrow_mut();
            let len = inner.x.len();

            if matches!(inner.commands[inner.commands.len() - 1], PathCommand::Move) {
                inner.x[len - 1] = p.x;
                inner.y[len - 1] = p.y;
                inner.weight[len - 1] = 1.0;
            } else {
                inner.close();

                let open_point_index = inner.x.len();

                inner.x.push(p.x);
                inner.y.push(p.y);
                inner.weight.push(1.0);

                inner.commands.push(PathCommand::Move);

                inner.open_point_index = open_point_index;
            }
        }

        self
    }

    #[inline]
    pub fn line_to(&mut self, p: Point) -> &mut Self {
        {
            let mut inner = self.inner.borrow_mut();

            inner.x.push(p.x);
            inner.y.push(p.y);
            inner.weight.push(1.0);

            inner.commands.push(PathCommand::Line);
        }

        self
    }

    #[inline]
    pub fn quad_to(&mut self, p1: Point, p2: Point) -> &mut Self {
        {
            let mut inner = self.inner.borrow_mut();

            inner.x.push(p1.x);
            inner.y.push(p1.y);
            inner.weight.push(1.0);

            inner.x.push(p2.x);
            inner.y.push(p2.y);
            inner.weight.push(1.0);

            inner.commands.push(PathCommand::Quad);
        }

        self
    }

    #[inline]
    pub fn cubic_to(&mut self, p1: Point, p2: Point, p3: Point) -> &mut Self {
        {
            let mut inner = self.inner.borrow_mut();

            inner.x.push(p1.x);
            inner.y.push(p1.y);
            inner.weight.push(1.0);

            inner.x.push(p2.x);
            inner.y.push(p2.y);
            inner.weight.push(1.0);

            inner.x.push(p3.x);
            inner.y.push(p3.y);
            inner.weight.push(1.0);

            inner.commands.push(PathCommand::Cubic);
        }

        self
    }

    #[inline]
    pub fn rat_quad_to(&mut self, p1: Point, p2: Point, weight: f32) -> &mut Self {
        {
            let mut inner = self.inner.borrow_mut();

            inner.x.push(p1.x * weight);
            inner.y.push(p1.y * weight);
            inner.weight.push(weight);

            inner.x.push(p2.x);
            inner.y.push(p2.y);
            inner.weight.push(1.0);

            inner.commands.push(PathCommand::Quad);
        }

        self
    }

    #[inline]
    pub fn rat_cubic_to(&mut self, p1: Point, p2: Point, p3: Point, w1: f32, w2: f32) -> &mut Self {
        {
            let mut inner = self.inner.borrow_mut();

            inner.x.push(p1.x * w1);
            inner.y.push(p1.y * w1);
            inner.weight.push(w1);

            inner.x.push(p2.x * w2);
            inner.y.push(p2.y * w2);
            inner.weight.push(w2);

            inner.x.push(p3.x);
            inner.y.push(p3.y);
            inner.weight.push(1.0);

            inner.commands.push(PathCommand::Cubic);
        }

        self
    }

    #[inline]
    pub fn build(&mut self) -> Path {
        let mut inner = self.inner.borrow_mut();

        inner.close();

        Path {
            inner: Rc::clone(&self.inner),
            transform: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math::Point;

    fn dist(p0: Point, p1: Point, p2: Point) -> f32 {
        let d10 = p1 - p0;
        let d21 = p2 - p1;

        (d21.x * d10.y - d10.x * d21.y).abs() / d21.len()
    }

    fn min_dist(p: Point, points: &[Point]) -> f32 {
        points
            .windows(2)
            .map(|window| dist(p, window[0], window[1]))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    fn eval_quad(t: f32, points: &[WeightedPoint; 3]) -> Point {
        let x = lerp(
            t,
            lerp(t, points[0].point.x, points[1].point.x),
            lerp(t, points[1].point.x, points[2].point.x),
        );
        let y = lerp(
            t,
            lerp(t, points[0].point.y, points[1].point.y),
            lerp(t, points[1].point.y, points[2].point.y),
        );

        Point { x, y }
    }

    macro_rules! curve {
        (
            ( $p0x:expr , $p0y:expr ) ,
            ( $p1x:expr , $p1y:expr ) ,
            ( $p2x:expr , $p2y:expr ) ,
            ( $p3x:expr , $p3y:expr ) $( , )?
        ) => {
            [
                WeightedPoint {
                    point: Point::new($p0x, $p0y),
                    weight: 1.0,
                },
                WeightedPoint {
                    point: Point::new($p1x, $p1y),
                    weight: 1.0,
                },
                WeightedPoint {
                    point: Point::new($p2x, $p2y),
                    weight: 1.0,
                },
                WeightedPoint {
                    point: Point::new($p3x, $p3y),
                    weight: 1.0,
                },
            ]
        };
        (
            ( $p0x:expr , $p0y:expr ) ,
            ( $p1x:expr , $p1y:expr ) ,
            ( $p2x:expr , $p2y:expr ) $( , )?
        ) => {
            [
                WeightedPoint {
                    point: Point::new($p0x, $p0y),
                    weight: 1.0,
                },
                WeightedPoint {
                    point: Point::new($p1x, $p1y),
                    weight: 1.0,
                },
                WeightedPoint {
                    point: Point::new($p2x, $p2y),
                    weight: 1.0,
                },
            ]
        };
        ( ( $p0x:expr , $p0y:expr ) , ( $p1x:expr , $p1y:expr ) $( , )? ) => {
            [
                WeightedPoint {
                    point: Point::new($p0x, $p0y),
                    weight: 1.0,
                },
                WeightedPoint {
                    point: Point::new($p1x, $p1y),
                    weight: 1.0,
                },
            ]
        };
    }

    #[test]
    fn quads() {
        let mut primitives = Primitives::default();

        let c0 = curve![(2.0, 0.0), (0.0, 1.0), (10.0, 1.0)];
        let c1 = curve![(10.0, 1.0), (20.0, 1.0), (18.0, 0.0)];

        primitives.push_quad(c0);
        primitives.push_quad(c1);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 9);

        assert_eq!(segments.x[0], 2.0);
        assert_eq!(segments.y[0], 0.0);
        assert_eq!(segments.x[8], 18.0);
        assert_eq!(segments.y[8], 0.0);

        let a = Point::new(segments.x[3], segments.y[3]);
        let b = Point::new(segments.x[5], segments.y[5]);

        assert!((a - b).len() > 10.0);

        let points: Vec<_> = segments
            .x
            .iter()
            .zip(segments.y.iter())
            .map(|(&x, &y)| Point::new(x, y))
            .collect();

        let min_c0 = (0..=50)
            .into_iter()
            .map(|i| {
                let t = i as f32 / 50.0;

                min_dist(eval_quad(t, &c0), &points)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let min_c1 = (0..=50)
            .into_iter()
            .map(|i| {
                let t = i as f32 / 50.0;

                min_dist(eval_quad(t, &c1), &points)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!(min_c0 < MAX_ERROR);
        assert!(min_c1 < MAX_ERROR);
    }

    #[test]
    fn two_splines() {
        let mut primitives = Primitives::default();

        primitives.push_quad(curve![(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)]);
        primitives.push_quad(curve![(3.0, 0.0), (4.0, 4.0), (5.0, 0.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 11);

        assert_eq!(segments.x[0], 0.0);
        assert_eq!(segments.y[0], 0.0);
        assert_eq!(segments.x[4], 2.0);
        assert_eq!(segments.y[4], 0.0);
        assert_eq!(segments.x[5], 3.0);
        assert_eq!(segments.y[5], 0.0);
        assert_eq!(segments.x[10], 5.0);
        assert_eq!(segments.y[10], 0.0);
    }

    #[test]
    fn collinear_quad() {
        let mut primitives = Primitives::default();

        primitives.push_quad(curve![(0.0, 0.0), (2.0, 0.0001), (1.0, 0.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 3);

        assert!((segments.x[1] - 1.25).abs() < 0.01);
        assert!((segments.y[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn overlapping_control_point_quad() {
        let mut primitives = Primitives::default();

        primitives.push_quad(curve![(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]);
        primitives.push_quad(curve![(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]);
        primitives.push_quad(curve![(1.0, 1.0), (2.0, 2.0), (2.0, 2.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 2);

        assert!((segments.x[0] - 0.0).abs() < 0.01);
        assert!((segments.y[0] - 0.0).abs() < 0.01);
        assert!((segments.x[1] - 2.0).abs() < 0.01);
        assert!((segments.y[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn rat_quad() {
        let mut primitives = Primitives::default();
        let weight = 10.0;

        primitives.push_quad([
            WeightedPoint {
                point: Point::new(0.0, 0.0),
                weight: 1.0,
            },
            WeightedPoint {
                point: Point::new(1.0 * weight, 2.0 * weight),
                weight,
            },
            WeightedPoint {
                point: Point::new(2.0, 0.0),
                weight: 1.0,
            },
        ]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 5);

        let points: Vec<_> = segments
            .x
            .iter()
            .zip(segments.y.iter())
            .map(|(&x, &y)| Point::new(x, y))
            .collect();

        assert!((points[2].x - 1.0).abs() <= 0.001);

        let distances: Vec<_> = points
            .windows(2)
            .map(|window| (window[1] - window[0]).len())
            .collect();

        assert!(distances[0] > 1.5);
        assert!(distances[1] < 0.2);
        assert!(distances[2] < 0.2);
        assert!(distances[3] > 1.5);
    }

    #[test]
    fn lines_and_quads() {
        let mut primitives = Primitives::default();

        primitives.push_line(curve![(-1.0, -2.0), (0.0, 0.0)]);
        primitives.push_quad(curve![(0.0, 0.0), (1.0, 2.0), (2.0, 0.0)]);
        primitives.push_line(curve![(2.0, 0.0), (3.0, -2.0)]);
        primitives.push_line(curve![(3.0, -2.0), (4.0, 2.0)]);
        primitives.push_line(curve![(4.0, 2.0), (5.0, -4.0)]);
        primitives.push_line(curve![(5.0, -4.0), (6.0, 0.0)]);
        primitives.push_quad(curve![(6.0, 0.0), (7.0, 4.0), (8.0, 0.0)]);
        primitives.push_line(curve![(8.0, 0.0), (9.0, -4.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 12);

        assert_eq!(segments.x[0], -1.0);
        assert_eq!(segments.y[0], -2.0);
        assert_eq!(segments.x[4], 3.0);
        assert_eq!(segments.y[4], -2.0);
        assert_eq!(segments.x[5], 4.0);
        assert_eq!(segments.y[5], 2.0);
        assert_eq!(segments.x[6], 5.0);
        assert_eq!(segments.y[6], -4.0);
        assert_eq!(segments.x[11], 9.0);
        assert_eq!(segments.y[11], -4.0);
    }

    #[test]
    fn cubic() {
        let mut primitives = Primitives::default();

        primitives.push_cubic(curve![(0.0, 0.0), (10.0, 6.0), (-2.0, 6.0), (8.0, 0.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 10);

        assert!(segments.x[2] > segments.x[7]);
        assert!(segments.x[3] > segments.x[6]);
        assert!(segments.x[4] > segments.x[5]);

        assert!(segments.y[0] < segments.y[1]);
        assert!(segments.y[1] < segments.y[2]);
        assert!(segments.y[2] < segments.y[3]);
        assert!(segments.y[3] < segments.y[4]);

        assert!(segments.y[5] > segments.y[6]);
        assert!(segments.y[6] > segments.y[7]);
        assert!(segments.y[7] > segments.y[8]);
        assert!(segments.y[8] > segments.y[9]);
    }

    #[test]
    fn rat_cubic_high() {
        let mut primitives = Primitives::default();
        let weight = 10.0;

        primitives.push_cubic([
            WeightedPoint {
                point: Point::new(0.0, 0.0),
                weight: 1.0,
            },
            WeightedPoint {
                point: Point::new(5.0 * weight, 3.0 * weight),
                weight,
            },
            WeightedPoint {
                point: Point::new(-1.0 * weight, 3.0 * weight),
                weight,
            },
            WeightedPoint {
                point: Point::new(4.0, 0.0),
                weight: 1.0,
            },
        ]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 45);
    }

    #[test]
    fn rat_cubic_low() {
        let mut primitives = Primitives::default();
        let weight = 0.5;

        primitives.push_cubic([
            WeightedPoint {
                point: Point::new(0.0, 0.0),
                weight: 1.0,
            },
            WeightedPoint {
                point: Point::new(5.0 * weight, 3.0 * weight),
                weight,
            },
            WeightedPoint {
                point: Point::new(-1.0 * weight, 3.0 * weight),
                weight,
            },
            WeightedPoint {
                point: Point::new(4.0, 0.0),
                weight: 1.0,
            },
        ]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 7);
    }

    #[test]
    fn collinear_cubic() {
        let mut primitives = Primitives::default();

        primitives.push_cubic(curve![(1.0, 0.0), (0.0, 0.0), (3.0, 0.0), (2.0, 0.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 5);

        assert_eq!(segments.x[0], 1.0);
        assert_eq!(segments.y[0], 0.0);

        assert!(segments.x[1] > 0.5);
        assert!(segments.x[1] < 1.0);
        assert_eq!(segments.y[1], 0.0);

        assert!(segments.x[2] > 1.0);
        assert!(segments.x[2] < 2.0);
        assert_eq!(segments.y[2], 0.0);

        assert!(segments.x[3] > 2.0);
        assert!(segments.x[3] < 2.5);
        assert_eq!(segments.y[3], 0.0);

        assert_eq!(segments.x[4], 2.0);
        assert_eq!(segments.y[4], 0.0);
    }

    #[test]
    fn overlapping_control_point_cubic_line() {
        let mut primitives = Primitives::default();

        primitives.push_cubic(curve![(0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0)]);
        primitives.push_cubic(curve![(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]);
        primitives.push_cubic(curve![(1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 2.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 9);

        for x in segments.x.windows(2) {
            assert!(x[0] < x[1]);
        }

        for y in segments.y.windows(2) {
            assert!(y[0] < y[1]);
        }

        for (&x, &y) in segments.x.iter().zip(segments.y.iter()) {
            assert_eq!(x, y);
        }

        assert!((segments.x[0] - 0.0).abs() < 0.01);
        assert!((segments.y[0] - 0.0).abs() < 0.01);
        assert!((segments.x[8] - 2.0).abs() < 0.01);
        assert!((segments.y[8] - 2.0).abs() < 0.01);
    }

    #[test]
    fn ring() {
        let mut primitives = Primitives::default();

        primitives.push_cubic(curve![(0.0, 2.0), (2.0, 2.0), (2.0, 2.0), (2.0, 0.0)]);
        primitives.push_cubic(curve![(2.0, 0.0), (2.0, -2.0), (2.0, -2.0), (0.0, -2.0)]);
        primitives.push_cubic(curve![(0.0, -2.0), (-2.0, -2.0), (-2.0, -2.0), (-2.0, 0.0)]);
        primitives.push_cubic(curve![(-2.0, 0.0), (-2.0, 2.0), (-2.0, 2.0), (0.0, 2.0)]);

        primitives.push_contour(Contour);

        primitives.push_cubic(curve![(0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)]);
        primitives.push_cubic(curve![(-1.0, 0.0), (-1.0, -1.0), (-1.0, -1.0), (0.0, -1.0)]);
        primitives.push_cubic(curve![(0.0, -1.0), (1.0, -1.0), (1.0, -1.0), (1.0, 0.0)]);
        primitives.push_cubic(curve![(1.0, 0.0), (1.0, 1.0), (1.0, 1.0), (0.0, 1.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.start_new_contour.len(), 30);

        assert_eq!(
            segments
                .start_new_contour
                .iter()
                .filter(|&&start_new_contour| start_new_contour)
                .count(),
            2
        );

        assert!(segments.start_new_contour[16]);
        assert!(segments.start_new_contour[29]);
    }

    #[test]
    fn ring_overlapping_start() {
        let mut primitives = Primitives::default();

        primitives.push_cubic(curve![(0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)]);
        primitives.push_cubic(curve![(-1.0, 0.0), (-1.0, -1.0), (-1.0, -1.0), (0.0, -1.0)]);
        primitives.push_cubic(curve![(0.0, -1.0), (1.0, -1.0), (1.0, -1.0), (1.0, 0.0)]);
        primitives.push_cubic(curve![(1.0, 0.0), (1.0, 1.0), (1.0, 1.0), (0.0, 1.0)]);

        primitives.push_contour(Contour);

        primitives.push_cubic(curve![(0.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 2.0)]);
        primitives.push_cubic(curve![(1.0, 2.0), (1.0, 3.0), (1.0, 3.0), (0.0, 3.0)]);
        primitives.push_cubic(curve![(0.0, 3.0), (-1.0, 3.0), (-1.0, 3.0), (-1.0, 2.0)]);
        primitives.push_cubic(curve![(-1.0, 2.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]);

        let segments = primitives.into_segments();

        assert_eq!(segments.start_new_contour.len(), 26);

        assert_eq!(
            segments
                .start_new_contour
                .iter()
                .filter(|&&start_new_contour| start_new_contour)
                .count(),
            2
        );

        assert!(segments.start_new_contour[12]);
        assert!(segments.start_new_contour[25]);
    }

    #[test]
    fn circle() {
        let mut primitives = Primitives::default();
        let radius = 50.0;
        let weight = 2.0f32.sqrt() / 2.0;

        primitives.push_quad([
            WeightedPoint {
                point: Point::new(radius, 0.0),
                weight: 1.0,
            },
            WeightedPoint {
                point: Point::new(0.0, 0.0),
                weight,
            },
            WeightedPoint {
                point: Point::new(0.0, radius),
                weight: 1.0,
            },
        ]);
        primitives.push_quad([
            WeightedPoint {
                point: Point::new(0.0, radius),
                weight: 1.0,
            },
            WeightedPoint {
                point: Point::new(0.0, 2.0 * radius * weight),
                weight,
            },
            WeightedPoint {
                point: Point::new(radius, 2.0 * radius),
                weight: 1.0,
            },
        ]);
        primitives.push_quad([
            WeightedPoint {
                point: Point::new(radius, 2.0 * radius),
                weight: 1.0,
            },
            WeightedPoint {
                point: Point::new(2.0 * radius * weight, 2.0 * radius * weight),
                weight,
            },
            WeightedPoint {
                point: Point::new(2.0 * radius, radius),
                weight: 1.0,
            },
        ]);
        primitives.push_quad([
            WeightedPoint {
                point: Point::new(2.0 * radius, radius),
                weight: 1.0,
            },
            WeightedPoint {
                point: Point::new(2.0 * radius * weight, 0.0),
                weight,
            },
            WeightedPoint {
                point: Point::new(radius, 0.0),
                weight: 1.0,
            },
        ]);

        let segments = primitives.into_segments();

        assert_eq!(segments.x.len(), 66);

        let points: Vec<_> = segments
            .x
            .iter()
            .zip(segments.y.iter())
            .map(|(&x, &y)| Point::new(x, y))
            .collect();

        let max_distance = points
            .windows(2)
            .map(|window| (window[1] - window[0]).len())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!(max_distance < 5.0);
    }

    #[test]
    fn transform_path() {
        let weight = 2.0f32.sqrt() / 2.0;
        let radius = 10.0;

        let mut builder = PathBuilder::new();

        builder.move_to(Point::new(radius, 0.0));
        builder.rat_quad_to(
            Point::new(radius, -radius),
            Point::new(0.0, -radius),
            weight,
        );
        builder.rat_quad_to(
            Point::new(-radius, -radius),
            Point::new(-radius, 0.0),
            weight,
        );
        builder.rat_quad_to(Point::new(-radius, radius), Point::new(0.0, radius), weight);
        builder.rat_quad_to(Point::new(radius, radius), Point::new(radius, 0.0), weight);

        let path = builder.build();

        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut ids = Vec::new();

        let id0 = GeomId::default();
        let id1 = id0.next();

        path.push_segments_to(&mut x, &mut y, id1, &mut ids);

        let orig_len = x.len();

        assert_eq!(ids[..ids.len() - 1], vec![Some(id1); ids.len() - 1]);
        assert_eq!(ids[ids.len() - 1], None);

        for (&x, &y) in x.iter().zip(y.iter()) {
            assert!((Point::new(x, y).len() - radius).abs() <= 0.1);
        }

        let dx = 5.0;
        let dy = 20.0;
        let translated_path = path.transform(&[1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0]);

        x.clear();
        y.clear();

        translated_path.push_segments_to(&mut x, &mut y, id0, &mut ids);

        for (&x, &y) in x.iter().zip(y.iter()) {
            assert!((Point::new(x - dx, y - dy).len() - radius).abs() <= 0.1);
        }

        let s = 2.0;
        let scaled_path = path.transform(&[s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 1.0]);

        x.clear();
        y.clear();

        scaled_path.push_segments_to(&mut x, &mut y, id0, &mut ids);

        for (&x, &y) in x.iter().zip(y.iter()) {
            assert!((Point::new(x, y).len() - s * radius).abs() <= 0.1);
        }

        let scaled_len = x.len();

        assert!(scaled_len > orig_len);
    }

    #[test]
    fn perspective_transform_path() {
        let weight = 2.0f32.sqrt() / 2.0;
        let radius = 10.0;
        let translation = 1000.0;

        let mut builder = PathBuilder::new();

        builder.move_to(Point::new(radius + translation, 0.0));
        builder.rat_quad_to(
            Point::new(radius + translation, -radius),
            Point::new(translation, -radius),
            weight,
        );
        builder.rat_quad_to(
            Point::new(-radius + translation, -radius),
            Point::new(-radius + translation, 0.0),
            weight,
        );
        builder.rat_quad_to(
            Point::new(-radius + translation, radius),
            Point::new(translation, radius),
            weight,
        );
        builder.rat_quad_to(
            Point::new(radius + translation, radius),
            Point::new(radius + translation, 0.0),
            weight,
        );

        let path = builder
            .build()
            .transform(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.001, 0.0, 1.0]);

        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut ids = Vec::new();

        path.push_segments_to(&mut x, &mut y, GeomId::default(), &mut ids);

        let mut points: Vec<_> = x
            .iter()
            .zip(y.iter())
            .map(|(&x, &y)| Point::new(x, y))
            .collect();
        points.pop(); // Remove duplicate point.

        let half_len = points.len() / 2;

        let min = (0..half_len)
            .into_iter()
            .map(|i| (points[i] - points[(i + half_len) % points.len()]).len())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max = (0..half_len)
            .into_iter()
            .map(|i| (points[i] - points[(i + half_len) % points.len()]).len())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!((min - radius / 2.0).abs() <= 0.2);
        assert!((max - radius).abs() <= 0.2);
    }
}
