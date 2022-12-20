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

use std::{cell::Cell, num::NonZeroU64};

use rayon::prelude::*;

use crate::{
    composition::InnerLayer,
    consts,
    math::AffineTransform,
    utils::{ExtendTuple10, ExtendTuple3},
    Path,
};

const MIN_LEN: usize = 1_024;

fn integers_between(a: f32, b: f32) -> u32 {
    let min = a.min(b);
    let max = a.max(b);

    0.max((max.ceil() - min.floor() - 1.0) as u32)
}

/// This returns a value that is similar (but not identical!) to the Manhattan Distance between
/// two segment endpoints. We'll call it the "Manhattan Block Distance". You can think of this as
/// the number of city blocks which are touched by a piecewise horizontal/vertical path between the
/// points.
///
/// # Examples
///
/// ```text
/// (0.5, 0.5) -> (0.3, 0.7)
/// ```
/// In this case, there are no integers between `0.5` and `0.3`, nor between `0.5` and `0.7`: both
/// points reside on the same "city block", so the length is `1`.
/// NOTE: by this metric, a degenerate segment `(0.5, 0.5) -> (0.5, 0.5)` also has length `1`.
///
/// ```text
/// (0.9, 1.9) -> (1.1, 0.1)
/// ```
/// The x-coordinates `0.9` and `1.1` are on adjacent "city blocks", which contributes `1` to
/// the segment length (even though the distance between them is only `0.2`). Similarly, the
/// y-coordinates `1.9` and `0.1` are on adjacent "city blocks", which also contributes `1` to the
/// segment length (even though the distance between them is `1.8`, almost `2`!).  This gives a total
/// segment length of `3`:
///   - `1` for the starting block
///   - `1` to move horizontally from `0.9` to `1.1`
///   - `1` to move vertically from `1.9` to `0.1`
fn manhattan_segment_length(p0x: f32, p1x: f32, p0y: f32, p1y: f32) -> u32 {
    integers_between(p0x, p1x) + integers_between(p0y, p1y) + 1
}

fn prefix_sum(vals: &mut [u32]) -> u32 {
    let mut sum = 0;
    for val in vals {
        sum += *val;
        *val = sum;
    }

    sum
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct GeomId(NonZeroU64);

impl GeomId {
    #[cfg(test)]
    pub fn get(self) -> u64 {
        self.0.get() - 1
    }

    #[inline]
    pub fn next(self) -> Self {
        Self(
            NonZeroU64::new(
                self.0
                    .get()
                    .checked_add(1)
                    .expect("id should never reach u64::MAX"),
            )
            .unwrap(),
        )
    }
}

impl Default for GeomId {
    #[inline]
    fn default() -> Self {
        Self(NonZeroU64::new(1).unwrap())
    }
}

#[derive(Debug, Default)]
pub struct SegmentBuffer {
    view: SegmentBufferView,
    cached_len: Cell<usize>,
    cached_until: Cell<usize>,
}

impl SegmentBuffer {
    // This type is only used in forma where it does not need `is_empty`.
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        if self.view.ids.len() <= self.cached_until.get() {
            self.cached_len.get()
        } else {
            let new_len = self.cached_len.get()
                + self.view.ids[self.cached_until.get()..]
                    .iter()
                    .filter(|id| id.is_some())
                    .count();

            self.cached_len.set(new_len);
            self.cached_until.set(self.view.ids.len());

            new_len
        }
    }

    #[inline]
    pub fn push_path(&mut self, id: GeomId, path: &Path) {
        path.push_segments_to(&mut self.view.x, &mut self.view.y, id, &mut self.view.ids);

        self.view.ids.resize(
            self.view.x.len().checked_sub(1).unwrap_or_default(),
            Some(id),
        );

        if self
            .view
            .ids
            .last()
            .map(Option::is_some)
            .unwrap_or_default()
        {
            self.view.ids.push(None);
        }
    }

    #[cfg(test)]
    pub fn push(&mut self, id: GeomId, segment: [crate::math::Point; 2]) {
        let new_point_needed =
            if let (Some(&x), Some(&y)) = (self.view.x.last(), self.view.y.last()) {
                let last_point = crate::math::Point { x, y };

                last_point != segment[0]
            } else {
                true
            };

        if new_point_needed {
            self.view.x.push(segment[0].x);
            self.view.y.push(segment[0].y);
        }

        self.view.x.push(segment[1].x);
        self.view.y.push(segment[1].y);

        if self.view.ids.len() >= 2 {
            match self.view.ids[self.view.ids.len() - 2] {
                Some(last_id) if last_id != id => {
                    self.view.ids.push(Some(id));
                    self.view.ids.push(None);
                }
                _ => {
                    self.view.ids.pop();
                    self.view.ids.push(Some(id));
                    self.view.ids.push(None);
                }
            }
        } else {
            self.view.ids.push(Some(id));
            self.view.ids.push(None);
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(GeomId) -> bool,
    {
        let len = self.view.x.len();
        let mut del = 0;
        let mut prev_id = None;

        for i in 0..len {
            // `None` IDs will always belong to the previous ID.
            // Thus, if an ID is removed here, its None will be removed as well.

            let id = self.view.ids[i];
            let should_retain = id
                .or(prev_id)
                .map(&mut f)
                .expect("consecutive None values should not exist in ids");
            prev_id = id;

            if !should_retain {
                del += 1;
                continue;
            }

            if del > 0 {
                self.view.x.swap(i - del, i);
                self.view.y.swap(i - del, i);
                self.view.ids.swap(i - del, i);
            }
        }

        if del > 0 {
            self.view.x.truncate(len - del);
            self.view.y.truncate(len - del);
            self.view.ids.truncate(len - del);
        }
    }

    pub fn fill_cpu_view<F>(mut self, layers: F) -> SegmentBufferView
    where
        F: Fn(GeomId) -> Option<InnerLayer> + Send + Sync,
    {
        let ps_layers = self.view.x.par_windows(2).with_min_len(MIN_LEN).zip_eq(
            self.view.y.par_windows(2).with_min_len(MIN_LEN).zip_eq(
                self.view.ids[..self.view.ids.len().checked_sub(1).unwrap_or_default()]
                    .par_iter()
                    .with_min_len(MIN_LEN),
            ),
        );
        let par_iter = ps_layers.map(|(xs, (ys, &id))| {
            let empty_line = Default::default();

            let p0x = xs[0];
            let p0y = ys[0];
            let p1x = xs[1];
            let p1y = ys[1];

            if id.is_none() {
                // Returns a length of 0 so that the line segments between two
                // polygonal chains generate no pixel segments.
                return empty_line;
            }

            let layer = match id.and_then(&layers) {
                Some(layer) => layer,
                None => return empty_line,
            };

            if let InnerLayer {
                is_enabled: false, ..
            } = layer
            {
                return empty_line;
            }

            let order = match layer.order {
                Some(order) => order.as_u32(),
                None => return empty_line,
            };

            fn transform_point(point: (f32, f32), transform: &AffineTransform) -> (f32, f32) {
                (
                    transform
                        .ux
                        .mul_add(point.0, transform.vx.mul_add(point.1, transform.tx)),
                    transform
                        .uy
                        .mul_add(point.0, transform.vy.mul_add(point.1, transform.ty)),
                )
            }

            let transform = layer
                .affine_transform
                .as_ref()
                .map(|transform| &transform.0);

            let (p0x, p0y, p1x, p1y) = if let Some(transform) = transform {
                let (p0x, p0y) = transform_point((p0x, p0y), transform);
                let (p1x, p1y) = transform_point((p1x, p1y), transform);

                (p0x, p0y, p1x, p1y)
            } else {
                (p0x, p0y, p1x, p1y)
            };

            if p0y == p1y {
                return empty_line;
            }

            let dx = p1x - p0x;
            let dy = p1y - p0y;
            let dx_recip = dx.recip();
            let dy_recip = dy.recip();

            // We compute the two line parameters that correspond to the first horizontal and first
            // vertical intersections with the pixel grid from point `p0` towards `p1`.
            let t_offset_x = if dx != 0.0 {
                ((p0x.ceil() - p0x) * dx_recip).max((p0x.floor() - p0x) * dx_recip)
            } else {
                0.0
            };
            let t_offset_y = if dy != 0.0 {
                ((p0y.ceil() - p0y) * dy_recip).max((p0y.floor() - p0y) * dy_recip)
            } else {
                0.0
            };

            let a = dx_recip.abs();
            let b = dy_recip.abs();
            let c = t_offset_x;
            let d = t_offset_y;

            let length = manhattan_segment_length(p0x, p1x, p0y, p1y);

            // Converting to sub-pixel space on the fly by multiplying with `PIXEL_WIDTH`.
            (
                order,
                p0x * consts::PIXEL_WIDTH as f32,
                p0y * consts::PIXEL_WIDTH as f32,
                dx * consts::PIXEL_WIDTH as f32,
                dy * consts::PIXEL_WIDTH as f32,
                a,
                b,
                c,
                d,
                length,
            )
        });

        ExtendTuple10::new((
            &mut self.view.orders,
            &mut self.view.x0,
            &mut self.view.y0,
            &mut self.view.dx,
            &mut self.view.dy,
            &mut self.view.a,
            &mut self.view.b,
            &mut self.view.c,
            &mut self.view.d,
            &mut self.view.lengths,
        ))
        .par_extend(par_iter);

        prefix_sum(&mut self.view.lengths);

        self.view
    }

    pub fn fill_gpu_view<F>(mut self, layers: F) -> SegmentBufferView
    where
        F: Fn(GeomId) -> Option<InnerLayer> + Send + Sync,
    {
        fn transform_point(point: (f32, f32), transform: &AffineTransform) -> (f32, f32) {
            (
                transform
                    .ux
                    .mul_add(point.0, transform.vx.mul_add(point.1, transform.tx)),
                transform
                    .uy
                    .mul_add(point.0, transform.vy.mul_add(point.1, transform.ty)),
            )
        }

        if !self.view.ids.is_empty() {
            let point = match self.view.ids[0].and_then(&layers) {
                None
                | Some(
                    InnerLayer {
                        is_enabled: false, ..
                    }
                    | InnerLayer { order: None, .. },
                ) => Default::default(),
                Some(InnerLayer {
                    affine_transform: None,
                    ..
                }) => [self.view.x[0], self.view.y[0]],
                Some(InnerLayer {
                    affine_transform: Some(transform),
                    ..
                }) => {
                    let (x, y) = transform_point((self.view.x[0], self.view.y[0]), &transform.0);
                    [x, y]
                }
            };
            self.view.points.push(point);
        }

        let ps_layers = self.view.x.par_windows(2).with_min_len(MIN_LEN).zip_eq(
            self.view
                .y
                .par_windows(2)
                .with_min_len(MIN_LEN)
                .zip_eq(self.view.ids.par_windows(2).with_min_len(MIN_LEN)),
        );
        let par_iter = ps_layers.map(|(xs, (ys, ids))| {
            const NONE: u32 = u32::MAX;
            let p0x = xs[0];
            let p0y = ys[0];
            let (p1x, p1y) = match ids[0].or(ids[1]).and_then(&layers) {
                None
                | Some(
                    InnerLayer {
                        is_enabled: false, ..
                    }
                    | InnerLayer { order: None, .. },
                ) => (0.0, 0.0),
                Some(InnerLayer {
                    affine_transform: None,
                    ..
                }) => (xs[1], ys[1]),
                Some(InnerLayer {
                    affine_transform: Some(transform),
                    ..
                }) => transform_point((xs[1], ys[1]), &transform.0),
            };

            let layer = match ids[0].and_then(&layers) {
                Some(layer) => layer,
                // Points at then end of segment chain have to be transformed for the compute shader.
                None => return (0, [p1x, p1y], NONE),
            };

            if let InnerLayer {
                is_enabled: false, ..
            } = layer
            {
                return (0, [p1x, p1y], NONE);
            }

            let order = match layer.order {
                Some(order) => order.as_u32(),
                None => return (0, [p1x, p1y], NONE),
            };

            let transform = layer
                .affine_transform
                .as_ref()
                .map(|transform| &transform.0);

            let (p0x, p0y) = if let Some(transform) = transform {
                transform_point((p0x, p0y), transform)
            } else {
                (p0x, p0y)
            };

            if p0y == p1y {
                return (0, [p1x, p1y], NONE);
            }
            let length = integers_between(p0x, p1x) + integers_between(p0y, p1y) + 1;

            (length, [p1x, p1y], order)
        });

        ExtendTuple3::new((
            &mut self.view.lengths,
            &mut self.view.points,
            &mut self.view.orders,
        ))
        .par_extend(par_iter);

        prefix_sum(&mut self.view.lengths);

        self.view
    }
}

// `x`, `y` and `ids` have the same size and encode polygonal chains.
// In `ids`, `None` identifies the last element of a chain.
//
// `points` and `lengths` have the same size and encode pixel segment
// generators.
#[derive(Debug, Default)]
pub struct SegmentBufferView {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub ids: Vec<Option<GeomId>>,
    pub orders: Vec<u32>,
    /// Lines' p0 x-coordinates multiplied by `PIXEL_WIDTH`.
    pub x0: Vec<f32>,
    /// Lines' p0 y-coordinates multiplied by `PIXEL_WIDTH`.
    pub y0: Vec<f32>,
    /// Line slopes' x-coordinates multiplied by `PIXEL_WIDTH`.
    pub dx: Vec<f32>,
    /// Line slopes' y-coordinates multiplied by `PIXEL_WIDTH`.
    pub dy: Vec<f32>,
    /// x-coordinate arithmetic progression coefficients. (`a` from `a * t + c`)
    pub a: Vec<f32>,
    /// y-coordinate arithmetic progression coefficients. (`b` from `b * t + d`)
    pub b: Vec<f32>,
    /// x-coordinate arithmetic progression coefficients. (`c` from `a * t + c`)
    pub c: Vec<f32>,
    /// y-coordinate arithmetic progression coefficients. (`d` from `b * t + d`)
    pub d: Vec<f32>,
    pub points: Vec<[f32; 2]>,
    /// Lengths in number of pixel segments that a line would produce. See
    /// [`manhattan_segment_length`].
    pub lengths: Vec<u32>,
}

impl SegmentBufferView {
    #[inline]
    pub fn recycle(mut self) -> SegmentBuffer {
        self.orders.clear();
        self.x0.clear();
        self.y0.clear();
        self.dx.clear();
        self.dy.clear();
        self.a.clear();
        self.b.clear();
        self.c.clear();
        self.d.clear();
        self.lengths.clear();
        self.points.clear();

        SegmentBuffer {
            view: self,
            ..Default::default()
        }
    }

    pub fn len(&self) -> usize {
        self.lengths.last().copied().unwrap_or_default() as usize
    }

    pub fn inner_len(&self) -> usize {
        self.x.len() - 1
    }
}
