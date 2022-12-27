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

use std::{
    borrow::Cow,
    cell::{Cell, RefCell, RefMut},
    collections::BTreeMap,
    convert::TryInto,
    mem,
    ops::ControlFlow,
    slice::ChunksExactMut,
};

use rayon::prelude::*;

use crate::{
    consts,
    styling::{BlendMode, Color, Fill, FillRule, Func, Props, Style},
    utils::simd::{f32x4, f32x8, i16x16, i32x8, i8x16, u32x4, u32x8, u8x32, Simd},
};

use self::layer_workbench::{OptimizerTileWriteOp, TileWriteOp};

use super::{
    buffer::layout::{Flusher, Layout, Slice, TileFill},
    Channel, PixelSegment, Rect,
};

mod layer_workbench;
#[macro_use]
mod styling;

use layer_workbench::{Context, LayerPainter, LayerWorkbench};

const PIXEL_AREA: usize = consts::PIXEL_WIDTH * consts::PIXEL_WIDTH;
const PIXEL_DOUBLE_AREA: usize = 2 * PIXEL_AREA;

// From Hacker's Delight, p. 378-380. 2 ^ 23 as f32.
const C23: u32 = 0x4B00_0000;

macro_rules! cols {
    ( & $array:expr, $x0:expr, $x1:expr ) => {{
        fn size_of_el<T: Simd>(_: impl AsRef<[T]>) -> usize {
            T::LANES
        }

        let from = $x0 * crate::consts::cpu::TILE_HEIGHT / size_of_el(&$array);
        let to = $x1 * crate::consts::cpu::TILE_HEIGHT / size_of_el(&$array);

        &$array[from..to]
    }};

    ( & mut $array:expr, $x0:expr, $x1:expr ) => {{
        fn size_of_el<T: Simd>(_: impl AsRef<[T]>) -> usize {
            T::LANES
        }

        let from = $x0 * crate::consts::cpu::TILE_HEIGHT / size_of_el(&$array);
        let to = $x1 * crate::consts::cpu::TILE_HEIGHT / size_of_el(&$array);

        &mut $array[from..to]
    }};
}

#[inline]
fn doubled_area_to_coverage(doubled_area: i32x8, fill_rule: FillRule) -> f32x8 {
    match fill_rule {
        FillRule::NonZero => {
            let doubled_area: f32x8 = doubled_area.into();
            (doubled_area * f32x8::splat((PIXEL_DOUBLE_AREA as f32).recip()))
                .abs()
                .clamp(f32x8::splat(0.0), f32x8::splat(1.0))
        }
        FillRule::EvenOdd => {
            let doubled_area: f32x8 = (i32x8::splat(PIXEL_DOUBLE_AREA as i32)
                - ((doubled_area & i32x8::splat(2 * PIXEL_DOUBLE_AREA as i32 - 1))
                    - i32x8::splat(PIXEL_DOUBLE_AREA as i32))
                .abs())
            .into();
            doubled_area * f32x8::splat((PIXEL_DOUBLE_AREA as f32).recip())
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[inline]
fn linear_to_srgb_approx_simdx8(l: f32x8) -> f32x8 {
    let a = f32x8::splat(0.201_017_72f32);
    let b = f32x8::splat(-0.512_801_47f32);
    let c = f32x8::splat(1.344_401f32);
    let d = f32x8::splat(-0.030_656_587f32);

    let s = l.sqrt();
    let s2 = l;
    let s3 = s2 * s;

    let m = l * f32x8::splat(12.92);
    let n = a.mul_add(s3, b.mul_add(s2, c.mul_add(s, d)));

    m.select(n, l.le(f32x8::splat(0.003_130_8)))
}

#[allow(clippy::many_single_char_names)]
#[inline]
fn linear_to_srgb_approx_simdx4(l: f32x4) -> f32x4 {
    let a = f32x4::splat(0.201_017_72f32);
    let b = f32x4::splat(-0.512_801_47f32);
    let c = f32x4::splat(1.344_401f32);
    let d = f32x4::splat(-0.030_656_587f32);

    let s = l.sqrt();
    let s2 = l;
    let s3 = s2 * s;

    let m = l * f32x4::splat(12.92);
    let n = a.mul_add(s3, b.mul_add(s2, c.mul_add(s, d)));

    m.select(n, l.le(f32x4::splat(0.003_130_8)))
}

// From Hacker's Delight, p. 378-380.

#[inline]
fn to_u32x8(val: f32x8) -> u32x8 {
    let max = f32x8::splat(f32::from(u8::MAX));
    let c23 = u32x8::splat(C23);

    let scaled = (val * max).clamp(f32x8::splat(0.0), max);
    let val = scaled + f32x8::from_bits(c23);

    val.to_bits()
}

#[inline]
fn to_u32x4(val: f32x4) -> u32x4 {
    let max = f32x4::splat(f32::from(u8::MAX));
    let c23 = u32x4::splat(C23);

    let scaled = (val * max).clamp(f32x4::splat(0.0), max);
    let val = scaled + f32x4::from_bits(c23);

    val.to_bits()
}

#[inline]
fn to_srgb_bytes(color: [f32; 4]) -> [u8; 4] {
    let linear = f32x4::new([color[0], color[1], color[2], 0.0]);
    let srgb = to_u32x4(linear_to_srgb_approx_simdx4(linear).set::<3>(color[3]));

    srgb.into()
}

pub trait LayerProps: Send + Sync {
    fn get(&self, layer_id: u32) -> Cow<'_, Props>;
    fn is_unchanged(&self, layer_id: u32) -> bool;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Cover {
    // Proportion of the pixel area covered for each pixel row of a tile row.
    // 0 is none, 16 is full coverage. Value above 16 happens when paths self-overlap.
    covers: [i8x16; consts::cpu::TILE_HEIGHT / i8x16::LANES],
}

impl Cover {
    pub fn as_slice_mut(&mut self) -> &mut [i8; consts::cpu::TILE_HEIGHT] {
        unsafe { mem::transmute(&mut self.covers) }
    }

    pub fn add_cover_to(&self, covers: &mut [i8x16]) {
        for (i, &cover) in self.covers.iter().enumerate() {
            covers[i] += cover;
        }
    }

    pub fn is_empty(&self, fill_rule: FillRule) -> bool {
        match fill_rule {
            FillRule::NonZero => self
                .covers
                .iter()
                .all(|&cover| cover.eq(i8x16::splat(0)).all()),
            FillRule::EvenOdd => self
                .covers
                .iter()
                .all(|&cover| (cover.abs() & i8x16::splat(31)).eq(i8x16::splat(0)).all()),
        }
    }

    pub fn is_full(&self, fill_rule: FillRule) -> bool {
        match fill_rule {
            FillRule::NonZero => self.covers.iter().all(|&cover| {
                cover
                    .abs()
                    .eq(i8x16::splat(consts::PIXEL_WIDTH as i8))
                    .all()
            }),
            FillRule::EvenOdd => self.covers.iter().any(|&cover| {
                (cover.abs() & i8x16::splat(0b1_1111))
                    .eq(i8x16::splat(0b1_0000))
                    .all()
            }),
        }
    }
}

impl PartialEq for Cover {
    fn eq(&self, other: &Self) -> bool {
        self.covers
            .iter()
            .zip(other.covers.iter())
            .all(|(t, o)| t.eq(*o).all())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CoverCarry {
    cover: Cover,
    layer_id: u32,
}

#[derive(Debug)]
pub(crate) struct Painter {
    doubled_areas: [i16x16; consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / i16x16::LANES],
    covers: [i8x16; (consts::cpu::TILE_WIDTH + 1) * consts::cpu::TILE_HEIGHT / i8x16::LANES],
    clip: Option<(
        [f32x8; consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
        u32,
    )>,
    red: [f32x8; consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
    green: [f32x8; consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
    blue: [f32x8; consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
    alpha: [f32x8; consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
    srgb: [u8x32; consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT * 4 / u8x32::LANES],
}

impl LayerPainter for Painter {
    fn clear_cells(&mut self) {
        self.doubled_areas
            .iter_mut()
            .for_each(|doubled_area| *doubled_area = i16x16::splat(0));
        self.covers
            .iter_mut()
            .for_each(|cover| *cover = i8x16::splat(0));
    }

    fn acc_segment(
        &mut self,
        segment: PixelSegment<{ consts::cpu::TILE_WIDTH }, { consts::cpu::TILE_HEIGHT }>,
    ) {
        let x = segment.local_x() as usize;
        let y = segment.local_y() as usize;

        let doubled_areas: &mut [i16; consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT] =
            unsafe { mem::transmute(&mut self.doubled_areas) };
        let covers: &mut [i8; (consts::cpu::TILE_WIDTH + 1) * consts::cpu::TILE_HEIGHT] =
            unsafe { mem::transmute(&mut self.covers) };

        doubled_areas[x * consts::cpu::TILE_HEIGHT + y] += segment.double_area();
        covers[(x + 1) * consts::cpu::TILE_HEIGHT + y] += segment.cover();
    }

    fn acc_cover(&mut self, cover: Cover) {
        cover.add_cover_to(&mut self.covers);
    }

    fn clear(&mut self, color: Color) {
        self.red.iter_mut().for_each(|r| *r = f32x8::splat(color.r));
        self.green
            .iter_mut()
            .for_each(|g| *g = f32x8::splat(color.g));
        self.blue
            .iter_mut()
            .for_each(|b| *b = f32x8::splat(color.b));
        self.alpha
            .iter_mut()
            .for_each(|alpha| *alpha = f32x8::splat(color.a));
    }

    fn paint_layer(
        &mut self,
        tile_x: usize,
        tile_y: usize,
        layer_id: u32,
        props: &Props,
        apply_clip: bool,
    ) -> Cover {
        let mut doubled_areas = [i32x8::splat(0); consts::cpu::TILE_HEIGHT / i32x8::LANES];
        let mut covers = [i8x16::splat(0); consts::cpu::TILE_HEIGHT / i8x16::LANES];
        let mut coverages = [f32x8::splat(0.0); consts::cpu::TILE_HEIGHT / f32x8::LANES];

        if let Some((_, last_layer)) = self.clip {
            if last_layer < layer_id {
                self.clip = None;
            }
        }

        for x in 0..=consts::cpu::TILE_WIDTH {
            if x != 0 {
                self.compute_doubled_areas(x - 1, &covers, &mut doubled_areas);

                for y in 0..coverages.len() {
                    coverages[y] = doubled_area_to_coverage(doubled_areas[y], props.fill_rule);

                    match &props.func {
                        Func::Draw(style) => {
                            if coverages[y].eq(f32x8::splat(0.0)).all() {
                                continue;
                            }

                            if apply_clip && self.clip.is_none() {
                                continue;
                            }

                            let fill = Self::fill_at(
                                x - 1 + tile_x * consts::cpu::TILE_WIDTH,
                                y * f32x8::LANES + tile_y * consts::cpu::TILE_HEIGHT,
                                style,
                            );

                            self.blend_at(x - 1, y, coverages, apply_clip, fill, style.blend_mode);
                        }
                        Func::Clip(layers) => {
                            self.clip_at(x - 1, y, coverages, layer_id + *layers as u32)
                        }
                    }
                }
            }

            let column = cols!(&self.covers, x, x + 1);
            for y in 0..column.len() {
                covers[y] += column[y];
            }
        }

        Cover { covers }
    }
}

impl Painter {
    pub fn new() -> Self {
        Self {
            doubled_areas: [i16x16::splat(0);
                consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / i16x16::LANES],
            covers: [i8x16::splat(0);
                (consts::cpu::TILE_WIDTH + 1) * consts::cpu::TILE_HEIGHT / i8x16::LANES],
            clip: None,
            red: [f32x8::splat(0.0);
                consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
            green: [f32x8::splat(0.0);
                consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
            blue: [f32x8::splat(0.0);
                consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
            alpha: [f32x8::splat(1.0);
                consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
            srgb: [u8x32::splat(0);
                consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT * 4 / u8x32::LANES],
        }
    }

    #[inline]
    fn fill_at(x: usize, y: usize, style: &Style) -> [f32x8; 4] {
        match &style.fill {
            Fill::Solid(color) => {
                let Color { r, g, b, a } = *color;
                [
                    f32x8::splat(r),
                    f32x8::splat(g),
                    f32x8::splat(b),
                    f32x8::splat(a),
                ]
            }
            Fill::Gradient(gradient) => gradient.color_at(x as f32, y as f32),
            Fill::Texture(texture) => texture.color_at(x as f32, y as f32),
        }
    }

    fn compute_doubled_areas(
        &self,
        x: usize,
        covers: &[i8x16; consts::cpu::TILE_HEIGHT / i8x16::LANES],
        doubled_areas: &mut [i32x8; consts::cpu::TILE_HEIGHT / i32x8::LANES],
    ) {
        let column = cols!(&self.doubled_areas, x, x + 1);
        for y in 0..covers.len() {
            let covers: [i32x8; 2] = covers[y].into();
            let column: [i32x8; 2] = column[y].into();

            for yy in 0..2 {
                doubled_areas[2 * y + yy] =
                    i32x8::splat(consts::PIXEL_DOUBLE_WIDTH as i32) * covers[yy] + column[yy];
            }
        }
    }

    fn blend_at(
        &mut self,
        x: usize,
        y: usize,
        coverages: [f32x8; consts::cpu::TILE_HEIGHT / f32x8::LANES],
        is_clipped: bool,
        fill: [f32x8; 4],
        blend_mode: BlendMode,
    ) {
        let dst_r = &mut cols!(&mut self.red, x, x + 1)[y];
        let dst_g = &mut cols!(&mut self.green, x, x + 1)[y];
        let dst_b = &mut cols!(&mut self.blue, x, x + 1)[y];
        let dst_a = &mut cols!(&mut self.alpha, x, x + 1)[y];

        let src_r = fill[0];
        let src_g = fill[1];
        let src_b = fill[2];
        let mut src_a = fill[3] * coverages[y];

        if is_clipped {
            if let Some((mask, _)) = self.clip {
                src_a *= cols!(&mask, x, x + 1)[y];
            }
        }

        let [blended_r, blended_g, blended_b] =
            blend_function!(blend_mode, *dst_r, *dst_g, *dst_b, src_r, src_g, src_b);

        let inv_dst_a = f32x8::splat(1.0) - *dst_a;
        let inv_dst_a_src_a = inv_dst_a * src_a;
        let inv_src_a = f32x8::splat(1.0) - src_a;
        let dst_a_src_a = *dst_a * src_a;

        let current_r = src_r.mul_add(inv_dst_a_src_a, blended_r * dst_a_src_a);
        let current_g = src_g.mul_add(inv_dst_a_src_a, blended_g * dst_a_src_a);
        let current_b = src_b.mul_add(inv_dst_a_src_a, blended_b * dst_a_src_a);

        *dst_r = dst_r.mul_add(inv_src_a, current_r);
        *dst_g = dst_g.mul_add(inv_src_a, current_g);
        *dst_b = dst_b.mul_add(inv_src_a, current_b);
        *dst_a = dst_a.mul_add(inv_src_a, src_a);
    }

    fn clip_at(
        &mut self,
        x: usize,
        y: usize,
        coverages: [f32x8; consts::cpu::TILE_HEIGHT / f32x8::LANES],
        last_layer_id: u32,
    ) {
        let clip = self.clip.get_or_insert_with(|| {
            (
                [f32x8::splat(0.0);
                    consts::cpu::TILE_WIDTH * consts::cpu::TILE_HEIGHT / f32x8::LANES],
                last_layer_id,
            )
        });
        cols!(&mut clip.0, x, x + 1)[y] = coverages[y];
    }

    fn compute_srgb(&mut self, channels: [Channel; 4]) {
        for ((((&red, &green), &blue), &alpha), srgb) in self
            .red
            .iter()
            .zip(self.green.iter())
            .zip(self.blue.iter())
            .zip(self.alpha.iter())
            .zip(self.srgb.iter_mut())
        {
            let red = linear_to_srgb_approx_simdx8(red);
            let green = linear_to_srgb_approx_simdx8(green);
            let blue = linear_to_srgb_approx_simdx8(blue);

            let unpacked = channels.map(|c| to_u32x8(c.select(red, green, blue, alpha)));

            *srgb = u8x32::from_u32_interleaved(unpacked);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paint_tile_row<S: LayerProps, L: Layout>(
        &mut self,
        workbench: &mut LayerWorkbench,
        tile_y: usize,
        mut segments: &[PixelSegment<{ consts::cpu::TILE_WIDTH }, { consts::cpu::TILE_HEIGHT }>],
        props: &S,
        channels: [Channel; 4],
        clear_color: Color,
        previous_clear_color: Option<Color>,
        cached_tiles: Option<&[CachedTile]>,
        row: ChunksExactMut<'_, Slice<'_, u8>>,
        crop: &Option<Rect>,
        flusher: Option<&dyn Flusher>,
    ) {
        // Map layer id to cover carry information.
        let mut covers_left_of_row: BTreeMap<u32, Cover> = BTreeMap::new();
        let tile_x_start = crop
            .as_ref()
            .map(|rect| rect.hor.start as i16)
            .unwrap_or_default();
        if let Ok(last_clipped_index) =
            super::search_last_by_key(segments, false, |segment| segment.tile_x() >= tile_x_start)
        {
            // Accumulate cover for clipped tiles.
            for segment in &segments[..=last_clipped_index] {
                let cover = covers_left_of_row.entry(segment.layer_id()).or_default();
                cover.as_slice_mut()[segment.local_y() as usize] += segment.cover();
            }

            segments = &segments[last_clipped_index + 1..];
        }

        workbench.init(
            covers_left_of_row
                .into_iter()
                .map(|(layer_id, cover)| CoverCarry { cover, layer_id }),
        );

        for (tile_x, slices) in row.enumerate() {
            if let Some(rect) = &crop {
                if !rect.hor.contains(&tile_x) {
                    continue;
                }
            }

            let current_segments =
                super::search_last_by_key(segments, tile_x as i16, |segment| segment.tile_x())
                    .map(|last_index| {
                        let current_segments = &segments[..=last_index];
                        segments = &segments[last_index + 1..];
                        current_segments
                    })
                    .unwrap_or(&[]);

            let context = Context {
                tile_x,
                tile_y,
                segments: current_segments,
                props,
                cached_clear_color: previous_clear_color,
                cached_tile: cached_tiles.map(|cached_tiles| &cached_tiles[tile_x]),
                channels,
                clear_color,
            };

            self.clip = None;

            match workbench.drive_tile_painting(self, &context) {
                TileWriteOp::None => (),
                TileWriteOp::Solid(color) => L::write(slices, flusher, TileFill::Solid(color)),
                TileWriteOp::ColorBuffer => {
                    self.compute_srgb(channels);
                    let colors: &[[u8; 4]] = unsafe {
                        std::slice::from_raw_parts(
                            self.srgb.as_ptr().cast(),
                            self.srgb.len() * mem::size_of::<u8x32>() / mem::size_of::<[u8; 4]>(),
                        )
                    };
                    L::write(slices, flusher, TileFill::Full(colors));
                }
            }
        }
    }
}

thread_local!(static PAINTER_WORKBENCH: RefCell<(Painter, LayerWorkbench)> = RefCell::new((
    Painter::new(),
    LayerWorkbench::new(),
)));

#[allow(clippy::too_many_arguments)]
fn print_row<S: LayerProps, L: Layout>(
    segments: &[PixelSegment<{ consts::cpu::TILE_WIDTH }, { consts::cpu::TILE_HEIGHT }>],
    channels: [Channel; 4],
    clear_color: Color,
    crop: &Option<Rect>,
    styles: &S,
    j: usize,
    row: ChunksExactMut<'_, Slice<'_, u8>>,
    previous_clear_color: Option<Color>,
    cached_tiles: Option<&[CachedTile]>,
    flusher: Option<&dyn Flusher>,
) {
    if let Some(rect) = crop {
        if !rect.vert.contains(&j) {
            return;
        }
    }

    let segments = super::search_last_by_key(segments, j as i16, |segment| segment.tile_y())
        .map(|end| {
            let result = super::search_last_by_key(&segments[..end], j as i16 - 1, |segment| {
                segment.tile_y()
            });
            let start = match result {
                Ok(i) => i + 1,
                Err(i) => i,
            };

            &segments[start..=end]
        })
        .unwrap_or(&[]);

    PAINTER_WORKBENCH.with(|pair| {
        let (mut painter, mut workbench) =
            RefMut::map_split(pair.borrow_mut(), |pair| (&mut pair.0, &mut pair.1));

        painter.paint_tile_row::<S, L>(
            &mut workbench,
            j,
            segments,
            styles,
            channels,
            clear_color,
            previous_clear_color,
            cached_tiles,
            row,
            crop,
            flusher,
        );
    });
}

#[derive(Clone, Debug, Default)]
pub struct CachedTile {
    // Bitfield used to store the existence of `layer_count` and `solid_color` values
    tags: Cell<u8>,
    // (0b0x)
    layer_count: Cell<[u8; 3]>,
    // (0bx0)
    solid_color: Cell<[u8; 4]>,
}

impl CachedTile {
    pub fn layer_count(&self) -> Option<u32> {
        let layer_count = self.layer_count.get();
        let layer_count = u32::from_le_bytes([layer_count[0], layer_count[1], layer_count[2], 0]);

        match self.tags.get() {
            0b10 | 0b11 => Some(layer_count),
            _ => None,
        }
    }

    pub fn solid_color(&self) -> Option<[u8; 4]> {
        match self.tags.get() {
            0b01 | 0b11 => Some(self.solid_color.get()),
            _ => None,
        }
    }

    pub fn update_layer_count(&self, layer_count: Option<u32>) -> Option<u32> {
        let previous_layer_count = self.layer_count();
        match layer_count {
            None => {
                self.tags.set(self.tags.get() & 0b01);
            }
            Some(layer_count) => {
                self.tags.set(self.tags.get() | 0b10);
                self.layer_count
                    .set(layer_count.to_le_bytes()[..3].try_into().unwrap());
            }
        };
        previous_layer_count
    }

    pub fn update_solid_color(&self, solid_color: Option<[u8; 4]>) -> Option<[u8; 4]> {
        let previous_solid_color = self.solid_color();
        match solid_color {
            None => {
                self.tags.set(self.tags.get() & 0b10);
            }
            Some(color) => {
                self.tags.set(self.tags.get() | 0b01);
                self.solid_color.set(color);
            }
        };
        previous_solid_color
    }

    pub fn convert_optimizer_op<P: LayerProps>(
        tile_op: ControlFlow<OptimizerTileWriteOp>,
        context: &Context<'_, P>,
    ) -> ControlFlow<TileWriteOp> {
        match tile_op {
            ControlFlow::Break(OptimizerTileWriteOp::Solid(color)) => {
                let color = to_srgb_bytes(context.channels.map(|c| color.channel(c)));
                let color_is_unchanged = context
                    .cached_tile
                    .as_ref()
                    .map(|cached_tile| cached_tile.update_solid_color(Some(color)) == Some(color))
                    .unwrap_or_default();

                if color_is_unchanged {
                    ControlFlow::Break(TileWriteOp::None)
                } else {
                    ControlFlow::Break(TileWriteOp::Solid(color))
                }
            }
            ControlFlow::Break(OptimizerTileWriteOp::None) => ControlFlow::Break(TileWriteOp::None),
            _ => {
                if let Some(cached_tile) = context.cached_tile {
                    cached_tile.update_solid_color(None);
                }

                ControlFlow::Continue(())
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn for_each_row<L: Layout, S: LayerProps>(
    layout: &mut L,
    buffer: &mut [u8],
    channels: [Channel; 4],
    flusher: Option<&dyn Flusher>,
    previous_clear_color: Option<Color>,
    cached_tiles: Option<RefMut<'_, Vec<CachedTile>>>,
    mut segments: &[PixelSegment<{ consts::cpu::TILE_WIDTH }, { consts::cpu::TILE_HEIGHT }>],
    clear_color: Color,
    crop: &Option<Rect>,
    styles: &S,
) {
    // Skip content with negative y coordinates.
    if let Ok(start) = super::search_last_by_key(segments, false, |segment| segment.tile_y() >= 0) {
        segments = &segments[start + 1..];
    }

    let width_in_tiles = layout.width_in_tiles();
    let row_of_tiles_len = width_in_tiles * layout.slices_per_tile();
    let mut slices = layout.slices(buffer);

    if let Some(mut cached_tiles) = cached_tiles {
        slices
            .par_chunks_mut(row_of_tiles_len)
            .zip_eq(cached_tiles.par_chunks_mut(width_in_tiles))
            .enumerate()
            .for_each(|(j, (row_of_tiles, cached_tiles))| {
                print_row::<S, L>(
                    segments,
                    channels,
                    clear_color,
                    crop,
                    styles,
                    j,
                    row_of_tiles.chunks_exact_mut(row_of_tiles.len() / width_in_tiles),
                    previous_clear_color,
                    Some(cached_tiles),
                    flusher,
                );
            });
    } else {
        slices
            .par_chunks_mut(row_of_tiles_len)
            .enumerate()
            .for_each(|(j, row_of_tiles)| {
                print_row::<S, L>(
                    segments,
                    channels,
                    clear_color,
                    crop,
                    styles,
                    j,
                    row_of_tiles.chunks_exact_mut(row_of_tiles.len() / width_in_tiles),
                    previous_clear_color,
                    None,
                    flusher,
                );
            });
    }
}

#[cfg(feature = "bench")]
pub fn painter_fill_at_bench(width: usize, height: usize, style: &Style) -> f32x8 {
    let mut sum = f32x8::indexed();
    for y in 0..width {
        for x in 0..height {
            for c in Painter::fill_at(x, y, styling) {
                sum += c;
            }
        }
    }
    sum
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::{collections::HashMap, iter};

    use crate::{
        consts::cpu::{TILE_HEIGHT, TILE_WIDTH},
        cpu::{buffer::layout::LinearLayout, Rasterizer, RGBA},
        math::Point,
        Composition, GeomId, Order, SegmentBuffer,
    };

    const RED: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    const RED_GREEN_50: Color = Color {
        r: 1.0,
        g: 0.5,
        b: 0.0,
        a: 1.0,
    };
    const RED_50: Color = Color {
        r: 0.5,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    const RED_50_GREEN_50: Color = Color {
        r: 0.5,
        g: 0.5,
        b: 0.0,
        a: 1.0,
    };
    const GREEN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    const GREEN_50: Color = Color {
        r: 0.0,
        g: 0.5,
        b: 0.0,
        a: 1.0,
    };
    const BLUE: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    const BLACK_ALPHA_50: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.5,
    };
    const BLACK_ALPHA_0: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    const BLACK_RGBA: [u8; 4] = [0, 0, 0, 255];
    const RED_RGBA: [u8; 4] = [255, 0, 0, 255];
    const GREEN_RGBA: [u8; 4] = [0, 255, 0, 255];
    const BLUE_RGBA: [u8; 4] = [0, 0, 255, 255];

    impl LayerProps for HashMap<u32, Style> {
        fn get(&self, layer_id: u32) -> Cow<'_, Props> {
            let style = self.get(&layer_id).unwrap().clone();

            Cow::Owned(Props {
                fill_rule: FillRule::NonZero,
                func: Func::Draw(style),
            })
        }

        fn is_unchanged(&self, _: u32) -> bool {
            false
        }
    }

    impl LayerProps for HashMap<u32, Props> {
        fn get(&self, layer_id: u32) -> Cow<'_, Props> {
            Cow::Owned(self.get(&layer_id).unwrap().clone())
        }

        fn is_unchanged(&self, _: u32) -> bool {
            false
        }
    }

    impl<F> LayerProps for F
    where
        F: Fn(u32) -> Style + Send + Sync,
    {
        fn get(&self, layer_id: u32) -> Cow<'_, Props> {
            let style = self(layer_id);

            Cow::Owned(Props {
                fill_rule: FillRule::NonZero,
                func: Func::Draw(style),
            })
        }

        fn is_unchanged(&self, _: u32) -> bool {
            false
        }
    }

    impl Painter {
        fn colors(&self) -> [[f32; 4]; TILE_WIDTH * TILE_HEIGHT] {
            let mut colors = [[0.0, 0.0, 0.0, 1.0]; TILE_WIDTH * TILE_HEIGHT];

            for (i, (((c0, c1), c2), alpha)) in self
                .red
                .iter()
                .copied()
                .flat_map(f32x8::to_array)
                .zip(self.green.iter().copied().flat_map(f32x8::to_array))
                .zip(self.blue.iter().copied().flat_map(f32x8::to_array))
                .zip(self.alpha.iter().copied().flat_map(f32x8::to_array))
                .enumerate()
            {
                colors[i] = [c0, c1, c2, alpha];
            }

            colors
        }
    }

    fn line_segments(
        points: &[(Point, Point)],
        same_layer: bool,
    ) -> Vec<PixelSegment<TILE_WIDTH, TILE_HEIGHT>> {
        let mut segment_buffer = SegmentBuffer::default();
        let mut composition = Composition::new();

        if same_layer {
            composition.get_mut_or_insert_default(Order::new(0).unwrap());

            for &(p0, p1) in points.iter() {
                segment_buffer.push(GeomId::default(), [p0, p1]);
            }
        } else {
            for (id, &(p0, p1)) in points.iter().enumerate() {
                composition.get_mut_or_insert_default(Order::new(id as u32).unwrap());
                segment_buffer.push(GeomId::new(id as u64), [p0, p1]);
            }
        }

        let (layers, geom_id_to_order) = composition.layers_for_segments();

        let lines = segment_buffer.fill_cpu_view(usize::MAX, usize::MAX, layers, &geom_id_to_order);

        let mut rasterizer = Rasterizer::default();
        rasterizer.rasterize(&lines);

        let mut segments: Vec<_> = rasterizer.segments().to_vec();
        segments.sort_unstable();

        segments
    }

    fn paint_tile(
        cover_carries: impl IntoIterator<Item = CoverCarry>,
        segments: &[PixelSegment<TILE_WIDTH, TILE_HEIGHT>],
        props: &impl LayerProps,
        clear_color: Color,
    ) -> [[f32; 4]; TILE_WIDTH * TILE_HEIGHT] {
        let mut painter = Painter::new();
        let mut workbench = LayerWorkbench::new();

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments,
            props,
            cached_clear_color: None,
            cached_tile: None,
            channels: RGBA,
            clear_color,
        };

        workbench.init(cover_carries);
        workbench.drive_tile_painting(&mut painter, &context);

        painter.colors()
    }

    fn coverage(double_area: i32, fill_rules: FillRule) -> f32 {
        let array = doubled_area_to_coverage(i32x8::splat(double_area), fill_rules).to_array();

        for val in array {
            assert_eq!(val, array[0]);
        }

        array[0]
    }

    #[test]
    fn double_area_non_zero() {
        let area = PIXEL_DOUBLE_AREA as i32;

        assert_eq!(coverage(-area * 2, FillRule::NonZero), 1.0);
        assert_eq!(coverage(-area * 3 / 2, FillRule::NonZero), 1.0);
        assert_eq!(coverage(-area, FillRule::NonZero), 1.0);
        assert_eq!(coverage(-area / 2, FillRule::NonZero), 0.5);
        assert_eq!(coverage(0, FillRule::NonZero), 0.0);
        assert_eq!(coverage(area / 2, FillRule::NonZero), 0.5);
        assert_eq!(coverage(area, FillRule::NonZero), 1.0);
        assert_eq!(coverage(area * 3 / 2, FillRule::NonZero), 1.0);
        assert_eq!(coverage(area * 2, FillRule::NonZero), 1.0);
    }

    #[test]
    fn double_area_even_odd() {
        let area = PIXEL_DOUBLE_AREA as i32;

        assert_eq!(coverage(-area * 2, FillRule::NonZero), 1.0);
        assert_eq!(coverage(-area * 3 / 2, FillRule::EvenOdd), 0.5);
        assert_eq!(coverage(-area, FillRule::EvenOdd), 1.0);
        assert_eq!(coverage(-area / 2, FillRule::EvenOdd), 0.5);
        assert_eq!(coverage(0, FillRule::EvenOdd), 0.0);
        assert_eq!(coverage(area / 2, FillRule::EvenOdd), 0.5);
        assert_eq!(coverage(area, FillRule::EvenOdd), 1.0);
        assert_eq!(coverage(area * 3 / 2, FillRule::EvenOdd), 0.5);
        assert_eq!(coverage(area * 2, FillRule::NonZero), 1.0);
    }

    #[test]
    fn carry_cover() {
        let mut cover_carry = CoverCarry {
            cover: Cover::default(),
            layer_id: 0,
        };
        cover_carry.cover.covers[0].as_mut_array()[1] = 16;
        cover_carry.layer_id = 1;

        let segments = line_segments(
            &[(Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32))],
            false,
        );

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(GREEN),
                ..Default::default()
            },
        );
        styles.insert(
            1,
            Style {
                fill: Fill::Solid(RED),
                ..Default::default()
            },
        );

        assert_eq!(
            paint_tile([cover_carry], &segments, &styles, BLACK)[0..2],
            [GREEN, RED].map(Color::to_array),
        );
    }

    #[test]
    fn overlapping_triangles() {
        let segments = line_segments(
            &[
                (Point::new(0.0, 0.0), Point::new(4.0, 4.0)),
                (Point::new(4.0, 0.0), Point::new(0.0, 4.0)),
            ],
            false,
        );

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(GREEN),
                ..Default::default()
            },
        );
        styles.insert(
            1,
            Style {
                fill: Fill::Solid(RED),
                ..Default::default()
            },
        );

        let colors = paint_tile([], &segments, &styles, BLACK);

        let row_start = 0;
        let row_end = 4;

        let mut column = 0;
        assert_eq!(
            colors[column + row_start..column + row_end],
            [GREEN_50, BLACK, BLACK, RED_50].map(Color::to_array),
        );

        column += TILE_HEIGHT;
        assert_eq!(
            colors[column + row_start..column + row_end],
            [GREEN, GREEN_50, RED_50, RED].map(Color::to_array),
        );

        column += TILE_HEIGHT;
        assert_eq!(
            colors[column + row_start..column + row_end],
            [GREEN, RED_50_GREEN_50, RED, RED].map(Color::to_array),
        );

        column += TILE_HEIGHT;
        assert_eq!(
            colors[column + row_start..column + row_end],
            [RED_50_GREEN_50, RED, RED, RED].map(Color::to_array),
        );
    }

    #[test]
    fn transparent_overlay() {
        let segments = line_segments(
            &[
                (Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32)),
                (Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32)),
            ],
            false,
        );

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(RED),
                ..Default::default()
            },
        );
        styles.insert(
            1,
            Style {
                fill: Fill::Solid(BLACK_ALPHA_50),
                ..Default::default()
            },
        );

        assert_eq!(
            paint_tile([], &segments, &styles, BLACK)[0],
            RED_50.to_array()
        );
    }

    #[test]
    fn linear_blend_over() {
        let segments = line_segments(
            &[
                (Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32)),
                (Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32)),
            ],
            false,
        );

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(RED),
                ..Default::default()
            },
        );
        styles.insert(
            1,
            Style {
                fill: Fill::Solid(Color { a: 0.5, ..GREEN }),
                ..Default::default()
            },
        );

        assert_eq!(
            paint_tile([], &segments, &styles, BLACK)[0],
            RED_50_GREEN_50.to_array()
        );
    }

    #[test]
    fn linear_blend_difference() {
        let segments = line_segments(
            &[
                (Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32)),
                (Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32)),
            ],
            false,
        );

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(RED),
                ..Default::default()
            },
        );
        styles.insert(
            1,
            Style {
                fill: Fill::Solid(Color { a: 0.5, ..GREEN }),
                blend_mode: BlendMode::Difference,
                ..Default::default()
            },
        );

        assert_eq!(
            paint_tile([], &segments, &styles, BLACK)[0],
            RED_GREEN_50.to_array()
        );
    }

    #[test]
    fn linear_blend_hue_white_opaque_brackground() {
        let segments = line_segments(
            &[(Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32))],
            false,
        );

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(Color { a: 0.5, ..GREEN }),
                blend_mode: BlendMode::Hue,
                ..Default::default()
            },
        );

        assert_eq!(
            paint_tile([], &segments, &styles, WHITE)[0],
            WHITE.to_array()
        );
    }

    #[test]
    fn linear_blend_hue_white_transparent_brackground() {
        let segments = line_segments(
            &[(Point::new(0.0, 0.0), Point::new(0.0, TILE_HEIGHT as f32))],
            false,
        );

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(Color { a: 0.5, ..GREEN }),
                blend_mode: BlendMode::Hue,
                ..Default::default()
            },
        );

        assert_eq!(
            paint_tile([], &segments, &styles, Color { a: 0.0, ..WHITE })[0],
            [0.5, 1.0, 0.5, 0.5],
        );
    }

    #[test]
    fn cover_carry_is_empty() {
        assert!(Cover {
            covers: [i8x16::splat(0); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::NonZero));
        assert!(!Cover {
            covers: [i8x16::splat(1); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::NonZero));
        assert!(!Cover {
            covers: [i8x16::splat(-1); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::NonZero));
        assert!(!Cover {
            covers: [i8x16::splat(16); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::NonZero));
        assert!(!Cover {
            covers: [i8x16::splat(-16); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::NonZero));

        assert!(Cover {
            covers: [i8x16::splat(0); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(1); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(-1); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(16); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(-16); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
        assert!(Cover {
            covers: [i8x16::splat(32); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
        assert!(Cover {
            covers: [i8x16::splat(-32); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(48); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(-48); TILE_HEIGHT / 16]
        }
        .is_empty(FillRule::EvenOdd));
    }

    #[test]
    fn cover_carry_is_full() {
        assert!(!Cover {
            covers: [i8x16::splat(0); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::NonZero));
        assert!(!Cover {
            covers: [i8x16::splat(1); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::NonZero));
        assert!(!Cover {
            covers: [i8x16::splat(-1); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::NonZero));
        assert!(Cover {
            covers: [i8x16::splat(16); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::NonZero));
        assert!(Cover {
            covers: [i8x16::splat(-16); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::NonZero));

        assert!(!Cover {
            covers: [i8x16::splat(0); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(1); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(-1); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
        assert!(Cover {
            covers: [i8x16::splat(16); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
        assert!(Cover {
            covers: [i8x16::splat(-16); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(32); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
        assert!(!Cover {
            covers: [i8x16::splat(-32); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
        assert!(Cover {
            covers: [i8x16::splat(48); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
        assert!(Cover {
            covers: [i8x16::splat(-48); TILE_HEIGHT / 16]
        }
        .is_full(FillRule::EvenOdd));
    }

    #[test]
    fn clip() {
        let segments = line_segments(
            &[
                (Point::new(0.0, 0.0), Point::new(4.0, 4.0)),
                (Point::new(0.0, 0.0), Point::new(0.0, 4.0)),
                (Point::new(0.0, 0.0), Point::new(0.0, 4.0)),
            ],
            false,
        );

        let mut props = HashMap::new();

        props.insert(
            0,
            Props {
                fill_rule: FillRule::NonZero,
                func: Func::Clip(2),
            },
        );
        props.insert(
            1,
            Props {
                fill_rule: FillRule::NonZero,
                func: Func::Draw(Style {
                    fill: Fill::Solid(GREEN),
                    is_clipped: true,
                    ..Default::default()
                }),
            },
        );
        props.insert(
            2,
            Props {
                fill_rule: FillRule::NonZero,
                func: Func::Draw(Style {
                    fill: Fill::Solid(RED),
                    is_clipped: true,
                    ..Default::default()
                }),
            },
        );
        props.insert(
            3,
            Props {
                fill_rule: FillRule::NonZero,
                func: Func::Draw(Style {
                    fill: Fill::Solid(GREEN),
                    ..Default::default()
                }),
            },
        );

        let mut painter = Painter::new();
        let mut workbench = LayerWorkbench::new();

        let mut context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &segments,
            props: &props,
            cached_clear_color: None,
            cached_tile: None,
            channels: RGBA,
            clear_color: BLACK,
        };

        workbench.drive_tile_painting(&mut painter, &context);

        let colors = painter.colors();
        let mut col = [BLACK.to_array(); 4];

        for i in 0..4 {
            col[i] = [0.5, 0.25, 0.0, 1.0];

            if i >= 1 {
                col[i - 1] = RED.to_array();
            }

            assert_eq!(colors[i * TILE_HEIGHT..i * TILE_HEIGHT + 4], col);
        }

        let segments = line_segments(&[(Point::new(4.0, 0.0), Point::new(4.0, 4.0))], false);

        context.tile_x = 1;
        context.segments = &segments;

        workbench.drive_tile_painting(&mut painter, &context);
        for i in 0..4 {
            assert_eq!(
                painter.colors()[i * TILE_HEIGHT..i * TILE_HEIGHT + 4],
                [RED.to_array(); 4]
            );
        }
    }

    #[test]
    fn f32_to_u8_scaled() {
        fn convert(val: f32) -> u8 {
            let vals: [u8; 4] = to_u32x4(f32x4::splat(val)).into();
            vals[0]
        }

        assert_eq!(convert(-0.001), 0);
        assert_eq!(convert(1.001), 255);

        for i in 0..255 {
            assert_eq!(convert(f32::from(i) * 255.0f32.recip()), i);
        }
    }

    #[test]
    fn srgb() {
        let premultiplied = [
            // Small values will still result in > 0 in sRGB.
            0.001 * 0.5,
            // Expected to be < 128.
            0.2 * 0.5,
            // Expected to be > 128.
            0.5 * 0.5,
            // Should convert linearly.
            0.5,
        ];

        assert_eq!(to_srgb_bytes(premultiplied), [2, 89, 137, 128]);
    }

    #[test]
    fn flusher() {
        macro_rules! seg {
            ( $j:expr, $i:expr ) => {
                PixelSegment::new($j, $i, 0, 0, 0, 0, 0)
            };
        }

        #[derive(Debug)]
        struct WhiteFlusher;

        impl Flusher for WhiteFlusher {
            fn flush(&self, slice: &mut [u8]) {
                for color in slice {
                    *color = 255u8;
                }
            }
        }

        let width = TILE_WIDTH + TILE_WIDTH / 2;
        let mut buffer = vec![0u8; width * TILE_HEIGHT * 4];
        let mut buffer_layout = LinearLayout::new(width, width * 4, TILE_HEIGHT);

        let segments = &[seg!(0, 0), seg!(0, 1), seg!(1, 0), seg!(1, 1)];

        for_each_row(
            &mut buffer_layout,
            &mut buffer,
            RGBA,
            Some(&WhiteFlusher),
            None,
            None,
            segments,
            BLACK_ALPHA_0,
            &None,
            &|_| Style::default(),
        );

        assert!(buffer.iter().all(|&color| color == 255u8));
    }

    #[test]
    fn flush_background() {
        #[derive(Debug)]
        struct WhiteFlusher;

        impl Flusher for WhiteFlusher {
            fn flush(&self, slice: &mut [u8]) {
                for color in slice {
                    *color = 255u8;
                }
            }
        }

        let mut buffer = vec![0u8; TILE_WIDTH * TILE_HEIGHT * 4];
        let mut buffer_layout = LinearLayout::new(TILE_WIDTH, TILE_WIDTH * 4, TILE_HEIGHT);

        for_each_row(
            &mut buffer_layout,
            &mut buffer,
            RGBA,
            Some(&WhiteFlusher),
            None,
            None,
            &[],
            BLACK_ALPHA_0,
            &None,
            &|_| Style::default(),
        );

        assert!(buffer.iter().all(|&color| color == 255u8));
    }

    #[test]
    fn skip_opaque_tiles() {
        let mut buffer = vec![0u8; TILE_WIDTH * TILE_HEIGHT * 3 * 4];

        let mut buffer_layout = LinearLayout::new(TILE_WIDTH * 3, TILE_WIDTH * 3 * 4, TILE_HEIGHT);

        let mut segments = vec![];
        for y in 0..TILE_HEIGHT {
            segments.push(PixelSegment::new(
                2,
                -1,
                0,
                TILE_WIDTH as u8 - 1,
                y as u8,
                0,
                consts::PIXEL_WIDTH as i8,
            ));
        }

        segments.push(PixelSegment::new(
            0,
            -1,
            0,
            TILE_WIDTH as u8 - 1,
            0,
            0,
            consts::PIXEL_WIDTH as i8,
        ));
        segments.push(PixelSegment::new(
            1,
            0,
            0,
            0,
            1,
            0,
            consts::PIXEL_WIDTH as i8,
        ));

        for y in 0..TILE_HEIGHT {
            segments.push(PixelSegment::new(
                2,
                1,
                0,
                TILE_WIDTH as u8 - 1,
                y as u8,
                0,
                -(consts::PIXEL_WIDTH as i8),
            ));
        }

        segments.sort();

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(BLUE),
                ..Default::default()
            },
        );
        styles.insert(
            1,
            Style {
                fill: Fill::Solid(GREEN),
                ..Default::default()
            },
        );
        styles.insert(
            2,
            Style {
                fill: Fill::Solid(RED),
                ..Default::default()
            },
        );

        for_each_row(
            &mut buffer_layout,
            &mut buffer,
            RGBA,
            None,
            None,
            None,
            &segments,
            BLACK,
            &None,
            &|layer| styles[&layer].clone(),
        );

        let tiles = buffer_layout.slices(&mut buffer);

        assert_eq!(
            tiles.iter().map(|slice| slice.to_vec()).collect::<Vec<_>>(),
            // First two tiles need to be completely red.
            iter::repeat(vec![RED_RGBA; TILE_WIDTH].concat())
                .take(TILE_HEIGHT)
                .chain(iter::repeat(vec![RED_RGBA; TILE_WIDTH].concat()).take(TILE_HEIGHT))
                .chain(
                    // The last tile contains one blue and one green line.
                    iter::once(vec![BLUE_RGBA; TILE_WIDTH].concat())
                        .chain(iter::once(vec![GREEN_RGBA; TILE_WIDTH].concat()))
                        // Followed by black lines (clear color).
                        .chain(
                            iter::repeat(vec![BLACK_RGBA; TILE_WIDTH].concat())
                                .take(TILE_HEIGHT - 2)
                        )
                )
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn crop() {
        let mut buffer = vec![0u8; TILE_WIDTH * TILE_HEIGHT * 9 * 4];

        let mut buffer_layout =
            LinearLayout::new(TILE_WIDTH * 3, TILE_WIDTH * 3 * 4, TILE_HEIGHT * 3);

        let mut segments = vec![];
        for j in 0..3 {
            for y in 0..TILE_HEIGHT {
                segments.push(PixelSegment::new(
                    0,
                    0,
                    j,
                    TILE_WIDTH as u8 - 1,
                    y as u8,
                    0,
                    consts::PIXEL_WIDTH as i8,
                ));
            }
        }

        segments.sort();

        let mut styles = HashMap::new();

        styles.insert(
            0,
            Style {
                fill: Fill::Solid(BLUE),
                ..Default::default()
            },
        );

        for_each_row(
            &mut buffer_layout,
            &mut buffer,
            RGBA,
            None,
            None,
            None,
            &segments,
            RED,
            &Some(Rect::new(
                TILE_WIDTH..TILE_WIDTH * 2 + TILE_WIDTH / 2,
                TILE_HEIGHT..TILE_HEIGHT * 2,
            )),
            &|layer| styles[&layer].clone(),
        );

        let tiles = buffer_layout.slices(&mut buffer);

        assert_eq!(
            tiles.iter().map(|slice| slice.to_vec()).collect::<Vec<_>>(),
            // First row of tiles needs to be completely black.
            iter::repeat(vec![0u8; TILE_WIDTH * 4])
                .take(TILE_HEIGHT * 3)
                // Second row begins with a black tile.
                .chain(iter::repeat(vec![0u8; TILE_WIDTH * 4]).take(TILE_HEIGHT))
                .chain(iter::repeat(vec![BLUE_RGBA; TILE_WIDTH].concat()).take(TILE_HEIGHT * 2))
                // Third row of tiles needs to be completely black as well.
                .chain(iter::repeat(vec![0u8; TILE_WIDTH * 4]).take(TILE_HEIGHT * 3))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn tiles_len() {
        let width = TILE_WIDTH * 4;
        let width_stride = TILE_WIDTH * 5 * 4;
        let height = TILE_HEIGHT * 8;

        let buffer_layout = LinearLayout::new(width, width_stride, height);

        assert_eq!(
            buffer_layout.width_in_tiles() * buffer_layout.height_in_tiles(),
            32
        );
    }

    #[test]
    fn cached_tiles() {
        const RED: [u8; 4] = [255, 0, 0, 255];

        let cached_tile = CachedTile::default();

        // Solid color
        assert_eq!(cached_tile.solid_color(), None);
        assert_eq!(cached_tile.update_solid_color(Some(RED)), None);
        assert_eq!(cached_tile.solid_color(), Some(RED));

        // Layer count
        assert_eq!(cached_tile.layer_count(), None);
        assert_eq!(cached_tile.update_layer_count(Some(2)), None);
        assert_eq!(cached_tile.layer_count(), Some(2));
    }
}
