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
    cell::Cell,
    mem,
    ops::{ControlFlow, RangeInclusive},
};

use rustc_hash::FxHashMap;

use crate::{
    consts,
    cpu::{self, Channel, PixelSegment},
    styling::{Color, FillRule, Func, Props, Style},
};

use self::passes::PassesSharedState;

use super::{CachedTile, Cover, CoverCarry, LayerProps};

mod passes;

pub(crate) trait LayerPainter {
    fn clear_cells(&mut self);
    fn acc_segment(
        &mut self,
        segment: PixelSegment<{ consts::cpu::TILE_WIDTH }, { consts::cpu::TILE_HEIGHT }>,
    );
    fn acc_cover(&mut self, cover: Cover);
    fn clear(&mut self, color: Color);
    fn paint_layer(
        &mut self,
        tile_x: usize,
        tile_y: usize,
        layer_id: u32,
        props: &Props,
        apply_clip: bool,
    ) -> Cover;
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Index(usize);

#[derive(Debug)]
struct MaskedCell<T> {
    val: T,
    mask: Cell<bool>,
}

#[derive(Debug, Default)]
pub struct MaskedVec<T> {
    vals: Vec<MaskedCell<T>>,
    skipped: Cell<usize>,
}

impl<T> MaskedVec<T> {
    pub fn len(&self) -> usize {
        self.vals.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.vals.iter().map(|cell| &cell.val)
    }

    pub fn iter_with_masks(&self) -> impl Iterator<Item = (&T, bool)> {
        self.vals
            .iter()
            .enumerate()
            .map(move |(i, cell)| (&cell.val, i >= self.skipped.get() && cell.mask.get()))
    }

    pub fn iter_masked(&self) -> impl DoubleEndedIterator<Item = (Index, &T)> {
        self.vals
            .iter()
            .enumerate()
            .skip(self.skipped.get())
            .filter_map(|(i, cell)| cell.mask.get().then_some((Index(i), &cell.val)))
    }

    pub fn clear(&mut self) {
        self.vals.clear();
        self.skipped.set(0);
    }

    pub fn set_mask(&self, i: Index, mask: bool) {
        self.vals[i.0].mask.set(mask);
    }

    pub fn skip_until(&self, i: Index) {
        self.skipped.set(i.0);
    }
}

impl<T: Copy + Ord + PartialEq> MaskedVec<T> {
    pub fn sort_and_dedup(&mut self) {
        self.vals.sort_unstable_by_key(|cell| cell.val);
        self.vals.dedup_by_key(|cell| cell.val);
    }
}

impl<A> Extend<A> for MaskedVec<A> {
    fn extend<T: IntoIterator<Item = A>>(&mut self, iter: T) {
        self.vals.extend(iter.into_iter().map(|val| MaskedCell {
            val,
            mask: Cell::new(true),
        }));
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum OptimizerTileWriteOp {
    None,
    Solid(Color),
}

#[derive(Debug, Eq, PartialEq)]
pub enum TileWriteOp {
    None,
    Solid([u8; 4]),
    ColorBuffer,
}

pub struct Context<'c, P: LayerProps> {
    pub tile_x: usize,
    pub tile_y: usize,
    pub segments: &'c [PixelSegment<{ consts::cpu::TILE_WIDTH }, { consts::cpu::TILE_HEIGHT }>],
    pub props: &'c P,
    pub cached_clear_color: Option<Color>,
    pub channels: [Channel; 4],
    pub cached_tile: Option<&'c CachedTile>,
    pub clear_color: Color,
}

#[derive(Debug, Default)]
pub struct LayerWorkbenchState {
    pub ids: MaskedVec<u32>,
    pub segment_ranges: FxHashMap<u32, RangeInclusive<usize>>,
    pub queue_indices: FxHashMap<u32, usize>,
    pub queue: Vec<CoverCarry>,
    next_queue: Vec<CoverCarry>,
}

impl LayerWorkbenchState {
    fn segments<'c, P: LayerProps>(
        &self,
        context: &'c Context<'_, P>,
        id: u32,
    ) -> Option<&'c [PixelSegment<{ consts::cpu::TILE_WIDTH }, { consts::cpu::TILE_HEIGHT }>]> {
        self.segment_ranges
            .get(&id)
            .map(|range| &context.segments[range.clone()])
    }

    fn cover(&self, id: u32) -> Option<&Cover> {
        self.queue_indices.get(&id).map(|&i| &self.queue[i].cover)
    }

    pub(crate) fn layer_is_full<'c, P: LayerProps>(
        &self,
        context: &'c Context<'_, P>,
        id: u32,
        fill_rule: FillRule,
    ) -> bool {
        self.segments(context, id).is_none()
            && self
                .cover(id)
                .map(|cover| cover.is_full(fill_rule))
                .unwrap_or_default()
    }
}

#[derive(Debug, Default)]
pub(crate) struct LayerWorkbench {
    state: LayerWorkbenchState,
    passes_shared_state: PassesSharedState,
}

impl LayerWorkbench {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn init(&mut self, cover_carries: impl IntoIterator<Item = CoverCarry>) {
        self.state.queue.clear();
        self.state.queue.extend(cover_carries);
    }

    fn next_tile(&mut self) {
        self.state.ids.clear();
        self.state.segment_ranges.clear();
        self.state.queue_indices.clear();

        mem::swap(&mut self.state.queue, &mut self.state.next_queue);

        self.state.next_queue.clear();

        self.passes_shared_state.reset();
    }

    fn cover_carry<'c, P: LayerProps>(
        &self,
        context: &'c Context<'_, P>,
        id: u32,
    ) -> Option<CoverCarry> {
        let mut acc_cover = Cover::default();

        if let Some(segments) = self.state.segments(context, id) {
            for segment in segments {
                acc_cover.as_slice_mut()[segment.local_y() as usize] += segment.cover();
            }
        }

        if let Some(cover) = self.state.cover(id) {
            cover.add_cover_to(&mut acc_cover.covers);
        }

        (!acc_cover.is_empty(context.props.get(id).fill_rule)).then_some(CoverCarry {
            cover: acc_cover,
            layer_id: id,
        })
    }

    fn optimization_passes<'c, P: LayerProps>(
        &mut self,
        context: &'c Context<'_, P>,
    ) -> ControlFlow<OptimizerTileWriteOp> {
        let state = &mut self.state;
        let passes_shared_state = &mut self.passes_shared_state;

        passes::tile_unchanged_pass(state, passes_shared_state, context)?;
        passes::skip_trivial_clips_pass(state, passes_shared_state, context)?;
        passes::skip_fully_covered_layers_pass(state, passes_shared_state, context)?;

        ControlFlow::Continue(())
    }

    fn populate_layers<'c, P: LayerProps>(&mut self, context: &'c Context<'_, P>) {
        let mut start = 0;
        while let Some(id) = context.segments.get(start).map(|s| s.layer_id()) {
            let diff =
                cpu::search_last_by_key(&context.segments[start..], id, |s| s.layer_id()).unwrap();

            self.state.segment_ranges.insert(id, start..=start + diff);

            start += diff + 1;
        }

        self.state.queue_indices.extend(
            self.state
                .queue
                .iter()
                .enumerate()
                .map(|(i, cover_carry)| (cover_carry.layer_id, i)),
        );

        self.state.ids.extend(
            self.state
                .segment_ranges
                .keys()
                .copied()
                .chain(self.state.queue_indices.keys().copied()),
        );

        self.state.ids.sort_and_dedup();
    }

    pub fn drive_tile_painting<'c, P: LayerProps>(
        &mut self,
        painter: &mut impl LayerPainter,
        context: &'c Context<'_, P>,
    ) -> TileWriteOp {
        self.populate_layers(context);

        if let ControlFlow::Break(tile_op) =
            CachedTile::convert_optimizer_op(self.optimization_passes(context), context)
        {
            for &id in self.state.ids.iter() {
                if let Some(cover_carry) = self.cover_carry(context, id) {
                    self.state.next_queue.push(cover_carry);
                }
            }

            self.next_tile();

            return tile_op;
        }

        painter.clear(context.clear_color);

        for (&id, mask) in self.state.ids.iter_with_masks() {
            if mask {
                painter.clear_cells();

                if let Some(segments) = self.state.segments(context, id) {
                    for &segment in segments {
                        painter.acc_segment(segment);
                    }
                }

                if let Some(&cover) = self.state.cover(id) {
                    painter.acc_cover(cover);
                }

                let props = context.props.get(id);
                let mut apply_clip = false;

                if let Func::Draw(Style { is_clipped, .. }) = props.func {
                    apply_clip =
                        is_clipped && !self.passes_shared_state.skip_clipping.contains(&id);
                }

                let cover =
                    painter.paint_layer(context.tile_x, context.tile_y, id, &props, apply_clip);

                if !cover.is_empty(props.fill_rule) {
                    self.state.next_queue.push(CoverCarry {
                        cover,
                        layer_id: id,
                    });
                }
            } else if let Some(cover_carry) = self.cover_carry(context, id) {
                self.state.next_queue.push(cover_carry);
            }
        }

        self.next_tile();

        TileWriteOp::ColorBuffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        consts::cpu::{TILE_HEIGHT, TILE_WIDTH},
        cpu::RGBA,
        styling::{BlendMode, Fill},
        utils::simd::{i8x16, Simd},
    };

    use std::borrow::Cow;

    const WHITEF: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    const BLACKF: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    const REDF: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    const RED: [u8; 4] = [255, 0, 0, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];

    impl<T: PartialEq, const N: usize> PartialEq<[T; N]> for MaskedVec<T> {
        fn eq(&self, other: &[T; N]) -> bool {
            self.iter_masked().map(|(_, val)| val).eq(other.iter())
        }
    }

    struct UnimplementedPainter;

    impl LayerPainter for UnimplementedPainter {
        fn clear_cells(&mut self) {
            unimplemented!();
        }

        fn acc_segment(&mut self, _segment: PixelSegment<TILE_WIDTH, TILE_HEIGHT>) {
            unimplemented!();
        }

        fn acc_cover(&mut self, _cover: Cover) {
            unimplemented!();
        }

        fn clear(&mut self, _color: Color) {
            unimplemented!();
        }

        fn paint_layer(
            &mut self,
            _tile_x: usize,
            _tile_y: usize,
            _layer_id: u32,
            _props: &Props,
            _apply_clip: bool,
        ) -> Cover {
            unimplemented!()
        }
    }

    #[test]
    fn masked_vec() {
        let mut v = MaskedVec::default();

        v.extend([1, 2, 3, 4, 5, 6, 7, 8, 9]);

        for (i, &val) in v.iter_masked() {
            if let 2 | 3 | 4 | 5 = val {
                v.set_mask(i, false);
            }

            if val == 3 {
                v.set_mask(i, true);
            }
        }

        assert_eq!(v, [1, 3, 6, 7, 8, 9]);

        for (i, &val) in v.iter_masked() {
            if let 3 | 7 = val {
                v.set_mask(i, false);
            }
        }

        assert_eq!(v, [1, 6, 8, 9]);

        for (i, &val) in v.iter_masked() {
            if val == 8 {
                v.skip_until(i);
            }
        }

        assert_eq!(v, [8, 9]);
    }

    enum CoverType {
        Partial,
        Full,
    }

    fn cover(layer_id: u32, cover_type: CoverType) -> CoverCarry {
        let cover = match cover_type {
            CoverType::Partial => Cover {
                covers: [i8x16::splat(1); TILE_HEIGHT / i8x16::LANES],
            },
            CoverType::Full => Cover {
                covers: [i8x16::splat(consts::PIXEL_WIDTH as i8); TILE_HEIGHT / i8x16::LANES],
            },
        };

        CoverCarry { cover, layer_id }
    }

    fn segment(layer_id: u32) -> PixelSegment<TILE_WIDTH, TILE_HEIGHT> {
        PixelSegment::new(layer_id, 0, 0, 0, 0, 0, 0)
    }

    #[test]
    fn populate_layers() {
        let mut workbench = LayerWorkbench::default();

        struct UnimplementedProps;

        impl LayerProps for UnimplementedProps {
            fn get(&self, _layer_id: u32) -> Cow<'_, Props> {
                unimplemented!()
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                unimplemented!()
            }
        }

        workbench.init([
            cover(0, CoverType::Partial),
            cover(3, CoverType::Partial),
            cover(4, CoverType::Partial),
        ]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[
                segment(0),
                segment(1),
                segment(1),
                segment(2),
                segment(5),
                segment(5),
                segment(5),
            ],
            props: &UnimplementedProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: None,
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        let segment_ranges = workbench.state.segment_ranges;
        let queue_indices = workbench.state.queue_indices;

        assert_eq!(workbench.state.ids, [0, 1, 2, 3, 4, 5]);

        assert_eq!(segment_ranges.len(), 4);
        assert_eq!(segment_ranges.get(&0).cloned(), Some(0..=0));
        assert_eq!(segment_ranges.get(&1).cloned(), Some(1..=2));
        assert_eq!(segment_ranges.get(&2).cloned(), Some(3..=3));
        assert_eq!(segment_ranges.get(&3).cloned(), None);
        assert_eq!(segment_ranges.get(&4).cloned(), None);
        assert_eq!(segment_ranges.get(&5).cloned(), Some(4..=6));

        assert_eq!(queue_indices.len(), 3);
        assert_eq!(queue_indices.get(&0).copied(), Some(0));
        assert_eq!(queue_indices.get(&1).copied(), None);
        assert_eq!(queue_indices.get(&2).copied(), None);
        assert_eq!(queue_indices.get(&3).copied(), Some(1));
        assert_eq!(queue_indices.get(&4).copied(), Some(2));
        assert_eq!(queue_indices.get(&5).copied(), None);
    }

    #[test]
    fn skip_unchanged() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, _layer_id: u32) -> Cow<'_, Props> {
                unimplemented!()
            }

            fn is_unchanged(&self, layer_id: u32) -> bool {
                layer_id < 5
            }
        }

        let cached_tiles = CachedTile::default();
        cached_tiles.update_layer_count(Some(4));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[segment(0), segment(1), segment(2), segment(3), segment(4)],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        // Optimization should fail because the number of layers changed.
        assert_eq!(
            passes::tile_unchanged_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context
            ),
            ControlFlow::Continue(()),
        );
        assert_eq!(cached_tiles.layer_count(), Some(5));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[segment(0), segment(1), segment(2), segment(3), segment(4)],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: BLACKF,
        };

        // Skip should occur because the previous pass updated the number of layers.
        assert_eq!(
            passes::tile_unchanged_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context
            ),
            ControlFlow::Break(OptimizerTileWriteOp::None),
        );
        assert_eq!(cached_tiles.layer_count(), Some(5));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[segment(1), segment(2), segment(3), segment(4), segment(5)],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.next_tile();
        workbench.populate_layers(&context);

        // Optimization should fail because at least one layer changed.
        assert_eq!(
            passes::tile_unchanged_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context
            ),
            ControlFlow::Continue(()),
        );
        assert_eq!(cached_tiles.layer_count(), Some(5));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[segment(0), segment(1), segment(2), segment(3), segment(4)],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: WHITEF,
        };

        workbench.next_tile();
        workbench.populate_layers(&context);

        // Optimization should fail because the clear color changed.
        assert_eq!(
            passes::tile_unchanged_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context
            ),
            ControlFlow::Continue(()),
        );
        assert_eq!(cached_tiles.layer_count(), Some(5));
    }

    #[test]
    fn skip_full_clip() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(match layer_id {
                    1 | 3 => Props {
                        func: Func::Clip(1),
                        ..Default::default()
                    },
                    _ => Props {
                        func: Func::Draw(Style {
                            is_clipped: layer_id == 2,
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                })
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                unimplemented!()
            }
        }

        workbench.init([
            cover(0, CoverType::Partial),
            cover(1, CoverType::Full),
            cover(2, CoverType::Partial),
            cover(3, CoverType::Full),
        ]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: None,
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        passes::skip_trivial_clips_pass(
            &mut workbench.state,
            &mut workbench.passes_shared_state,
            &context,
        );

        let skip_clipping = workbench.passes_shared_state.skip_clipping;

        assert_eq!(workbench.state.ids, [0, 2]);
        assert!(!skip_clipping.contains(&0));
        assert!(skip_clipping.contains(&2));
    }

    #[test]
    fn skip_layer_outside_of_clip() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, _layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(Props {
                    func: Func::Draw(Style {
                        is_clipped: true,
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                unimplemented!()
            }
        }

        workbench.init([cover(0, CoverType::Partial), cover(1, CoverType::Partial)]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: None,
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        passes::skip_trivial_clips_pass(
            &mut workbench.state,
            &mut workbench.passes_shared_state,
            &context,
        );

        assert_eq!(workbench.state.ids, []);
    }

    #[test]
    fn skip_without_layer_usage() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(match layer_id {
                    1 | 4 => Props {
                        func: Func::Clip(1),
                        ..Default::default()
                    },
                    _ => Props::default(),
                })
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                unimplemented!()
            }
        }

        workbench.init([
            cover(0, CoverType::Partial),
            cover(1, CoverType::Partial),
            cover(3, CoverType::Partial),
            cover(4, CoverType::Partial),
        ]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: None,
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        passes::skip_trivial_clips_pass(
            &mut workbench.state,
            &mut workbench.passes_shared_state,
            &context,
        );

        assert_eq!(workbench.state.ids, [0, 3]);
    }

    #[test]
    fn skip_everything_below_opaque() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, _layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(Props::default())
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                false
            }
        }

        workbench.init([
            cover(0, CoverType::Partial),
            cover(1, CoverType::Partial),
            cover(2, CoverType::Full),
        ]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[segment(3)],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: None,
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        assert_eq!(
            passes::skip_fully_covered_layers_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context,
            ),
            ControlFlow::Continue(()),
        );

        assert_eq!(workbench.state.ids, [2, 3]);
    }

    #[test]
    fn blend_top_full_layers() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(Props {
                    func: Func::Draw(Style {
                        fill: Fill::Solid(Color {
                            r: 0.5,
                            g: 0.5,
                            b: 0.5,
                            a: 0.5,
                        }),
                        blend_mode: match layer_id {
                            0 => BlendMode::Over,
                            1 => BlendMode::Multiply,
                            _ => unimplemented!(),
                        },
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                false
            }
        }

        workbench.init([cover(0, CoverType::Full), cover(1, CoverType::Full)]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: None,
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        assert_eq!(
            passes::skip_fully_covered_layers_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context,
            ),
            ControlFlow::Break(OptimizerTileWriteOp::Solid(Color {
                r: 0.28125,
                g: 0.28125,
                b: 0.28125,
                a: 0.75
            })),
        );
    }

    #[test]
    fn blend_top_full_layers_with_clear_color() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, _layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(Props {
                    func: Func::Draw(Style {
                        fill: Fill::Solid(Color {
                            r: 0.5,
                            g: 0.5,
                            b: 0.5,
                            a: 0.5,
                        }),
                        blend_mode: BlendMode::Multiply,
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                false
            }
        }

        workbench.init([cover(0, CoverType::Full), cover(1, CoverType::Full)]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(WHITEF),
            cached_tile: None,
            channels: RGBA,
            clear_color: WHITEF,
        };

        workbench.populate_layers(&context);

        assert_eq!(
            passes::skip_fully_covered_layers_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context,
            ),
            ControlFlow::Break(OptimizerTileWriteOp::Solid(Color {
                r: 0.5625,
                g: 0.5625,
                b: 0.5625,
                a: 1.0
            })),
        );
    }

    #[test]
    fn skip_fully_covered_layers_clip() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(Props {
                    func: match layer_id {
                        0 => Func::Clip(1),
                        1 => Func::Draw(Style {
                            blend_mode: BlendMode::Multiply,
                            ..Default::default()
                        }),
                        _ => unimplemented!(),
                    },
                    ..Default::default()
                })
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                false
            }
        }

        workbench.init([cover(0, CoverType::Partial), cover(1, CoverType::Full)]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(WHITEF),
            cached_tile: None,
            channels: RGBA,
            clear_color: WHITEF,
        };

        workbench.populate_layers(&context);

        assert_eq!(
            passes::skip_fully_covered_layers_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context,
            ),
            ControlFlow::Continue(()),
        );
    }

    #[test]
    fn skip_clip_then_blend() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(Props {
                    func: match layer_id {
                        0 => Func::Clip(1),
                        1 => Func::Draw(Style {
                            fill: Fill::Solid(Color {
                                r: 0.5,
                                g: 0.5,
                                b: 0.5,
                                a: 0.5,
                            }),
                            blend_mode: BlendMode::Multiply,
                            ..Default::default()
                        }),
                        _ => unimplemented!(),
                    },
                    ..Default::default()
                })
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                false
            }
        }

        workbench.init([cover(0, CoverType::Partial), cover(1, CoverType::Full)]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(WHITEF),
            cached_tile: None,
            channels: RGBA,
            clear_color: WHITEF,
        };

        assert_eq!(
            workbench.drive_tile_painting(&mut UnimplementedPainter, &context),
            TileWriteOp::Solid([224, 224, 224, 255]),
        );
    }

    #[test]
    fn skip_visible_is_unchanged() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, layer_id: u32) -> Cow<'_, Props> {
                if layer_id == 2 {
                    return Cow::Owned(Props {
                        func: Func::Draw(Style {
                            fill: Fill::Solid(REDF),
                            ..Default::default()
                        }),
                        ..Default::default()
                    });
                }

                Cow::Owned(Props::default())
            }

            fn is_unchanged(&self, layer_id: u32) -> bool {
                layer_id != 0
            }
        }

        workbench.init([
            cover(0, CoverType::Partial),
            cover(1, CoverType::Partial),
            cover(2, CoverType::Full),
        ]);

        let cached_tiles = CachedTile::default();
        cached_tiles.update_layer_count(Some(3));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        // Tile has changed because layer 0 changed.
        assert_eq!(
            passes::tile_unchanged_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context
            ),
            ControlFlow::Continue(()),
        );
        // However, we can still skip drawing because everything visible is unchanged.
        assert_eq!(
            passes::skip_fully_covered_layers_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context,
            ),
            ControlFlow::Break(OptimizerTileWriteOp::None),
        );

        cached_tiles.update_layer_count(Some(2));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        // Tile has changed because layer 0 changed and number of layers has changed.
        assert_eq!(
            passes::tile_unchanged_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context
            ),
            ControlFlow::Continue(()),
        );
        // We can still skip the tile because any newly added layer is covered by an opaque layer.
        assert_eq!(
            passes::skip_fully_covered_layers_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context,
            ),
            ControlFlow::Break(OptimizerTileWriteOp::None),
        );

        cached_tiles.update_layer_count(Some(4));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        // Tile has changed because layer 0 changed and number of layers has changed.
        assert_eq!(
            passes::tile_unchanged_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context
            ),
            ControlFlow::Continue(()),
        );
        // This time we cannot skip because there might have been a visible layer
        // last frame that is now removed.
        assert_eq!(
            passes::skip_fully_covered_layers_pass(
                &mut workbench.state,
                &mut workbench.passes_shared_state,
                &context,
            ),
            ControlFlow::Break(OptimizerTileWriteOp::Solid(REDF)),
        );
    }

    #[test]
    fn skip_solid_color_is_unchanged() {
        let mut workbench = LayerWorkbench::default();

        struct TestProps;

        impl LayerProps for TestProps {
            fn get(&self, _layer_id: u32) -> Cow<'_, Props> {
                Cow::Owned(Props {
                    func: Func::Draw(Style {
                        fill: Fill::Solid(REDF),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
            }

            fn is_unchanged(&self, _layer_id: u32) -> bool {
                false
            }
        }

        workbench.init([cover(0, CoverType::Full)]);

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: None,
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        // We can't skip drawing because we don't have any cached tile.
        assert_eq!(
            workbench.drive_tile_painting(&mut UnimplementedPainter, &context),
            TileWriteOp::Solid(RED),
        );

        let cached_tiles = CachedTile::default();
        cached_tiles.update_layer_count(Some(0));
        cached_tiles.update_solid_color(Some(WHITE));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        // We can't skip drawing because the tile solid color (RED) is different from the previous one (WHITE).
        assert_eq!(
            workbench.drive_tile_painting(&mut UnimplementedPainter, &context),
            TileWriteOp::Solid(RED),
        );

        cached_tiles.update_layer_count(Some(0));
        cached_tiles.update_solid_color(Some(RED));

        let context = Context {
            tile_x: 0,
            tile_y: 0,
            segments: &[],
            props: &TestProps,
            cached_clear_color: Some(BLACKF),
            cached_tile: Some(&cached_tiles),
            channels: RGBA,
            clear_color: BLACKF,
        };

        workbench.populate_layers(&context);

        // We can skip drawing because the tile solid color is unchanged.
        assert_eq!(
            workbench.drive_tile_painting(&mut UnimplementedPainter, &context),
            TileWriteOp::None,
        );
    }
}
