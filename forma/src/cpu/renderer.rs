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
    cell::{RefCell, RefMut},
    ops::Range,
    rc::Rc,
};

use rustc_hash::FxHashMap;

use crate::{
    consts,
    cpu::painter::{self, CachedTile, LayerProps},
    styling::{Color, Props},
    utils::{Order, SmallBitSet},
    Composition, Layer,
};

use super::{
    buffer::{layout::Layout, Buffer, BufferLayerCache},
    Channel, Rasterizer,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Rect {
    pub(crate) hor: Range<usize>,
    pub(crate) vert: Range<usize>,
}

impl Rect {
    /// The resulting rectangle is currently approximated to the tile grid.
    pub fn new(horizontal: Range<usize>, vertical: Range<usize>) -> Self {
        Self {
            hor: horizontal.start / consts::cpu::TILE_WIDTH
                ..(horizontal.end + consts::cpu::TILE_WIDTH - 1) / consts::cpu::TILE_WIDTH,
            vert: vertical.start / consts::cpu::TILE_HEIGHT
                ..(vertical.end + consts::cpu::TILE_HEIGHT - 1) / consts::cpu::TILE_HEIGHT,
        }
    }
}

#[derive(Debug, Default)]
pub struct Renderer {
    rasterizer: Rasterizer<{ consts::cpu::TILE_WIDTH }, { consts::cpu::TILE_HEIGHT }>,
    buffers_with_caches: Rc<RefCell<SmallBitSet>>,
}

impl Renderer {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn create_buffer_layer_cache(&mut self) -> Option<BufferLayerCache> {
        self.buffers_with_caches
            .borrow_mut()
            .first_empty_slot()
            .map(|id| BufferLayerCache::new(id, Rc::downgrade(&self.buffers_with_caches)))
    }

    pub fn render<L>(
        &mut self,
        composition: &mut Composition,
        buffer: &mut Buffer<'_, '_, L>,
        mut channels: [Channel; 4],
        clear_color: Color,
        crop: Option<Rect>,
    ) where
        L: Layout,
    {
        // If `clear_color` has alpha = 1 we can upgrade the alpha channel to `Channel::One`
        // in order to skip reading the alpha channel.
        if clear_color.a == 1.0 {
            channels = channels.map(|c| match c {
                Channel::Alpha => Channel::One,
                c => c,
            });
        }

        if let Some(layer_cache) = buffer.layer_cache.as_ref() {
            let tiles_len = buffer.layout.width_in_tiles() * buffer.layout.height_in_tiles();
            let cache = &layer_cache.cache;

            cache
                .borrow_mut()
                .tiles
                .resize(tiles_len, CachedTile::default());

            if cache.borrow().width != Some(buffer.layout.width())
                || cache.borrow().height != Some(buffer.layout.height())
            {
                cache.borrow_mut().width = Some(buffer.layout.width());
                cache.borrow_mut().height = Some(buffer.layout.height());

                layer_cache.clear();
            }
        }

        composition.compact_geom();
        composition
            .shared_state
            .borrow_mut()
            .props_interner
            .compact();

        let layers = &composition.layers;
        let shared_state = &mut *composition.shared_state.borrow_mut();
        let segment_buffer = &mut shared_state.segment_buffer;
        let geom_id_to_order = &shared_state.geom_id_to_order;
        let rasterizer = &mut self.rasterizer;

        struct CompositionContext<'l> {
            layers: &'l FxHashMap<Order, Layer>,
            cache_id: Option<u8>,
        }

        impl LayerProps for CompositionContext<'_> {
            #[inline]
            fn get(&self, id: u32) -> Cow<'_, Props> {
                Cow::Borrowed(
                    self.layers
                        .get(&Order::new(id).expect("PixelSegment layer_id cannot overflow Order"))
                        .map(|layer| &layer.props)
                        .expect(
                            "Layers outside of HashMap should not produce visible PixelSegments",
                        ),
                )
            }

            #[inline]
            fn is_unchanged(&self, id: u32) -> bool {
                match self.cache_id {
                    None => false,
                    Some(cache_id) => self
                        .layers
                        .get(&Order::new(id).expect("PixelSegment layer_id cannot overflow Order"))
                        .map(|layer| layer.is_unchanged(cache_id))
                        .expect(
                            "Layers outside of HashMap should not produce visible PixelSegments",
                        ),
                }
            }
        }

        let context = CompositionContext {
            layers,
            cache_id: buffer.layer_cache.as_ref().map(|cache| cache.id),
        };

        // `take()` sets the RefCell's content with `Default::default()` which is cheap for Option.
        let taken_buffer = segment_buffer
            .take()
            .expect("segment_buffer should not be None");

        *segment_buffer = {
            let segment_buffer_view = {
                duration!("gfx", "SegmentBuffer::fill_cpu_view");
                taken_buffer.fill_cpu_view(
                    buffer.layout.width(),
                    buffer.layout.height(),
                    context.layers,
                    geom_id_to_order,
                )
            };

            {
                duration!("gfx", "Rasterizer::rasterize");
                rasterizer.rasterize(&segment_buffer_view);
            }
            {
                duration!("gfx", "Rasterizer::sort");
                rasterizer.sort();
            }

            let previous_clear_color = buffer
                .layer_cache
                .as_ref()
                .and_then(|layer_cache| layer_cache.cache.borrow().clear_color);

            let cached_tiles = buffer.layer_cache.as_ref().map(|layer_cache| {
                RefMut::map(layer_cache.cache.borrow_mut(), |cache| &mut cache.tiles)
            });

            {
                duration!("gfx", "painter::for_each_row");
                painter::for_each_row(
                    buffer.layout,
                    buffer.buffer,
                    channels,
                    buffer.flusher.as_deref(),
                    previous_clear_color,
                    cached_tiles,
                    rasterizer.segments(),
                    clear_color,
                    &crop,
                    &context,
                );
            }

            Some(segment_buffer_view.recycle())
        };

        if let Some(buffer_layer_cache) = &buffer.layer_cache {
            buffer_layer_cache.cache.borrow_mut().clear_color = Some(clear_color);

            for layer in composition.layers.values_mut() {
                layer.set_is_unchanged(buffer_layer_cache.id, layer.inner.is_enabled);
            }
        }
    }
}
