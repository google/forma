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
    cell::RefCell,
    rc::{Rc, Weak},
};

use crate::{styling::Color, utils::SmallBitSet};

use super::painter::CachedTile;

pub mod layout;

use self::layout::{Flusher, Layout};

/// A short-lived description of the buffer being rendered into for the current frame.
///
/// # Examples
///
/// ```
/// # use forma_render::cpu::buffer::{BufferBuilder, layout::LinearLayout};
/// let width = 100;
/// let height = 100;
/// let mut buffer = vec![0; 100 * 4 * 100];
///
/// let _buffer = BufferBuilder::new(
///     &mut buffer,
///     &mut LinearLayout::new(width, width * 4, height),
/// ).build();
/// ```
#[derive(Debug)]
pub struct Buffer<'b, 'l, L: Layout> {
    pub(crate) buffer: &'b mut [u8],
    pub(crate) layout: &'l mut L,
    pub(crate) layer_cache: Option<BufferLayerCache>,
    pub(crate) flusher: Option<Box<dyn Flusher>>,
}

/// A builder for the [`Buffer`].
///
/// # Examples
///
/// ```
/// # use forma_render::cpu::buffer::{BufferBuilder, layout::LinearLayout};
/// let width = 100;
/// let height = 100;
/// let mut buffer = vec![0; 100 * 4 * 100];
/// let mut layout = LinearLayout::new(width, width * 4, height);
/// let _buffer = BufferBuilder::new(&mut buffer, &mut layout).build();
/// ```
#[derive(Debug)]
pub struct BufferBuilder<'b, 'l, L: Layout> {
    buffer: Buffer<'b, 'l, L>,
}

impl<'b, 'l, L: Layout> BufferBuilder<'b, 'l, L> {
    #[inline]
    pub fn new(buffer: &'b mut [u8], layout: &'l mut L) -> Self {
        Self {
            buffer: Buffer {
                buffer,
                layout,
                layer_cache: None,
                flusher: None,
            },
        }
    }

    #[inline]
    pub fn layer_cache(mut self, layer_cache: BufferLayerCache) -> Self {
        self.buffer.layer_cache = Some(layer_cache);
        self
    }

    #[inline]
    pub fn flusher(mut self, flusher: Box<dyn Flusher>) -> Self {
        self.buffer.flusher = Some(flusher);
        self
    }

    #[inline]
    pub fn build(self) -> Buffer<'b, 'l, L> {
        self.buffer
    }
}

#[derive(Debug)]
struct IdDropper {
    id: u8,
    buffers_with_caches: Weak<RefCell<SmallBitSet>>,
}

impl Drop for IdDropper {
    fn drop(&mut self) {
        if let Some(buffers_with_caches) = Weak::upgrade(&self.buffers_with_caches) {
            buffers_with_caches.borrow_mut().remove(self.id);
        }
    }
}

#[derive(Debug)]
pub(crate) struct CacheInner {
    pub clear_color: Option<Color>,
    pub tiles: Vec<CachedTile>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    _id_dropper: IdDropper,
}

/// A per-[`Buffer`] cache that enables forma to skip rendering to buffer
/// regions that have not changed.
///
/// # Examples
///
/// ```
/// # use forma_render::{
/// #     cpu::{buffer::{BufferBuilder, layout::LinearLayout}, Renderer, RGBA},
/// #     styling::Color, Composition,
/// # };
/// let mut buffer = vec![0; 4];
///
/// let mut composition = Composition::new();
/// let mut renderer = Renderer::new();
/// let layer_cache = renderer.create_buffer_layer_cache().unwrap();
///
/// renderer.render(
///     &mut composition,
///     &mut BufferBuilder::new(&mut buffer, &mut LinearLayout::new(1, 1 * 4, 1))
///         .layer_cache(layer_cache.clone())
///         .build(),
///     RGBA,
///     Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
///     None,
/// );
///
/// // Rendered white on first frame.
/// assert_eq!(buffer, [255; 4]);
///
/// // Reset buffer manually.
/// buffer = vec![0; 4];
///
/// renderer.render(
///     &mut composition,
///     &mut BufferBuilder::new(&mut buffer, &mut LinearLayout::new(1, 1 * 4, 1))
///         .layer_cache(layer_cache.clone())
///         .build(),
///     RGBA,
///     Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
///     None,
/// );
///
/// // Skipped rendering on second frame since nothing changed.
/// assert_eq!(buffer, [0; 4]);
/// ```
#[derive(Clone, Debug)]
pub struct BufferLayerCache {
    pub(crate) id: u8,
    pub(crate) cache: Rc<RefCell<CacheInner>>,
}

impl BufferLayerCache {
    pub(crate) fn new(id: u8, buffers_with_caches: Weak<RefCell<SmallBitSet>>) -> Self {
        Self {
            id,
            cache: Rc::new(RefCell::new(CacheInner {
                clear_color: None,
                tiles: Vec::new(),
                width: None,
                height: None,
                _id_dropper: IdDropper {
                    id,
                    buffers_with_caches,
                },
            })),
        }
    }

    #[inline]
    pub fn clear(&self) {
        let mut cache = self.cache.borrow_mut();

        cache.clear_color = None;
        cache.tiles.fill(CachedTile::default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem;

    fn new_cache(bit_set: &Rc<RefCell<SmallBitSet>>) -> BufferLayerCache {
        bit_set
            .borrow_mut()
            .first_empty_slot()
            .map(|id| BufferLayerCache::new(id, Rc::downgrade(bit_set)))
            .unwrap()
    }

    #[test]
    fn clone_and_drop() {
        let bit_set = Rc::new(RefCell::new(SmallBitSet::default()));

        let cache0 = new_cache(&bit_set);
        let cache1 = new_cache(&bit_set);
        let cache2 = new_cache(&bit_set);

        assert!(bit_set.borrow().contains(&0));
        assert!(bit_set.borrow().contains(&1));
        assert!(bit_set.borrow().contains(&2));

        mem::drop(cache0.clone());
        mem::drop(cache1.clone());
        mem::drop(cache2.clone());

        assert!(bit_set.borrow().contains(&0));
        assert!(bit_set.borrow().contains(&1));
        assert!(bit_set.borrow().contains(&2));

        mem::drop(cache1);

        assert!(bit_set.borrow().contains(&0));
        assert!(!bit_set.borrow().contains(&1));
        assert!(bit_set.borrow().contains(&2));

        let cache1 = new_cache(&bit_set);

        assert!(bit_set.borrow().contains(&0));
        assert!(bit_set.borrow().contains(&1));
        assert!(bit_set.borrow().contains(&2));

        mem::drop(cache0);
        mem::drop(cache1);
        mem::drop(cache2);

        assert!(!bit_set.borrow().contains(&0));
        assert!(!bit_set.borrow().contains(&1));
        assert!(!bit_set.borrow().contains(&2));
    }
}
