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

#[cfg(test)]
use std::cell::Ref;
use std::{cell::RefCell, rc::Rc};

use rustc_hash::FxHashMap;

use crate::{segment::GeomId, styling::Props, utils::SmallBitSet, Order};

mod interner;
mod layer;
mod state;

pub use self::{
    interner::{Interned, Interner},
    layer::{InnerLayer, Layer},
    state::{LayerSharedState, LayerSharedStateInner},
};

const LINES_GARBAGE_THRESHOLD: usize = 2;

/// A composition is an ordered collection of [`Layer`]s. It is the only means through which
/// content can be rendered.
///
/// The composition works similarly to a `HashMap<Order, Layer>`.
///
/// # Examples
///
/// ```
/// # use forma_render::prelude::*;
/// let mut composition = Composition::new();
///
/// let layer0 = composition.create_layer();
/// let layer1 = composition.create_layer();
///
/// assert!(composition.insert(Order::new(0).unwrap(), layer0).is_none());
/// assert!(composition.insert(Order::new(0).unwrap(), layer1).is_some()); // Some(layer0)
/// ```
#[derive(Debug, Default)]
pub struct Composition {
    pub(crate) layers: FxHashMap<Order, Layer>,
    pub(crate) shared_state: Rc<RefCell<LayerSharedStateInner>>,
}

impl Composition {
    /// Creates a new composition.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let composition = Composition::new();
    ///
    /// assert!(composition.is_empty());
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`Layer`] which is [enabled](Layer::is_enabled) by default.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let _layer = composition.create_layer();
    /// ```
    #[inline]
    pub fn create_layer(&mut self) -> Layer {
        let (geom_id, props) = {
            let mut state = self.shared_state.borrow_mut();

            let geom_id = state.new_geom_id();
            let props = state.props_interner.get(Props::default());

            (geom_id, props)
        };

        Layer {
            inner: InnerLayer::default(),
            shared_state: LayerSharedState::new(Rc::clone(&self.shared_state)),
            geom_id,
            props,
            is_unchanged: SmallBitSet::default(),
            lines_count: 0,
        }
    }

    /// Checks if the composition is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let composition = Composition::new();
    ///
    /// assert!(composition.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns the numbers of [`Layer`]s in this composition.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let composition = Composition::new();
    ///
    /// assert_eq!(composition.len(), 0);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Inserts a `layer` at specified `order` and optionally returns the layer already at `order` if any.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let mut layer0 = composition.create_layer();
    /// let layer1 = composition.create_layer();
    ///
    /// layer0.disable();
    ///
    /// assert!(composition.insert(Order::new(0).unwrap(), layer0).is_none());
    ///
    /// match composition.insert(Order::new(0).unwrap(), layer1) {
    ///     Some(layer0) if !layer0.is_enabled() => (),
    ///     _ => unreachable!(),
    /// }
    /// ```
    #[inline]
    pub fn insert(&mut self, order: Order, mut layer: Layer) -> Option<Layer> {
        assert_eq!(
            &layer.shared_state, &self.shared_state,
            "Layer was crated by a different Composition"
        );

        layer.set_order(Some(order));

        self.layers.insert(order, layer).map(|mut layer| {
            layer.set_order(None);

            layer
        })
    }

    /// Removes a `layer` at specified `order` and optionally returns the layer already at `order` if any.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let layer0 = composition.create_layer();
    /// let layer1 = composition.create_layer();
    ///
    /// assert!(composition.insert(Order::new(0).unwrap(), layer0).is_none());
    /// assert!(composition.insert(Order::new(0).unwrap(), layer1).is_some()); // Some(layer0)
    /// ```
    #[inline]
    pub fn remove(&mut self, order: Order) -> Option<Layer> {
        self.layers.remove(&order).map(|mut layer| {
            layer.set_order(None);

            layer
        })
    }

    /// Optionally recover the order of presently-stored [`Layer`] by its `geom_id`. Bear in mind
    /// that [clearing](Layer::clear) its geometry will reset the ID which will need to be
    /// [refreshed](Layer::geom_id).
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let layer = composition.create_layer();
    ///
    /// let id0 = layer.geom_id();
    /// let order = Order::new(0).unwrap();
    ///
    /// assert_eq!(composition.get_order_if_stored(id0), None);
    ///
    /// // Post-insertion, the geometry is accessible by ID.
    /// composition.insert(Order::new(0).unwrap(), layer);
    /// assert_eq!(composition.get_order_if_stored(id0), Some(order));
    ///
    /// let layer = composition.get_mut(order).unwrap();
    ///
    /// layer.clear();
    ///
    /// let id1 = layer.geom_id();
    ///
    /// assert_ne!(id0, id1);
    ///
    /// // Old ID cannot be used any longer.
    /// assert_eq!(composition.get_order_if_stored(id0), None);
    ///
    /// // Since is has not been removed, the geometry is still accessible by ID.
    /// assert_eq!(composition.get_order_if_stored(id1), Some(order));
    ///
    /// // Post-remove, the geometry is no longer accessible.
    /// composition.remove(order);
    /// assert_eq!(composition.get_order_if_stored(id1), None);
    /// ```
    #[inline]
    pub fn get_order_if_stored(&self, geom_id: GeomId) -> Option<Order> {
        self.shared_state
            .borrow()
            .geom_id_to_order
            .get(&geom_id)
            .copied()
            .flatten()
    }

    /// Optionally returns a reference to the layer stored at specified `order`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let layer = composition.create_layer();
    /// let order = Order::new(0).unwrap();
    ///
    /// composition.insert(order, layer);
    ///
    /// assert!(composition.get(order).is_some());
    /// ```
    #[inline]
    pub fn get(&self, order: Order) -> Option<&Layer> {
        self.layers.get(&order)
    }

    /// Optionally returns a mutable reference to the layer stored at specified `order`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let layer = composition.create_layer();
    /// let order = Order::new(0).unwrap();
    ///
    /// composition.insert(order, layer);
    ///
    /// assert!(composition.get_mut(order).is_some());
    /// ```
    #[inline]
    pub fn get_mut(&mut self, order: Order) -> Option<&mut Layer> {
        self.layers.get_mut(&order)
    }

    /// Creates and inserts a default layer into the composition, returning a mutable reference
    /// to it. it is a combined form of [`Self::create_layer`] and
    /// [`Self::insert`]/[`Self::get_mut`].
    ///
    /// It is especially useful when wanting to update a specific layer for multiple frames.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let line = PathBuilder::new().line_to(Point::new(10.0, 10.0)).build();
    ///
    /// composition.get_mut_or_insert_default(Order::new(0).unwrap()).insert(&line);
    /// ```
    #[inline]
    pub fn get_mut_or_insert_default(&mut self, order: Order) -> &mut Layer {
        if !self.layers.contains_key(&order) {
            let layer = self.create_layer();
            self.insert(order, layer);
        }

        self.get_mut(order).unwrap()
    }

    /// Returns an [`ExactSizeIterator`] over all order/layer reference pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// composition.get_mut_or_insert_default(Order::new(0).unwrap());
    /// composition.get_mut_or_insert_default(Order::new(1).unwrap());
    ///
    /// assert_eq!(composition.layers().count(), 2);
    /// ```
    #[inline]
    pub fn layers(&self) -> impl ExactSizeIterator<Item = (Order, &Layer)> + '_ {
        self.layers.iter().map(|(&order, layer)| (order, layer))
    }

    /// Returns an [`ExactSizeIterator`] over all order/mutable layer reference pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// composition.get_mut_or_insert_default(Order::new(0).unwrap());
    /// composition.get_mut_or_insert_default(Order::new(1).unwrap());
    ///
    /// assert_eq!(composition.layers_mut().count(), 2);
    /// ```
    #[inline]
    pub fn layers_mut(&mut self) -> impl ExactSizeIterator<Item = (Order, &mut Layer)> + '_ {
        self.layers.iter_mut().map(|(&order, layer)| (order, layer))
    }

    fn builder_len(&self) -> usize {
        self.shared_state
            .borrow()
            .segment_buffer
            .as_ref()
            .expect("lines_builder should not be None")
            .len()
    }

    fn actual_len(&self) -> usize {
        self.layers.values().map(|layer| layer.lines_count).sum()
    }

    /// Forces the composition to run geometry garbage collection if more than half the memory
    /// occupied by it is not accessible anymore by the end-user. (i.e. by [`Layer::clear`] or
    /// [`Layer::drop`])
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// // Insertions and removals/clears happen here.
    ///
    /// composition.compact_geom();
    /// ```
    #[inline]
    pub fn compact_geom(&mut self) {
        if self.builder_len() >= self.actual_len() * LINES_GARBAGE_THRESHOLD {
            let state = &mut *self.shared_state.borrow_mut();
            let lines_builder = &mut state.segment_buffer;
            let geom_id_to_order = &mut state.geom_id_to_order;

            lines_builder
                .as_mut()
                .expect("lines_builder should not be None")
                .retain(|id| geom_id_to_order.contains_key(&id));
        }
    }

    #[cfg(test)]
    pub fn layers_for_segments(
        &self,
    ) -> (
        &FxHashMap<Order, Layer>,
        Ref<FxHashMap<GeomId, Option<Order>>>,
    ) {
        (
            &self.layers,
            Ref::map(self.shared_state.borrow(), |state| &state.geom_id_to_order),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        consts::cpu::{TILE_HEIGHT, TILE_WIDTH},
        cpu::{
            buffer::{layout::LinearLayout, BufferBuilder},
            Renderer, RGBA,
        },
        math::{GeomPresTransform, Point},
        styling::{Color, Fill, FillRule, Func, Style},
        Path, PathBuilder,
    };

    const BLACK_SRGB: [u8; 4] = [0x00, 0x00, 0x00, 0xFF];
    const GRAY_SRGB: [u8; 4] = [0xBB, 0xBB, 0xBB, 0xFF];
    const GRAY_ALPHA_50_SRGB: [u8; 4] = [0xBB, 0xBB, 0xBB, 0x80];
    const WHITE_ALPHA_0_SRGB: [u8; 4] = [0xFF, 0xFF, 0xFF, 0x00];
    const RED_SRGB: [u8; 4] = [0xFF, 0x00, 0x00, 0xFF];
    const GREEN_SRGB: [u8; 4] = [0x00, 0xFF, 0x00, 0xFF];
    const RED_50_GREEN_50_SRGB: [u8; 4] = [0xBB, 0xBB, 0x00, 0xFF];

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
    const GRAY: Color = Color {
        r: 0.5,
        g: 0.5,
        b: 0.5,
        a: 1.0,
    };
    const WHITE_TRANSPARENT: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 0.0,
    };
    const RED: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    const GREEN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };

    fn pixel_path(x: i32, y: i32) -> Path {
        let mut builder = PathBuilder::new();

        builder.move_to(Point::new(x as f32, y as f32));
        builder.line_to(Point::new(x as f32, (y + 1) as f32));
        builder.line_to(Point::new((x + 1) as f32, (y + 1) as f32));
        builder.line_to(Point::new((x + 1) as f32, y as f32));
        builder.line_to(Point::new(x as f32, y as f32));

        builder.build()
    }

    fn solid(color: Color) -> Props {
        Props {
            func: Func::Draw(Style {
                fill: Fill::Solid(color),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn composition_len() {
        let mut composition = Composition::new();

        assert!(composition.is_empty());
        assert_eq!(composition.len(), 0);

        composition.get_mut_or_insert_default(Order::new(0).unwrap());

        assert!(!composition.is_empty());
        assert_eq!(composition.len(), 1);
    }

    #[test]
    fn background_color_clear() {
        let mut buffer = [GREEN_SRGB].concat();
        let mut layout = LinearLayout::new(1, 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            RED,
            None,
        );

        assert_eq!(buffer, [RED_SRGB].concat());
    }

    #[test]
    fn background_color_clear_when_changed() {
        let mut buffer = [GREEN_SRGB].concat();
        let mut layout = LinearLayout::new(1, 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();
        let layer_cache = renderer.create_buffer_layer_cache().unwrap();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone())
                .build(),
            RGBA,
            RED,
            None,
        );

        assert_eq!(buffer, [RED_SRGB].concat());

        buffer = [GREEN_SRGB].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone())
                .build(),
            RGBA,
            RED,
            None,
        );

        // Skip clearing if the color is the same.
        assert_eq!(buffer, [GREEN_SRGB].concat());

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache)
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [BLACK_SRGB].concat());
    }

    #[test]
    fn one_pixel() {
        let mut buffer = [GREEN_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(1, 0)).set_props(solid(RED));

        composition.insert(Order::new(0).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            GREEN,
            None,
        );

        assert_eq!(buffer, [GREEN_SRGB, RED_SRGB, GREEN_SRGB].concat());
    }

    #[test]
    fn two_pixels_same_layer() {
        let mut buffer = [GREEN_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);
        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let mut layer = composition.create_layer();
        layer
            .insert(&pixel_path(1, 0))
            .insert(&pixel_path(2, 0))
            .set_props(solid(RED));

        composition.insert(Order::new(0).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            GREEN,
            None,
        );

        assert_eq!(buffer, [GREEN_SRGB, RED_SRGB, RED_SRGB].concat());
    }

    #[test]
    fn one_pixel_translated() {
        let mut buffer = [GREEN_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let mut layer = composition.create_layer();
        layer
            .insert(&pixel_path(1, 0))
            .set_props(solid(RED))
            .set_transform(GeomPresTransform::try_from([1.0, 0.0, 0.0, 1.0, 0.5, 0.0]).unwrap());

        composition.insert(Order::new(0).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            GREEN,
            None,
        );

        assert_eq!(
            buffer,
            [GREEN_SRGB, RED_50_GREEN_50_SRGB, RED_50_GREEN_50_SRGB].concat()
        );
    }

    #[test]
    fn one_pixel_rotated() {
        let mut buffer = [GREEN_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let angle = -std::f32::consts::PI / 2.0;

        let mut layer = composition.create_layer();
        layer
            .insert(&pixel_path(-1, 1))
            .set_props(solid(RED))
            .set_transform(
                GeomPresTransform::try_from([
                    angle.cos(),
                    -angle.sin(),
                    angle.sin(),
                    angle.cos(),
                    0.0,
                    0.0,
                ])
                .unwrap(),
            );

        composition.insert(Order::new(0).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            GREEN,
            None,
        );

        assert_eq!(buffer, [GREEN_SRGB, RED_SRGB, GREEN_SRGB].concat());
    }

    #[test]
    fn clear_and_resize() {
        let mut buffer = [GREEN_SRGB; 4].concat();
        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let order0 = Order::new(0).unwrap();
        let order1 = Order::new(1).unwrap();
        let order2 = Order::new(2).unwrap();

        let mut layer0 = composition.create_layer();
        layer0.insert(&pixel_path(0, 0)).set_props(solid(RED));

        composition.insert(order0, layer0);

        let mut layer1 = composition.create_layer();
        layer1.insert(&pixel_path(1, 0)).set_props(solid(RED));

        composition.insert(order1, layer1);

        let mut layer2 = composition.create_layer();
        layer2
            .insert(&pixel_path(2, 0))
            .insert(&pixel_path(3, 0))
            .set_props(solid(RED));

        composition.insert(order2, layer2);

        let mut layout = LinearLayout::new(4, 4 * 4, 1);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            GREEN,
            None,
        );

        assert_eq!(buffer, [RED_SRGB, RED_SRGB, RED_SRGB, RED_SRGB].concat());
        assert_eq!(composition.builder_len(), 16);
        assert_eq!(composition.actual_len(), 16);

        buffer = [GREEN_SRGB; 4].concat();

        let mut layout = LinearLayout::new(4, 4 * 4, 1);

        composition.get_mut(order0).unwrap().clear();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            GREEN,
            None,
        );

        assert_eq!(buffer, [GREEN_SRGB, RED_SRGB, RED_SRGB, RED_SRGB].concat());
        assert_eq!(composition.builder_len(), 16);
        assert_eq!(composition.actual_len(), 12);

        buffer = [GREEN_SRGB; 4].concat();

        let mut layout = LinearLayout::new(4, 4 * 4, 1);

        composition.get_mut(order2).unwrap().clear();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            GREEN,
            None,
        );

        assert_eq!(
            buffer,
            [GREEN_SRGB, RED_SRGB, GREEN_SRGB, GREEN_SRGB].concat()
        );
        assert_eq!(composition.builder_len(), 4);
        assert_eq!(composition.actual_len(), 4);
    }

    #[test]
    fn clear_twice() {
        let mut composition = Composition::new();

        let order = Order::new(0).unwrap();

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(0, 0)).set_props(solid(RED));

        composition.insert(order, layer);

        assert_eq!(composition.actual_len(), 4);

        composition.get_mut(order).unwrap().clear();

        assert_eq!(composition.actual_len(), 0);

        composition.get_mut(order).unwrap().clear();

        assert_eq!(composition.actual_len(), 0);
    }

    #[test]
    fn insert_over_layer() {
        let mut buffer = [BLACK_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(0, 0)).set_props(solid(RED));

        composition.insert(Order::new(0).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [RED_SRGB, BLACK_SRGB, BLACK_SRGB].concat());

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(1, 0)).set_props(solid(GREEN));

        buffer = [BLACK_SRGB; 3].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [RED_SRGB, BLACK_SRGB, BLACK_SRGB].concat());

        composition.insert(Order::new(0).unwrap(), layer);

        buffer = [BLACK_SRGB; 3].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [BLACK_SRGB, GREEN_SRGB, BLACK_SRGB].concat());
    }

    #[test]
    fn layer_replace_remove() {
        let mut buffer = [BLACK_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(0, 0)).set_props(solid(RED));

        composition.insert(Order::new(0).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [RED_SRGB, BLACK_SRGB, BLACK_SRGB].concat());

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(1, 0)).set_props(solid(GREEN));

        let _old_layer = composition.insert(Order::new(0).unwrap(), layer);

        buffer = [BLACK_SRGB; 3].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [BLACK_SRGB, GREEN_SRGB, BLACK_SRGB].concat());

        let _old_layer = composition.remove(Order::new(0).unwrap());

        buffer = [BLACK_SRGB; 3].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [BLACK_SRGB, BLACK_SRGB, BLACK_SRGB].concat());
    }

    #[test]
    fn layer_clear() {
        let mut buffer = [BLACK_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let order = Order::new(0).unwrap();

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(0, 0)).set_props(solid(RED));

        composition.insert(order, layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [RED_SRGB, BLACK_SRGB, BLACK_SRGB].concat());

        composition
            .get_mut(order)
            .unwrap()
            .insert(&pixel_path(1, 0));

        buffer = [BLACK_SRGB; 3].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [RED_SRGB, RED_SRGB, BLACK_SRGB].concat());

        composition.get_mut(order).unwrap().clear();

        buffer = [BLACK_SRGB; 3].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [BLACK_SRGB, BLACK_SRGB, BLACK_SRGB].concat());

        composition
            .get_mut(order)
            .unwrap()
            .insert(&pixel_path(2, 0));

        buffer = [BLACK_SRGB; 3].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [BLACK_SRGB, BLACK_SRGB, RED_SRGB].concat());
    }

    #[test]
    fn geom_id() {
        let mut composition = Composition::new();

        let mut layer = composition.create_layer();

        layer.insert(&PathBuilder::new().build());
        let geom_id0 = layer.geom_id();

        layer.insert(&PathBuilder::new().build());
        let geom_id1 = layer.geom_id();

        assert_eq!(geom_id0, geom_id1);

        layer.clear();

        assert_ne!(layer.geom_id(), geom_id0);

        layer.insert(&PathBuilder::new().build());
        let geom_id2 = layer.geom_id();

        assert_ne!(geom_id0, geom_id2);

        let order = Order::new(0).unwrap();
        composition.insert(order, layer);

        assert_eq!(composition.get_order_if_stored(geom_id2), Some(order));

        let layer = composition.create_layer();
        composition.insert(order, layer);

        assert_eq!(composition.get_order_if_stored(geom_id2), None);
    }

    #[test]
    fn srgb_alpha_blending() {
        let mut buffer = [BLACK_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let mut layer = composition.create_layer();
        layer
            .insert(&pixel_path(0, 0))
            .set_props(solid(BLACK_ALPHA_50));

        composition.insert(Order::new(0).unwrap(), layer);

        let mut layer = composition.create_layer();

        layer.insert(&pixel_path(1, 0)).set_props(solid(GRAY));

        composition.insert(Order::new(1).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            WHITE_TRANSPARENT,
            None,
        );

        assert_eq!(
            buffer,
            [GRAY_ALPHA_50_SRGB, GRAY_SRGB, WHITE_ALPHA_0_SRGB].concat()
        );
    }

    #[test]
    fn render_changed_layers_only() {
        let mut buffer = [BLACK_SRGB; 3 * TILE_WIDTH * TILE_HEIGHT].concat();
        let mut layout = LinearLayout::new(3 * TILE_WIDTH, 3 * TILE_WIDTH * 4, TILE_HEIGHT);
        let mut composition = Composition::new();
        let mut renderer = Renderer::new();
        let layer_cache = renderer.create_buffer_layer_cache();

        let mut layer = composition.create_layer();
        layer
            .insert(&pixel_path(0, 0))
            .insert(&pixel_path(TILE_WIDTH as i32, 0))
            .set_props(solid(RED));

        composition.insert(Order::new(0).unwrap(), layer);

        let order = Order::new(1).unwrap();

        let mut layer = composition.create_layer();
        layer
            .insert(&pixel_path(TILE_WIDTH as i32 + 1, 0))
            .insert(&pixel_path(2 * TILE_WIDTH as i32, 0))
            .set_props(solid(GREEN));

        composition.insert(order, layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone().unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], RED_SRGB);
        assert_eq!(buffer[TILE_WIDTH * 4..TILE_WIDTH * 4 + 4], RED_SRGB);
        assert_eq!(
            buffer[(TILE_WIDTH + 1) * 4..(TILE_WIDTH + 1) * 4 + 4],
            GREEN_SRGB
        );
        assert_eq!(
            buffer[2 * TILE_WIDTH * 4..2 * TILE_WIDTH * 4 + 4],
            GREEN_SRGB
        );

        let mut buffer = [BLACK_SRGB; 3 * TILE_WIDTH * TILE_HEIGHT].concat();

        composition.get_mut(order).unwrap().set_props(solid(RED));

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], BLACK_SRGB);
        assert_eq!(buffer[TILE_WIDTH * 4..TILE_WIDTH * 4 + 4], RED_SRGB);
        assert_eq!(
            buffer[(TILE_WIDTH + 1) * 4..(TILE_WIDTH + 1) * 4 + 4],
            RED_SRGB
        );
        assert_eq!(buffer[2 * TILE_WIDTH * 4..2 * TILE_WIDTH * 4 + 4], RED_SRGB);
    }

    #[test]
    fn insert_remove_same_order_will_not_render_again() {
        let mut buffer = [BLACK_SRGB; 3].concat();
        let mut layout = LinearLayout::new(3, 3 * 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();
        let layer_cache = renderer.create_buffer_layer_cache().unwrap();

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(0, 0)).set_props(solid(RED));

        composition.insert(Order::new(0).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [RED_SRGB, BLACK_SRGB, BLACK_SRGB].concat());

        let layer = composition.remove(Order::new(0).unwrap()).unwrap();
        composition.insert(Order::new(0).unwrap(), layer);

        buffer = [BLACK_SRGB; 3].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache)
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer, [BLACK_SRGB, BLACK_SRGB, BLACK_SRGB].concat());
    }

    #[test]
    fn clear_emptied_tiles() {
        let mut buffer = [BLACK_SRGB; 2 * TILE_WIDTH * TILE_HEIGHT].concat();
        let mut layout = LinearLayout::new(2 * TILE_WIDTH, 2 * TILE_WIDTH * 4, TILE_HEIGHT);
        let mut composition = Composition::new();
        let mut renderer = Renderer::new();
        let layer_cache = renderer.create_buffer_layer_cache();

        let order = Order::new(0).unwrap();

        let mut layer = composition.create_layer();
        layer
            .insert(&pixel_path(0, 0))
            .set_props(solid(RED))
            .insert(&pixel_path(TILE_WIDTH as i32, 0));

        composition.insert(order, layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone().unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], RED_SRGB);

        composition.get_mut(order).unwrap().set_transform(
            GeomPresTransform::try_from([1.0, 0.0, 0.0, 1.0, TILE_WIDTH as f32, 0.0]).unwrap(),
        );

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone().unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], BLACK_SRGB);

        composition.get_mut(order).unwrap().set_transform(
            GeomPresTransform::try_from([1.0, 0.0, 0.0, 1.0, -(TILE_WIDTH as f32), 0.0]).unwrap(),
        );

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone().unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], RED_SRGB);

        composition.get_mut(order).unwrap().set_transform(
            GeomPresTransform::try_from([1.0, 0.0, 0.0, 1.0, 0.0, TILE_HEIGHT as f32]).unwrap(),
        );

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], BLACK_SRGB);
    }

    #[test]
    fn separate_layer_caches() {
        let mut buffer = [BLACK_SRGB; TILE_WIDTH * TILE_HEIGHT].concat();
        let mut layout = LinearLayout::new(TILE_WIDTH, TILE_WIDTH * 4, TILE_HEIGHT);
        let mut composition = Composition::new();
        let mut renderer = Renderer::new();
        let layer_cache0 = renderer.create_buffer_layer_cache();
        let layer_cache1 = renderer.create_buffer_layer_cache();

        let order = Order::new(0).unwrap();

        let mut layer = composition.create_layer();
        layer.insert(&pixel_path(0, 0)).set_props(solid(RED));

        composition.insert(order, layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache0.clone().unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], RED_SRGB);

        let mut buffer = [BLACK_SRGB; TILE_WIDTH * TILE_HEIGHT].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache0.clone().unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], BLACK_SRGB);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache1.clone().unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], RED_SRGB);

        composition
            .get_mut(order)
            .unwrap()
            .set_transform(GeomPresTransform::try_from([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).unwrap());

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache0.unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], BLACK_SRGB);
        assert_eq!(buffer[4..8], RED_SRGB);

        let mut buffer = [BLACK_SRGB; TILE_WIDTH * TILE_HEIGHT].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache1.unwrap())
                .build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], BLACK_SRGB);
        assert_eq!(buffer[4..8], RED_SRGB);
    }

    #[test]
    fn draw_if_width_or_height_change() {
        let mut buffer = [BLACK_SRGB; 1].concat();
        let mut layout = LinearLayout::new(1, 4, 1);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();
        let layer_cache = renderer.create_buffer_layer_cache().unwrap();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone())
                .build(),
            RGBA,
            RED,
            None,
        );

        assert_eq!(buffer[0..4], RED_SRGB);

        buffer = [BLACK_SRGB; 1].concat();

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone())
                .build(),
            RGBA,
            RED,
            None,
        );

        assert_eq!(buffer[0..4], BLACK_SRGB);

        buffer = [BLACK_SRGB; 2].concat();
        layout = LinearLayout::new(2, 2 * 4, 1);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache.clone())
                .build(),
            RGBA,
            RED,
            None,
        );

        assert_eq!(buffer[0..8], [RED_SRGB; 2].concat());

        buffer = [BLACK_SRGB; 2].concat();
        layout = LinearLayout::new(1, 4, 2);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout)
                .layer_cache(layer_cache)
                .build(),
            RGBA,
            RED,
            None,
        );

        assert_eq!(buffer[0..8], [RED_SRGB; 2].concat());
    }

    #[test]
    fn even_odd() {
        let mut builder = PathBuilder::new();

        builder.move_to(Point::new(0.0, 0.0));
        builder.line_to(Point::new(0.0, TILE_HEIGHT as f32));
        builder.line_to(Point::new(3.0 * TILE_WIDTH as f32, TILE_HEIGHT as f32));
        builder.line_to(Point::new(3.0 * TILE_WIDTH as f32, 0.0));
        builder.line_to(Point::new(TILE_WIDTH as f32, 0.0));
        builder.line_to(Point::new(TILE_WIDTH as f32, TILE_HEIGHT as f32));
        builder.line_to(Point::new(2.0 * TILE_WIDTH as f32, TILE_HEIGHT as f32));
        builder.line_to(Point::new(2.0 * TILE_WIDTH as f32, 0.0));
        builder.line_to(Point::new(0.0, 0.0));

        let path = builder.build();

        let mut buffer = [BLACK_SRGB; 3 * TILE_WIDTH * TILE_HEIGHT].concat();
        let mut layout = LinearLayout::new(3 * TILE_WIDTH, 3 * TILE_WIDTH * 4, TILE_HEIGHT);

        let mut composition = Composition::new();
        let mut renderer = Renderer::new();

        let mut layer = composition.create_layer();
        layer.insert(&path).set_props(Props {
            fill_rule: FillRule::EvenOdd,
            func: Func::Draw(Style {
                fill: Fill::Solid(RED),
                ..Default::default()
            }),
        });

        composition.insert(Order::new(0).unwrap(), layer);

        renderer.render(
            &mut composition,
            &mut BufferBuilder::new(&mut buffer, &mut layout).build(),
            RGBA,
            BLACK,
            None,
        );

        assert_eq!(buffer[0..4], RED_SRGB);
        assert_eq!(buffer[TILE_WIDTH * 4..(TILE_WIDTH * 4 + 4)], BLACK_SRGB);
        assert_eq!(buffer[2 * TILE_WIDTH * 4..2 * TILE_WIDTH * 4 + 4], RED_SRGB);
    }
}
