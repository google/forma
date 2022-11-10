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

use crate::{math::GeomPresTransform, styling::Props, utils::SmallBitSet, GeomId, Order, Path};

use super::{Interned, LayerSharedState};

/// Small, compact version of a [`Layer`] that implements [`Send`] and is used by the
/// [`SegmentBuffer`].
///
/// This type should be as small as possible since it gets copied on use and thus
/// should only contain information needed by the [`SegmentBuffer`].
///
/// [`SegmentBuffer`]: crate::segment::SegmentBuffer
#[derive(Clone, Debug)]
pub struct InnerLayer {
    pub is_enabled: bool,
    pub affine_transform: Option<GeomPresTransform>,
    pub order: Option<Order>,
}

impl Default for InnerLayer {
    fn default() -> Self {
        Self {
            is_enabled: true,
            affine_transform: None,
            order: None,
        }
    }
}

/// A layer is a *reusable* collection of geometry (i.e. [`Path`]s) with common properties and
/// order in the paint stack.
///
/// They are created by calling [`Composition::create_layer`] or
/// [`Composition::get_mut_or_insert_default`].
///
/// [`Composition::create_layer`]: crate::Composition::create_layer
/// [`Composition::get_mut_or_insert_default`]: crate::Composition::get_mut_or_insert_default
///
/// # Examples
///
/// ```
/// # use forma_render::prelude::*;
/// let mut composition = Composition::new();
///
/// let _layer = composition.get_mut_or_insert_default(Order::new(0).unwrap());
/// ```
#[derive(Debug)]
pub struct Layer {
    pub(crate) inner: InnerLayer,
    pub(crate) shared_state: LayerSharedState,
    pub(crate) geom_id: GeomId,
    pub(crate) props: Interned<Props>,
    pub(crate) is_unchanged: SmallBitSet,
    pub(crate) lines_count: usize,
}

impl Layer {
    /// Inserts `path` into the geometry of the layer.
    ///
    /// The inserted paths basically contribute to a single internal path containing the geometry
    /// of all the paths.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let line0 = PathBuilder::new().line_to(Point::new(10.0, 10.0)).build();
    /// let line1 = PathBuilder::new().line_to(Point::new(10.0, 0.0)).build();
    ///
    /// composition
    ///     .get_mut_or_insert_default(Order::new(0).unwrap())
    ///     .insert(&line0)
    ///     .insert(&line1);
    /// ```
    pub fn insert(&mut self, path: &Path) -> &mut Self {
        {
            let mut state = self.shared_state.inner();
            let builder = state
                .segment_buffer
                .as_mut()
                .expect("lines_builder should not be None");

            let old_len = builder.len();
            builder.push_path(self.geom_id, path);
            let len = builder.len() - old_len;

            state
                .geom_id_to_order
                .insert(self.geom_id, self.inner.order);

            self.lines_count += len;
        }

        self.is_unchanged.clear();
        self
    }

    /// Clears the geometry stored in the layer and resets its [geometry ID](Self::geom_id).
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let mut layer = composition.create_layer();
    ///
    /// let initial_id = layer.geom_id();
    ///
    /// layer.clear();
    ///
    /// assert_ne!(layer.geom_id(), initial_id);
    /// ```
    pub fn clear(&mut self) -> &mut Self {
        {
            let mut state = self.shared_state.inner();

            state.geom_id_to_order.remove(&self.geom_id);

            self.geom_id = state.new_geom_id();
            state
                .geom_id_to_order
                .insert(self.geom_id, self.inner.order);
        }

        self.lines_count = 0;

        self.is_unchanged.clear();
        self
    }

    pub(crate) fn set_order(&mut self, order: Option<Order>) {
        if order.is_some() && self.inner.order != order {
            self.inner.order = order;
            self.is_unchanged.clear();
        }

        self.shared_state
            .inner()
            .geom_id_to_order
            .insert(self.geom_id, order);
    }

    /// Returns the layer's geometry ID.
    ///
    /// Used to retrieve the layer's [`Order`] if stored in a [`Composition`](crate::Composition).
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let order = Order::new(0).unwrap();
    /// let layer = composition.get_mut_or_insert_default(order);
    /// let id = layer.geom_id();
    ///
    /// assert_eq!(composition.get_order_if_stored(id), Some(order));
    /// ```
    pub fn geom_id(&self) -> GeomId {
        self.geom_id
    }

    pub(crate) fn is_unchanged(&self, cache_id: u8) -> bool {
        self.is_unchanged.contains(&cache_id)
    }

    pub(crate) fn set_is_unchanged(&mut self, cache_id: u8, is_unchanged: bool) -> bool {
        if is_unchanged {
            self.is_unchanged.insert(cache_id)
        } else {
            self.is_unchanged.remove(cache_id)
        }
    }

    /// Returns `true` if the layer is enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let layer = composition.create_layer();
    ///
    /// assert!(layer.is_enabled());
    /// ```
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.inner.is_enabled
    }

    /// Sets the layer's enabled state.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let mut layer = composition.create_layer();
    ///
    /// layer.set_is_enabled(false);
    ///
    /// assert!(!layer.is_enabled());
    /// ```
    #[inline]
    pub fn set_is_enabled(&mut self, is_enabled: bool) -> &mut Self {
        self.inner.is_enabled = is_enabled;
        self
    }

    /// Disables the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let mut layer = composition.create_layer();
    ///
    /// layer.disable();
    ///
    /// assert!(!layer.is_enabled());
    /// ```
    #[inline]
    pub fn disable(&mut self) -> &mut Self {
        self.set_is_enabled(false)
    }

    /// Enables the layer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let mut layer = composition.create_layer();
    ///
    /// layer.disable();
    /// layer.enable();
    ///
    /// assert!(layer.is_enabled());
    /// ```
    #[inline]
    pub fn enable(&mut self) -> &mut Self {
        self.set_is_enabled(true)
    }

    /// Returns the layer's transform.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let layer = composition.create_layer();
    ///
    /// assert!(layer.transform().is_identity());
    /// ```
    #[inline]
    pub fn transform(&self) -> GeomPresTransform {
        self.inner.affine_transform.unwrap_or_default()
    }

    /// Sets the layer's transform.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let mut layer = composition.create_layer();
    ///
    /// layer.set_transform(AffineTransform::from([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).try_into().unwrap());
    /// ```
    #[inline]
    pub fn set_transform(&mut self, transform: GeomPresTransform) -> &mut Self {
        // We want to perform a cheap check for the common case without hampering this function too
        // much.
        #[allow(clippy::float_cmp)]
        let affine_transform = (!transform.is_identity()).then_some(transform);

        if self.inner.affine_transform != affine_transform {
            self.is_unchanged.clear();
            self.inner.affine_transform = affine_transform;
        }

        self
    }

    /// Returns the layer's properties.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let layer = composition.create_layer();
    ///
    /// assert_eq!(layer.props().fill_rule, FillRule::NonZero);
    /// ```
    #[inline]
    pub fn props(&self) -> &Props {
        &self.props
    }

    /// Sets the layer's properties.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::prelude::*;
    /// let mut composition = Composition::new();
    ///
    /// let mut layer = composition.create_layer();
    ///
    /// layer.set_props(Props {
    ///     fill_rule: FillRule::EvenOdd,
    ///     ..Default::default()
    /// });
    /// ```
    #[inline]
    pub fn set_props(&mut self, props: Props) -> &mut Self {
        if *self.props != props {
            self.is_unchanged.clear();
            self.props = self.shared_state.inner().props_interner.get(props);
        }

        self
    }
}

impl Drop for Layer {
    fn drop(&mut self) {
        self.shared_state
            .inner()
            .geom_id_to_order
            .remove(&self.geom_id);
    }
}
