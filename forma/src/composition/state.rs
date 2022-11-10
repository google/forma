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
    cell::{RefCell, RefMut},
    mem,
    rc::Rc,
};

use rustc_hash::FxHashMap;

use crate::{styling::Props, GeomId, Order, SegmentBuffer};

use super::Interner;

#[derive(Debug)]
pub struct LayerSharedStateInner {
    pub segment_buffer: Option<SegmentBuffer>,
    pub geom_id_to_order: FxHashMap<GeomId, Option<Order>>,
    pub props_interner: Interner<Props>,
    geom_id_generator: GeomId,
}

impl Default for LayerSharedStateInner {
    fn default() -> Self {
        Self {
            segment_buffer: Some(SegmentBuffer::default()),
            geom_id_to_order: FxHashMap::default(),
            props_interner: Interner::default(),
            geom_id_generator: GeomId::default(),
        }
    }
}

impl LayerSharedStateInner {
    pub fn new_geom_id(&mut self) -> GeomId {
        let prev = self.geom_id_generator;
        mem::replace(&mut self.geom_id_generator, prev.next())
    }
}

#[derive(Debug, Default)]
pub struct LayerSharedState {
    inner: Rc<RefCell<LayerSharedStateInner>>,
}

impl LayerSharedState {
    pub fn new(inner: Rc<RefCell<LayerSharedStateInner>>) -> Self {
        Self { inner }
    }

    pub fn inner(&mut self) -> RefMut<'_, LayerSharedStateInner> {
        self.inner.borrow_mut()
    }
}

impl PartialEq<Rc<RefCell<LayerSharedStateInner>>> for LayerSharedState {
    fn eq(&self, other: &Rc<RefCell<LayerSharedStateInner>>) -> bool {
        Rc::ptr_eq(&self.inner, other)
    }
}

// Safe as long as `inner` can only be accessed by `&mut self`.
unsafe impl Sync for LayerSharedState {}
