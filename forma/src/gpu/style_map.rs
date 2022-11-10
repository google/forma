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
    collections::{HashMap, HashSet},
    fmt,
};

use etagere::{AllocId, Allocation, AtlasAllocator};
use rustc_hash::FxHashMap;

use crate::{
    styling::{Fill, Func, GradientType, Image, ImageId, Props},
    utils::Order,
    Layer,
};

const ATLAS_SIZE: i32 = 4_096;

#[derive(Debug, Default)]
struct PropHeader {
    func: u32,
    is_clipped: u32,
    fill_rule: u32,
    fill_type: u32,
    blend_mode: u32,
    gradient_stop_count: u32,
}

impl PropHeader {
    fn to_bits(&self) -> u32 {
        fn prepend<const B: u32>(current: u32, value: u32) -> u32 {
            #[cfg(test)]
            {
                assert!(
                    (value >> B) == 0,
                    "Unable to store {} with {} bits",
                    value,
                    B
                );
                assert!(
                    (current >> (u32::BITS - B)) == 0,
                    "Prepending {} bit would drop the mos significan bits of {}",
                    B,
                    value
                );
            }
            (current << B) + value
        }
        let header = 0;
        let header = prepend::<1>(header, self.func);
        let header = prepend::<1>(header, self.is_clipped);
        let header = prepend::<1>(header, self.fill_rule);
        let header = prepend::<2>(header, self.fill_type);
        let header = prepend::<4>(header, self.blend_mode);

        prepend::<16>(header, self.gradient_stop_count)
    }
}

struct ImageAllocator {
    allocator: AtlasAllocator,
    image_to_alloc: HashMap<ImageId, Allocation>,
    unused_allocs: HashSet<AllocId>,
    new_allocs: Vec<(Image, [u32; 4])>,
}

impl ImageAllocator {
    fn new() -> ImageAllocator {
        ImageAllocator {
            allocator: AtlasAllocator::new(etagere::size2(ATLAS_SIZE, ATLAS_SIZE)),
            image_to_alloc: HashMap::new(),
            unused_allocs: HashSet::new(),
            new_allocs: Vec::new(),
        }
    }

    fn start_populate(&mut self) {
        self.new_allocs.clear();
        self.unused_allocs = self.allocator.iter().map(|alloc| alloc.id).collect();
    }

    fn end_populate(&mut self) {
        for alloc_id in &self.unused_allocs {
            self.allocator.deallocate(*alloc_id);
        }
    }

    fn get_or_create_alloc(&mut self, image: &Image) -> &Allocation {
        let new_allocs = &mut self.new_allocs;
        let allocator = &mut self.allocator;

        let allocation = self.image_to_alloc.entry(image.id()).or_insert_with(|| {
            let allocation = allocator
                .allocate(etagere::size2(image.width() as i32, image.height() as i32))
                .expect("Texture does not fit in the atlas");

            let rect = allocation.rectangle.to_u32();
            new_allocs.push((
                image.clone(),
                [
                    rect.min.x,
                    rect.min.y,
                    rect.min.x + image.width(),
                    rect.min.y + image.height(),
                ],
            ));
            allocation
        });

        self.unused_allocs.remove(&allocation.id);

        allocation
    }
}

impl fmt::Debug for ImageAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageAllocator")
            .field("allocator", &"{}")
            .field("image_to_alloc", &self.image_to_alloc)
            .field("unused_allocs", &self.unused_allocs)
            .field("new_allocs", &self.new_allocs)
            .finish()
    }
}

#[derive(Debug)]
pub struct StyleMap {
    // Position of the style in the u32 buffer, by order value.
    style_offsets: Vec<u32>,
    styles: Vec<u32>,
    image_allocator: ImageAllocator,
}

impl StyleMap {
    pub fn new() -> StyleMap {
        StyleMap {
            style_offsets: Vec::new(),
            styles: Vec::new(),
            image_allocator: ImageAllocator::new(),
        }
    }

    pub fn push(&mut self, props: &Props) {
        let style = match &props.func {
            Func::Clip(order) => {
                self.styles.push(
                    PropHeader {
                        func: 1,
                        fill_rule: props.fill_rule as u32,
                        ..Default::default()
                    }
                    .to_bits(),
                );
                self.styles.push(*order as u32);
                return;
            }
            Func::Draw(style) => style,
        };
        self.styles.push(
            PropHeader {
                func: 0,
                fill_rule: props.fill_rule as u32,
                is_clipped: u32::from(style.is_clipped),
                fill_type: match &style.fill {
                    Fill::Solid(_) => 0,
                    Fill::Gradient(gradient) => match gradient.r#type() {
                        GradientType::Linear => 1,
                        GradientType::Radial => 2,
                    },
                    Fill::Texture(_) => 3,
                },
                blend_mode: style.blend_mode as u32,
                gradient_stop_count: match &style.fill {
                    Fill::Gradient(gradient) => gradient.colors_with_stops().len() as u32,
                    _ => 0,
                },
            }
            .to_bits(),
        );

        match &style.fill {
            Fill::Solid(color) => self.styles.extend(color.to_array().map(f32::to_bits)),
            Fill::Gradient(gradient) => {
                self.styles
                    .extend(gradient.start().to_array().map(f32::to_bits));
                self.styles
                    .extend(gradient.end().to_array().map(f32::to_bits));
                gradient
                    .colors_with_stops()
                    .iter()
                    .for_each(|(color, stop)| {
                        self.styles.extend(color.to_array().map(f32::to_bits));
                        self.styles.push(stop.to_bits());
                    });
            }
            Fill::Texture(texture) => {
                self.styles
                    .extend(texture.transform.to_array().map(f32::to_bits));

                let image = &texture.image;
                let alloc = self.image_allocator.get_or_create_alloc(image);

                let min = alloc.rectangle.min.cast::<f32>();
                self.styles.extend(
                    [
                        min.x,
                        min.y,
                        min.x + image.width() as f32,
                        min.y + image.height() as f32,
                    ]
                    .map(f32::to_bits),
                );
            }
        }
    }

    pub fn populate(&mut self, layers: &FxHashMap<Order, Layer>) {
        self.style_offsets.clear();
        self.styles.clear();
        self.image_allocator.start_populate();

        let mut props_set = FxHashMap::default();

        for (order, layer) in layers.iter() {
            let order = order.as_u32();
            let props = layer.props();

            if self.style_offsets.len() <= order as usize {
                self.style_offsets.resize(order as usize + 1, 0);
            }

            let offset = *props_set.entry(props).or_insert_with(|| {
                let offset = self.styles.len() as u32;
                self.push(props);
                offset
            });

            self.style_offsets[order as usize] = offset;
        }

        self.image_allocator.end_populate();
    }

    pub fn style_offsets(&self) -> &[u32] {
        &self.style_offsets
    }

    pub fn styles(&self) -> &[u32] {
        &self.styles
    }

    pub fn new_allocs(&self) -> &[(Image, [u32; 4])] {
        self.image_allocator.new_allocs.as_ref()
    }
}
