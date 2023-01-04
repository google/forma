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

#![doc(test(attr(deny(warnings))))]

//! forma is a high-performance vector-graphics renderer with a CPU & GPU back-end.
//!
//! ## Example
//!
//! The following example renders two overlapping rectangles and highlights the most common API
//! usage:
//!
//! ```
//! # use forma_render as forma;
//! use forma::{cpu::{buffer::{BufferBuilder, layout::LinearLayout}, Renderer, RGBA}, prelude::*};
//!
//! fn rect(x: f32, y: f32, width: f32, height: f32) -> Path {
//!     PathBuilder::new()
//!         .move_to(Point::new(x, y))
//!         .line_to(Point::new(x + width, y))
//!         .line_to(Point::new(x + width, y + height))
//!         .line_to(Point::new(x, y + height))
//!         .build()
//! }
//!
//! fn solid(r: f32, g: f32, b: f32) -> Props {
//!     Props {
//!         func: Func::Draw(Style {
//!             fill: Fill::Solid(Color { r, g, b, a: 1.0 }),
//!             ..Default::default()
//!         }),
//!         ..Default::default()
//!     }
//! }
//!
//! // The composition is akin to a `HashMap<Order, Layer>`. Layers can be inserted and
//! // removed from the composition by their orders.
//! let mut composition = Composition::new();
//! let mut renderer = Renderer::new();
//!
//! // The layer cache enables updating only tiles that changed from last frame.
//! let layer_cache = renderer.create_buffer_layer_cache().unwrap();
//!
//! composition
//!     .get_mut_or_insert_default(Order::new(0).unwrap())
//!     .insert(&rect(50.0, 50.0, 100.0, 50.0))
//!     .set_props(solid(1.0, 0.0, 0.0));
//!
//! composition
//!     .get_mut_or_insert_default(Order::new(1).unwrap())
//!     .insert(&rect(100.0, 50.0, 100.0, 50.0))
//!     .set_props(solid(0.0, 0.0, 1.0));
//!
//! let width = 250;
//! let height = 150;
//! let mut buffer = vec![0; width * height * 4]; // 4 bytes per pixel.
//!
//! renderer.render(
//!     &mut composition,
//!     // Stride is width * 4 bytes per pixel.
//!     &mut BufferBuilder::new(&mut buffer, &mut LinearLayout::new(width, width * 4, height))
//!         .layer_cache(layer_cache.clone())
//!         .build(),
//!     RGBA,
//!     Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
//!     None,
//! );
//!
//! // Background is white.
//! assert_eq!(buffer.chunks(4).next().unwrap(), [255, 255, 255, 255]);
//!
//! // First rectangle is red.
//! let index = 75 + 75 * width;
//! assert_eq!(buffer.chunks(4).nth(index).unwrap(), [255, 0, 0, 255]);
//!
//! // Overlap is blue.
//! let index = 125 + 75 * width;
//! assert_eq!(buffer.chunks(4).nth(index).unwrap(), [0, 0, 255, 255]);
//!
//! // Second rectangle is blue.
//! let index = 175 + 75 * width;
//! assert_eq!(buffer.chunks(4).nth(index).unwrap(), [0, 0, 255, 255]);
//! ```
//!
//! ## Reusing the composition
//!
//! For best possible performance, reusing the composition is essential. This may mean that in some
//! cases one might have to remove, then re-insert some layers around in order to achieve the
//! desired ordering of layers.
//!
//! For simple cases, [Layer::set_is_enabled] can provide an alternative to removing and
//! re-inserting layers into the composition.

#[cfg(target_os = "fuchsia")]
macro_rules! duration {
    ($category:expr, $name:expr $(, $key:expr => $val:expr)*) => {
        fuchsia_trace::duration!($category, $name $(, $key => $val)*)
    }
}

#[cfg(not(target_os = "fuchsia"))]
macro_rules! duration {
    ($category:expr, $name:expr $(, $key:expr => $val:expr)*) => {};
}

mod composition;
pub mod consts;
pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod math;
mod path;
mod segment;
pub mod styling;
mod utils;

pub(crate) use self::segment::{SegmentBuffer, SegmentBufferView};

pub use self::{
    composition::{Composition, Layer},
    path::{Path, PathBuilder},
    segment::GeomId,
    utils::{Order, OrderError},
};

pub mod prelude {
    pub use crate::cpu::{
        buffer::{
            layout::{Layout, LinearLayout},
            BufferBuilder, BufferLayerCache,
        },
        Channel, BGR0, BGR1, BGRA, RGB0, RGB1, RGBA,
    };
    #[cfg(feature = "gpu")]
    pub use crate::gpu::Timings;
    pub use crate::{
        math::{AffineTransform, GeomPresTransform, Point},
        styling::{
            BlendMode, Color, Fill, FillRule, Func, Gradient, GradientBuilder, GradientType, Image,
            Props, Style, Texture,
        },
        Composition, Layer, Order, Path, PathBuilder,
    };
}
