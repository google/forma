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

use std::{collections::HashSet, time::Duration};

use forma::{cpu, prelude::*};
use wasm_bindgen::{prelude::*, Clamped};

#[path = "../../demo/src/demos/circles.rs"]
pub mod circles;
#[path = "../../demo/src/demos/spaceship.rs"]
pub mod spaceship;
mod utils;

use circles::Circles;
use spaceship::Spaceship;
pub use wasm_bindgen_rayon::init_thread_pool;
use winit::event::VirtualKeyCode;

struct Keyboard {
    pressed: HashSet<VirtualKeyCode>,
}

impl Keyboard {
    fn new() -> Self {
        Self {
            pressed: HashSet::new(),
        }
    }

    fn is_key_down(&self, key: VirtualKeyCode) -> bool {
        self.pressed.contains(&key)
    }
}

trait App {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn set_width(&mut self, width: usize);
    fn set_height(&mut self, height: usize);
    fn compose(&mut self, composition: &mut Composition, elapsed: Duration, keyboard: &Keyboard);
}

#[wasm_bindgen]
pub struct Context {
    composition: Composition,
    renderer: cpu::Renderer,
    layout: LinearLayout,
    layer_cache: BufferLayerCache,
    app: Box<dyn App>,
    buffer: Vec<u8>,
    width: usize,
    height: usize,
    was_cleared: bool,
}

#[wasm_bindgen]
pub fn context_new_circles(width: usize, height: usize, count: usize) -> Context {
    utils::set_panic_hook();

    let buffer = vec![0u8; width * 4 * height];
    let layout = LinearLayout::new(width, width * 4, height);

    let composition = Composition::new();
    let mut renderer = cpu::Renderer::new();
    let layer_cache = renderer.create_buffer_layer_cache().unwrap();

    let app: Box<dyn App> = Box::new(Circles::new(count));

    Context {
        composition,
        renderer,
        layout,
        layer_cache,
        app,
        buffer,
        width,
        height,
        was_cleared: false,
    }
}

#[wasm_bindgen]
pub fn context_new_spaceship(width: usize, height: usize) -> Context {
    utils::set_panic_hook();

    let buffer = vec![0u8; width * 4 * height];
    let layout = LinearLayout::new(width, width * 4, height);

    let composition = Composition::new();
    let mut renderer = cpu::Renderer::new();
    let layer_cache = renderer.create_buffer_layer_cache().unwrap();

    let app: Box<dyn App> = Box::new(Spaceship::new());

    Context {
        composition,
        renderer,
        layout,
        layer_cache,
        app,
        buffer,
        width,
        height,
        was_cleared: false,
    }
}

#[wasm_bindgen]
pub fn context_draw(
    context: &mut Context,
    width: usize,
    height: usize,
    elapsed: f64,
    force_clear: bool,
    controls: u8,
) -> Clamped<Vec<u8>> {
    if context.width != width || context.height != height {
        context.buffer = vec![0u8; width * 4 * height];
        context.layout = LinearLayout::new(width, width * 4, height);

        context.app.set_width(width);
        context.app.set_height(height);
    }

    if force_clear {
        for pixel in context.buffer.chunks_mut(4) {
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
            pixel[3] = 0;
        }

        context.was_cleared = true;
    } else {
        if context.was_cleared {
            context.layer_cache.clear();
            context.was_cleared = false;
        }
    }

    let mut pressed = HashSet::new();

    if controls & 0b1000 != 0 {
        pressed.insert(VirtualKeyCode::Up);
    }
    if controls & 0b0100 != 0 {
        pressed.insert(VirtualKeyCode::Right);
    }
    if controls & 0b0010 != 0 {
        pressed.insert(VirtualKeyCode::Down);
    }
    if controls & 0b0001 != 0 {
        pressed.insert(VirtualKeyCode::Left);
    }

    context.app.compose(
        &mut context.composition,
        Duration::from_secs_f64(elapsed / 1000.0),
        &Keyboard { pressed },
    );

    context.renderer.render(
        &mut context.composition,
        &mut BufferBuilder::new(&mut context.buffer, &mut context.layout)
            .layer_cache(context.layer_cache.clone())
            .build(),
        RGB1,
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 0.0,
        },
        None,
    );

    Clamped(context.buffer.clone())
}
