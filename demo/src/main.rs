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
    collections::HashSet,
    fmt,
    path::PathBuf,
    str::FromStr,
    time::{Duration, Instant},
};

use clap::{Parser, Subcommand};
use forma::prelude::*;
use runner::{CpuRunner, GpuRunner};
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

pub mod demos;
mod runner;

enum Device {
    Cpu,
    GpuLowPower,
    GpuHighPerformance,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::GpuLowPower => write!(f, "gpu low power"),
            Device::GpuHighPerformance => write!(f, "gpu high performance"),
        }
    }
}

impl FromStr for Device {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "c" | "cpu" => Ok(Device::Cpu),
            "l" | "gpu-low" => Ok(Device::GpuLowPower),
            "h" | "gpu-high" => Ok(Device::GpuHighPerformance),
            _ => Err("must be c|cpu or l|gpu-low or h|gpu-high"),
        }
    }
}

#[derive(Parser)]
#[clap(about = "forma demo with multiple modes")]
struct Demo {
    /// Device to run the demo on
    #[clap(default_value = "cpu")]
    device: Device,
    #[clap(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    /// Renders random circles
    Circles {
        /// Amount of circles to draw
        #[clap(default_value = "100")]
        count: usize,
    },
    /// Renders an SVG
    Svg {
        /// .svg input file
        #[clap(parse(from_os_str))]
        file: PathBuf,
        /// Scale of the SVG
        #[clap(short, long, default_value = "1.0")]
        scale: f32,
    },
    /// Renders a spaceship game
    Spaceship,
    /// Renders a rotating texture
    Texture,
}

trait App {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn set_width(&mut self, width: usize);
    fn set_height(&mut self, height: usize);
    fn compose(&mut self, composition: &mut Composition, elapsed: Duration, keyboard: &Keyboard);
}

trait Runner {
    fn resize(&mut self, width: u32, height: u32);
    fn render(&mut self, app: &mut dyn App, elapsed: Duration, keyboard: &Keyboard);
}

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

    fn on_keyboard_input(&mut self, input: winit::event::KeyboardInput) {
        if let Some(code) = input.virtual_keycode {
            match input.state {
                ElementState::Pressed => self.pressed.insert(code),
                ElementState::Released => self.pressed.remove(&code),
            };
        }
    }
}

pub fn to_linear(rgb: [u8; 3]) -> Color {
    fn conv(l: u8) -> f32 {
        let l = f32::from(l) * 255.0f32.recip();

        if l <= 0.04045 {
            l * 12.92f32.recip()
        } else {
            ((l + 0.055) * 1.055f32.recip()).powf(2.4)
        }
    }

    Color {
        r: conv(rgb[0]),
        g: conv(rgb[1]),
        b: conv(rgb[2]),
        a: 1.0,
    }
}

fn main() {
    let opts = Demo::parse();

    let mut app: Box<dyn App> = match opts.mode {
        Mode::Circles { count } => Box::new(demos::Circles::new(count)),
        Mode::Svg { file, scale } => Box::new(demos::Svg::new(file, scale)),
        Mode::Spaceship => Box::new(demos::Spaceship::new()),
        Mode::Texture {} => Box::new(demos::Texture::new()),
    };

    let width = app.width();
    let height = app.height();

    let event_loop = EventLoop::new();
    let mut runner: Box<dyn Runner> = match opts.device {
        Device::Cpu => Box::new(CpuRunner::new(&event_loop, width as u32, height as u32)),
        Device::GpuLowPower => Box::new(GpuRunner::new(
            &event_loop,
            width as u32,
            height as u32,
            wgpu::PowerPreference::LowPower,
        )),
        Device::GpuHighPerformance => Box::new(GpuRunner::new(
            &event_loop,
            width as u32,
            height as u32,
            wgpu::PowerPreference::HighPerformance,
        )),
    };

    let mut instant = Instant::now();
    let mut keyboard = Keyboard::new();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event:
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                keyboard.on_keyboard_input(input);
            }
            Event::WindowEvent {
                event:
                    WindowEvent::Resized(size)
                    | WindowEvent::ScaleFactorChanged {
                        new_inner_size: &mut size,
                        ..
                    },
                ..
            } => {
                runner.resize(size.width, size.height);

                app.set_width(size.width as usize);
                app.set_height(size.height as usize);
            }
            Event::MainEventsCleared => {
                let elapsed = instant.elapsed();
                instant = Instant::now();

                runner.render(&mut *app, elapsed, &keyboard);
            }
            _ => (),
        }
    });
}
