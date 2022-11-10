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

use std::time::Duration;

use forma::prelude::*;
use rand::prelude::*;

use crate::{App, Keyboard};

fn circle(x: f32, y: f32, radius: f32) -> Path {
    let weight = 2.0f32.sqrt() / 2.0;

    let mut builder = PathBuilder::new();

    builder.move_to(Point::new(x + radius, y));
    builder.rat_quad_to(
        Point::new(x + radius, y - radius),
        Point::new(x, y - radius),
        weight,
    );
    builder.rat_quad_to(
        Point::new(x - radius, y - radius),
        Point::new(x - radius, y),
        weight,
    );
    builder.rat_quad_to(
        Point::new(x - radius, y + radius),
        Point::new(x, y + radius),
        weight,
    );
    builder.rat_quad_to(
        Point::new(x + radius, y + radius),
        Point::new(x + radius, y),
        weight,
    );

    builder.build()
}

pub struct Circles {
    count: usize,
    width: usize,
    height: usize,
    needs_composition: bool,
}

impl Circles {
    pub fn new(count: usize) -> Self {
        Self {
            count,
            width: 1000,
            height: 1000,
            needs_composition: true,
        }
    }
}

impl App for Circles {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn set_width(&mut self, width: usize) {
        if self.width == width {
            return;
        }

        self.width = width;
        self.needs_composition = true;
    }

    fn set_height(&mut self, height: usize) {
        if self.height == height {
            return;
        }

        self.height = height;
        self.needs_composition = true;
    }

    fn compose(&mut self, composition: &mut Composition, _: Duration, _: &Keyboard) {
        if !self.needs_composition {
            return;
        }

        let radius_range = 10.0..50.0;

        let mut rng = StdRng::seed_from_u64(42);

        for order in 0..self.count {
            let color = Color {
                r: rng.gen(),
                g: rng.gen(),
                b: rng.gen(),
                a: 0.2,
            };

            composition
                .get_mut_or_insert_default(Order::new(order as u32).unwrap())
                .clear()
                .insert(&circle(
                    rng.gen_range(0.0..App::width(self) as f32),
                    rng.gen_range(0.0..App::height(self) as f32),
                    rng.gen_range(radius_range.clone()),
                ))
                .set_props(Props {
                    fill_rule: FillRule::NonZero,
                    func: Func::Draw(Style {
                        fill: Fill::Solid(color),
                        ..Default::default()
                    }),
                });
        }

        self.needs_composition = false;
    }
}
