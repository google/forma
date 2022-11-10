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
    path::{self, PathBuf},
    time::Duration,
};

use forma::{prelude::*, styling};
use image::GenericImageView;

use crate::{App, Keyboard};

#[derive(Debug)]
pub struct Texture {
    width: usize,
    height: usize,
    time: Duration,
    image: Image,
}

impl Texture {
    pub fn new() -> Self {
        Self {
            width: 1000,
            height: 1000,
            time: Duration::ZERO,
            image: load_image(&PathBuf::from("assets/image/butterfly.jpg")),
        }
    }

    fn transform(&self, t: f32) -> AffineTransform {
        let scale = 1.5 - t.cos();
        let mut t = AffineTransform {
            ux: t.cos() * scale,
            uy: t.sin() * scale,
            vx: -t.sin() * scale,
            vy: t.cos() * scale,
            tx: 0.0,
            ty: 0.0,
        };

        let (x, y) = (self.width as f32 * 0.5, self.height as f32 * 0.5);
        (t.tx, t.ty) = (
            self.image.width() as f32 * 0.5 - t.ux * x - t.vx * y,
            self.image.height() as f32 * 0.5 - t.uy * x - t.vy * y,
        );
        t
    }
}

impl Default for Texture {
    fn default() -> Self {
        Self::new()
    }
}

fn load_image(file_path: &path::Path) -> Image {
    let img = image::io::Reader::open(file_path)
        .expect("Unable to open file")
        .decode()
        .expect("Unable to decode file");

    let data: Vec<_> = img
        .to_rgb8()
        .pixels()
        .map(|p| [p.0[0], p.0[1], p.0[2], 255])
        .collect();
    Image::from_srgba(&data[..], img.width() as usize, img.height() as usize).unwrap()
}

impl App for Texture {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn set_width(&mut self, width: usize) {
        self.width = width;
    }

    fn set_height(&mut self, height: usize) {
        self.height = height;
    }

    fn compose(&mut self, composition: &mut Composition, elapsed: Duration, _: &Keyboard) {
        const PAD: f32 = 32.0;

        let (w, h) = (self.width as f32, self.height as f32);
        let layer = composition
            .get_mut_or_insert_default(Order::new(0).unwrap())
            .clear()
            .insert(
                &PathBuilder::new()
                    .move_to(Point { x: PAD, y: PAD })
                    .line_to(Point { x: w - PAD, y: PAD })
                    .line_to(Point {
                        x: w - PAD,
                        y: h - PAD,
                    })
                    .line_to(Point { x: PAD, y: h - PAD })
                    .build(),
            );

        self.time += elapsed;

        layer.set_props(Props {
            fill_rule: FillRule::NonZero,
            func: Func::Draw(Style {
                fill: Fill::Texture(styling::Texture {
                    transform: self.transform(self.time.as_secs_f32()),
                    image: self.image.clone(),
                }),
                ..Default::default()
            }),
        });
    }
}
