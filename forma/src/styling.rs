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
    fmt, hash,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use crate::{
    math::{AffineTransform, Point},
    utils::CanonBits,
};
#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Eq for Color {}

impl PartialEq for Color {
    fn eq(&self, other: &Self) -> bool {
        self.r == other.r && self.g == other.g && self.b == other.b && self.a == other.a
    }
}

impl hash::Hash for Color {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.r.to_canon_bits().hash(state);
        self.g.to_canon_bits().hash(state);
        self.b.to_canon_bits().hash(state);
        self.a.to_canon_bits().hash(state);
    }
}

impl Default for Color {
    fn default() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

impl Default for FillRule {
    fn default() -> Self {
        Self::NonZero
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GradientType {
    Linear,
    Radial,
}

const NO_STOP: f32 = -1.0;

#[derive(Clone, Debug)]
pub struct GradientBuilder {
    r#type: GradientType,
    start: Point,
    end: Point,
    stops: Vec<(Color, f32)>,
}

impl GradientBuilder {
    pub fn new(start: Point, end: Point) -> Self {
        Self {
            r#type: GradientType::Linear,
            start,
            end,
            stops: Vec::new(),
        }
    }

    pub fn r#type(&mut self, r#type: GradientType) -> &mut Self {
        self.r#type = r#type;
        self
    }

    pub fn color(&mut self, color: Color) -> &mut Self {
        self.stops.push((color, NO_STOP));
        self
    }

    pub fn color_with_stop(&mut self, color: Color, stop: f32) -> &mut Self {
        if !(0.0..=1.0).contains(&stop) {
            panic!("gradient stops must be between 0.0 and 1.0");
        }

        self.stops.push((color, stop));
        self
    }

    pub fn build(mut self) -> Option<Gradient> {
        if self.stops.len() < 2 {
            return None;
        }

        let stop_increment = 1.0 / (self.stops.len() - 1) as f32;
        for (i, (_, stop)) in self.stops.iter_mut().enumerate() {
            if *stop == NO_STOP {
                *stop = i as f32 * stop_increment;
            }
        }

        Some(Gradient {
            r#type: self.r#type,
            start: self.start,
            end: self.end,
            stops: self.stops.into(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Gradient {
    pub(crate) r#type: GradientType,
    pub(crate) start: Point,
    pub(crate) end: Point,
    pub(crate) stops: Arc<[(Color, f32)]>,
}

impl Eq for Gradient {}

impl PartialEq for Gradient {
    fn eq(&self, other: &Self) -> bool {
        self.r#type == other.r#type
            && self.start == other.start
            && self.end == other.end
            && self.stops == other.stops
    }
}

impl hash::Hash for Gradient {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.r#type.hash(state);
        self.start.hash(state);
        self.end.hash(state);

        self.stops.len().hash(state);
        for (color, stop) in self.stops.iter() {
            (color, stop.to_canon_bits()).hash(state);
        }
    }
}

impl Gradient {
    pub fn r#type(&self) -> GradientType {
        self.r#type
    }

    pub fn start(&self) -> Point {
        self.start
    }

    pub fn end(&self) -> Point {
        self.end
    }

    #[inline]
    pub fn colors_with_stops(&self) -> &[(Color, f32)] {
        &self.stops
    }
}

#[derive(Debug)]
pub enum ImageError {
    SizeMismatch {
        len: usize,
        width: usize,
        height: usize,
    },
    TooLarge,
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeMismatch { len, width, height } => {
                write!(
                    f,
                    "buffer has {} pixels, which does not match \
                     the specified width ({}) and height ({})",
                    len, width, height
                )
            }
            Self::TooLarge => {
                write!(
                    f,
                    "image dimensions exceed what is addressable \
                     with f32; try to reduce the image size."
                )
            }
        }
    }
}

/// f16 value without denormals and within 0 and one.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct f16(u16);

impl f16 {
    #[inline]
    pub fn to_f32(self) -> f32 {
        if self.0 != 0 {
            f32::from_bits(0x3800_0000 + (u32::from(self.0) << 13))
        } else {
            0.0
        }
    }
}

impl From<f32> for f16 {
    fn from(val: f32) -> Self {
        if val != 0.0 {
            f16(((val.to_bits() - 0x3800_0000) >> 13) as u16)
        } else {
            f16(0)
        }
    }
}

/// Transforms `sRGB` component into linear space.
fn to_linear(l: u8) -> f32 {
    let l = f32::from(l) * 255.0f32.recip();
    if l <= 0.04045 {
        l * 12.92f32.recip()
    } else {
        ((l + 0.055) * 1.055f32.recip()).powf(2.4)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ImageId(usize);

impl ImageId {
    fn new() -> ImageId {
        static GENERATOR: AtomicUsize = AtomicUsize::new(0);

        ImageId(GENERATOR.fetch_add(1, Ordering::SeqCst))
    }
}

#[derive(Clone, Debug)]
pub struct Image {
    /// Pixels RGBA.
    pub(crate) data: Arc<[[f16; 4]]>,
    /// Largest x coordinate within the Image.
    pub(crate) max_x: f32,
    /// Largest y coordinate within the Image.
    pub(crate) max_y: f32,
    /// Width of the image in pixels.
    pub(crate) width: u32,
    /// Unique identifier for this image.
    pub(crate) id: ImageId,
}

impl Eq for Image {}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.data.as_ptr() == other.data.as_ptr()
            && self.max_x == other.max_x
            && self.max_y == other.max_y
    }
}

impl hash::Hash for Image {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.data.as_ptr().hash(state);
        self.max_x.to_canon_bits().hash(state);
        self.max_y.to_canon_bits().hash(state);
    }
}

impl Image {
    /// Creates an image from `sRGB` color channels and linear alpha.
    /// The boxed array size must match the image dimensions.
    pub fn from_srgba(data: &[[u8; 4]], width: usize, height: usize) -> Result<Self, ImageError> {
        let to_alpha = |a| f32::from(a) * f32::from(u8::MAX).recip();
        let data = data
            .iter()
            .map(|c| {
                [
                    to_linear(c[0]),
                    to_linear(c[1]),
                    to_linear(c[2]),
                    to_alpha(c[3]),
                ]
                .map(f16::from)
            })
            .collect();
        Self::new(data, width, height)
    }

    pub fn from_linear_rgba(
        data: &[[f32; 4]],
        width: usize,
        height: usize,
    ) -> Result<Self, ImageError> {
        let data = data.iter().map(|c| c.map(f16::from)).collect();
        Self::new(data, width, height)
    }

    fn new(data: Arc<[[f16; 4]]>, width: usize, height: usize) -> Result<Self, ImageError> {
        match width * height {
            len if len > u32::MAX as usize => Err(ImageError::TooLarge),
            len if len != data.len() => Err(ImageError::SizeMismatch {
                len: data.len(),
                width,
                height,
            }),
            _ => Ok(Image {
                data,
                max_x: width as f32 - 1.0,
                max_y: height as f32 - 1.0,
                width: width as u32,
                id: ImageId::new(),
            }),
        }
    }

    pub fn id(&self) -> ImageId {
        self.id
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.max_y as u32 + 1
    }

    pub(crate) fn data(&self) -> &[[f16; 4]] {
        self.data.as_ref()
    }
}

/// Describes how to shade a surface using a bitmap image.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Texture {
    /// Transformation from screen-space to texture-space.
    pub transform: AffineTransform,
    /// Image shared with zero or more textures.
    pub image: Image,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Fill {
    Solid(Color),
    Gradient(Gradient),
    Texture(Texture),
}

impl Default for Fill {
    fn default() -> Self {
        Self::Solid(Color::default())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlendMode {
    Over,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Over
    }
}

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Style {
    pub is_clipped: bool,
    pub fill: Fill,
    pub blend_mode: BlendMode,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Func {
    Draw(Style),
    // Clips the subsequent layer with this one.
    // From this order up to to order + n included are affected, if
    // their `is_clipped` property is `true`.
    Clip(usize),
}

impl Default for Func {
    fn default() -> Self {
        Self::Draw(Style::default())
    }
}

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Props {
    pub fill_rule: FillRule,
    pub func: Func,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn f16_error() {
        // Error for the 256 values of u8 alpha is low.
        let alpha_mse = (0u8..=255u8)
            .map(|u| f32::from(u) / 255.0)
            .map(|v| (v - f16::from(v).to_f32()))
            .map(|d| d * d)
            .sum::<f32>()
            / 256.0;
        assert!(alpha_mse < 5e-8, "alpha_mse: {}", alpha_mse);

        // Values for 256 values of u8 alpha are distinct.
        let alpha_distinct = (0u8..=255u8)
            .map(|a| f16::from(f32::from(a) / 255.0))
            .collect::<HashSet<f16>>()
            .len();
        assert_eq!(alpha_distinct, 256);

        // Error for the 256 value of u8 sRGB is low.
        let component_mse = (0u8..=255u8)
            .map(to_linear)
            .map(|v| (v - f16::from(v).to_f32()))
            .map(|d| d * d)
            .sum::<f32>()
            / 256.0;
        assert!(component_mse < 3e-8, "component_mse: {}", component_mse);

        // Values for 256 values of u8 sRGB are distinct.
        let component_distinct = (0u8..=255u8)
            .map(|c| f16::from(to_linear(c)))
            .collect::<HashSet<f16>>()
            .len();
        assert_eq!(component_distinct, 256);

        // Min and max values are intact.
        assert_eq!(f16::from(0.0).to_f32(), 0.0);
        assert_eq!(f16::from(1.0).to_f32(), 1.0);
    }

    #[test]
    fn f16_conversion() {
        for i in 0..255 {
            let value = (i as f32) / 255.0;
            let value_f16 = f16::from(value);
            assert!(half::f16::from_f32(value).to_bits().abs_diff(value_f16.0) <= 1);
            assert_eq!(
                half::f16::from_bits(value_f16.0).to_f32(),
                value_f16.to_f32()
            );
        }
    }
}
