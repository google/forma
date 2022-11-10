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
    f32, hash,
    ops::{Add, Div, Mul, Sub},
};

use crate::utils::CanonBits;

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Eq for Point {}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl hash::Hash for Point {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.x.to_canon_bits().hash(state);
        self.y.to_canon_bits().hash(state);
    }
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub(crate) fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }
}

#[allow(clippy::many_single_char_names)]
fn approx_atan2(y: f32, x: f32) -> f32 {
    let x_abs = x.abs();
    let y_abs = y.abs();

    let a = x_abs.min(y_abs) / x_abs.max(y_abs);
    let s = a * a;
    let mut r = s
        .mul_add(-0.046_496_473, 0.159_314_22)
        .mul_add(s, -0.327_622_77)
        .mul_add(s * a, a);

    if y_abs > x_abs {
        r = f32::consts::FRAC_PI_2 - r;
    }

    if x < 0.0 {
        r = f32::consts::PI - r;
    }

    if y < 0.0 {
        r = -r;
    }

    r
}

// Restrict the Point functions visibility as we do not want to run into the business of delivering
// a linear algebra package.
impl Point {
    pub(crate) fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub(crate) fn angle(self) -> Option<f32> {
        (self.len() >= f32::EPSILON).then(|| approx_atan2(self.y, self.x))
    }
}

impl Add<Point> for Point {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub<Point> for Point {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<f32> for Point {
    type Output = Self;

    #[inline]
    fn mul(self, other: f32) -> Self {
        Self {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl Div<f32> for Point {
    type Output = Self;

    #[inline]
    fn div(self, other: f32) -> Self {
        Self {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[test]
    fn add() {
        assert_eq!(
            Point { x: 32.0, y: 126.0 },
            Point { x: 13.0, y: 27.0 } + Point { x: 19.0, y: 99.0 }
        );
    }

    #[test]
    fn sub() {
        assert_eq!(
            Point { x: -6.0, y: -72.0 },
            Point { x: 13.0, y: 27.0 } - Point { x: 19.0, y: 99.0 }
        );
    }

    #[test]
    fn mul() {
        assert_eq!(Point { x: 3.0, y: 21.0 }, Point { x: 1.0, y: 7.0 } * 3.0);
    }

    #[test]
    fn div() {
        assert_eq!(Point { x: 1.0, y: 7.0 }, Point { x: 3.0, y: 21.0 } / 3.0);
    }

    #[test]
    fn angle() {
        assert_eq!(Some(0.0), Point { x: 1.0, y: 0.0 }.angle());
        assert_eq!(Some(0.0), Point { x: 1e10, y: 0.0 }.angle());
        assert_eq!(Some(PI), Point { x: -1.0, y: 0.0 }.angle());
        assert_eq!(Some(FRAC_PI_2), Point { x: 0.0, y: 1.0 }.angle());
        assert_eq!(Some(-FRAC_PI_2), Point { x: 0.0, y: -1.0 }.angle());
        assert!((FRAC_PI_4 - Point { x: 1.0, y: 1.0 }.angle().unwrap()).abs() < 1e-3);
        assert!((-FRAC_PI_4 - Point { x: 1.0, y: -1.0 }.angle().unwrap()).abs() < 1e-3);
    }
}
