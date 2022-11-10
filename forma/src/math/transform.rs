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

use std::{convert::TryFrom, error::Error, fmt, hash};

use crate::{consts, math::Point, path::MAX_ERROR, utils::CanonBits};

const MAX_SCALING_FACTOR_X: f32 = 1.0 + MAX_ERROR / consts::MAX_WIDTH as f32;
const MAX_SCALING_FACTOR_Y: f32 = 1.0 + MAX_ERROR / consts::MAX_HEIGHT as f32;

/// 2D transformation that preserves parallel lines.
///
/// Such a transformation can combine translation, scale, flip, rotate and shears.
/// It is represented by a 3 by 3 matrix where the last row is [0, 0, 1].
///
/// ```text
/// [ x' ]   [ u.x v.x t.x ] [ x ]
/// [ y' ] = [ u.y v.y t.y ] [ y ]
/// [ 1  ]   [   0   0   1 ] [ 1 ]
/// ```
#[derive(Copy, Clone, Debug)]
pub struct AffineTransform {
    pub ux: f32,
    pub uy: f32,
    pub vx: f32,
    pub vy: f32,
    pub tx: f32,
    pub ty: f32,
}

impl AffineTransform {
    pub(crate) fn transform(&self, point: Point) -> Point {
        Point {
            x: self.ux.mul_add(point.x, self.vx.mul_add(point.y, self.tx)),
            y: self.uy.mul_add(point.x, self.vy.mul_add(point.y, self.ty)),
        }
    }

    pub(crate) fn is_identity(&self) -> bool {
        *self == Self::default()
    }

    pub fn to_array(&self) -> [f32; 6] {
        [self.ux, self.uy, self.vx, self.vy, self.tx, self.ty]
    }
}

impl Eq for AffineTransform {}

impl PartialEq for AffineTransform {
    fn eq(&self, other: &Self) -> bool {
        self.ux == other.ux
            && self.uy == other.uy
            && self.vx == other.vx
            && self.vy == other.vy
            && self.tx == other.tx
            && self.ty == other.ty
    }
}

impl hash::Hash for AffineTransform {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.ux.to_canon_bits().hash(state);
        self.uy.to_canon_bits().hash(state);
        self.vx.to_canon_bits().hash(state);
        self.vy.to_canon_bits().hash(state);
        self.tx.to_canon_bits().hash(state);
        self.ty.to_canon_bits().hash(state);
    }
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self {
            ux: 1.0,
            vx: 0.0,
            tx: 0.0,
            uy: 0.0,
            vy: 1.0,
            ty: 0.0,
        }
    }
}

impl From<[f32; 6]> for AffineTransform {
    fn from(transform: [f32; 6]) -> Self {
        Self {
            ux: transform[0],
            uy: transform[2],
            vx: transform[1],
            vy: transform[3],
            tx: transform[4],
            ty: transform[5],
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum GeomPresTransformError {
    ExceededScalingFactor { x: bool, y: bool },
}

impl fmt::Display for GeomPresTransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeomPresTransformError::ExceededScalingFactor { x: true, y: false } => {
                write!(f, "exceeded scaling factor on the X axis (-1.0 to 1.0)")
            }
            GeomPresTransformError::ExceededScalingFactor { x: false, y: true } => {
                write!(f, "exceeded scaling factor on the Y axis (-1.0 to 1.0)")
            }
            GeomPresTransformError::ExceededScalingFactor { x: true, y: true } => {
                write!(f, "exceeded scaling factor on both axis (-1.0 to 1.0)")
            }
            _ => panic!("cannot display invalid GeomPresTransformError"),
        }
    }
}

impl Error for GeomPresTransformError {}

#[derive(Default, Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeomPresTransform(pub(crate) AffineTransform);

impl GeomPresTransform {
    /// ```text
    /// [ x' ]   [ t.0 t.1 t.4 ] [ x ]
    /// [ y' ] = [ t.2 t.3 t.5 ] [ y ]
    /// [ 1  ]   [   0   0   1 ] [ 1 ]
    /// ```
    #[inline]
    pub fn new(mut transform: [f32; 9]) -> Option<Self> {
        (transform[6].abs() <= f32::EPSILON && transform[7].abs() <= f32::EPSILON)
            .then(|| {
                if (transform[8] - 1.0).abs() > f32::EPSILON {
                    let recip = transform[8].recip();
                    for val in &mut transform[..6] {
                        *val *= recip;
                    }
                }

                Self::try_from(AffineTransform {
                    ux: transform[0],
                    vx: transform[1],
                    uy: transform[3],
                    vy: transform[4],
                    tx: transform[2],
                    ty: transform[5],
                })
                .ok()
            })
            .flatten()
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        self.0.is_identity()
    }

    pub(crate) fn transform(&self, point: Point) -> Point {
        self.0.transform(point)
    }

    #[inline]
    pub fn to_array(&self) -> [f32; 6] {
        [
            self.0.ux, self.0.vx, self.0.uy, self.0.vy, self.0.tx, self.0.ty,
        ]
    }
}

impl TryFrom<[f32; 6]> for GeomPresTransform {
    type Error = GeomPresTransformError;
    fn try_from(transform: [f32; 6]) -> Result<Self, Self::Error> {
        GeomPresTransform::try_from(AffineTransform::from(transform))
    }
}

impl TryFrom<AffineTransform> for GeomPresTransform {
    type Error = GeomPresTransformError;

    fn try_from(t: AffineTransform) -> Result<Self, Self::Error> {
        let scales_up_x = t.ux * t.ux + t.uy * t.uy > MAX_SCALING_FACTOR_X;
        let scales_up_y = t.vx * t.vx + t.vy * t.vy > MAX_SCALING_FACTOR_Y;

        (!scales_up_x && !scales_up_y).then_some(Self(t)).ok_or(
            GeomPresTransformError::ExceededScalingFactor {
                x: scales_up_x,
                y: scales_up_y,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_identity() {
        let transform = GeomPresTransform::default();

        assert_eq!(
            transform.transform(Point::new(2.0, 3.0)),
            Point::new(2.0, 3.0)
        );
    }

    #[test]
    fn as_slice() {
        let slice = [0.1, 0.5, 0.4, 0.3, 0.7, 0.9];

        assert_eq!(
            slice,
            GeomPresTransform::try_from(slice).unwrap().to_array()
        );
    }

    #[test]
    fn scale_translate() {
        let transform = GeomPresTransform::try_from([0.1, 0.5, 0.4, 0.3, 0.5, 0.6]).unwrap();

        assert_eq!(
            transform.transform(Point::new(2.0, 3.0)),
            Point::new(2.2, 2.3)
        );
    }

    #[test]
    fn wrong_scaling_factor() {
        let transform = [
            0.1,
            MAX_SCALING_FACTOR_Y.sqrt(),
            MAX_SCALING_FACTOR_X.sqrt(),
            0.1,
            0.5,
            0.0,
        ];

        assert_eq!(
            GeomPresTransform::try_from(transform),
            Err(GeomPresTransformError::ExceededScalingFactor { x: true, y: true })
        );
    }

    #[test]
    fn wrong_scaling_factor_x() {
        let transform = [0.1, 0.0, MAX_SCALING_FACTOR_X.sqrt(), 0.0, 0.5, 0.0];

        assert_eq!(
            GeomPresTransform::try_from(transform),
            Err(GeomPresTransformError::ExceededScalingFactor { x: true, y: false })
        );
    }

    #[test]
    fn wrong_scaling_factor_y() {
        let transform = [0.0, MAX_SCALING_FACTOR_Y.sqrt(), 0.0, 0.1, 0.5, 0.0];

        assert_eq!(
            GeomPresTransform::try_from(transform),
            Err(GeomPresTransformError::ExceededScalingFactor { x: false, y: true })
        );
    }

    #[test]
    fn correct_scaling_factor() {
        let transform = [1.0, MAX_SCALING_FACTOR_Y.sqrt(), 0.0, 0.0, 0.5, 0.0];

        assert_eq!(
            transform,
            GeomPresTransform::try_from(transform).unwrap().to_array()
        );
    }
}
