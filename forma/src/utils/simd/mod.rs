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

#![allow(non_camel_case_types)]

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64;
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "fma",
    ),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
mod auto;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod avx;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64::*;
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "fma",
    ),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128"),
)))]
pub use auto::*;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx",
    target_feature = "avx2",
    target_feature = "fma",
))]
pub use avx::*;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm32::*;

pub trait Simd {
    const LANES: usize;
}

impl Simd for u8x32 {
    const LANES: usize = 32;
}

impl Simd for i8x16 {
    const LANES: usize = 16;
}

impl Simd for i16x16 {
    const LANES: usize = 16;
}

impl Simd for i32x8 {
    const LANES: usize = 8;
}

impl Simd for f32x8 {
    const LANES: usize = 8;
}

#[cfg(test)]
mod tests {
    use std::f32::INFINITY;

    use super::*;

    #[test]
    fn f32x8_splat() {
        for v in [1.0, 0.0, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(f32x8::splat(v).to_array(), [v; 8]);
        }
    }

    #[test]
    fn f32x8_indexed() {
        let index: Vec<f32> = (0..8).map(|v| v as f32).collect();
        assert_eq!(f32x8::indexed().to_array(), index[..]);
    }

    #[test]
    fn f32x8_from_array() {
        let value = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0];
        assert_eq!(f32x8::from_array(value).to_array(), value);
    }

    #[test]
    fn u32x8_splat() {
        assert_eq!([0; 8], u32x8::splat(0).to_array());
        assert_eq!([u32::MAX; 8], u32x8::splat(u32::MAX).to_array());
        assert_eq!([u32::MIN; 8], u32x8::splat(u32::MIN).to_array());
    }

    #[test]
    fn u32x8_mul_add() {
        let a = u32x8::from(f32x8::from_array([
            10.0, 20.0, 30.0, 50.0, 70.0, 110.0, 130.0, 170.0,
        ]));
        let b = u32x8::from(f32x8::from_array([
            19.0, 23.0, 29.0, 31.0, 37.0, 41.0, 43.0, 47.0,
        ]));
        let c = u32x8::from(f32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
        let expected = [191, 462, 873, 1554, 2595, 4516, 5597, 7998];
        assert_eq!(expected, u32x8::mul_add(a, b, c).to_array());

        let a = u32x8::from(f32x8::splat(0x007f_ffff as f32));
        let b = u32x8::from(f32x8::splat(0x200 as f32));
        let c = u32x8::from(f32x8::splat(0x1ff as f32));
        let expected = [u32::MAX; 8];
        assert_eq!(expected, u32x8::mul_add(a, b, c).to_array());
    }

    #[test]
    fn u32x8_from_f32x8() {
        let values = u32x8::from(f32x8::from_array([
            -INFINITY,
            -f32::MAX,
            -2.5,
            -1.0,
            -0.5,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
        ]));
        assert_eq!(values.to_array(), [0u32; 8]);

        let f32_before = |f| f32::from_bits(f32::to_bits(f) - 1);
        let values = u32x8::from(f32x8::from_array([
            0.0,
            f32::MIN_POSITIVE,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            f32_before(1.),
        ]));
        assert_eq!(values.to_array(), [0u32; 8]);

        let values = u32x8::from(f32x8::from_array([
            1.0,
            1.4,
            1.5,
            f32_before(2.0),
            2.0,
            2.4,
            2.5,
            f32_before(3.0),
        ]));
        assert_eq!(values.to_array(), [1, 1, 1, 1, 2, 2, 2, 2]);

        const MAX_INT_F32: u32 = 1u32 << 24;
        for value in (MAX_INT_F32 - 255)..MAX_INT_F32 {
            assert_eq!(
                [value; 8],
                u32x8::from(f32x8::splat(value as f32)).to_array()
            );
        }
    }
}
