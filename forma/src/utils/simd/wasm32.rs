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
    arch::wasm32::{self, *},
    array, mem,
    ops::{Add, AddAssign, BitAnd, BitOr, BitOrAssign, BitXor, Div, Mul, MulAssign, Neg, Not, Sub},
};

#[derive(Clone, Copy, Debug)]
pub struct m8x16(v128);

impl m8x16 {
    pub fn all(self) -> bool {
        u8x16_all_true(self.0)
    }
}

impl Default for m8x16 {
    fn default() -> Self {
        Self(u8x16_splat(0))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct m32x4(v128);

#[derive(Clone, Copy, Debug)]
pub struct m32x8([v128; 2]);

impl m32x8 {
    pub fn all(self) -> bool {
        u32x4_all_true(self.0[0]) && u32x4_all_true(self.0[1])
    }

    pub fn any(self) -> bool {
        v128_any_true(self.0[0]) || v128_any_true(self.0[1])
    }
}

impl Not for m32x8 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self([v128_not(self.0[0]), v128_not(self.0[1])])
    }
}

impl BitOr for m32x8 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self([v128_or(self.0[0], rhs.0[0]), v128_or(self.0[1], rhs.0[1])])
    }
}

impl BitOrAssign for m32x8 {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl BitXor for m32x8 {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self([v128_xor(self.0[0], rhs.0[0]), v128_xor(self.0[1], rhs.0[1])])
    }
}

impl Default for m32x8 {
    fn default() -> Self {
        Self([u32x4_splat(0), u32x4_splat(0)])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u8x32([v128; 2]);

impl u8x32 {
    pub fn splat(val: u8) -> Self {
        Self([u8x16_splat(val), u8x16_splat(val)])
    }

    pub fn from_u32_interleaved(vals: [u32x8; 4]) -> Self {
        let mask = i32x4_splat(0xFF);

        let narrowed: [_; 4] = array::from_fn(|i| {
            i16x8_narrow_i32x4(v128_and(vals[i].0[0], mask), v128_and(vals[i].0[1], mask))
        });

        let bytes_low = u8x16_narrow_i16x8(narrowed[0], narrowed[1]);
        let bytes_high = u8x16_narrow_i16x8(narrowed[2], narrowed[3]);

        Self([
            u8x16_shuffle::<0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27>(
                bytes_low, bytes_high,
            ),
            u8x16_shuffle::<4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31>(
                bytes_low, bytes_high,
            ),
        ])
    }
}

impl Default for u8x32 {
    fn default() -> Self {
        Self::splat(0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u32x4(v128);

impl u32x4 {
    pub fn splat(val: u32) -> Self {
        Self(u32x4_splat(val))
    }
}

impl From<u32x4> for [u8; 4] {
    fn from(val: u32x4) -> Self {
        let val = v128_and(val.0, u32x4_splat(0xFF));

        let _i16 = i16x8_narrow_i32x4(val, val);
        let _u8 = u8x16_narrow_i16x8(_i16, _i16);

        u32x4_extract_lane::<0>(_u8).to_ne_bytes()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u32x8([v128; 2]);

impl u32x8 {
    pub fn splat(val: u32) -> Self {
        Self([u32x4_splat(val), u32x4_splat(val)])
    }

    pub fn to_array(self) -> [u32; 8] {
        unsafe { mem::transmute(self.0) }
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self([
            u32x4_add(u32x4_mul(self.0[0], a.0[0]), b.0[0]),
            u32x4_add(u32x4_mul(self.0[1], a.0[1]), b.0[1]),
        ])
    }
}

impl From<f32x8> for u32x8 {
    fn from(val: f32x8) -> Self {
        Self([f32x4_convert_u32x4(val.0[0]), f32x4_convert_u32x4(val.0[1])])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i8x16(v128);

impl i8x16 {
    #[cfg(test)]
    pub fn as_mut_array(&mut self) -> &mut [i8; 16] {
        std::mem::transmute(&mut self.0)
    }

    pub fn splat(val: i8) -> Self {
        Self(i8x16_splat(val))
    }

    pub fn eq(self, other: Self) -> m8x16 {
        m8x16(i8x16_eq(self.0, other.0))
    }

    pub fn abs(self) -> Self {
        Self(i8x16_abs(self.0))
    }
}

impl Default for i8x16 {
    fn default() -> Self {
        Self::splat(0)
    }
}

impl Add for i8x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(i8x16_add(self.0, rhs.0))
    }
}

impl AddAssign for i8x16 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl BitAnd for i8x16 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(v128_and(self.0, rhs.0))
    }
}

impl From<i8x16> for [i32x8; 2] {
    fn from(val: i8x16) -> Self {
        i16x16([
            i16x8_extend_low_i8x16(val.0),
            i16x8_extend_high_i8x16(val.0),
        ])
        .into()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i16x16([v128; 2]);

impl i16x16 {
    pub fn splat(val: i16) -> Self {
        Self([i16x8_splat(val), i16x8_splat(val)])
    }
}

impl Default for i16x16 {
    fn default() -> Self {
        Self::splat(0)
    }
}

impl From<i16x16> for [i32x8; 2] {
    fn from(val: i16x16) -> Self {
        [
            i32x8([
                i32x4_extend_low_i16x8(val.0[0]),
                i32x4_extend_high_i16x8(val.0[0]),
            ]),
            i32x8([
                i32x4_extend_low_i16x8(val.0[1]),
                i32x4_extend_high_i16x8(val.0[1]),
            ]),
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i32x8([v128; 2]);

impl i32x8 {
    pub fn splat(val: i32) -> Self {
        Self([i32x4_splat(val), i32x4_splat(val)])
    }

    pub fn eq(self, other: Self) -> m32x8 {
        m32x8([
            i32x4_eq(self.0[0], other.0[0]),
            i32x4_eq(self.0[1], other.0[1]),
        ])
    }

    pub fn shr<const N: i32>(self) -> Self {
        Self([
            i32x4_shr(self.0[0], N as u32),
            i32x4_shr(self.0[1], N as u32),
        ])
    }

    pub fn abs(self) -> Self {
        Self([i32x4_abs(self.0[0]), i32x4_abs(self.0[1])])
    }
}

impl Default for i32x8 {
    fn default() -> Self {
        Self::splat(0)
    }
}

impl Add for i32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([
            i32x4_add(self.0[0], rhs.0[0]),
            i32x4_add(self.0[1], rhs.0[1]),
        ])
    }
}

impl Sub for i32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            i32x4_sub(self.0[0], rhs.0[0]),
            i32x4_sub(self.0[1], rhs.0[1]),
        ])
    }
}

impl Mul for i32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self([
            i32x4_mul(self.0[0], rhs.0[0]),
            i32x4_mul(self.0[1], rhs.0[1]),
        ])
    }
}

impl BitAnd for i32x8 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self([v128_and(self.0[0], rhs.0[0]), v128_and(self.0[1], rhs.0[1])])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct f32x4(v128);

impl f32x4 {
    pub fn new(vals: [f32; 4]) -> Self {
        Self(wasm32::f32x4(vals[0], vals[1], vals[2], vals[3]))
    }

    pub fn splat(val: f32) -> Self {
        Self(f32x4_splat(val))
    }

    pub fn from_bits(val: u32x4) -> Self {
        Self(val.0)
    }

    pub fn to_bits(self) -> u32x4 {
        u32x4(self.0)
    }

    pub fn set<const INDEX: usize>(self, val: f32) -> Self {
        Self(f32x4_replace_lane::<INDEX>(self.0, val))
    }

    pub fn le(self, other: Self) -> m32x4 {
        m32x4(f32x4_le(self.0, other.0))
    }

    pub fn select(self, other: Self, mask: m32x4) -> Self {
        Self(v128_bitselect(self.0, other.0, mask.0))
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(f32x4_min(f32x4_max(self.0, min.0), max.0))
    }

    pub fn sqrt(self) -> Self {
        Self(f32x4_sqrt(self.0))
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(f32x4_add(f32x4_mul(self.0, a.0), b.0))
    }
}

impl Add for f32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(f32x4_add(self.0, rhs.0))
    }
}

impl Mul for f32x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(f32x4_mul(self.0, rhs.0))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct f32x8([v128; 2]);

impl f32x8 {
    pub fn splat(val: f32) -> Self {
        Self([f32x4_splat(val), f32x4_splat(val)])
    }

    pub fn indexed() -> Self {
        const INDICES: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        Self::from_array(INDICES)
    }

    pub fn from_bits(val: u32x8) -> Self {
        Self([val.0[0], val.0[1]])
    }

    pub fn to_bits(self) -> u32x8 {
        u32x8([self.0[0], self.0[1]])
    }

    pub fn from_array(val: [f32; 8]) -> Self {
        Self([
            wasm32::f32x4(val[0], val[1], val[2], val[3]),
            wasm32::f32x4(val[4], val[5], val[6], val[7]),
        ])
    }

    #[cfg(test)]
    pub fn to_array(self) -> [f32; 8] {
        unsafe { mem::transmute(self.0) }
    }

    pub fn eq(self, other: Self) -> m32x8 {
        m32x8([
            f32x4_eq(self.0[0], other.0[0]),
            f32x4_eq(self.0[1], other.0[1]),
        ])
    }

    pub fn lt(self, other: Self) -> m32x8 {
        m32x8([
            f32x4_lt(self.0[0], other.0[0]),
            f32x4_lt(self.0[1], other.0[1]),
        ])
    }

    pub fn le(self, other: Self) -> m32x8 {
        m32x8([
            f32x4_le(self.0[0], other.0[0]),
            f32x4_le(self.0[1], other.0[1]),
        ])
    }

    pub fn select(self, other: Self, mask: m32x8) -> Self {
        Self([
            v128_bitselect(self.0[0], other.0[0], mask.0[0]),
            v128_bitselect(self.0[1], other.0[1], mask.0[1]),
        ])
    }

    pub fn abs(self) -> Self {
        Self([f32x4_abs(self.0[0]), f32x4_abs(self.0[1])])
    }

    pub fn min(self, other: Self) -> Self {
        Self([
            f32x4_min(self.0[0], other.0[0]),
            f32x4_min(self.0[1], other.0[1]),
        ])
    }

    pub fn max(self, other: Self) -> Self {
        Self([
            f32x4_max(self.0[0], other.0[0]),
            f32x4_max(self.0[1], other.0[1]),
        ])
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.min(max).max(min)
    }

    pub fn sqrt(self) -> Self {
        Self([f32x4_sqrt(self.0[0]), f32x4_sqrt(self.0[1])])
    }

    pub fn recip(self) -> Self {
        Self([
            f32x4_div(f32x4_splat(1.0), self.0[0]),
            f32x4_div(f32x4_splat(1.0), self.0[1]),
        ])
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self([
            f32x4_add(f32x4_mul(self.0[0], a.0[0]), b.0[0]),
            f32x4_add(f32x4_mul(self.0[1], a.0[1]), b.0[1]),
        ])
    }
}

impl Default for f32x8 {
    fn default() -> Self {
        Self::splat(0.0)
    }
}

impl Add for f32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self([
            f32x4_add(self.0[0], rhs.0[0]),
            f32x4_add(self.0[1], rhs.0[1]),
        ])
    }
}

impl AddAssign for f32x8 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for f32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self([
            f32x4_sub(self.0[0], rhs.0[0]),
            f32x4_sub(self.0[1], rhs.0[1]),
        ])
    }
}

impl Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self([
            f32x4_mul(self.0[0], rhs.0[0]),
            f32x4_mul(self.0[1], rhs.0[1]),
        ])
    }
}

impl MulAssign for f32x8 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for f32x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self([
            f32x4_div(self.0[0], rhs.0[0]),
            f32x4_div(self.0[1], rhs.0[1]),
        ])
    }
}

impl Neg for f32x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self([f32x4_neg(self.0[0]), f32x4_neg(self.0[1])])
    }
}

impl BitOr for f32x8 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self([v128_or(self.0[0], rhs.0[0]), v128_or(self.0[1], rhs.0[1])])
    }
}

impl BitOrAssign for f32x8 {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl From<i32x8> for f32x8 {
    fn from(val: i32x8) -> Self {
        Self([f32x4_convert_i32x4(val.0[0]), f32x4_convert_i32x4(val.0[1])])
    }
}
