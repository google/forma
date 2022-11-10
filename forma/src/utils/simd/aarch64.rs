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
    arch::aarch64::*,
    array, mem,
    ops::{Add, AddAssign, BitAnd, BitOr, BitOrAssign, BitXor, Div, Mul, MulAssign, Neg, Not, Sub},
};

#[derive(Clone, Copy, Debug)]
pub struct m8x16(uint8x16_t);

impl m8x16 {
    pub fn all(self) -> bool {
        unsafe { vminvq_u8(self.0) != 0 }
    }
}

impl Default for m8x16 {
    fn default() -> Self {
        Self(unsafe { vdupq_n_u8(0) })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct m32x4(uint32x4_t);

#[derive(Clone, Copy, Debug)]
pub struct m32x8([uint32x4_t; 2]);

impl m32x8 {
    pub fn all(self) -> bool {
        unsafe { vminvq_u32(self.0[0]) != 0 && vminvq_u32(self.0[1]) != 0 }
    }

    pub fn any(self) -> bool {
        unsafe { vmaxvq_u32(self.0[0]) != 0 || vmaxvq_u32(self.0[1]) != 0 }
    }
}

impl Not for m32x8 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(unsafe { [vmvnq_u32(self.0[0]), vmvnq_u32(self.0[1])] })
    }
}

impl BitOr for m32x8 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(unsafe {
            [
                vorrq_u32(self.0[0], rhs.0[0]),
                vorrq_u32(self.0[1], rhs.0[1]),
            ]
        })
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
        Self(unsafe {
            [
                veorq_u32(self.0[0], rhs.0[0]),
                veorq_u32(self.0[1], rhs.0[1]),
            ]
        })
    }
}

impl Default for m32x8 {
    fn default() -> Self {
        Self(unsafe { [vdupq_n_u32(0), vdupq_n_u32(0)] })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u8x32([uint8x16_t; 2]);

impl u8x32 {
    pub fn splat(val: u8) -> Self {
        Self(unsafe { [vdupq_n_u8(val), vdupq_n_u8(val)] })
    }

    pub fn from_u32_interleaved(vals: [u32x8; 4]) -> Self {
        unsafe {
            let bytes: [_; 4] = array::from_fn(|i| {
                vmovn_u16(vcombine_u16(
                    vmovn_u32(vals[i].0[0]),
                    vmovn_u32(vals[i].0[1]),
                ))
            });

            let mut result = Self::splat(0);

            vst4_u8(
                result.0.as_mut_ptr().cast(),
                uint8x8x4_t(bytes[0], bytes[1], bytes[2], bytes[3]),
            );

            result
        }
    }
}

impl Default for u8x32 {
    fn default() -> Self {
        Self::splat(0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u32x4(uint32x4_t);

impl u32x4 {
    pub fn splat(val: u32) -> Self {
        Self(unsafe { vdupq_n_u32(val) })
    }
}

impl From<u32x4> for [u8; 4] {
    fn from(val: u32x4) -> Self {
        unsafe {
            let _u16 = vmovn_u32(val.0);
            let _u32 = vget_lane_u32::<0>(vreinterpret_u32_u8(vmovn_u16(vcombine_u16(_u16, _u16))));

            _u32.to_ne_bytes()
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u32x8([uint32x4_t; 2]);

impl u32x8 {
    pub fn splat(val: u32) -> Self {
        Self(unsafe { [vdupq_n_u32(val), vdupq_n_u32(val)] })
    }

    pub fn to_array(self) -> [u32; 8] {
        unsafe { mem::transmute(self.0) }
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe {
            [
                vmlaq_u32(b.0[0], self.0[0], a.0[0]),
                vmlaq_u32(b.0[1], self.0[1], a.0[1]),
            ]
        })
    }
}

impl From<f32x8> for u32x8 {
    fn from(val: f32x8) -> Self {
        Self(unsafe { [vcvtq_u32_f32(val.0[0]), vcvtq_u32_f32(val.0[1])] })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i8x16(int8x16_t);

impl i8x16 {
    #[cfg(test)]
    pub fn as_mut_array(&mut self) -> &mut [i8; 16] {
        unsafe { mem::transmute(&mut self.0) }
    }

    pub fn splat(val: i8) -> Self {
        Self(unsafe { vdupq_n_s8(val) })
    }

    pub fn eq(self, other: Self) -> m8x16 {
        m8x16(unsafe { vceqq_s8(self.0, other.0) })
    }

    pub fn abs(self) -> Self {
        Self(unsafe { vabsq_s8(self.0) })
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
        Self(unsafe { vaddq_s8(self.0, rhs.0) })
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
        Self(unsafe { vandq_s8(self.0, rhs.0) })
    }
}

impl From<i8x16> for [i32x8; 2] {
    fn from(val: i8x16) -> Self {
        i16x16(unsafe { [vmovl_s8(vget_low_s8(val.0)), vmovl_s8(vget_high_s8(val.0))] }).into()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i16x16([int16x8_t; 2]);

impl i16x16 {
    pub fn splat(val: i16) -> Self {
        Self(unsafe { [vdupq_n_s16(val), vdupq_n_s16(val)] })
    }
}

impl Default for i16x16 {
    fn default() -> Self {
        Self::splat(0)
    }
}

impl From<i16x16> for [i32x8; 2] {
    fn from(val: i16x16) -> Self {
        unsafe {
            [
                i32x8([
                    vmovl_s16(vget_low_s16(val.0[0])),
                    vmovl_s16(vget_high_s16(val.0[0])),
                ]),
                i32x8([
                    vmovl_s16(vget_low_s16(val.0[1])),
                    vmovl_s16(vget_high_s16(val.0[1])),
                ]),
            ]
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i32x8([int32x4_t; 2]);

impl i32x8 {
    pub fn splat(val: i32) -> Self {
        Self(unsafe { [vdupq_n_s32(val), vdupq_n_s32(val)] })
    }

    pub fn eq(self, other: Self) -> m32x8 {
        m32x8(unsafe {
            [
                vceqq_s32(self.0[0], other.0[0]),
                vceqq_s32(self.0[1], other.0[1]),
            ]
        })
    }

    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { [vshrq_n_s32::<N>(self.0[0]), vshrq_n_s32::<N>(self.0[1])] })
    }

    pub fn abs(self) -> Self {
        Self(unsafe { [vabsq_s32(self.0[0]), vabsq_s32(self.0[1])] })
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
        Self(unsafe {
            [
                vaddq_s32(self.0[0], rhs.0[0]),
                vaddq_s32(self.0[1], rhs.0[1]),
            ]
        })
    }
}

impl Sub for i32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(unsafe {
            [
                vsubq_s32(self.0[0], rhs.0[0]),
                vsubq_s32(self.0[1], rhs.0[1]),
            ]
        })
    }
}

impl Mul for i32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(unsafe {
            [
                vmulq_s32(self.0[0], rhs.0[0]),
                vmulq_s32(self.0[1], rhs.0[1]),
            ]
        })
    }
}

impl BitAnd for i32x8 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(unsafe {
            [
                vandq_s32(self.0[0], rhs.0[0]),
                vandq_s32(self.0[1], rhs.0[1]),
            ]
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct f32x4(float32x4_t);

impl f32x4 {
    pub fn new(vals: [f32; 4]) -> Self {
        Self(unsafe { vld1q_f32(vals.as_ptr()) })
    }

    pub fn splat(val: f32) -> Self {
        Self(unsafe { vdupq_n_f32(val) })
    }

    pub fn from_bits(val: u32x4) -> Self {
        Self(unsafe { vreinterpretq_f32_u32(val.0) })
    }

    pub fn to_bits(self) -> u32x4 {
        u32x4(unsafe { vreinterpretq_u32_f32(self.0) })
    }

    pub fn set<const INDEX: i32>(self, val: f32) -> Self {
        Self(unsafe { vsetq_lane_f32::<INDEX>(val, self.0) })
    }

    pub fn le(self, other: Self) -> m32x4 {
        m32x4(unsafe { vcleq_f32(self.0, other.0) })
    }

    pub fn select(self, other: Self, mask: m32x4) -> Self {
        Self(unsafe { vbslq_f32(mask.0, self.0, other.0) })
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(unsafe { vminq_f32(vmaxq_f32(self.0, min.0), max.0) })
    }

    pub fn sqrt(self) -> Self {
        Self(unsafe { vsqrtq_f32(self.0) })
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { vfmaq_f32(b.0, self.0, a.0) })
    }
}

impl Add for f32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(unsafe { vaddq_f32(self.0, rhs.0) })
    }
}

impl Mul for f32x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(unsafe { vmulq_f32(self.0, rhs.0) })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct f32x8([float32x4_t; 2]);

impl f32x8 {
    pub fn splat(val: f32) -> Self {
        Self(unsafe { [vdupq_n_f32(val), vdupq_n_f32(val)] })
    }

    pub fn indexed() -> Self {
        const INDICES: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        Self::from_array(INDICES)
    }

    pub fn from_bits(val: u32x8) -> Self {
        Self(unsafe {
            [
                vreinterpretq_f32_u32(val.0[0]),
                vreinterpretq_f32_u32(val.0[1]),
            ]
        })
    }

    pub fn to_bits(self) -> u32x8 {
        u32x8(unsafe {
            [
                vreinterpretq_u32_f32(self.0[0]),
                vreinterpretq_u32_f32(self.0[1]),
            ]
        })
    }

    pub fn from_array(val: [f32; 8]) -> Self {
        Self(unsafe { [vld1q_f32(val.as_ptr()), vld1q_f32(val[4..].as_ptr())] })
    }

    #[cfg(test)]
    pub fn to_array(self) -> [f32; 8] {
        unsafe { mem::transmute(self.0) }
    }

    pub fn eq(self, other: Self) -> m32x8 {
        m32x8(unsafe {
            [
                vceqq_f32(self.0[0], other.0[0]),
                vceqq_f32(self.0[1], other.0[1]),
            ]
        })
    }

    pub fn lt(self, other: Self) -> m32x8 {
        m32x8(unsafe {
            [
                vcltq_f32(self.0[0], other.0[0]),
                vcltq_f32(self.0[1], other.0[1]),
            ]
        })
    }

    pub fn le(self, other: Self) -> m32x8 {
        m32x8(unsafe {
            [
                vcleq_f32(self.0[0], other.0[0]),
                vcleq_f32(self.0[1], other.0[1]),
            ]
        })
    }

    pub fn select(self, other: Self, mask: m32x8) -> Self {
        Self(unsafe {
            [
                vbslq_f32(mask.0[0], self.0[0], other.0[0]),
                vbslq_f32(mask.0[1], self.0[1], other.0[1]),
            ]
        })
    }

    pub fn abs(self) -> Self {
        Self(unsafe { [vabsq_f32(self.0[0]), vabsq_f32(self.0[1])] })
    }

    pub fn min(self, other: Self) -> Self {
        Self(unsafe {
            [
                vminq_f32(self.0[0], other.0[0]),
                vminq_f32(self.0[1], other.0[1]),
            ]
        })
    }

    pub fn max(self, other: Self) -> Self {
        Self(unsafe {
            [
                vmaxq_f32(self.0[0], other.0[0]),
                vmaxq_f32(self.0[1], other.0[1]),
            ]
        })
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.min(max).max(min)
    }

    pub fn sqrt(self) -> Self {
        Self(unsafe { [vsqrtq_f32(self.0[0]), vsqrtq_f32(self.0[1])] })
    }

    pub fn recip(self) -> Self {
        Self(unsafe { [vrecpeq_f32(self.0[0]), vrecpeq_f32(self.0[1])] })
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe {
            [
                vfmaq_f32(b.0[0], self.0[0], a.0[0]),
                vfmaq_f32(b.0[1], self.0[1], a.0[1]),
            ]
        })
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
        Self(unsafe {
            [
                vaddq_f32(self.0[0], rhs.0[0]),
                vaddq_f32(self.0[1], rhs.0[1]),
            ]
        })
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
        Self(unsafe {
            [
                vsubq_f32(self.0[0], rhs.0[0]),
                vsubq_f32(self.0[1], rhs.0[1]),
            ]
        })
    }
}

impl Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(unsafe {
            [
                vmulq_f32(self.0[0], rhs.0[0]),
                vmulq_f32(self.0[1], rhs.0[1]),
            ]
        })
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
        Self(unsafe {
            [
                vdivq_f32(self.0[0], rhs.0[0]),
                vdivq_f32(self.0[1], rhs.0[1]),
            ]
        })
    }
}

impl Neg for f32x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(unsafe { [vnegq_f32(self.0[0]), vnegq_f32(self.0[1])] })
    }
}

impl BitOr for f32x8 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(unsafe {
            [
                vreinterpretq_f32_u32(vorrq_u32(
                    vreinterpretq_u32_f32(self.0[0]),
                    vreinterpretq_u32_f32(rhs.0[0]),
                )),
                vreinterpretq_f32_u32(vorrq_u32(
                    vreinterpretq_u32_f32(self.0[1]),
                    vreinterpretq_u32_f32(rhs.0[1]),
                )),
            ]
        })
    }
}

impl BitOrAssign for f32x8 {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl From<i32x8> for f32x8 {
    fn from(val: i32x8) -> Self {
        Self(unsafe { [vcvtq_f32_s32(val.0[0]), vcvtq_f32_s32(val.0[1])] })
    }
}
