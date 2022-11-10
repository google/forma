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
    arch::x86_64::*,
    mem,
    ops::{Add, AddAssign, BitAnd, BitOr, BitOrAssign, BitXor, Div, Mul, MulAssign, Neg, Not, Sub},
    ptr,
};

#[derive(Clone, Copy, Debug)]
pub struct m8x16(__m128i);

impl m8x16 {
    pub fn all(self) -> bool {
        unsafe { _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, _mm_setzero_si128())) == 0 }
    }
}

impl Default for m8x16 {
    fn default() -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct m32x4(__m128i);

#[derive(Clone, Copy, Debug)]
pub struct m32x8(__m256i);

impl m32x8 {
    pub fn all(self) -> bool {
        unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi32(self.0, _mm256_setzero_si256())) == 0 }
    }

    pub fn any(self) -> bool {
        unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi32(self.0, _mm256_setzero_si256())) != -1 }
    }
}

impl Not for m32x8 {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(unsafe { _mm256_xor_si256(self.0, _mm256_cmpeq_epi32(self.0, self.0)) })
    }
}

impl BitOr for m32x8 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm256_or_si256(self.0, rhs.0) })
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
        Self(unsafe { _mm256_xor_si256(self.0, rhs.0) })
    }
}

impl Default for m32x8 {
    fn default() -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u8x32(__m256i);

impl u8x32 {
    pub fn splat(val: u8) -> Self {
        Self(unsafe { _mm256_set1_epi8(i8::from_ne_bytes(val.to_ne_bytes())) })
    }

    pub fn from_u32_interleaved(vals: [u32x8; 4]) -> Self {
        unsafe {
            let mask = _mm256_set1_epi32(0xFF);

            let bytes = _mm256_packus_epi16(
                _mm256_packus_epi32(
                    _mm256_and_si256(vals[0].0, mask),
                    _mm256_and_si256(vals[1].0, mask),
                ),
                _mm256_packus_epi32(
                    _mm256_and_si256(vals[2].0, mask),
                    _mm256_and_si256(vals[3].0, mask),
                ),
            );

            let shuffle = _mm256_set_epi8(
                15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                13, 9, 5, 1, 12, 8, 4, 0,
            );

            Self(_mm256_shuffle_epi8(bytes, shuffle))
        }
    }
}

impl Default for u8x32 {
    fn default() -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i8x16(__m128i);

impl i8x16 {
    #[cfg(test)]
    pub fn as_mut_array(&mut self) -> &mut [i8; 16] {
        unsafe { mem::transmute(&mut self.0) }
    }

    pub fn splat(val: i8) -> Self {
        Self(unsafe { _mm_set1_epi8(val) })
    }

    pub fn eq(self, other: Self) -> m8x16 {
        m8x16(unsafe { _mm_cmpeq_epi8(self.0, other.0) })
    }

    pub fn abs(self) -> Self {
        Self(unsafe { _mm_abs_epi8(self.0) })
    }
}

impl Default for i8x16 {
    fn default() -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }
}

impl Add for i8x16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_add_epi8(self.0, rhs.0) })
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
        Self(unsafe { _mm_and_si128(self.0, rhs.0) })
    }
}

impl From<i8x16> for [i32x8; 2] {
    fn from(val: i8x16) -> Self {
        unsafe {
            let _i8x16_lo = _mm_unpacklo_epi64(val.0, _mm_setzero_si128());
            let _i8x16_hi = _mm_unpackhi_epi64(val.0, _mm_setzero_si128());

            [
                i32x8(_mm256_cvtepi8_epi32(_i8x16_lo)),
                i32x8(_mm256_cvtepi8_epi32(_i8x16_hi)),
            ]
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i16x16(__m256i);

impl i16x16 {
    pub fn splat(val: i16) -> Self {
        Self(unsafe { _mm256_set1_epi16(val) })
    }
}

impl Default for i16x16 {
    fn default() -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }
}

impl From<i16x16> for [i32x8; 2] {
    fn from(val: i16x16) -> Self {
        unsafe {
            let mut _i16x8_lo = _mm_undefined_si128();
            let mut _i16x8_hi = _mm_undefined_si128();

            _mm256_storeu2_m128i(
                ptr::addr_of_mut!(_i16x8_hi),
                ptr::addr_of_mut!(_i16x8_lo),
                val.0,
            );

            [
                i32x8(_mm256_cvtepi16_epi32(_i16x8_lo)),
                i32x8(_mm256_cvtepi16_epi32(_i16x8_hi)),
            ]
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct i32x8(__m256i);

impl i32x8 {
    pub fn splat(val: i32) -> Self {
        Self(unsafe { _mm256_set1_epi32(val) })
    }

    pub fn eq(self, other: Self) -> m32x8 {
        m32x8(unsafe { _mm256_cmpeq_epi32(self.0, other.0) })
    }

    pub fn shr<const N: i32>(self) -> Self {
        Self(unsafe { _mm256_srav_epi32(self.0, _mm256_set1_epi32(N)) })
    }

    pub fn abs(self) -> Self {
        Self(unsafe { _mm256_abs_epi32(self.0) })
    }
}

impl Default for i32x8 {
    fn default() -> Self {
        Self(unsafe { _mm256_setzero_si256() })
    }
}

impl Add for i32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm256_add_epi32(self.0, rhs.0) })
    }
}

impl Sub for i32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm256_sub_epi32(self.0, rhs.0) })
    }
}

impl Mul for i32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm256_mullo_epi32(self.0, rhs.0) })
    }
}

impl BitAnd for i32x8 {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm256_and_si256(self.0, rhs.0) })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u32x4(__m128i);

impl u32x4 {
    pub fn splat(val: u32) -> Self {
        Self(unsafe { _mm_set1_epi32(val as i32) })
    }
}

impl From<u32x4> for [u8; 4] {
    fn from(val: u32x4) -> Self {
        unsafe {
            let mask = _mm_set1_epi32(0xFF);
            let val = _mm_and_si128(val.0, mask);

            let val = _mm_packus_epi32(val, val);
            let val = _mm_packus_epi16(val, val);

            _mm_cvtsi128_si32(val).to_ne_bytes()
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct u32x8(__m256i);

impl u32x8 {
    pub fn splat(val: u32) -> Self {
        Self(unsafe { _mm256_set1_epi32(val as i32) })
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm256_add_epi32(_mm256_mullo_epi32(self.0, a.0), b.0) })
    }

    pub fn to_array(self) -> [u32; 8] {
        unsafe { mem::transmute(self.0) }
    }
}

impl From<f32x8> for u32x8 {
    fn from(val: f32x8) -> Self {
        // Sets negative value to 0 to prevent _mm256_cvttps_epi32 from
        // returning negative values.
        Self(unsafe { _mm256_cvttps_epi32(val.max(f32x8::splat(0.0)).0) })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct f32x4(__m128);

impl f32x4 {
    pub fn new(vals: [f32; 4]) -> Self {
        Self(unsafe { _mm_set_ps(vals[3], vals[2], vals[1], vals[0]) })
    }

    pub fn splat(val: f32) -> Self {
        Self(unsafe { _mm_set1_ps(val) })
    }

    pub fn from_bits(val: u32x4) -> Self {
        Self(unsafe { _mm_castsi128_ps(val.0) })
    }

    pub fn to_bits(self) -> u32x4 {
        u32x4(unsafe { _mm_castps_si128(self.0) })
    }

    pub fn set<const INDEX: i32>(self, val: f32) -> Self {
        Self(unsafe {
            _mm_castsi128_ps(_mm_insert_epi32::<INDEX>(
                _mm_castps_si128(self.0),
                val.to_bits() as i32,
            ))
        })
    }

    pub fn le(self, other: Self) -> m32x4 {
        m32x4(unsafe { _mm_castps_si128(_mm_cmp_ps(self.0, other.0, _CMP_LE_OQ)) })
    }

    pub fn select(self, other: Self, mask: m32x4) -> Self {
        Self(unsafe { _mm_blendv_ps(other.0, self.0, _mm_castsi128_ps(mask.0)) })
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(unsafe { _mm_min_ps(_mm_max_ps(self.0, min.0), max.0) })
    }

    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm_sqrt_ps(self.0) })
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm_fmadd_ps(self.0, a.0, b.0) })
    }
}

impl Add for f32x4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_add_ps(self.0, rhs.0) })
    }
}

impl Mul for f32x4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm_mul_ps(self.0, rhs.0) })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct f32x8(__m256);

impl f32x8 {
    pub fn splat(val: f32) -> Self {
        Self(unsafe { _mm256_set1_ps(val) })
    }

    pub fn indexed() -> Self {
        Self(unsafe { _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0) })
    }

    pub fn from_bits(val: u32x8) -> Self {
        Self(unsafe { _mm256_castsi256_ps(val.0) })
    }

    pub fn to_bits(self) -> u32x8 {
        u32x8(unsafe { _mm256_castps_si256(self.0) })
    }

    pub fn from_array(val: [f32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_ps(val.as_ptr()) })
    }

    #[cfg(test)]
    pub fn to_array(self) -> [f32; 8] {
        unsafe { mem::transmute(self.0) }
    }

    pub fn eq(self, other: Self) -> m32x8 {
        m32x8(unsafe { _mm256_castps_si256(_mm256_cmp_ps(self.0, other.0, _CMP_EQ_OQ)) })
    }

    pub fn lt(self, other: Self) -> m32x8 {
        m32x8(unsafe { _mm256_castps_si256(_mm256_cmp_ps(self.0, other.0, _CMP_LT_OQ)) })
    }

    pub fn le(self, other: Self) -> m32x8 {
        m32x8(unsafe { _mm256_castps_si256(_mm256_cmp_ps(self.0, other.0, _CMP_LE_OQ)) })
    }

    pub fn select(self, other: Self, mask: m32x8) -> Self {
        Self(unsafe { _mm256_blendv_ps(other.0, self.0, _mm256_castsi256_ps(mask.0)) })
    }

    pub fn abs(self) -> Self {
        Self(unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0) })
    }

    pub fn min(self, other: Self) -> Self {
        Self(unsafe { _mm256_min_ps(self.0, other.0) })
    }

    pub fn max(self, other: Self) -> Self {
        Self(unsafe { _mm256_max_ps(self.0, other.0) })
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.min(max).max(min)
    }

    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm256_sqrt_ps(self.0) })
    }

    pub fn recip(self) -> Self {
        Self(unsafe { _mm256_rcp_ps(self.0) })
    }

    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(unsafe { _mm256_fmadd_ps(self.0, a.0, b.0) })
    }
}

impl Default for f32x8 {
    fn default() -> Self {
        Self(unsafe { _mm256_setzero_ps() })
    }
}

impl Add for f32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm256_add_ps(self.0, rhs.0) })
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
        Self(unsafe { _mm256_sub_ps(self.0, rhs.0) })
    }
}

impl Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm256_mul_ps(self.0, rhs.0) })
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
        Self(unsafe { _mm256_div_ps(self.0, rhs.0) })
    }
}

impl Neg for f32x8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(unsafe { _mm256_xor_ps(self.0, _mm256_set1_ps(-0.0)) })
    }
}

impl BitOr for f32x8 {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(unsafe { _mm256_or_ps(self.0, rhs.0) })
    }
}

impl BitOrAssign for f32x8 {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl From<i32x8> for f32x8 {
    fn from(val: i32x8) -> Self {
        Self(unsafe { _mm256_cvtepi32_ps(val.0) })
    }
}
