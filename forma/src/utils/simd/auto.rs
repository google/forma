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

use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitOrAssign, BitXor, Div, Mul, MulAssign, Neg, Not, Sub,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct m8x16([u8; 16]);

impl m8x16 {
    pub fn all(self) -> bool {
        self.0.iter().all(|&val| val == u8::MAX)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct m32x4([u32; 4]);

#[derive(Clone, Copy, Debug, Default)]
pub struct m32x8([u32; 8]);

impl m32x8 {
    pub fn all(self) -> bool {
        self.0.iter().all(|&val| val == u32::MAX)
    }

    pub fn any(self) -> bool {
        self.0.iter().any(|&val| val == u32::MAX)
    }
}

impl Not for m32x8 {
    type Output = Self;

    fn not(mut self) -> Self::Output {
        self.0.iter_mut().for_each(|t| *t = !*t);
        self
    }
}

impl BitOr for m32x8 {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t |= o);
        self
    }
}

impl BitOrAssign for m32x8 {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl BitXor for m32x8 {
    type Output = Self;

    fn bitxor(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t ^= o);
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct u8x8([u8; 8]);

impl From<f32x8> for u8x8 {
    fn from(val: f32x8) -> Self {
        Self([
            val.0[0].round() as u8,
            val.0[1].round() as u8,
            val.0[2].round() as u8,
            val.0[3].round() as u8,
            val.0[4].round() as u8,
            val.0[5].round() as u8,
            val.0[6].round() as u8,
            val.0[7].round() as u8,
        ])
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct u8x32([u8; 32]);

impl u8x32 {
    pub fn splat(val: u8) -> Self {
        Self([
            val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val,
            val, val, val, val, val, val, val, val, val, val, val, val, val, val, val,
        ])
    }

    pub fn from_u32_interleaved(vals: [u32x8; 4]) -> Self {
        Self([
            vals[0].0[0] as u8,
            vals[1].0[0] as u8,
            vals[2].0[0] as u8,
            vals[3].0[0] as u8,
            vals[0].0[1] as u8,
            vals[1].0[1] as u8,
            vals[2].0[1] as u8,
            vals[3].0[1] as u8,
            vals[0].0[2] as u8,
            vals[1].0[2] as u8,
            vals[2].0[2] as u8,
            vals[3].0[2] as u8,
            vals[0].0[3] as u8,
            vals[1].0[3] as u8,
            vals[2].0[3] as u8,
            vals[3].0[3] as u8,
            vals[0].0[4] as u8,
            vals[1].0[4] as u8,
            vals[2].0[4] as u8,
            vals[3].0[4] as u8,
            vals[0].0[5] as u8,
            vals[1].0[5] as u8,
            vals[2].0[5] as u8,
            vals[3].0[5] as u8,
            vals[0].0[6] as u8,
            vals[1].0[6] as u8,
            vals[2].0[6] as u8,
            vals[3].0[6] as u8,
            vals[0].0[7] as u8,
            vals[1].0[7] as u8,
            vals[2].0[7] as u8,
            vals[3].0[7] as u8,
        ])
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct i8x16([i8; 16]);

impl i8x16 {
    pub fn splat(val: i8) -> Self {
        Self([
            val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val,
        ])
    }

    #[cfg(test)]
    pub fn as_mut_array(&mut self) -> &mut [i8; 16] {
        &mut self.0
    }

    pub fn eq(self, other: Self) -> m8x16 {
        m8x16([
            if self.0[0] == other.0[0] { u8::MAX } else { 0 },
            if self.0[1] == other.0[1] { u8::MAX } else { 0 },
            if self.0[2] == other.0[2] { u8::MAX } else { 0 },
            if self.0[3] == other.0[3] { u8::MAX } else { 0 },
            if self.0[4] == other.0[4] { u8::MAX } else { 0 },
            if self.0[5] == other.0[5] { u8::MAX } else { 0 },
            if self.0[6] == other.0[6] { u8::MAX } else { 0 },
            if self.0[7] == other.0[7] { u8::MAX } else { 0 },
            if self.0[8] == other.0[8] { u8::MAX } else { 0 },
            if self.0[9] == other.0[9] { u8::MAX } else { 0 },
            if self.0[10] == other.0[10] {
                u8::MAX
            } else {
                0
            },
            if self.0[11] == other.0[11] {
                u8::MAX
            } else {
                0
            },
            if self.0[12] == other.0[12] {
                u8::MAX
            } else {
                0
            },
            if self.0[13] == other.0[13] {
                u8::MAX
            } else {
                0
            },
            if self.0[14] == other.0[14] {
                u8::MAX
            } else {
                0
            },
            if self.0[15] == other.0[15] {
                u8::MAX
            } else {
                0
            },
        ])
    }

    pub fn abs(mut self) -> Self {
        self.0.iter_mut().for_each(|val| *val = val.abs());
        self
    }
}

impl Add for i8x16 {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t += o);
        self
    }
}

impl AddAssign for i8x16 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl BitAnd for i8x16 {
    type Output = Self;

    fn bitand(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t &= o);
        self
    }
}

impl From<i8x16> for [i32x8; 2] {
    fn from(val: i8x16) -> Self {
        [
            i32x8([
                val.0[0] as i32,
                val.0[1] as i32,
                val.0[2] as i32,
                val.0[3] as i32,
                val.0[4] as i32,
                val.0[5] as i32,
                val.0[6] as i32,
                val.0[7] as i32,
            ]),
            i32x8([
                val.0[8] as i32,
                val.0[9] as i32,
                val.0[10] as i32,
                val.0[11] as i32,
                val.0[12] as i32,
                val.0[13] as i32,
                val.0[14] as i32,
                val.0[15] as i32,
            ]),
        ]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct i16x16([i16; 16]);

impl i16x16 {
    pub fn splat(val: i16) -> Self {
        Self([
            val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val,
        ])
    }
}

impl From<i16x16> for [i32x8; 2] {
    fn from(val: i16x16) -> Self {
        [
            i32x8([
                val.0[0] as i32,
                val.0[1] as i32,
                val.0[2] as i32,
                val.0[3] as i32,
                val.0[4] as i32,
                val.0[5] as i32,
                val.0[6] as i32,
                val.0[7] as i32,
            ]),
            i32x8([
                val.0[8] as i32,
                val.0[9] as i32,
                val.0[10] as i32,
                val.0[11] as i32,
                val.0[12] as i32,
                val.0[13] as i32,
                val.0[14] as i32,
                val.0[15] as i32,
            ]),
        ]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct i32x8([i32; 8]);

impl i32x8 {
    pub fn splat(val: i32) -> Self {
        Self([val, val, val, val, val, val, val, val])
    }

    pub fn eq(self, other: Self) -> m32x8 {
        m32x8([
            if self.0[0] == other.0[0] { u32::MAX } else { 0 },
            if self.0[1] == other.0[1] { u32::MAX } else { 0 },
            if self.0[2] == other.0[2] { u32::MAX } else { 0 },
            if self.0[3] == other.0[3] { u32::MAX } else { 0 },
            if self.0[4] == other.0[4] { u32::MAX } else { 0 },
            if self.0[5] == other.0[5] { u32::MAX } else { 0 },
            if self.0[6] == other.0[6] { u32::MAX } else { 0 },
            if self.0[7] == other.0[7] { u32::MAX } else { 0 },
        ])
    }

    pub fn shr<const N: i32>(mut self) -> Self {
        self.0.iter_mut().for_each(|t| *t >>= N);
        self
    }

    pub fn abs(mut self) -> Self {
        self.0.iter_mut().for_each(|t| *t = t.abs());
        self
    }
}

impl Add for i32x8 {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t += o);
        self
    }
}

impl Sub for i32x8 {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t -= o);
        self
    }
}

impl Mul for i32x8 {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t *= o);
        self
    }
}

impl BitAnd for i32x8 {
    type Output = Self;

    fn bitand(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t &= o);
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct u32x4([u32; 4]);

impl u32x4 {
    pub fn splat(val: u32) -> Self {
        Self([val, val, val, val])
    }
}

impl From<u32x4> for [u8; 4] {
    fn from(val: u32x4) -> Self {
        [
            val.0[0] as u8,
            val.0[1] as u8,
            val.0[2] as u8,
            val.0[3] as u8,
        ]
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct u32x8([u32; 8]);

impl u32x8 {
    pub fn splat(val: u32) -> Self {
        Self([val, val, val, val, val, val, val, val])
    }

    pub fn to_array(self) -> [u32; 8] {
        self.0
    }

    pub fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0
            .iter_mut()
            .zip(a.0.iter().zip(b.0.iter()))
            .for_each(|(t, (&a, &b))| *t = *t * a + b);
        self
    }
}

impl From<f32x8> for u32x8 {
    fn from(val: f32x8) -> Self {
        Self([
            val.0[0] as u32,
            val.0[1] as u32,
            val.0[2] as u32,
            val.0[3] as u32,
            val.0[4] as u32,
            val.0[5] as u32,
            val.0[6] as u32,
            val.0[7] as u32,
        ])
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct f32x4([f32; 4]);

impl f32x4 {
    pub fn new(vals: [f32; 4]) -> Self {
        Self(vals)
    }

    pub fn splat(val: f32) -> Self {
        Self([val, val, val, val])
    }

    pub fn from_bits(val: u32x4) -> Self {
        Self([
            f32::from_bits(val.0[0]),
            f32::from_bits(val.0[1]),
            f32::from_bits(val.0[2]),
            f32::from_bits(val.0[3]),
        ])
    }

    pub fn to_bits(self) -> u32x4 {
        u32x4([
            self.0[0].to_bits(),
            self.0[1].to_bits(),
            self.0[2].to_bits(),
            self.0[3].to_bits(),
        ])
    }

    pub fn set<const INDEX: i32>(mut self, val: f32) -> Self {
        self.0[INDEX as usize] = val;
        self
    }

    pub fn le(self, other: Self) -> m32x4 {
        m32x4([
            if self.0[0] <= other.0[0] { u32::MAX } else { 0 },
            if self.0[1] <= other.0[1] { u32::MAX } else { 0 },
            if self.0[2] <= other.0[2] { u32::MAX } else { 0 },
            if self.0[3] <= other.0[3] { u32::MAX } else { 0 },
        ])
    }

    pub fn select(self, other: Self, mask: m32x4) -> Self {
        Self([
            if mask.0[0] == u32::MAX {
                self.0[0]
            } else {
                other.0[0]
            },
            if mask.0[1] == u32::MAX {
                self.0[1]
            } else {
                other.0[1]
            },
            if mask.0[2] == u32::MAX {
                self.0[2]
            } else {
                other.0[2]
            },
            if mask.0[3] == u32::MAX {
                self.0[3]
            } else {
                other.0[3]
            },
        ])
    }

    pub fn clamp(mut self, min: Self, max: Self) -> Self {
        self.0
            .iter_mut()
            .zip(min.0.iter().zip(max.0.iter()))
            .for_each(|(t, (&min, &max))| *t = t.clamp(min, max));
        self
    }

    pub fn sqrt(mut self) -> Self {
        self.0.iter_mut().for_each(|val| *val = val.sqrt());
        self
    }

    pub fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0
            .iter_mut()
            .zip(a.0.iter().zip(b.0.iter()))
            .for_each(|(t, (&a, &b))| *t = t.mul_add(a, b));
        self
    }
}

impl Add for f32x4 {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t += o);
        self
    }
}

impl Mul for f32x4 {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t *= o);
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct f32x8([f32; 8]);

impl f32x8 {
    pub fn splat(val: f32) -> Self {
        Self([val, val, val, val, val, val, val, val])
    }

    pub fn indexed() -> Self {
        Self::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    }

    pub fn from_array(val: [f32; 8]) -> Self {
        Self(val)
    }

    pub fn from_bits(val: u32x8) -> Self {
        Self([
            f32::from_bits(val.0[0]),
            f32::from_bits(val.0[1]),
            f32::from_bits(val.0[2]),
            f32::from_bits(val.0[3]),
            f32::from_bits(val.0[4]),
            f32::from_bits(val.0[5]),
            f32::from_bits(val.0[6]),
            f32::from_bits(val.0[7]),
        ])
    }

    pub fn to_bits(self) -> u32x8 {
        u32x8([
            self.0[0].to_bits(),
            self.0[1].to_bits(),
            self.0[2].to_bits(),
            self.0[3].to_bits(),
            self.0[4].to_bits(),
            self.0[5].to_bits(),
            self.0[6].to_bits(),
            self.0[7].to_bits(),
        ])
    }

    #[cfg(test)]
    pub fn to_array(self) -> [f32; 8] {
        self.0
    }

    pub fn eq(self, other: Self) -> m32x8 {
        m32x8([
            if self.0[0] == other.0[0] { u32::MAX } else { 0 },
            if self.0[1] == other.0[1] { u32::MAX } else { 0 },
            if self.0[2] == other.0[2] { u32::MAX } else { 0 },
            if self.0[3] == other.0[3] { u32::MAX } else { 0 },
            if self.0[4] == other.0[4] { u32::MAX } else { 0 },
            if self.0[5] == other.0[5] { u32::MAX } else { 0 },
            if self.0[6] == other.0[6] { u32::MAX } else { 0 },
            if self.0[7] == other.0[7] { u32::MAX } else { 0 },
        ])
    }

    pub fn lt(self, other: Self) -> m32x8 {
        m32x8([
            if self.0[0] < other.0[0] { u32::MAX } else { 0 },
            if self.0[1] < other.0[1] { u32::MAX } else { 0 },
            if self.0[2] < other.0[2] { u32::MAX } else { 0 },
            if self.0[3] < other.0[3] { u32::MAX } else { 0 },
            if self.0[4] < other.0[4] { u32::MAX } else { 0 },
            if self.0[5] < other.0[5] { u32::MAX } else { 0 },
            if self.0[6] < other.0[6] { u32::MAX } else { 0 },
            if self.0[7] < other.0[7] { u32::MAX } else { 0 },
        ])
    }

    pub fn le(self, other: Self) -> m32x8 {
        m32x8([
            if self.0[0] <= other.0[0] { u32::MAX } else { 0 },
            if self.0[1] <= other.0[1] { u32::MAX } else { 0 },
            if self.0[2] <= other.0[2] { u32::MAX } else { 0 },
            if self.0[3] <= other.0[3] { u32::MAX } else { 0 },
            if self.0[4] <= other.0[4] { u32::MAX } else { 0 },
            if self.0[5] <= other.0[5] { u32::MAX } else { 0 },
            if self.0[6] <= other.0[6] { u32::MAX } else { 0 },
            if self.0[7] <= other.0[7] { u32::MAX } else { 0 },
        ])
    }

    pub fn select(self, other: Self, mask: m32x8) -> Self {
        Self([
            if mask.0[0] == u32::MAX {
                self.0[0]
            } else {
                other.0[0]
            },
            if mask.0[1] == u32::MAX {
                self.0[1]
            } else {
                other.0[1]
            },
            if mask.0[2] == u32::MAX {
                self.0[2]
            } else {
                other.0[2]
            },
            if mask.0[3] == u32::MAX {
                self.0[3]
            } else {
                other.0[3]
            },
            if mask.0[4] == u32::MAX {
                self.0[4]
            } else {
                other.0[4]
            },
            if mask.0[5] == u32::MAX {
                self.0[5]
            } else {
                other.0[5]
            },
            if mask.0[6] == u32::MAX {
                self.0[6]
            } else {
                other.0[6]
            },
            if mask.0[7] == u32::MAX {
                self.0[7]
            } else {
                other.0[7]
            },
        ])
    }

    pub fn abs(mut self) -> Self {
        self.0.iter_mut().for_each(|val| *val = val.abs());
        self
    }

    pub fn min(mut self, other: Self) -> Self {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(t, &o)| *t = t.min(o));
        self
    }

    pub fn max(mut self, other: Self) -> Self {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(t, &o)| *t = t.max(o));
        self
    }

    pub fn clamp(mut self, min: Self, max: Self) -> Self {
        self.0
            .iter_mut()
            .zip(min.0.iter().zip(max.0.iter()))
            .for_each(|(t, (&min, &max))| *t = t.clamp(min, max));
        self
    }

    pub fn sqrt(mut self) -> Self {
        self.0.iter_mut().for_each(|val| *val = val.sqrt());
        self
    }

    pub fn recip(mut self) -> Self {
        self.0.iter_mut().for_each(|val| *val = val.recip());
        self
    }

    pub fn mul_add(mut self, a: Self, b: Self) -> Self {
        self.0
            .iter_mut()
            .zip(a.0.iter().zip(b.0.iter()))
            .for_each(|(t, (&a, &b))| *t = t.mul_add(a, b));
        self
    }
}

impl Add for f32x8 {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t += o);
        self
    }
}

impl AddAssign for f32x8 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for f32x8 {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t -= o);
        self
    }
}

impl Mul for f32x8 {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t *= o);
        self
    }
}

impl MulAssign for f32x8 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for f32x8 {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(t, &o)| *t /= o);
        self
    }
}

impl Neg for f32x8 {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.0.iter_mut().for_each(|t| *t = -*t);
        self
    }
}

impl BitOr for f32x8 {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self.0.iter_mut().zip(rhs.0.iter()).for_each(|(t, &o)| {
            let t_bytes = t.to_ne_bytes();
            let o_bytes = o.to_ne_bytes();

            *t = f32::from_ne_bytes([
                t_bytes[0] | o_bytes[0],
                t_bytes[1] | o_bytes[1],
                t_bytes[2] | o_bytes[2],
                t_bytes[3] | o_bytes[3],
            ]);
        });
        self
    }
}

impl BitOrAssign for f32x8 {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl From<i32x8> for f32x8 {
    fn from(val: i32x8) -> Self {
        Self([
            val.0[0] as f32,
            val.0[1] as f32,
            val.0[2] as f32,
            val.0[3] as f32,
            val.0[4] as f32,
            val.0[5] as f32,
            val.0[6] as f32,
            val.0[7] as f32,
        ])
    }
}
