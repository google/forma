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

use crate::utils::simd::f32x8;

pub(crate) trait Mask {
    fn zero() -> Self;
    fn one() -> Self;
}

impl Mask for f32x8 {
    #[inline]
    fn zero() -> Self {
        f32x8::splat(0.0)
    }

    #[inline]
    fn one() -> Self {
        f32x8::splat(1.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Channel {
    Red,
    Green,
    Blue,
    Alpha,
    Zero,
    One,
}

impl Channel {
    pub(crate) fn select<M: Mask>(self, r: M, g: M, b: M, a: M) -> M {
        match self {
            Channel::Red => r,
            Channel::Green => g,
            Channel::Blue => b,
            Channel::Alpha => a,
            Channel::Zero => M::zero(),
            Channel::One => M::one(),
        }
    }
}

pub const RGBA: [Channel; 4] = [Channel::Red, Channel::Green, Channel::Blue, Channel::Alpha];
pub const BGRA: [Channel; 4] = [Channel::Blue, Channel::Green, Channel::Red, Channel::Alpha];
pub const RGB0: [Channel; 4] = [Channel::Red, Channel::Green, Channel::Blue, Channel::Zero];
pub const BGR0: [Channel; 4] = [Channel::Blue, Channel::Green, Channel::Red, Channel::Zero];
pub const RGB1: [Channel; 4] = [Channel::Red, Channel::Green, Channel::Blue, Channel::One];
pub const BGR1: [Channel; 4] = [Channel::Blue, Channel::Green, Channel::Red, Channel::One];
