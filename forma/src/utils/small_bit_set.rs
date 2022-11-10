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

use std::mem;

type Container = u32;

#[derive(Clone, Debug, Default)]
pub struct SmallBitSet {
    bit_set: Container,
}

impl SmallBitSet {
    pub fn clear(&mut self) {
        self.bit_set = 0;
    }

    pub const fn contains(&self, val: &u8) -> bool {
        (self.bit_set >> *val as Container) & 0b1 != 0
    }

    pub fn insert(&mut self, val: u8) -> bool {
        if val as usize >= mem::size_of_val(&self.bit_set) * 8 {
            return false;
        }

        self.bit_set |= 0b1 << u32::from(val);

        true
    }

    pub fn remove(&mut self, val: u8) -> bool {
        if val as usize >= mem::size_of_val(&self.bit_set) * 8 {
            return false;
        }

        self.bit_set &= !(0b1 << u32::from(val));

        true
    }

    pub fn first_empty_slot(&mut self) -> Option<u8> {
        let slot = self.bit_set.trailing_ones() as u8;

        self.insert(slot).then_some(slot)
    }
}
