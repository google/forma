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

#![allow(clippy::assertions_on_constants)]

use std::mem;

use crate::cpu::PixelSegment;

pub(crate) const PIXEL_WIDTH: usize = 16;
pub(crate) const PIXEL_DOUBLE_WIDTH: usize = PIXEL_WIDTH * 2;
pub(crate) const PIXEL_SHIFT: usize = PIXEL_WIDTH.trailing_zeros() as usize;

pub const MAX_WIDTH: usize = 1 << 16;
pub const MAX_HEIGHT: usize = 1 << 15;

pub(crate) const MAX_WIDTH_SHIFT: usize = MAX_WIDTH.trailing_zeros() as usize;
pub(crate) const MAX_HEIGHT_SHIFT: usize = MAX_HEIGHT.trailing_zeros() as usize;

pub mod cpu {
    pub const TILE_WIDTH: usize = 16;
    const _: () = assert!(TILE_WIDTH % 16 == 0);
    const _: () = assert!(TILE_WIDTH <= 128);

    pub(crate) const TILE_WIDTH_SHIFT: usize = TILE_WIDTH.trailing_zeros() as usize;

    pub const TILE_HEIGHT: usize = 16;
    const _: () = assert!(TILE_HEIGHT % 16 == 0);
    const _: () = assert!(TILE_HEIGHT <= 128);

    pub(crate) const TILE_HEIGHT_SHIFT: usize = TILE_HEIGHT.trailing_zeros() as usize;
}

pub mod gpu {
    pub const TILE_WIDTH: usize = 16;
    pub const TILE_HEIGHT: usize = 4;
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum BitField {
    TileY,
    TileX,
    LayerId,
    LocalX,
    LocalY,
    DoubleAreaMultiplier,
    Cover,
}

#[derive(Debug)]
pub(crate) struct BitFieldMap {
    lengths: [usize; 7],
}

impl BitFieldMap {
    #[inline]
    pub const fn new<const TW: usize, const TH: usize>() -> Self {
        let tile_width_shift = TW.trailing_zeros() as usize;
        let tile_height_shift = TH.trailing_zeros() as usize;

        let mut lengths = [
            MAX_HEIGHT_SHIFT - tile_height_shift,
            MAX_WIDTH_SHIFT - tile_width_shift,
            0,
            tile_width_shift,
            tile_height_shift,
            ((PIXEL_WIDTH + 1) * 2).next_power_of_two().trailing_zeros() as usize,
            ((PIXEL_WIDTH + 1) * 2).next_power_of_two().trailing_zeros() as usize,
        ];

        let layer_id_len = mem::size_of::<PixelSegment<TW, TH>>() * 8
            - lengths[BitField::TileY as usize]
            - lengths[BitField::TileX as usize]
            - lengths[BitField::LocalX as usize]
            - lengths[BitField::LocalY as usize]
            - lengths[BitField::DoubleAreaMultiplier as usize]
            - lengths[BitField::Cover as usize];

        lengths[BitField::LayerId as usize] = layer_id_len;

        Self { lengths }
    }

    #[inline]
    pub const fn get(&self, bit_field: BitField) -> usize {
        self.lengths[bit_field as usize]
    }

    #[inline]
    pub const fn get_by_index(&self, i: usize) -> usize {
        self.lengths[i]
    }
}

pub const LAYER_LIMIT: usize = (1
    << BitFieldMap::new::<{ cpu::TILE_WIDTH }, { cpu::TILE_HEIGHT }>().get(BitField::LayerId))
    - 1;
const _: () = assert!(
    LAYER_LIMIT
        == (1
            << BitFieldMap::new::<{ gpu::TILE_WIDTH }, { gpu::TILE_HEIGHT }>()
                .get(BitField::LayerId))
            - 1,
    "LAYER_LIMIT must be the same both on cpu and gpu",
);
