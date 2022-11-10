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

use std::{cmp::Ordering, fmt, mem};

#[cfg(test)]
use bytemuck::{Pod, Zeroable};

use crate::consts::{BitField, BitFieldMap};

// Tile coordinates are signed integers stored with a bias.
// The value range goes from -1 to 2^bits - 2 inclusive.
const TILE_BIAS: i16 = 1;

/// Pixel-bounded line segment with `TW` tile width and `TH` tile height.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[cfg_attr(test, derive(Pod, Zeroable))]
#[repr(transparent)]
pub struct PixelSegment<const TW: usize, const TH: usize>(u64);

impl<const TW: usize, const TH: usize> PixelSegment<TW, TH> {
    const BIT_FIELD_MAP: BitFieldMap = BitFieldMap::new::<TW, TH>();

    #[inline]
    pub fn new(
        layer_id: u32,
        tile_x: i16,
        tile_y: i16,
        local_x: u8,
        local_y: u8,
        double_area_multiplier: u8,
        cover: i8,
    ) -> Self {
        let mut val = 0;

        val |= ((1 << Self::BIT_FIELD_MAP.get(BitField::TileY)) - 1)
            & (tile_y + TILE_BIAS).max(0) as u64;

        val <<= Self::BIT_FIELD_MAP.get(BitField::TileX);
        val |= ((1 << Self::BIT_FIELD_MAP.get(BitField::TileX)) - 1) as u64
            & (tile_x + TILE_BIAS).max(0) as u64;

        val <<= Self::BIT_FIELD_MAP.get(BitField::LayerId);
        val |= ((1 << Self::BIT_FIELD_MAP.get(BitField::LayerId)) - 1) as u64 & u64::from(layer_id);

        val <<= Self::BIT_FIELD_MAP.get(BitField::LocalX);
        val |= ((1 << Self::BIT_FIELD_MAP.get(BitField::LocalX)) - 1) as u64 & u64::from(local_x);

        val <<= Self::BIT_FIELD_MAP.get(BitField::LocalY);
        val |= ((1 << Self::BIT_FIELD_MAP.get(BitField::LocalY)) - 1) as u64 & u64::from(local_y);

        val <<= Self::BIT_FIELD_MAP.get(BitField::DoubleAreaMultiplier);
        val |= ((1 << Self::BIT_FIELD_MAP.get(BitField::DoubleAreaMultiplier)) - 1) as u64
            & u64::from(double_area_multiplier);

        val <<= Self::BIT_FIELD_MAP.get(BitField::Cover);
        val |= ((1 << Self::BIT_FIELD_MAP.get(BitField::Cover)) - 1) as u64 & cover as u64;

        Self(val)
    }

    #[inline]
    const fn shift_left_for(self, bit_field: BitField) -> u32 {
        let mut amount = 0;
        let mut i = 0;

        while i < bit_field as usize {
            amount += Self::BIT_FIELD_MAP.get_by_index(i);
            i += 1;
        }

        amount as u32
    }

    #[inline]
    const fn shift_right_for(self, bit_field: BitField) -> u32 {
        (mem::size_of::<Self>() * 8 - Self::BIT_FIELD_MAP.get(bit_field)) as u32
    }
    #[inline]
    const fn extract(self, bit_field: BitField) -> u64 {
        self.0 << self.shift_left_for(bit_field) >> self.shift_right_for(bit_field)
    }

    #[inline]
    const fn extract_signed(self, bit_field: BitField) -> i64 {
        (self.0 as i64) << self.shift_left_for(bit_field) >> self.shift_right_for(bit_field)
    }

    #[inline]
    pub fn layer_id(self) -> u32 {
        self.extract(BitField::LayerId) as u32
    }

    #[inline]
    pub fn tile_x(self) -> i16 {
        self.extract(BitField::TileX) as i16 - TILE_BIAS
    }

    #[inline]
    pub fn tile_y(self) -> i16 {
        self.extract(BitField::TileY) as i16 - TILE_BIAS
    }

    #[inline]
    pub fn local_x(self) -> u8 {
        self.extract(BitField::LocalX) as u8
    }

    #[inline]
    pub fn local_y(self) -> u8 {
        self.extract(BitField::LocalY) as u8
    }

    #[inline]
    fn double_area_multiplier(self) -> u8 {
        self.extract(BitField::DoubleAreaMultiplier) as u8
    }

    #[inline]
    pub fn double_area(self) -> i16 {
        i16::from(self.double_area_multiplier()) * i16::from(self.cover())
    }

    #[inline]
    pub fn cover(self) -> i8 {
        self.extract_signed(BitField::Cover) as i8
    }
}

impl<const TW: usize, const TH: usize> fmt::Debug for PixelSegment<TW, TH> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PixelSegment")
            .field("layer_id", &self.layer_id())
            .field("tile_x", &self.tile_x())
            .field("tile_y", &self.tile_y())
            .field("local_x", &self.local_x())
            .field("local_y", &self.local_y())
            .field("double_area", &self.double_area())
            .field("cover", &self.cover())
            .finish()
    }
}

impl<const TW: usize, const TH: usize> PartialOrd for PixelSegment<TW, TH> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const TW: usize, const TH: usize> Ord for PixelSegment<TW, TH> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by (tile_y, tile_x, layer_id).
        let offset = Self::BIT_FIELD_MAP.get(BitField::LocalX)
            + Self::BIT_FIELD_MAP.get(BitField::LocalY)
            + Self::BIT_FIELD_MAP.get(BitField::DoubleAreaMultiplier)
            + Self::BIT_FIELD_MAP.get(BitField::Cover);

        (self.0 >> offset).cmp(&(other.0 >> offset))
    }
}

impl<const TW: usize, const TH: usize> From<&PixelSegment<TW, TH>> for u64 {
    fn from(segment: &PixelSegment<TW, TH>) -> Self {
        segment.0
    }
}

#[inline]
pub fn search_last_by_key<F, K, const TW: usize, const TH: usize>(
    segments: &[PixelSegment<TW, TH>],
    key: K,
    mut f: F,
) -> Result<usize, usize>
where
    F: FnMut(&PixelSegment<TW, TH>) -> K,
    K: Ord,
{
    let mut len = segments.len();
    if len == 0 {
        return Err(0);
    }

    let mut start = 0;
    while len > 1 {
        let half = len / 2;
        let mid = start + half;
        (start, len) = match f(&segments[mid]).cmp(&key) {
            Ordering::Greater => (start, half),
            _ => (mid, len - half),
        };
    }

    match f(&segments[start]).cmp(&key) {
        Ordering::Less => Err(start + 1),
        Ordering::Equal => Ok(start),
        Ordering::Greater => Err(start),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::consts::{
        self,
        cpu::{TILE_HEIGHT, TILE_WIDTH},
    };

    #[test]
    fn pixel_segment() {
        let layer_id = 3;
        let tile_x = 4;
        let tile_y = 5;
        let local_x = 6;
        let local_y = 7;
        let double_area_multiplier = 8;
        let cover = 9;

        let pixel_segment = PixelSegment::<TILE_WIDTH, TILE_HEIGHT>::new(
            layer_id,
            tile_x,
            tile_y,
            local_x,
            local_y,
            double_area_multiplier,
            cover,
        );

        assert_eq!(pixel_segment.layer_id(), layer_id);
        assert_eq!(pixel_segment.tile_x(), tile_x);
        assert_eq!(pixel_segment.tile_y(), tile_y);
        assert_eq!(pixel_segment.local_x(), local_x);
        assert_eq!(pixel_segment.local_y(), local_y);
        assert_eq!(
            pixel_segment.double_area(),
            i16::from(double_area_multiplier) * i16::from(cover)
        );
        assert_eq!(pixel_segment.cover(), cover);
    }

    #[test]
    fn pixel_segment_max() {
        let layer_id = consts::LAYER_LIMIT as u32;
        let tile_x =
            (1 << (BitFieldMap::new::<TILE_WIDTH, TILE_HEIGHT>().get(BitField::TileX) - 1)) - 1;
        let tile_y =
            (1 << (BitFieldMap::new::<TILE_WIDTH, TILE_HEIGHT>().get(BitField::TileY) - 1)) - 1;
        let local_x =
            (1 << BitFieldMap::new::<TILE_WIDTH, TILE_HEIGHT>().get(BitField::LocalX)) - 1;
        let local_y =
            (1 << BitFieldMap::new::<TILE_WIDTH, TILE_HEIGHT>().get(BitField::LocalY)) - 1;
        let double_area_multiplier = consts::PIXEL_DOUBLE_WIDTH as u8;
        let cover = consts::PIXEL_WIDTH as i8;

        let pixel_segment = PixelSegment::<TILE_WIDTH, TILE_HEIGHT>::new(
            layer_id,
            tile_x,
            tile_y,
            local_x,
            local_y,
            double_area_multiplier,
            cover,
        );

        assert_eq!(pixel_segment.layer_id(), layer_id);
        assert_eq!(pixel_segment.tile_x(), tile_x);
        assert_eq!(pixel_segment.tile_y(), tile_y);
        assert_eq!(pixel_segment.local_x(), local_x);
        assert_eq!(pixel_segment.local_y(), local_y);
        assert_eq!(
            pixel_segment.double_area(),
            i16::from(double_area_multiplier) * i16::from(cover)
        );
        assert_eq!(pixel_segment.cover(), cover);
    }

    #[test]
    fn pixel_segment_min() {
        let layer_id = 0;
        let tile_x = -1;
        let tile_y = -1;
        let local_x = 0;
        let local_y = 0;
        let double_area_multiplier = 0;
        let cover = -(consts::PIXEL_WIDTH as i8);

        let pixel_segment = PixelSegment::<TILE_WIDTH, TILE_HEIGHT>::new(
            layer_id,
            tile_x,
            tile_y,
            local_x,
            local_y,
            double_area_multiplier,
            cover,
        );

        assert_eq!(pixel_segment.layer_id(), layer_id);
        assert_eq!(pixel_segment.tile_x(), -1);
        assert_eq!(pixel_segment.tile_y(), -1);
        assert_eq!(pixel_segment.local_x(), local_x);
        assert_eq!(pixel_segment.local_y(), local_y);
        assert_eq!(
            pixel_segment.double_area(),
            i16::from(double_area_multiplier) * i16::from(cover)
        );
        assert_eq!(pixel_segment.cover(), cover);
    }

    #[test]
    fn pixel_segment_clipping() {
        let tile_x = -2;
        let tile_y = -2;

        let pixel_segment =
            PixelSegment::<TILE_WIDTH, TILE_HEIGHT>::new(0, tile_x, tile_y, 0, 0, 0, 0);

        assert_eq!(
            pixel_segment.tile_x(),
            -1,
            "negative tile coord clipped to -1"
        );
        assert_eq!(
            pixel_segment.tile_y(),
            -1,
            "negative tile coord clipped to -1"
        );

        let tile_x = i16::MIN;
        let tile_y = i16::MIN;

        let pixel_segment =
            PixelSegment::<TILE_WIDTH, TILE_HEIGHT>::new(0, tile_x, tile_y, 0, 0, 0, 0);

        assert_eq!(
            pixel_segment.tile_x(),
            -1,
            "negative tile coord clipped to -1"
        );
        assert_eq!(
            pixel_segment.tile_y(),
            -1,
            "negative tile coord clipped to -1"
        );
    }

    #[test]
    fn search_last_by_key_test() {
        let size = 50;
        let segments: Vec<_> = (0..(size * 2))
            .map(|i| PixelSegment::<TILE_WIDTH, TILE_HEIGHT>::new(i / 2, 0, 0, 0, 0, 0, 0))
            .collect();
        for i in 0..size {
            assert_eq!(
                Ok((i * 2 + 1) as usize),
                search_last_by_key(segments.as_slice(), i, |ps| ps.layer_id())
            );
        }
    }
}
