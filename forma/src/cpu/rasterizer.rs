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

use crumsort::ParCrumSort;
use rayon::prelude::*;

use crate::{consts, utils::PrefixScanIter, SegmentBufferView};

use super::PixelSegment;

// This finds the ith term in the ordered union of two sequences:
//
// 1. a * t + c
// 2. b * t + d
//
// It works by estimating the amount of items that came from sequence 1 and
// sequence 2 such that the next item will be the ith. This results in two
// indices from each sequence. The final step is to simply pick the smaller one
// which naturally comes next.
#[allow(clippy::too_many_arguments)]
fn find(
    i: i32,
    a_over_a_b: f64,
    b_over_a_b: f64,
    c_d_over_a_b: f64,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
) -> f32 {
    let i = i as f32;

    // Index estimation requires extra bits of information to work correctly for
    // cases where e.g. a + b would lose information.
    let ja = if b.is_finite() {
        b_over_a_b.mul_add(i as f64, -c_d_over_a_b).ceil() as f32
    } else {
        i
    };
    let jb = if a.is_finite() {
        a_over_a_b.mul_add(i as f64, c_d_over_a_b).ceil() as f32
    } else {
        i
    };

    let guess_a = a.mul_add(ja, c);
    let guess_b = b.mul_add(jb, d);

    guess_a.min(guess_b)
}

fn get_ith_pixel_segment_params(i: u32, a: f32, b: f32, c: f32, d: f32) -> [f32; 2] {
    // This ensures that for `i = 0` the first parameter will always be `0.0`.
    let i = i as i32 - i32::from(c != 0.0) - i32::from(d != 0.0);

    let sum_recip = (a as f64 + b as f64).recip();
    let a_over_a_b = a as f64 * sum_recip;
    let b_over_a_b = b as f64 * sum_recip;
    let c_d_over_a_b = (c as f64 - d as f64) * sum_recip;

    let [t0, t1] = [i, i + 1].map(|i| find(i, a_over_a_b, b_over_a_b, c_d_over_a_b, a, b, c, d));

    // We want to ensure that the pixel segments end exactly at `0.0` and `0.0`.
    [t0.max(0.0), t1.min(1.0)]
}

fn round(v: f32) -> i32 {
    unsafe { (v + 0.5).floor().to_int_unchecked() }
}

#[derive(Debug, Default)]
pub struct Rasterizer<const TW: usize, const TH: usize> {
    segments: Vec<PixelSegment<TW, TH>>,
}

impl<const TW: usize, const TH: usize> Rasterizer<TW, TH> {
    pub fn segments(&self) -> &[PixelSegment<TW, TH>] {
        self.segments.as_slice()
    }

    #[inline(never)]
    pub fn rasterize(&mut self, segment_buffer_view: &SegmentBufferView) {
        // Shard the workload into set of similar output size in PixelSegment.
        let iter = PrefixScanIter::new(&segment_buffer_view.lengths);

        iter.into_par_iter()
            .with_min_len(256)
            // `line_i`th line segment, `seg_i`th pixel segment within that line.
            .map(|(line_i, seg_i)| {
                let line_i = line_i as usize;

                let [t0, t1] = get_ith_pixel_segment_params(
                    seg_i,
                    segment_buffer_view.a[line_i],
                    segment_buffer_view.b[line_i],
                    segment_buffer_view.c[line_i],
                    segment_buffer_view.d[line_i],
                );

                let x0f = t0.mul_add(
                    segment_buffer_view.dx[line_i],
                    segment_buffer_view.x0[line_i],
                );
                let y0f = t0.mul_add(
                    segment_buffer_view.dy[line_i],
                    segment_buffer_view.y0[line_i],
                );
                let x1f = t1.mul_add(
                    segment_buffer_view.dx[line_i],
                    segment_buffer_view.x0[line_i],
                );
                let y1f = t1.mul_add(
                    segment_buffer_view.dy[line_i],
                    segment_buffer_view.y0[line_i],
                );

                let x0_sub = round(x0f);
                let x1_sub = round(x1f);
                let y0_sub = round(y0f);
                let y1_sub = round(y1f);

                let border_x = x0_sub.min(x1_sub) >> consts::PIXEL_SHIFT;
                let border_y = y0_sub.min(y1_sub) >> consts::PIXEL_SHIFT;

                let tile_x = (border_x >> TW.trailing_zeros() as i32) as i16;
                let tile_y = (border_y >> TH.trailing_zeros() as i32) as i16;
                let local_x = (border_x & (TW - 1) as i32) as u8;
                let local_y = (border_y & (TH - 1) as i32) as u8;

                let border = (border_x << consts::PIXEL_SHIFT) + consts::PIXEL_WIDTH as i32;
                let height = y1_sub - y0_sub;

                let double_area_multiplier =
                    ((x1_sub - x0_sub).abs() + 2 * (border - x0_sub.max(x1_sub))) as u8;
                let cover = height as i8;

                PixelSegment::new(
                    segment_buffer_view.orders[line_i],
                    tile_x,
                    tile_y,
                    local_x,
                    local_y,
                    double_area_multiplier,
                    cover,
                )
            })
            .collect_into_vec(&mut self.segments);
    }

    #[inline]
    pub fn sort(&mut self) {
        self.segments.par_crumsort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        composition::Composition,
        consts::cpu::{TILE_HEIGHT, TILE_WIDTH},
        math::Point,
        segment::{GeomId, SegmentBuffer},
        utils::Order,
    };

    fn segments(p0: Point, p1: Point) -> Vec<PixelSegment<TILE_WIDTH, TILE_HEIGHT>> {
        let mut segment_buffer = SegmentBuffer::default();
        let mut composition = Composition::new();

        composition.get_mut_or_insert_default(Order::new(0).unwrap());

        segment_buffer.push(GeomId::default(), [p0, p1]);

        let (layers, geom_id_to_order) = composition.layers_for_segments();

        let lines = segment_buffer.fill_cpu_view(usize::MAX, usize::MAX, layers, &geom_id_to_order);

        let mut rasterizer = Rasterizer::default();
        rasterizer.rasterize(&lines);

        rasterizer.segments().to_vec()
    }

    fn areas_and_covers(segments: &[PixelSegment<TILE_WIDTH, TILE_HEIGHT>]) -> Vec<(i16, i8)> {
        segments
            .iter()
            .map(|&segment| (segment.double_area(), segment.cover()))
            .collect()
    }

    #[test]
    fn find_first_7() {
        let a = 2.0;
        let b = 3.0;
        let c = 0.2;
        let d = 0.1;

        let sum_recip = (a as f64 + b as f64).recip();
        let a_over_a_b = a as f64 * sum_recip;
        let b_over_a_b = b as f64 * sum_recip;
        let c_d_over_a_b = (c as f64 - d as f64) * sum_recip;

        assert_eq!(
            (0..7)
                .into_iter()
                .map(|i| find(i - 1, a_over_a_b, b_over_a_b, c_d_over_a_b, a, b, c, d))
                .collect::<Vec<_>>(),
            [0.1, 0.2, 2.2, 3.1, 4.2, 6.1, 6.2],
        );
    }

    #[test]
    fn find_ab_large_ratio() {
        let a = 16_777_216.0;
        let b = 0.000_1;
        let c = 10.0;
        let d = 0.000_01;

        let sum_recip = (a as f64 + b as f64).recip();
        let a_over_a_b = a as f64 * sum_recip;
        let b_over_a_b = b as f64 * sum_recip;
        let c_d_over_a_b = (c as f64 - d as f64) * sum_recip;

        assert_eq!(
            (2..4)
                .into_iter()
                .map(|i| find(i - 1, a_over_a_b, b_over_a_b, c_d_over_a_b, a, b, c, d))
                .collect::<Vec<_>>(),
            [0.000_21, 0.000_31],
        );
    }

    #[test]
    fn area_cover_octant_1() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(3.0, 2.0))),
            [
                (11 * 16, 11),
                (5 * 8 + 2 * (5 * 8), 5),
                (5 * 8, 5),
                (11 * 16, 11)
            ],
        );
    }

    #[test]
    fn area_cover_octant_2() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(2.0, 3.0))),
            [
                (16 * 11 + 2 * (16 * 5), 16),
                (8 * 5, 8),
                (8 * 5 + 2 * (8 * 11), 8),
                (16 * 11, 16)
            ],
        );
    }

    #[test]
    fn area_cover_octant_3() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(-2.0, 3.0))),
            [
                (16 * 11, 16),
                (8 * 5 + 2 * (8 * 11), 8),
                (8 * 5, 8),
                (16 * 11 + 2 * (16 * 5), 16)
            ],
        );
    }

    #[test]
    fn area_cover_octant_4() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(-3.0, 2.0))),
            [
                (11 * 16, 11),
                (5 * 8, 5),
                (5 * 8 + 2 * (5 * 8), 5),
                (11 * 16, 11)
            ],
        );
    }

    #[test]
    fn area_cover_octant_5() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(3.0, 2.0), Point::new(0.0, 0.0))),
            [
                (-(11 * 16), -11),
                (-(5 * 8), -5),
                (-(5 * 8 + 2 * (5 * 8)), -5),
                (-(11 * 16), -11)
            ],
        );
    }

    #[test]
    fn area_cover_octant_6() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(2.0, 3.0), Point::new(0.0, 0.0))),
            [
                (-(16 * 11), -16),
                (-(8 * 5 + 2 * (8 * 11)), -8),
                (-(8 * 5), -8),
                (-(16 * 11 + 2 * (16 * 5)), -16),
            ],
        );
    }

    #[test]
    fn area_cover_octant_7() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 3.0), Point::new(2.0, 0.0))),
            [
                (-(16 * 11 + 2 * (16 * 5)), -16),
                (-(8 * 5), -8),
                (-(8 * 5 + 2 * (8 * 11)), -8),
                (-(16 * 11), -16),
            ],
        );
    }

    #[test]
    fn area_cover_octant_8() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 2.0), Point::new(3.0, 0.0))),
            [
                (-(11 * 16), -11),
                (-(5 * 8 + 2 * (5 * 8)), -5),
                (-(5 * 8), -5),
                (-(11 * 16), -11)
            ],
        );
    }

    #[test]
    fn area_cover_axis_0() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(1.0, 0.0))),
            []
        );
    }

    #[test]
    fn area_cover_axis_45() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(1.0, 1.0))),
            [(16 * 16, 16)],
        );
    }

    #[test]
    fn area_cover_axis_90() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(0.0, 1.0))),
            [(2 * 16 * 16, 16)],
        );
    }

    #[test]
    fn area_cover_axis_135() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(-1.0, 1.0))),
            [(16 * 16, 16)],
        );
    }

    #[test]
    fn area_cover_axis_180() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(-1.0, 0.0))),
            []
        );
    }

    #[test]
    fn area_cover_axis_225() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(1.0, 1.0), Point::new(0.0, 0.0))),
            [(-(16 * 16), -16)],
        );
    }

    #[test]
    fn area_cover_axis_270() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 1.0), Point::new(0.0, 0.0))),
            [(2 * -(16 * 16), -16)],
        );
    }

    #[test]
    fn area_cover_axis_315() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 1.0), Point::new(1.0, 0.0))),
            [(-(16 * 16), -16)],
        );
    }

    fn tiles(segments: &[PixelSegment<TILE_WIDTH, TILE_HEIGHT>]) -> Vec<(i16, i16, u8, u8)> {
        segments
            .iter()
            .map(|&segment| {
                (
                    segment.tile_x(),
                    segment.tile_y(),
                    segment.local_x(),
                    segment.local_y(),
                )
            })
            .collect()
    }

    #[test]
    fn tile_octant_1() {
        assert_eq!(
            tiles(&segments(
                Point::new(TILE_WIDTH as f32, TILE_HEIGHT as f32),
                Point::new(TILE_WIDTH as f32 + 3.0, TILE_HEIGHT as f32 + 2.0),
            )),
            [(1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1), (1, 1, 2, 1)],
        );
    }

    #[test]
    fn tile_octant_2() {
        assert_eq!(
            tiles(&segments(
                Point::new(TILE_WIDTH as f32, TILE_HEIGHT as f32),
                Point::new(TILE_WIDTH as f32 + 2.0, TILE_HEIGHT as f32 + 3.0),
            )),
            [(1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 1), (1, 1, 1, 2)],
        );
    }

    #[test]
    fn tile_octant_3() {
        assert_eq!(
            tiles(&segments(
                Point::new(-(TILE_WIDTH as f32), TILE_HEIGHT as f32),
                Point::new(-(TILE_WIDTH as f32) - 2.0, TILE_HEIGHT as f32 + 3.0),
            )),
            [
                (-1, 1, TILE_WIDTH as u8 - 1, 0),
                (-1, 1, TILE_WIDTH as u8 - 1, 1),
                (-1, 1, TILE_WIDTH as u8 - 2, 1),
                (-1, 1, TILE_WIDTH as u8 - 2, 2),
            ],
        );
    }

    #[test]
    fn tile_octant_4() {
        assert_eq!(
            tiles(&segments(
                Point::new(-(TILE_WIDTH as f32), TILE_HEIGHT as f32),
                Point::new(-(TILE_WIDTH as f32) - 3.0, TILE_HEIGHT as f32 + 2.0),
            )),
            [
                (-1, 1, TILE_WIDTH as u8 - 1, 0),
                (-1, 1, TILE_WIDTH as u8 - 2, 0),
                (-1, 1, TILE_WIDTH as u8 - 2, 1),
                (-1, 1, TILE_WIDTH as u8 - 3, 1),
            ],
        );
    }

    #[test]
    fn tile_octant_5() {
        assert_eq!(
            tiles(&segments(
                Point::new(-(TILE_WIDTH as f32), TILE_HEIGHT as f32),
                Point::new(-(TILE_WIDTH as f32) - 3.0, TILE_HEIGHT as f32 - 2.0),
            )),
            [
                (-1, 0, TILE_WIDTH as u8 - 1, TILE_HEIGHT as u8 - 1),
                (-1, 0, TILE_WIDTH as u8 - 2, TILE_HEIGHT as u8 - 1),
                (-1, 0, TILE_WIDTH as u8 - 2, TILE_HEIGHT as u8 - 2),
                (-1, 0, TILE_WIDTH as u8 - 3, TILE_HEIGHT as u8 - 2),
            ],
        );
    }

    #[test]
    fn tile_octant_6() {
        assert_eq!(
            tiles(&segments(
                Point::new(-(TILE_WIDTH as f32), TILE_HEIGHT as f32),
                Point::new(-(TILE_WIDTH as f32) - 2.0, TILE_HEIGHT as f32 - 3.0),
            )),
            [
                (-1, 0, TILE_WIDTH as u8 - 1, TILE_HEIGHT as u8 - 1),
                (-1, 0, TILE_WIDTH as u8 - 1, TILE_HEIGHT as u8 - 2),
                (-1, 0, TILE_WIDTH as u8 - 2, TILE_HEIGHT as u8 - 2),
                (-1, 0, TILE_WIDTH as u8 - 2, TILE_HEIGHT as u8 - 3),
            ],
        );
    }

    #[test]
    fn tile_octant_7() {
        assert_eq!(
            tiles(&segments(
                Point::new(TILE_WIDTH as f32, TILE_HEIGHT as f32),
                Point::new(TILE_WIDTH as f32 + 2.0, (TILE_HEIGHT as f32) - 3.0),
            )),
            [
                (1, 0, 0, TILE_HEIGHT as u8 - 1),
                (1, 0, 0, TILE_HEIGHT as u8 - 2),
                (1, 0, 1, TILE_HEIGHT as u8 - 2),
                (1, 0, 1, TILE_HEIGHT as u8 - 3),
            ],
        );
    }

    #[test]
    fn tile_octant_8() {
        assert_eq!(
            tiles(&segments(
                Point::new(TILE_WIDTH as f32, TILE_HEIGHT as f32),
                Point::new(TILE_WIDTH as f32 + 3.0, (TILE_HEIGHT as f32) - 2.0),
            )),
            [
                (1, 0, 0, TILE_HEIGHT as u8 - 1),
                (1, 0, 1, TILE_HEIGHT as u8 - 1),
                (1, 0, 1, TILE_HEIGHT as u8 - 2),
                (1, 0, 2, TILE_HEIGHT as u8 - 2),
            ],
        );
    }

    #[test]
    fn start_and_end_not_on_pixel_border() {
        assert_eq!(
            areas_and_covers(&segments(Point::new(0.5, 0.25), Point::new(4.0, 2.0)))[0],
            (4 * 8, 4),
        );

        assert_eq!(
            areas_and_covers(&segments(Point::new(0.0, 0.0), Point::new(3.5, 1.75)))[4],
            (4 * 8 + 2 * (4 * 8), 4),
        );
    }
}
