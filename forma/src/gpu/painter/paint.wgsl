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

let TILE_WIDTH = 16u;
let TILE_HEIGHT = 4u;
let TILE_WIDTH_SHIFT = 4u;
let TILE_HEIGHT_SHIFT = 2u;

let MAX_WIDTH_SHIFT = 16u;
let MAX_HEIGHT_SHIFT = 15u;

let BLOCK_LEN = 64u;
let BLOCK_SHIFT = 6u;
let BLOCK_MASK = 63u;
let QUEUES_LEN = 128u;
let QUEUES_MASK = 127u;

let PIXEL_WIDTH = 16;
// Reciprocal of two times the number of subpixel:
// DOUBLE_AREA_RECIP is 1.0 / (2.0 * PIXEL_WIDTH * PIXEL_WIDTH )
let PIXEL_DOUBLE_AREA_RECIP = 0.001953125;

// Reciprocal of the altas dimension:
// 1 / 4096
let ATLAS_SIZE_RECIP = 0.000244140625;

let LAYER_ID_NONE = 0xffffffffu;

struct PixelSegment {
    lo: u32,
    hi: u32,
}

let LAYER_ID_BIT_SIZE = 21u;
let DOUBLE_AREA_MULTIPLIER_BIT_SIZE = 6u;
let COVER_BIT_SIZE = 6u;

fn pixelSegmentTileX(seg: PixelSegment) -> i32 {
    return extractBits(
        i32(seg.hi),
        32u - (MAX_WIDTH_SHIFT - TILE_WIDTH_SHIFT) -
            (MAX_HEIGHT_SHIFT - TILE_HEIGHT_SHIFT),
        MAX_WIDTH_SHIFT - TILE_WIDTH_SHIFT,
    ) - 1;
}

fn pixelSegmentTileY(seg: PixelSegment) -> i32 {
    return extractBits(
        i32(seg.hi),
        32u - (MAX_HEIGHT_SHIFT - TILE_HEIGHT_SHIFT),
        MAX_HEIGHT_SHIFT - TILE_HEIGHT_SHIFT,
    ) - 1;
}

fn pixelSegmentLayerId(seg: PixelSegment) -> u32 {
    let lo = extractBits(
        seg.lo,
        TILE_WIDTH_SHIFT + TILE_HEIGHT_SHIFT +
            DOUBLE_AREA_MULTIPLIER_BIT_SIZE + COVER_BIT_SIZE,
        32u - TILE_WIDTH_SHIFT - TILE_HEIGHT_SHIFT -
            DOUBLE_AREA_MULTIPLIER_BIT_SIZE - COVER_BIT_SIZE,
    );

    return insertBits(
        lo,
        seg.hi,
        32u - TILE_WIDTH_SHIFT - TILE_HEIGHT_SHIFT -
            DOUBLE_AREA_MULTIPLIER_BIT_SIZE - COVER_BIT_SIZE,
        32u - (MAX_WIDTH_SHIFT - TILE_WIDTH_SHIFT) -
            (MAX_HEIGHT_SHIFT - TILE_HEIGHT_SHIFT),
    );
}

fn pixelSegmentLocalX(seg: PixelSegment) -> u32 {
    return extractBits(
        seg.lo,
        TILE_HEIGHT_SHIFT + DOUBLE_AREA_MULTIPLIER_BIT_SIZE + COVER_BIT_SIZE,
        TILE_WIDTH_SHIFT,
    );
}

fn pixelSegmentLocalY(seg: PixelSegment) -> u32 {
    return extractBits(
        seg.lo,
        DOUBLE_AREA_MULTIPLIER_BIT_SIZE + COVER_BIT_SIZE,
        TILE_HEIGHT_SHIFT,
    );
}

fn pixelSegmentDoubleAreaMultiplier(seg: PixelSegment) -> u32 {
    return extractBits(
        seg.lo,
        COVER_BIT_SIZE,
        DOUBLE_AREA_MULTIPLIER_BIT_SIZE,
    );
}

fn pixelSegmentCover(seg: PixelSegment) -> i32 {
    return extractBits(i32(seg.lo), 0u, COVER_BIT_SIZE);
}

struct OptimizedSegment {
    lo: u32,
    hi: u32,
}

let DOUBLE_AREA_BIT_SIZE = 12u;
let DOUBLE_AREA_OFFSET = 20u;
let COVER_OFFSET = 26u;

fn optimizedSegment(
    tile_x: i32,
    layer_id: u32,
    local_x: u32,
    local_y: u32,
    double_area: i32,
    cover: i32,
) -> OptimizedSegment {
    var lo = local_y;

    lo = insertBits(lo, local_x, TILE_HEIGHT_SHIFT, TILE_WIDTH_SHIFT);
    lo = u32(insertBits(
        i32(lo),
        tile_x,
        TILE_WIDTH_SHIFT + TILE_HEIGHT_SHIFT,
        MAX_WIDTH_SHIFT - TILE_WIDTH_SHIFT,
    ));
    lo = u32(insertBits(
        i32(lo),
        double_area,
        DOUBLE_AREA_OFFSET,
        DOUBLE_AREA_BIT_SIZE,
    ));

    var hi = layer_id;

    hi = u32(insertBits(i32(hi), cover, COVER_OFFSET, COVER_BIT_SIZE));

    return OptimizedSegment(lo, hi);
}

fn optimizedSegmentTileX(seg: OptimizedSegment) -> i32 {
    return extractBits(
        i32(seg.lo),
        TILE_WIDTH_SHIFT + TILE_HEIGHT_SHIFT,
        MAX_WIDTH_SHIFT - TILE_WIDTH_SHIFT,
    );
}

fn optimizedSegmentLayerId(seg: OptimizedSegment) -> u32 {
    return extractBits(seg.hi, 0u, LAYER_ID_BIT_SIZE);
}

fn optimizedSegmentLocalX(seg: OptimizedSegment) -> u32 {
    return extractBits(seg.lo, TILE_HEIGHT_SHIFT, TILE_WIDTH_SHIFT);
}

fn optimizedSegmentLocalY(seg: OptimizedSegment) -> u32 {
    return extractBits(seg.lo, 0u, TILE_HEIGHT_SHIFT);
}

fn optimizedSegmentDoubleArea(seg: OptimizedSegment) -> i32 {
    return extractBits(i32(seg.lo), DOUBLE_AREA_OFFSET, DOUBLE_AREA_BIT_SIZE);
}

fn optimizedSegmentCover(seg: OptimizedSegment) -> i32 {
    return extractBits(i32(seg.hi), COVER_OFFSET, COVER_BIT_SIZE);
}

struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

struct Config {
    segments_len: u32,
    width: u32,
    height: u32,
    _padding: u32,
    clear_color: Color,
}

struct Style {
    fill_rule: u32,
    color: Color,
    blend_mode: u32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage> segments: array<PixelSegment>;
@group(0) @binding(2) var<storage> style_indices: array<u32>;
@group(0) @binding(3) var<storage> styles: array<u32>;
@group(0) @binding(4) var atlas: texture_2d<f32>;
@group(0) @binding(5) var atlas_sampler: sampler;
@group(0) @binding(6) var image: texture_storage_2d<rgba16float, write>;

var<workgroup> segment_block: array<OptimizedSegment, BLOCK_LEN>;
var<private> segment_index: u32;
var<private> block_index: u32;

// Returns how many colors and stops the gradient has.
// Returns 0 when the fill type is not a gradient.
fn getGradientStopsCount(style_header: u32) -> u32 {
    let STYLE_STOPS_COUNT_BITS = 16u;
    let STYLE_STOPS_COUNT_OFFSET = 0u;
    return extractBits(style_header, STYLE_STOPS_COUNT_OFFSET, STYLE_STOPS_COUNT_BITS);
}

// Returns `paint::BlendMode` ordinal.
fn getBlendMode(style_header:u32) -> u32 {
    let STYLE_BLEND_MODE_BITS = 4u;
    let STYLE_BLEND_MODE_OFFSET = 16u; // STYLE_STOPS_COUNT_BITS + STYLE_STOPS_COUNT_OFFSET.
    return extractBits(style_header, STYLE_BLEND_MODE_OFFSET, STYLE_BLEND_MODE_BITS);
}

// Returns the fill function by position in the following list:
// [Solid, Linear gradient, Radial gradient, Texture]
fn getFillType(style_header: u32) -> u32 {
    let STYLE_FILL_BITS = 2u;
    let STYLE_FILL_OFFSET = 20u; // STYLE_BLEND_MODE_BITS + STYLE_BLEND_MODE_OFFSET.
    return extractBits(style_header, STYLE_FILL_OFFSET, STYLE_FILL_BITS);
}

// Returns 1 for `FillRule::EvenOdd` and 0 for `FillRile::NonZero`.
fn getFillRule(style_header: u32) -> u32 {
    let STYLE_FILL_RULE_BITS = 1u;
    let STYLE_FILL_RULE_OFFSET = 22u; // STYLE_FILL_BITS + STYLE_FILL_OFFSET.
    return extractBits(style_header, STYLE_FILL_RULE_OFFSET, STYLE_FILL_RULE_BITS);
}

// Retuns `Style::is_clipped` value.
fn getIsClipped(style_header: u32) -> bool {
    let IS_CLIPPED_BITS = 1u;
    let IS_CLIPPED_OFFSET = 23u; // STYLE_FILL_RULE_BITS + STYLE_FILL_RULE_BITS.
    return bool(extractBits(style_header, IS_CLIPPED_OFFSET, IS_CLIPPED_BITS));
}

// Returns 0 for `Func::Draw` and 1 for `Func::Clip`.
fn getFunc(style_header: u32) -> u32 {
    let FUNC_BITS = 1u;
    let FUNC_OFFSET = 24u;
    return extractBits(style_header, FUNC_OFFSET, FUNC_BITS);
}

// Return the `Func::Clip` payload.
fn getClipValue(offset:u32) -> u32 {
    return styles[offset + 1u];
}

// Reads a vector from the style buffer at the given offset.
fn getVec4F32(offset:u32) -> vec4<f32> {
    return vec4(
        bitcast<f32>(styles[offset]),
        bitcast<f32>(styles[offset + 1u]),
        bitcast<f32>(styles[offset + 2u]),
        bitcast<f32>(styles[offset + 3u]),
    );
}

// Returns the color used by solid fill function.
fn getSolidColor(offset: u32) -> vec4<f32> {
    return getVec4F32(offset + 1u);
}

// Returns the two 2D points for the gradient packed into a vector.
fn getGradientStartEnd(offset: u32) -> vec4<f32> {
    return getVec4F32(offset + 1u);
}

// Returns the color of the Nth gradient stop.
fn getGradientColor(offset: u32, stop_idx: u32) -> vec4<f32> {
    let SKIP_HEADER = 1u;
    let SKIP_START_END = 4u;
    let offset = offset + SKIP_HEADER + SKIP_START_END + stop_idx * 5u;
    return getVec4F32(offset);
}

// Returns the value the Nth gradient stop.
fn getGradientStop(offset: u32, stop_idx: u32) -> f32 {
    let SKIP_HEADER = 1u;
    let SKIP_START_END = 4u;
    let SKIP_COLOR = 4u;
    let offset = offset + SKIP_HEADER + SKIP_START_END + stop_idx * 5u + SKIP_COLOR;
    return bitcast<f32>(styles[offset]);
}

fn getTextureRotation(offset: u32) -> mat2x2<f32> {
    return mat2x2<f32>(
        bitcast<f32>(styles[offset + 1u]),
        bitcast<f32>(styles[offset + 3u]),
        bitcast<f32>(styles[offset + 2u]),
        bitcast<f32>(styles[offset + 4u]),
    );
}

fn getTextureTranslation(offset: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>(styles[offset + 5u]),
        bitcast<f32>(styles[offset + 6u]),
    );
}

fn getTextureRect(offset: u32) -> vec4<f32> {
    return getVec4F32(offset + 7u);
}

fn loadSegments(tile_y: i32, local_index: u32) -> bool {
    if block_index > (config.segments_len >> BLOCK_SHIFT) {
        return false;
    }

    let i = block_index * BLOCK_LEN + local_index;
    var opt_seg = optimizedSegment(
        -2,
        0u,
        0u,
        0u,
        0,
        0,
    );

    workgroupBarrier();

    if i < config.segments_len {
        let seg = segments[i];

        if pixelSegmentTileY(seg) == tile_y {
            let cover = pixelSegmentCover(seg);
            let double_area = i32(pixelSegmentDoubleAreaMultiplier(seg)) * cover;

            opt_seg = optimizedSegment(
                pixelSegmentTileX(seg),
                pixelSegmentLayerId(seg),
                pixelSegmentLocalX(seg),
                pixelSegmentLocalY(seg),
                double_area,
                cover,
            );
        }
    }

    segment_block[local_index] = opt_seg;

    workgroupBarrier();

    block_index++;

    return true;
}

fn clearColor() -> vec4<f32> {
    return vec4(
        config.clear_color.r,
        config.clear_color.g,
        config.clear_color.b,
        config.clear_color.a,
    );
}

var<workgroup> queues_layer_id_buffer: array<u32, QUEUES_LEN>;
var<workgroup> queues_cover_buffer: array<atomic<u32>, QUEUES_LEN>;

// Incides into the `queues_layer_id_buffer` and `queues_cover_buffer` ring
// buffers.
struct Queues {
    start0: u32,
    end0: u32,
    start1: u32,
}

struct Painter {
    queues: Queues,
    double_area: i32,
    cover: i32,
    color: vec4<f32>,
    clip_pixel_coverage: f32,
    clip_last_layer_id: u32,
}

fn areaToCoverage(double_area: i32, fill_rule: u32) -> f32 {
    switch fill_rule {
        // NonZero
        case 0u {
            return clamp(abs(f32(double_area) * PIXEL_DOUBLE_AREA_RECIP), 0.0, 1.0);
        }
        // EvenOdd
        default {
            // Coverage computation breaks pixels into 16 by 16 sub-pixels.
            // `double_area` is twice the number of sub-pixels covered.
            // Full coverage is 512 = 16 x 16 x 2, and a pixel can be covered multiple
            // times in case of windings.
            //
            // Returns a triangular wave function of period 1024,
            // going from 0.0 to 1.0 over the range 0..=512,
            // and from 1.0 to 0.0 over the range 512..=1024.
            return f32(512 - abs((double_area & 1023) - 512)) * PIXEL_DOUBLE_AREA_RECIP;
        }
    }
}

fn lum(color: vec3<f32>) -> f32 {
    return fma(
        color.r,
        0.3,
        fma(color.g, 0.59, color.b * 0.11),
    );
}

fn sat(color: vec3<f32>) -> f32 {
    return max(color.r, max(color.g, color.b)) -
        min(color.r, min(color.g, color.b));
}

fn clipColor(color: vec3<f32>) -> vec3<f32> {
    let l = lum(color);
    let n = min(color.r, min(color.g, color.b));
    let x = max(color.r, max(color.g, color.b));
    let l_1 = l - 1.0;
    let x_l_recip = 1.0 / (x - l);
    let l_n_recip_l = 1.0 / (l - n) * l;

    return select(
        select(
            color,
            fma(
                vec3(l_n_recip_l),
                color - vec3(l),
                vec3(l),
            ),
            n < 0.0,
        ),
        fma(
            vec3(x_l_recip),
            fma(
                vec3(l),
                vec3(l_1) - color,
                color,
            ),
            vec3(l),
        ),
        x > 1.0,
    );
}

fn setLum(color: vec3<f32>, l: f32) -> vec3<f32> {
    let d = l - lum(color);
    return clipColor(color + vec3(d));
}

fn setSat(color: vec3<f32>, s: f32) -> vec3<f32> {
    let c_min = min(color.r, min(color.g, color.b));
    let c_max = max(color.r, max(color.g, color.b));
    let c_mid = color.r + color.g + color.b - c_min - c_max;

    let min_lt_max = c_min < c_max;
    let s_mid = select(
        0.0,
        fma(s, -c_min, s * c_mid) / (c_max - c_min),
        min_lt_max,
    );
    let s_max = select(0.0, s, min_lt_max);

    return select(
        select(vec3(s_mid), vec3(0.0), color == vec3(c_min)),
        vec3(s_max),
        color == vec3(c_max),
    );
}

fn blend(dst: vec4<f32>, src: vec4<f32>, blend_mode: u32) -> vec4<f32> {
    let inv_dst_a = 1.0 - dst.a;
    let inv_dst_a_src_a = inv_dst_a * src.a;
    let inv_src_a = 1.0 - src.a;
    let dst_a_src_a = dst.a * src.a;

    var color: vec3<f32>;
    switch blend_mode {
        // Over
        case 0u {
            color = src.rgb;
        }
        // Multiply
        case 1u {
            color = dst.rgb * src.rgb;
        }
        // Screen
        case 2u {
            color = fma(dst.rgb, -src.rgb, dst.rgb) + src.rgb;
        }
        // Overlay
        case 3u {
            color = 2.0 * select(
                (dst.rgb + src.rgb -
                    fma(dst.rgb, src.rgb, vec3(0.5))),
                dst.rgb * src.rgb,
                dst.rgb <= vec3(0.5),
            );
        }
        // Darken
        case 4u {
            color = min(dst.rgb, src.rgb);
        }
        // Lighten
        case 5u {
            color = max(dst.rgb, src.rgb);
        }
        // ColorDodge
        case 6u {
            color = select(
                min(vec3(1.0), dst.rgb / (vec3(1.0) - src.rgb)),
                vec3(1.0),
                src.rgb == vec3(1.0),
            );
        }
        // ColorBurn
        case 7u {
            color = select(
                vec3(1.0) - min(
                    vec3(1.0),
                    (vec3(1.0) - dst.rgb) / src.rgb,
                ),
                vec3(0.0),
                src.rgb == vec3(0.0),
            );
        }
        // HardLight
        case 8u {
            color = 2.0 * select(
                dst.rgb + src.rgb -
                    fma(dst.rgb, src.rgb, vec3(0.5)),
                dst.rgb * src.rgb,
                src.rgb <= vec3(0.5),
            );
        }
        // SoftLight
        case 9u {
            let d = select(
                sqrt(dst.rgb),
                dst.rgb * fma(
                    fma(vec3(16.0), dst.rgb, vec3(-12.0)),
                    dst.rgb,
                    vec3(4.0),
                ),
                dst.rgb <= vec3(0.25),
            );
            color = select(
                fma(
                    d - dst.rgb,
                    fma(vec3(2.0), src.rgb, vec3(-1.0)),
                    dst.rgb,
                ),
                fma(
                    dst.rgb - vec3(1.0),
                    fma(src.rgb, vec3(-2.0), vec3(1.0)) *
                        dst.rgb,
                    dst.rgb
                ),
                src.rgb <= vec3(0.5),
            );
        }
        // Difference
        case 10u {
            color = abs(dst.rgb - src.rgb);
        }
        // Exclusion
        case 11u {
            color = fma(
                dst.rgb,
                fma(vec3(-2.0), src.rgb, vec3(1.0)),
                src.rgb,
            );
        }
        // Hue
        case 12u {
            color = setLum(setSat(src.rgb, sat(dst.rgb)), lum(dst.rgb));
        }
        // Saturation
        case 13u {
            color = setLum(setSat(dst.rgb, sat(src.rgb)), lum(dst.rgb));
        }
        // Color
        case 14u {
            color = setLum(src.rgb, lum(dst.rgb));
        }
        // Luminosity
        default {
            color = setLum(dst.rgb, lum(src.rgb));
        }
    }

    let current = fma(src.rgb, vec3(inv_dst_a_src_a), color.rgb * dst_a_src_a);

    return fma(dst, vec4(inv_src_a), vec4(current, src.a));
}

fn painterPushCover(
    painter: ptr<function, Painter>,
    layer_id: u32,
    style_header: u32,
    local_id: vec2<u32>,
) {
    queues_layer_id_buffer[(*painter).queues.start1] = layer_id;

    if local_id.x == 0u && local_id.y == 0u {
        atomicStore(&queues_cover_buffer[(*painter).queues.start1], 0u);
    }

    workgroupBarrier();

    if local_id.x == (TILE_WIDTH - 1u) {
        atomicOr(
            &queues_cover_buffer[(*painter).queues.start1],
            u32(((*painter).cover & 255) << (local_id.y << 3u)),
        );
    }

    workgroupBarrier();

    // FillRule::NonZero is 0, FillRule::EvenOdd is 1.
    let mask = select(0xffffffffu, 0x1f1f1f1fu, getFillRule(style_header) == 1u);

    (*painter).queues.start1 = (
        (*painter).queues.start1 +
        u32(bool(atomicLoad(&queues_cover_buffer[(*painter).queues.start1]) & mask))
    ) & QUEUES_MASK;
}

fn painterBlendLayer(
    painter: ptr<function, Painter>,
    layer_id: u32,
    pixel_coords: vec2<u32>,
    local_id: vec2<u32>,
) {
    let style_offset = style_indices[layer_id];
    let style_header = styles[style_offset];
    painterPushCover(painter, layer_id, style_header, local_id);

    if layer_id > (*painter).clip_last_layer_id {
        // Layers with `is_clipped: true` are not drawn.
        (*painter).clip_pixel_coverage = 0.0;
    }

    if getFunc(style_header) == 0u /* Func::Draw */ {
        var coverage =  areaToCoverage((*painter).double_area, getFillRule(style_header)) *
            select(1.0, (*painter).clip_pixel_coverage, getIsClipped(style_header));

        if coverage > 0.0 {
            var src: vec4<f32>;
            // Select the default branch when `getFunc(style_header)` is 1 which
            // means the function is `Func::Clip`.
            let fill_type = getFillType(style_header);
            switch fill_type {
                // Solid color.
                case 0u {
                    src = getSolidColor(style_offset);
                }

                // Gradients.
                case 1u, 2u {
                    let start_end = getGradientStartEnd(style_offset);
                    let start = start_end.xy;
                    let end = start_end.zw;
                    let d = end - start;
                    let p = vec2<f32>(pixel_coords) - start;
                    var t: f32;
                    switch fill_type {
                        // Linear gradient.
                        case 1u: {
                            t = clamp(dot(p, d) / dot(d, d), 0.0, 1.0);
                        }
                        // Linear gradient.
                        default {
                            t = sqrt(dot(p, p) / dot(d, d));
                        }
                    }
                    var i: u32 = getGradientStopsCount(style_header) - 1u;
                    loop {
                        if i <= 0u | getGradientStop(style_offset, i) < t { break; }
                        i--;
                    }
                    let from_color = getGradientColor(style_offset, i);
                    let from_stop = getGradientStop(style_offset, i);
                    let to_color = getGradientColor(style_offset, i + 1u);
                    let to_stop = getGradientStop(style_offset, i + 1u);
                    let t = (t - from_stop) / (to_stop - from_stop);
                    src = mix(from_color, to_color, t);
                }
                // Texture.
                default {
                    let r = getTextureRotation(style_offset);
                    let t = getTextureTranslation(style_offset);
                    let rect = getTextureRect(style_offset);
                    var p = vec2<f32>(pixel_coords) * r + t + rect.xy;

                    p.x = clamp(p.x, rect.x + 0.5, rect.z - 0.5);
                    p.y = clamp(p.y, rect.y + 0.5, rect.w - 0.5);

                    src = textureSampleLevel(atlas, atlas_sampler, p * ATLAS_SIZE_RECIP, 0.0);
                }
            }

            src.a *= coverage;
            (*painter).color = blend((*painter).color, src, getBlendMode(style_header));
        }
    } else {
        (*painter).clip_pixel_coverage = areaToCoverage((*painter).double_area, getFillRule(style_header));
        (*painter).clip_last_layer_id = getClipValue(style_indices[layer_id]) + layer_id;
    }

    (*painter).double_area = 0;
    (*painter).cover = 0;
}

fn painterPopQueueUntil(
    painter: ptr<function, Painter>,
    layer_id: u32,
    pixel_coords: vec2<u32>,
    local_id: vec2<u32>,
) {
    while (*painter).queues.start0 != (*painter).queues.end0 {
        let current_layer_id =
            queues_layer_id_buffer[(*painter).queues.start0];
        if (current_layer_id > layer_id) { break; }

        let shift = local_id.y << 3u;
        let cover = i32(queues_cover_buffer[(*painter).queues.start0]) <<
            (24u - shift) >> 24u;

        (*painter).double_area += cover * 2 * PIXEL_WIDTH;
        (*painter).cover += cover;

        if current_layer_id < layer_id {
            painterBlendLayer(painter, current_layer_id, pixel_coords, local_id);
        }

        (*painter).queues.start0 = ((*painter).queues.start0 + 1u) &
            QUEUES_MASK;
    }
}

// Accumulates cover from pixel segments with negative tile.x.
// Ignores segments from different rows.
fn painterNegativeCovers(
    painter: ptr<function, Painter>,
    tile: vec2<i32>,
    local_index: u32,
    local_id: vec2<u32>,
) {
    var seg: OptimizedSegment;
    var layer_id = LAYER_ID_NONE;
    loop {
        var should_break = false;
        loop {
            seg = segment_block[segment_index];

            should_break = optimizedSegmentTileX(seg) >= 0;

            if should_break || segment_index == BLOCK_LEN { break; }

            segment_index += 1u;

            let current_layer_id = optimizedSegmentLayerId(seg);

            if current_layer_id != layer_id {
                // All segments for the `layer_id` have been processed.
                // We can decide whenever to shade the pixel or not for this layer.
                if layer_id != LAYER_ID_NONE {
                    let style_header = styles[style_indices[layer_id]];
                    painterPushCover(
                        painter,
                        layer_id,
                        style_header,
                        local_id,
                    );
                    (*painter).cover = 0;
                }

                layer_id = current_layer_id;
            }

            let cover = select(
                0,
                optimizedSegmentCover(seg),
                optimizedSegmentLocalY(seg) == local_id.y,
            );

            (*painter).cover += cover;
        }

        if segment_index == BLOCK_LEN {
            should_break = !loadSegments(tile.y, local_index);
            segment_index = 0u;
        }

        if should_break {
            if layer_id != LAYER_ID_NONE {
                let style_header = styles[style_indices[layer_id]];
                painterPushCover(painter, layer_id, style_header, local_id);
                (*painter).cover = 0;
            }

            break;
        }
    }
}

// Accumulates cover from pixel segments and compute shading.
// Ignores segments from different rows.
fn painterPaintTile(
    painter: ptr<function, Painter>,
    tile: vec2<i32>,
    local_index: u32,
    pixel_coords: vec2<u32>,
    local_id: vec2<u32>,
) {
    var seg: OptimizedSegment;
    var layer_id = LAYER_ID_NONE;
    (*painter).clip_pixel_coverage = 0.0;
    (*painter).clip_last_layer_id = 0u;
    loop {
        var should_break = false;
        loop {
            seg = segment_block[segment_index];

            should_break = optimizedSegmentTileX(seg) != tile.x;

            if should_break || segment_index == BLOCK_LEN { break; }

            segment_index += 1u;

            let current_layer_id = optimizedSegmentLayerId(seg);

            if current_layer_id != layer_id {
                // All segments for the `layer_id` have been processed.
                // We can decide whenever to shade the pixel or not for this layer.
                if layer_id != LAYER_ID_NONE {
                    painterBlendLayer(painter, layer_id, pixel_coords, local_id);
                }

                painterPopQueueUntil(painter, current_layer_id, pixel_coords, local_id);

                layer_id = current_layer_id;
            }

            let local_x = optimizedSegmentLocalX(seg);
            let local_y = optimizedSegmentLocalY(seg);

            (*painter).double_area += select(
                0,
                optimizedSegmentDoubleArea(seg),
                local_id.x == local_x && local_id.y == local_y,
            );

            let cover = optimizedSegmentCover(seg);

            (*painter).double_area += 2 * PIXEL_WIDTH * select(
                0,
                cover,
                local_id.x > local_x && local_id.y == local_y,
            );
            (*painter).cover += select(
                0,
                cover,
                local_id.y == local_y,
            );
        }

        if segment_index == BLOCK_LEN {
            should_break = !loadSegments(tile.y, local_index);
            segment_index = 0u;
        }

        if should_break {
            if layer_id != LAYER_ID_NONE {
                painterBlendLayer(painter, layer_id, pixel_coords, local_id);
            }

            painterPopQueueUntil(painter, LAYER_ID_NONE, pixel_coords, local_id);

            break;
        }
    }
}

fn findStartOfTileRow(tile_y: i32) -> u32 {
    if config.segments_len == 0u {
        return 0u;
    }

    var end = config.segments_len - 1u;

    var start = 0u;
    while start < end {
        let mid = (start + end) >> 1u;

        if pixelSegmentTileY(segments[mid]) < tile_y {
            start = mid + 1u;
        } else {
            end = mid;
        }
    }

    return start;
}

// Paints an entire row of tiles.
@compute @workgroup_size(16, 4)
fn paint(
    @builtin(local_invocation_id) local_id_vec: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id_vec: vec3<u32>,
) {
    let local_id = local_id_vec.xy;
    var tile = vec2(-1, i32(workgroup_id_vec.x));
    let tile_row_len = (config.width + TILE_WIDTH - 1u) / TILE_WIDTH;

    let start_index = findStartOfTileRow(tile.y);
    block_index = start_index >> BLOCK_SHIFT;

    loadSegments(tile.y, local_index);
    segment_index = start_index & BLOCK_MASK;

    var painter: Painter;
    painter.queues = Queues(0u, 0u, 0u);
    painter.double_area = 0;
    painter.cover = 0;

    painterNegativeCovers(&painter, tile, local_index, local_id);

    painter.cover = 0;
    painter.queues.end0 = painter.queues.start1;
    tile.x += 1;

    while u32(tile.x) <= tile_row_len {
        painter.color = clearColor();
        let pixel_coords = vec2<i32>(local_id) + tile * vec2(
            i32(TILE_WIDTH),
            i32(TILE_HEIGHT),
        );
        painterPaintTile(&painter, tile, local_index, vec2<u32>(pixel_coords), local_id);
        textureStore(image, pixel_coords, painter.color);

        painter.queues.end0 = painter.queues.start1;

        tile.x += 1;
    }
}
