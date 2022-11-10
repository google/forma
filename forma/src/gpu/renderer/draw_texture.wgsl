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

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var vertex_output: VertexOutput;

    vertex_output.uv = vec2(
        f32((i32(vertex_index) << 1u) & 2),
        f32(i32(vertex_index) & 2),
    );

    let pos = 2.0 * vertex_output.uv - vec2(1.0, 1.0);
    vertex_output.pos = vec4(pos.x, pos.y, 0.0, 1.0);

    vertex_output.uv.y = 1.0 - vertex_output.uv.y;

    return vertex_output;
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(image, smp, uv);
}
