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

mod conveyor_sort;
mod painter;
mod rasterizer;
mod renderer;
mod style_map;

pub(crate) use style_map::StyleMap;

pub use self::renderer::{Renderer, Timings};

pub(crate) struct TimeStamp<'qs> {
    query_set: &'qs wgpu::QuerySet,
    start_index: u32,
    end_index: u32,
}
pub(crate) trait GpuContext {
    type Data<'s>;

    fn init(device: &wgpu::Device) -> Self;
    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        time_stamp: Option<TimeStamp>,
        data: &mut Self::Data<'_>,
    );
}
