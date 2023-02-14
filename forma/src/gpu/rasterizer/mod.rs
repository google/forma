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

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{segment::SegmentBufferView, utils::DivCeil};

use super::{GpuContext, TimeStamp};

const WORKGROUP_SIZE: u32 = 256;
const MAX_INVOCATION_COUNT: u32 = 16;

#[derive(Debug)]
pub struct Data<'s> {
    pub segment_buffer_view: &'s SegmentBufferView,
    pub pixel_segment_buffer: wgpu::Buffer,
}

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Config {
    pub lines_len: u32,
    pub segments_len: u32,
}

impl Config {
    pub fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[derive(Debug)]
pub struct RasterizerContext {
    prepare_lines_pipeline: wgpu::ComputePipeline,
    prepare_lines_layout: wgpu::BindGroupLayout,
    rasterizer_pipeline: wgpu::ComputePipeline,
    rasterizer_layout: wgpu::BindGroupLayout,
}

impl GpuContext for RasterizerContext {
    type Data<'s> = Data<'s>;

    fn init(device: &wgpu::Device) -> Self {
        let rasterizer_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rasterizer.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("rasterizer.wgsl"))),
        });

        let prepare_lines_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prepareLines"),
                layout: None,
                module: &rasterizer_module,
                entry_point: "prepareLines",
            });
        let prepare_lines_layout = prepare_lines_pipeline.get_bind_group_layout(0);

        let rasterizer_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rasterize"),
                layout: None,
                module: &rasterizer_module,
                entry_point: "rasterize",
            });
        let rasterizer_layout = rasterizer_pipeline.get_bind_group_layout(0);

        RasterizerContext {
            prepare_lines_pipeline,
            prepare_lines_layout,
            rasterizer_pipeline,
            rasterizer_layout,
        }
    }

    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        time_stamp: Option<super::TimeStamp>,
        data: &mut Data,
    ) {
        let config = Config {
            lines_len: data.segment_buffer_view.inner_len() as _,
            segments_len: data.segment_buffer_view.len() as _,
        };
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: config.as_byte_slice(),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points"),
            contents: bytemuck::cast_slice(&data.segment_buffer_view.points),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let order_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("orders"),
            contents: bytemuck::cast_slice(&data.segment_buffer_view.orders),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // The struct rasterizer.wgsl:Lines has 9 x 32 bits fields.
        let line_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lines"),
            usage: wgpu::BufferUsages::STORAGE,
            size: (config.lines_len * 9 * 4) as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.prepare_lines_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: line_buffer.as_entire_binding(),
                },
            ],
        });

        if let Some(TimeStamp {
            query_set,
            start_index,
            ..
        }) = time_stamp
        {
            encoder.write_timestamp(query_set, start_index);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prepare_lines"),
            });
            pass.set_pipeline(&self.prepare_lines_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            pass.dispatch_workgroups(
                config
                    .lines_len
                    .div_ceil_(WORKGROUP_SIZE)
                    .min(MAX_INVOCATION_COUNT),
                1,
                1,
            );
        }

        if let Some(TimeStamp {
            query_set,
            end_index,
            ..
        }) = time_stamp
        {
            encoder.write_timestamp(query_set, end_index);
        }

        let line_len_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line_lens"),
            contents: bytemuck::cast_slice(&data.segment_buffer_view.lengths),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.rasterizer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: line_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: line_len_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: data.pixel_segment_buffer.as_entire_binding(),
                },
            ],
        });

        if let Some(TimeStamp {
            query_set,
            start_index,
            ..
        }) = time_stamp
        {
            encoder.write_timestamp(query_set, start_index);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rasterize"),
            });
            pass.set_pipeline(&self.rasterizer_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            pass.dispatch_workgroups(
                (data.segment_buffer_view.len() as u32)
                    .div_ceil_(WORKGROUP_SIZE)
                    .min(MAX_INVOCATION_COUNT),
                1,
                1,
            );
        }

        if let Some(TimeStamp {
            query_set,
            end_index,
            ..
        }) = time_stamp
        {
            encoder.write_timestamp(query_set, end_index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{distributions::Uniform, prelude::*};

    use crate::{
        consts::gpu::{TILE_HEIGHT, TILE_WIDTH},
        cpu::{PixelSegment, Rasterizer},
        math::Point,
        segment::{GeomId, SegmentBuffer},
        Composition, Order, PathBuilder,
    };

    fn run_rasterizer(
        segment_buffer_view: SegmentBufferView,
    ) -> (
        Vec<PixelSegment<TILE_WIDTH, TILE_HEIGHT>>,
        SegmentBufferView,
    ) {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .unwrap();

        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());

        let context = RasterizerContext::init(&device);
        let segment_buffer_len_padded = segment_buffer_view.len();
        let size = (segment_buffer_len_padded * std::mem::size_of::<u64>()) as wgpu::BufferAddress;

        let pixel_segment_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut data = Data {
            segment_buffer_view: &segment_buffer_view,
            pixel_segment_buffer,
        };
        context.encode(&device, &mut encoder, None, &mut data);

        encoder.copy_buffer_to_buffer(&data.pixel_segment_buffer, 0, &staging_buffer, 0, size);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let mut actual_segments = {
            let buffer_slice = staging_buffer.slice(..);

            buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
            device.poll(wgpu::Maintain::Wait);

            bytemuck::cast_slice(&buffer_slice.get_mapped_range()).to_vec()
        };

        staging_buffer.unmap();
        actual_segments.sort();

        (actual_segments, segment_buffer_view)
    }

    #[test]
    fn rasterize_triangle() {
        let path = PathBuilder::new()
            .move_to(Point { x: 1.0, y: 1.0 })
            .line_to(Point { x: 1.0, y: 4.0 })
            .line_to(Point { x: 2.0, y: 4.0 })
            .build();

        let mut segment_buffer = SegmentBuffer::default();
        let mut composition = Composition::new();

        segment_buffer.push_path(GeomId::default(), &path);

        composition.get_mut_or_insert_default(Order::new(1).unwrap());

        let (layers, geom_id_to_order) = composition.layers_for_segments();

        let segment_buffer_view =
            segment_buffer.fill_gpu_view(usize::MAX, usize::MAX, layers, &geom_id_to_order);

        let (actual_segments, _) = run_rasterizer(segment_buffer_view);
        let expected_segments = [
            PixelSegment::new(1, 0, 0, 1, 1, 32, 16),
            PixelSegment::new(1, 0, 0, 1, 2, 32, 16),
            PixelSegment::new(1, 0, 0, 1, 3, 32, 16),
            PixelSegment::new(1, 0, 0, 1, 3, 5, -16),
            PixelSegment::new(1, 0, 0, 1, 2, 16, -16),
            PixelSegment::new(1, 0, 0, 1, 1, 27, -16),
        ]
        .to_vec();

        assert_eq!(expected_segments, actual_segments);
    }

    #[test]
    fn rasterize_random_quad() {
        let mut segment_buffer = SegmentBuffer::default();
        let mut composition = Composition::new();
        let mut rng = SmallRng::seed_from_u64(0);

        // Using a small range to workaround the precision difference between CPU and GPU
        // that sometimes lead to rounding differences.
        let coord_range = Uniform::new(-8i32, 16);
        let mut rnd_point = || Point {
            x: rng.sample(coord_range) as f32,
            y: rng.sample(coord_range) as f32,
        };

        let mut geom_id = GeomId::default();

        for _ in 0..4096 {
            let path = PathBuilder::new()
                .move_to(rnd_point())
                .line_to(rnd_point())
                .line_to(rnd_point())
                .line_to(rnd_point())
                .build();

            composition.get_mut_or_insert_default(Order::new(geom_id.get() as u32).unwrap());

            segment_buffer.push_path(geom_id, &path);
            geom_id = geom_id.next();
        }

        let (layers, geom_id_to_order) = composition.layers_for_segments();

        let mut gpu_segments = {
            let segment_buffer_view =
                segment_buffer.fill_gpu_view(usize::MAX, usize::MAX, layers, &geom_id_to_order);
            let (segments, segment_buffer_view) = run_rasterizer(segment_buffer_view);
            segment_buffer = segment_buffer_view.recycle();

            segments
        };

        let mut cpu_segments = {
            let segment_buffer_view =
                segment_buffer.fill_cpu_view(usize::MAX, usize::MAX, layers, &geom_id_to_order);
            let mut rasterizer = Rasterizer::<TILE_WIDTH, TILE_HEIGHT>::default();
            rasterizer.rasterize(&segment_buffer_view);

            rasterizer.segments().to_vec()
        };

        // Only test significant pixel segments, i.e. ones with `double_area` and `cover` different
        // from 0.
        let is_significant = |seg: &PixelSegment<TILE_WIDTH, TILE_HEIGHT>| {
            seg.double_area() != 0 && seg.cover() != 0
        };

        cpu_segments.sort();
        gpu_segments.sort();

        cpu_segments
            .iter()
            .filter(|seg| is_significant(seg))
            .zip(gpu_segments.iter().filter(|seg| is_significant(seg)))
            .enumerate()
            .for_each(|(i, (c, g))| {
                assert_eq!(c, g, "Segment {}", i);
            });
    }
}
