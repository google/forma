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

use std::{borrow::Cow, mem, slice};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{consts, segment::SegmentBufferView};

use super::{GpuContext, TimeStamp};

#[derive(Debug)]
pub struct Data<'s> {
    pub segment_buffer_view: &'s SegmentBufferView,
    pub pixel_segment_buffer: wgpu::Buffer,
    pub style_buffer: wgpu::Buffer,
    pub style_offset_buffer: wgpu::Buffer,
    pub atlas_texture: wgpu::TextureView,
    pub output_texture: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub clear_color: Color,
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Style {
    pub fill_rule: u32,
    pub color: Color,
    pub blend_mode: u32,
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct Config {
    pub segments_len: u32,
    pub width: u32,
    pub height: u32,
    pub _padding: u32,
    pub clear_color: Color,
}

impl Config {
    pub fn as_byte_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts((self as *const _) as *const u8, mem::size_of::<Self>()) }
    }
}

#[derive(Debug)]
pub struct PaintContext {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuContext for PaintContext {
    type Data<'s> = Data<'s>;

    fn init(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paint.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("paint.wgsl"))),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "paint",
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Self {
            pipeline,
            bind_group_layout,
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
            segments_len: data.segment_buffer_view.len() as _,
            width: data.width,
            height: data.height,
            clear_color: data.clear_color,
            _padding: 0,
        };

        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: config.as_byte_slice(),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data.pixel_segment_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: data.style_offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: data.style_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&data.atlas_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&data.output_texture),
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
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            pass.dispatch_workgroups(
                ((data.height as usize + consts::gpu::TILE_HEIGHT - 1) / consts::gpu::TILE_HEIGHT)
                    as _,
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
