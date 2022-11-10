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

use std::{borrow::Cow, convert::TryFrom, mem};

use bytemuck::{Pod, Zeroable};
use ramhorns::{Content, Template};
use wgpu::util::DeviceExt;

use crate::{segment::SegmentBufferView, utils::DivCeil};

use super::{GpuContext, TimeStamp};

const MAX_WORKGROUPS: usize = 65_536;
pub const BLOCK_SIZE: BlockSize = BlockSize::new(64, 9);

#[derive(Debug)]
pub struct Data<'s> {
    pub segment_buffer_view: &'s SegmentBufferView,
    pub pixel_segment_buffer: wgpu::Buffer,
    pub swap_buffer: wgpu::Buffer,
    pub offset_buffer: wgpu::Buffer,
}

#[derive(Content, Debug)]
pub struct BlockSize {
    block_width: u32,
    block_height: u32,
    pub block_len: u32,
}

impl BlockSize {
    pub const fn new(block_width: u32, block_height: u32) -> Self {
        Self {
            block_width,
            block_height,
            block_len: block_width * block_height,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Config {
    len: u32,
    len_in_blocks: u32,
    n_way: u32,
}

impl Config {
    pub fn new(len: usize) -> Option<Self> {
        let len_in_blocks = u32::try_from(len.div_ceil_(BLOCK_SIZE.block_len as _)).ok()?;

        Some(Self {
            len: len as _,
            len_in_blocks,
            n_way: 0,
        })
    }

    pub fn workgroup_size(&self) -> u32 {
        self.len_in_blocks.min(MAX_WORKGROUPS as u32)
    }
}

#[derive(Debug)]
pub struct SortContext {
    block_sort_pipeline: wgpu::ComputePipeline,
    block_sort_bind_group_layout: wgpu::BindGroupLayout,
    find_merge_offsets_pipeline: wgpu::ComputePipeline,
    find_merge_offsets_bind_group_layout: wgpu::BindGroupLayout,
    merge_blocks_pipeline: wgpu::ComputePipeline,
    merge_blocks_bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuContext for SortContext {
    type Data<'s> = Data<'s>;

    fn init(device: &wgpu::Device) -> Self {
        let template = Template::new(include_str!("sort.wgsl")).unwrap();

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(template.render(&BLOCK_SIZE))),
        });

        let block_sort_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: "blockSort",
            });

        let block_sort_bind_group_layout = block_sort_pipeline.get_bind_group_layout(0);

        let find_merge_offsets_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: "findMergeOffsets",
            });

        let find_merge_offsets_bind_group_layout =
            find_merge_offsets_pipeline.get_bind_group_layout(0);

        let merge_blocks_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: "mergeBlocks",
            });

        let merge_blocks_bind_group_layout = merge_blocks_pipeline.get_bind_group_layout(0);

        Self {
            block_sort_pipeline,
            block_sort_bind_group_layout,
            find_merge_offsets_pipeline,
            find_merge_offsets_bind_group_layout,
            merge_blocks_pipeline,
            merge_blocks_bind_group_layout,
        }
    }

    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        time_stamp: Option<super::TimeStamp>,
        data: &mut Data,
    ) {
        let mut config =
            Config::new(data.segment_buffer_view.len()).expect("numbers length too high");

        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let block_sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.block_sort_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.pixel_segment_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config_buffer.as_entire_binding(),
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
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.block_sort_pipeline);
            cpass.set_bind_group(0, &block_sort_bind_group, &[]);

            cpass.dispatch_workgroups(config.workgroup_size(), 1, 1);
        }

        let rounds = config.len_in_blocks.next_power_of_two().trailing_zeros();
        let max_rounds = device
            .limits()
            .max_storage_buffer_binding_size
            .div_ceil_(BLOCK_SIZE.block_len)
            .next_power_of_two()
            .trailing_zeros();

        for round in 0..max_rounds {
            config.n_way = 1 << (round + 1);

            let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&config),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            });

            let find_merge_offsets_bind_group =
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.find_merge_offsets_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: if round % 2 == 0 {
                                data.pixel_segment_buffer.as_entire_binding()
                            } else {
                                data.swap_buffer.as_entire_binding()
                            },
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: config_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: data.offset_buffer.as_entire_binding(),
                        },
                    ],
                });

            {
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&self.find_merge_offsets_pipeline);
                cpass.set_bind_group(0, &find_merge_offsets_bind_group, &[]);

                cpass.dispatch_workgroups(
                    config.workgroup_size().div_ceil_(BLOCK_SIZE.block_width),
                    1,
                    1,
                );
            }

            let merge_blocks_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.merge_blocks_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: if round % 2 == 0 {
                            data.pixel_segment_buffer.as_entire_binding()
                        } else {
                            data.swap_buffer.as_entire_binding()
                        },
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: if round % 2 == 0 {
                            data.swap_buffer.as_entire_binding()
                        } else {
                            data.pixel_segment_buffer.as_entire_binding()
                        },
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: data.offset_buffer.as_entire_binding(),
                    },
                ],
            });

            {
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&self.merge_blocks_pipeline);
                cpass.set_bind_group(0, &merge_blocks_bind_group, &[]);

                cpass.dispatch_workgroups(config.workgroup_size(), 1, 1);
            }

            if round == max_rounds - 1 {
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

        if rounds % 2 != 0 {
            mem::swap(&mut data.pixel_segment_buffer, &mut data.swap_buffer);
        }
    }
}
