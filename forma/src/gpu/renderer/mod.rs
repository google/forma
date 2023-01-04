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

use std::{borrow::Cow, mem, num::NonZeroU32, slice, time::Duration};

use anyhow::Error;
use wgpu::util::DeviceExt;

use crate::{segment::SegmentBufferView, styling::Color, utils::DivCeil, Composition};

use super::{conveyor_sort, painter, rasterizer, GpuContext, StyleMap, TimeStamp};

#[derive(Debug)]
pub struct Timings {
    pub rasterize: Duration,
    pub sort: Duration,
    pub paint: Duration,
    pub render: Duration,
}

impl Timings {
    pub(crate) const fn size() -> usize {
        mem::size_of::<Timings>() / mem::size_of::<u64>()
    }
}

#[derive(Debug)]
struct RenderContext {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

#[derive(Debug, Default)]
struct Resources {
    atlas: Option<wgpu::Texture>,
    texture: Option<(wgpu::Texture, u32, u32)>,
}

#[derive(Debug)]
pub struct Renderer {
    rasterizer: rasterizer::RasterizerContext,
    sort: conveyor_sort::SortContext,
    paint: painter::PaintContext,
    render: RenderContext,
    common: Resources,
    styling_map: StyleMap,
    has_timestamp_query: bool,
}

impl Renderer {
    pub fn minimum_device(adapter: &wgpu::Adapter) -> (wgpu::DeviceDescriptor, bool) {
        let adapter_features = adapter.features();
        let has_timestamp_query = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let desc = wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::TIMESTAMP_QUERY & adapter_features,
            limits: wgpu::Limits {
                max_texture_dimension_2d: 4096,
                max_storage_buffer_binding_size: 1 << 30,
                ..wgpu::Limits::downlevel_defaults()
            },
        };

        (desc, has_timestamp_query)
    }

    pub fn new(
        device: &wgpu::Device,
        swap_chain_format: wgpu::TextureFormat,
        has_timestamp_query: bool,
    ) -> Self {
        let rasterizer = rasterizer::RasterizerContext::init(device);
        let sort = conveyor_sort::SortContext::init(device);
        let paint = painter::PaintContext::init(device);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw_texture.wgsl"))),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(swap_chain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let render = RenderContext {
            pipeline,
            bind_group_layout,
            sampler,
        };

        Self {
            rasterizer,
            sort,
            paint,
            render,
            common: Resources::default(),
            styling_map: StyleMap::new(),
            has_timestamp_query,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_inner(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        segment_buffer_view: &SegmentBufferView,
        background_color: painter::Color,
    ) -> Result<Option<Timings>, Error> {
        let timestamp_context = self.has_timestamp_query.then(|| {
            let timestamp = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                count: Timings::size() as u32,
                ty: wgpu::QueryType::Timestamp,
            });

            let timestamp_period = queue.get_timestamp_period();

            let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: mem::size_of::<Timings>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            (timestamp, timestamp_period, data_buffer)
        });

        if let Some((texture, current_width, current_height)) = self.common.texture.as_ref() {
            if *current_width != width || *current_height != height {
                texture.destroy();

                self.common.texture = None;
            }
        }

        let output_texture = self
            .common
            .texture
            .get_or_insert_with(|| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                });

                (texture, width, height)
            })
            .0
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.render.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.render.sampler),
                },
            ],
            label: None,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let atlas_texture = self.common.atlas.get_or_insert_with(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("atlas"),
                size: wgpu::Extent3d {
                    width: 4096,
                    height: 4096,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            })
        });

        for (image, [xmin, ymin, _, _]) in self.styling_map.new_allocs() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: atlas_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: *xmin,
                        y: *ymin,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                unsafe {
                    slice::from_raw_parts(
                        (image.data() as *const _) as *const u8,
                        mem::size_of_val(image.data()),
                    )
                },
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(4 * 2 * image.width()).unwrap()),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: image.width(),
                    height: image.height(),
                    depth_or_array_layers: 1,
                },
            );
        }

        // Number of segments to be generated, out of the prefix sum.
        let segments_len = segment_buffer_view.len();
        if segments_len > 0 && width != 0 && height != 0 {
            let segments_blocks =
                segments_len.div_ceil_(conveyor_sort::BLOCK_SIZE.block_len as usize);

            let pixel_segment_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (segments_len * std::mem::size_of::<u64>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let mut data = rasterizer::Data {
                segment_buffer_view,
                pixel_segment_buffer,
            };
            self.rasterizer.encode(
                device,
                &mut encoder,
                timestamp_context
                    .as_ref()
                    .map(|(query_set, _, _)| TimeStamp {
                        query_set,
                        start_index: 0,
                        end_index: 1,
                    }),
                &mut data,
            );

            let slice_size = segments_len * std::mem::size_of::<u64>();
            let size = slice_size as wgpu::BufferAddress;

            let swap_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let offset_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: ((segments_blocks + 1) * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let pixel_segment_buffer = data.pixel_segment_buffer;
            let mut data = conveyor_sort::Data {
                segment_buffer_view,
                pixel_segment_buffer,
                swap_buffer,
                offset_buffer,
            };
            self.sort.encode(
                device,
                &mut encoder,
                timestamp_context
                    .as_ref()
                    .map(|(query_set, _, _)| TimeStamp {
                        query_set,
                        start_index: 2,
                        end_index: 3,
                    }),
                &mut data,
            );

            let style_offset_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(self.styling_map.style_offsets()),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let style_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(self.styling_map.styles()),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let pixel_segment_buffer = data.pixel_segment_buffer;
            let mut data = painter::Data {
                segment_buffer_view,
                pixel_segment_buffer,
                style_buffer,
                style_offset_buffer,
                atlas_texture: atlas_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                output_texture,
                width,
                height,
                clear_color: background_color,
            };
            self.paint.encode(
                device,
                &mut encoder,
                timestamp_context
                    .as_ref()
                    .map(|(query_set, _, _)| TimeStamp {
                        query_set,
                        start_index: 4,
                        end_index: 5,
                    }),
                &mut data,
            );
        }

        if let Some((timestamp, _, _)) = timestamp_context.as_ref() {
            encoder.write_timestamp(timestamp, 6);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.render.pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        if let Some((timestamp, _, _)) = timestamp_context.as_ref() {
            encoder.write_timestamp(timestamp, 7);
        }

        if let Some((timestamp, _, data_buffer)) = &timestamp_context {
            encoder.resolve_query_set(timestamp, 0..Timings::size() as u32, data_buffer, 0);
        }

        queue.submit(Some(encoder.finish()));

        let timings = timestamp_context
            .as_ref()
            .map(|(_, timestamp_period, data_buffer)| {
                use bytemuck::{Pod, Zeroable};

                #[repr(C)]
                #[derive(Clone, Copy, Debug, Pod, Zeroable)]
                struct TimestampData {
                    start: u64,
                    end: u64,
                }

                data_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());

                device.poll(wgpu::Maintain::Wait);

                let view = data_buffer.slice(..).get_mapped_range();
                let timestamps: [TimestampData; Timings::size() / 2] = *bytemuck::from_bytes(&view);
                let durations = timestamps.map(|timestamp| {
                    let nanos = (timestamp.end - timestamp.start) as f32 * timestamp_period;
                    Duration::from_nanos(nanos as u64)
                });

                Timings {
                    rasterize: durations[0],
                    sort: durations[1],
                    paint: durations[2],
                    render: durations[3],
                }
            });

        Ok(timings)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        composition: &mut Composition,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &wgpu::Surface,
        width: u32,
        height: u32,
        clear_color: Color,
    ) -> Option<Timings> {
        let frame = surface.get_current_texture().unwrap();
        let timings = self.render_to_texture(
            composition,
            device,
            queue,
            &frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            width,
            height,
            clear_color,
        );

        frame.present();

        timings
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_to_texture(
        &mut self,
        composition: &mut Composition,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        clear_color: Color,
    ) -> Option<Timings> {
        composition.compact_geom();
        composition
            .shared_state
            .borrow_mut()
            .props_interner
            .compact();

        let layers = &composition.layers;
        let shared_state = &mut *composition.shared_state.borrow_mut();
        let segment_buffer = &mut shared_state.segment_buffer;
        let geom_id_to_order = &shared_state.geom_id_to_order;
        let segment_buffer_view = segment_buffer
            .take()
            .expect("segment_buffer should not be None")
            .fill_gpu_view(width as usize, height as usize, layers, geom_id_to_order);

        self.styling_map.populate(layers);

        let timings = self
            .render_inner(
                device,
                queue,
                view,
                width,
                height,
                &segment_buffer_view,
                painter::Color {
                    r: clear_color.r,
                    g: clear_color.g,
                    b: clear_color.b,
                    a: clear_color.a,
                },
            )
            .unwrap();

        *segment_buffer = Some(segment_buffer_view.recycle());

        timings
    }
}
