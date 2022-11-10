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

//! End to end rendering test for forma GPU and CPU.
//! A report is written to ${TARGET}/tmp/tests/report.html

use std::{
    cell::RefCell,
    env,
    fmt::Debug,
    fs,
    fs::{create_dir_all, remove_dir_all},
    num::NonZeroU32,
    path,
    path::PathBuf,
    sync::Mutex,
};

use anyhow::{anyhow, Context};
use forma::{cpu, gpu, prelude::*};
use image::RgbaImage;
use once_cell::sync::OnceCell;
use serde::Serialize;

pub const WIDTH: f32 = 64.0;
pub const HEIGHT: f32 = 64.0;
pub const PADDING: f32 = 8.0;

fn cpu_render(composition: &mut Composition, width: usize, height: usize) -> RgbaImage {
    let mut data = vec![0; width * 4 * height];
    let mut layout = LinearLayout::new(width, width * 4, height);
    let mut buffer = BufferBuilder::new(&mut data, &mut layout).build();
    let mut renderer = cpu::Renderer::new();
    renderer.render(
        composition,
        &mut buffer,
        RGBA,
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 0.0,
        },
        None,
    );

    RgbaImage::from_raw(width as u32, height as u32, data).unwrap()
}

fn gpu_render(composition: &mut Composition, width: usize, height: usize) -> RgbaImage {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("failed to find an appropriate adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features: Default::default(),
            limits: wgpu::Limits {
                max_texture_dimension_2d: 8192,
                max_storage_buffer_binding_size: 1 << 30,
                ..wgpu::Limits::downlevel_defaults()
            },
        },
        None,
    ))
    .expect("failed to get device");

    let texture_desc = wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: width as u32,
            height: width as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
    };
    let texture = device.create_texture(&texture_desc);

    let mut renderer = gpu::Renderer::new(&device, texture_desc.format, false);
    renderer.render_to_texture(
        composition,
        &device,
        &queue,
        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
        width as u32,
        height as u32,
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 0.0,
        },
    );

    let output_buffer_size = (4 * width * height) as wgpu::BufferAddress;
    let output_buffer_desc = wgpu::BufferDescriptor {
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    };
    let output_buffer = device.create_buffer(&output_buffer_desc);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * width as u32),
                rows_per_image: NonZeroU32::new(width as u32),
            },
        },
        texture_desc.size,
    );
    queue.submit(Some(encoder.finish()));
    // We need to scope the mapping variables so that we can
    // unmap the buffer
    let image = {
        let buffer_slice = output_buffer.slice(..);

        buffer_slice.map_async(wgpu::MapMode::Read, |_| ());

        device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        RgbaImage::from_raw(width as u32, height as u32, data.to_vec()).unwrap()
    };
    output_buffer.unmap();
    image
}

fn compare_images(expected: &RgbaImage, actual: &RgbaImage, tolerance: u8) -> anyhow::Result<()> {
    let offenders = expected
        .pixels()
        .zip(actual.pixels())
        .filter(|(e, a)| {
            e.0.iter()
                .zip(a.0.iter())
                .any(|(e, a)| e.abs_diff(*a) > tolerance)
        })
        .count();
    if offenders > 0 {
        Err(anyhow!(
            "{} pixels differs have a channel that differ by more than {} units",
            offenders,
            tolerance
        ))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
enum RendererType {
    Cpu,
    Gpu,
}

#[derive(Serialize)]
struct TestReportEntry {
    // Unique name for this test.
    name: String,
    // Actual and expected strings contains the image encoded as
    // valid `src` attribute for an HTML `img` element.
    cpu_actual: String,
    // `None` when no reference image file exist.
    // CPU reference is required for test success.
    // It is typically missing when a new test is created.
    cpu_expected: Option<String>,
    gpu_actual: String,
    // `None` when no reference exist.
    gpu_expected: Option<String>,
    // "OK" or "KO".
    status: String,
    // Test status message. Empty for successful tests.
    message: String,
}

// Report collecting all information about all tests that is used for
// HTML report generation.
struct TestReport {
    output_dir: PathBuf,
    entries: Mutex<Vec<TestReportEntry>>,
}

// Object implementing soft failure so that we can collect
pub struct TestEnv {
    test_name: String,
    failures: RefCell<Vec<String>>,
}

impl TestEnv {
    pub fn new<T: Into<String>>(test_name: T) -> TestEnv {
        TestEnv {
            test_name: test_name.into(),
            failures: RefCell::new(vec![]),
        }
    }
    pub fn test_render<F>(self, compose: F)
    where
        F: Fn(&mut Composition),
    {
        if let Err(err) = test_render(compose, &self.test_name) {
            self.failures.borrow_mut().push(format!("{:?}", err));
        }
    }

    pub fn test_render_param<F, T: Debug>(&self, compose: F, t: T)
    where
        F: Fn(&mut Composition),
    {
        let t = format!("{:?}", t).replace('\"', "");
        if let Err(err) = test_render(compose, &format!("{}__{}", self.test_name, t)) {
            self.failures.borrow_mut().push(format!("{:?}", err));
        }
    }
}

// Implements soft failure so that all test can run.
impl Drop for TestEnv {
    fn drop(&mut self) {
        if self.failures.borrow().is_empty() {
            return;
        }

        println!("Test {}:", self.test_name);
        self.failures
            .borrow()
            .iter()
            .enumerate()
            .for_each(|(i, e)| println!(" - Error [{}]: {}", i, e));
        println!("Report generated at target/tmp/tests/report.html");
        panic!();
    }
}

fn test_render<F>(compose: F, test_name: &str) -> anyhow::Result<()>
where
    F: Fn(&mut Composition),
{
    let cpu_actual = {
        let mut composition = Composition::new();
        compose(&mut composition);
        cpu_render(&mut composition, WIDTH as usize, HEIGHT as usize)
    };

    let gpu_actual = {
        let mut composition = Composition::new();
        compose(&mut composition);
        gpu_render(&mut composition, WIDTH as usize, HEIGHT as usize)
    };

    let tolerance = 8;
    let result = (|| {
        let expected_cpu = expected_image(test_name, RendererType::Cpu)
            .context("CPU reference image is missing.")?;
        compare_images(&expected_cpu, &cpu_actual, tolerance)
            .context("CPU result differs from the reference image.")?;
        let expected_gpu = expected_image(test_name, RendererType::Gpu).unwrap_or(expected_cpu);
        compare_images(&expected_gpu, &gpu_actual, tolerance)
            .context("GPU result differs from the reference image.")
    })();
    add_result_to_report(test_name, &cpu_actual, &gpu_actual, tolerance, &result);
    result
}

/// Path to the reference image to which the rendering is compared to.
fn expected_image_path(test_name: &str, renderer: RendererType) -> PathBuf {
    let renderer = format!("{:?}", renderer).to_lowercase();
    env::current_dir()
        .unwrap()
        .join("expected")
        .join(format!("{}__{}.png", test_name, renderer))
}

/// Returns the reference image buffer if the such image exist.
fn expected_image(test_name: &str, renderer: RendererType) -> anyhow::Result<RgbaImage> {
    let path = expected_image_path(test_name, renderer);
    if !path.exists() {
        return Err(anyhow!("Unable to open file {:?}", path));
    }
    // Panic if the file exists but is not valid.
    Ok(image::io::Reader::open(path)
        .context("Unable to open file")?
        .decode()
        .context("Unable to open file")?
        .into_rgba8())
}

fn add_result_to_report(
    test_name: &str,
    cpu_actual: &RgbaImage,
    gpu_actual: &RgbaImage,
    tolerance: u8,
    status: &anyhow::Result<()>,
) {
    static REPORT: OnceCell<Mutex<TestReport>> = OnceCell::new();
    let lock = REPORT
        .get_or_init(|| {
            let output_dir =
                path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(env!("CARGO_CRATE_NAME"));
            let _ = remove_dir_all(&output_dir);
            create_dir_all(&output_dir).unwrap();
            Mutex::new(TestReport::new(output_dir))
        })
        .lock();
    let report = lock.unwrap();

    let cpu_expected = expected_image_path(test_name, RendererType::Cpu);
    let gpu_expected = expected_image_path(test_name, RendererType::Gpu);
    report.add_result(
        test_name,
        cpu_expected.exists().then_some(cpu_expected),
        cpu_actual,
        gpu_expected.exists().then_some(gpu_expected),
        gpu_actual,
        tolerance,
        status,
    );
}

impl TestReport {
    fn new(output_dir: path::PathBuf) -> TestReport {
        TestReport {
            output_dir,
            entries: Mutex::new(vec![]),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_result(
        &self,
        test_name: &str,
        cpu_expected: Option<PathBuf>,
        cpu_actual: &RgbaImage,
        gpu_expected: Option<PathBuf>,
        gpu_actual: &RgbaImage,
        tolerance: u8,
        status: &anyhow::Result<()>,
    ) {
        let path = |dir, suffix| {
            self.output_dir
                .join(dir)
                .join(format!("{}__{}.png", test_name, suffix))
        };
        let to_base64_img_src = |path: &path::PathBuf| {
            format!(
                "data:image/png;base64, {}",
                base64::encode(fs::read(path).unwrap())
            )
        };
        let save_actual = |image: &RgbaImage, suffix| {
            let path = path("actual", suffix);
            fs::create_dir_all(&path.parent().unwrap()).unwrap();
            image.save(&path).unwrap();
            to_base64_img_src(&path)
        };
        let cpu_actual_b64 = save_actual(cpu_actual, "cpu");
        let gpu_actual_b64 = if compare_images(cpu_actual, gpu_actual, tolerance).is_ok() {
            // No need for a GPU reference to be written. This will ease the maintenance of
            // expected results directory by copying files from the `/actual` directory with:
            // rsync --delete target/tmp/tests/actual/ e2e_tests/expected/
            cpu_actual_b64.clone()
        } else {
            save_actual(gpu_actual, "gpu")
        };
        let entry = TestReportEntry {
            name: test_name.to_string(),
            cpu_actual: cpu_actual_b64,
            cpu_expected: cpu_expected.as_ref().map(to_base64_img_src),
            gpu_actual: gpu_actual_b64,
            gpu_expected: gpu_expected
                .or(cpu_expected)
                .as_ref()
                .map(to_base64_img_src),
            status: match status {
                Ok(_) => "OK".to_owned(),
                Err(_) => "KO".to_owned(),
            },
            message: match status {
                Ok(_) => "".to_owned(),
                Err(e) => format!("{:?}", e),
            },
        };
        let mut entries = self.entries.lock().unwrap();
        entries.push(entry);

        // Update the report every time.
        // The is no hook in the test framework to generate this report at the end.
        let report = include_str!("report.html").replace(
            "[/* generated */]",
            &serde_json::to_string_pretty(entries.as_slice()).unwrap(),
        );
        fs::write(&self.output_dir.join("report.html"), report).unwrap();
    }
}
