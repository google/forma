![forma logo](assets/images/logo.png?raw=true)

[![crates.io badge](https://img.shields.io/crates/v/forma-render?style=for-the-badge)](https://crates.io/crates/forma-render) [![](https://dcbadge.vercel.app/api/server/CYtcmqgh)](https://discord.gg/CYtcmqgh)

A (thouroughly) parallelized **experimental** Rust vector-graphics renderer with both a software (CPU) and hardware (GPU)
back-end having the following goals, in this order:

  1. **Portability**; supporting Fuchsia, Linux, macOS, Windows, Android & iOS.
  2. **Performance**; making use of compute-focused pipeline that is highly parallelized both at the instruction-level and the thread-level.
  3. **Simplicity**; implementing an easy-to-understand 4-stage pipeline.
  4. **Size**; minimizing the number of dependencies and focusing on vector-graphics only.

It relies on Rust's SIMD auto-vectorization/intrinsics and [Rayon] to have good performance on the CPU, while using [WebGPU] ([wgpu]) to take advantage of the GPU.

[Rayon]: https://github.com/rayon-rs/rayon
[WebGPU]: https://github.com/gpuweb/gpuweb
[wgpu]: https://wgpu.rs/

## Getting started

Add the following to your `Cargo.toml` dependencies:

```toml
forma = { version = "0.1.0", package = "forma-render" }
```

## 4-stage Pipeline

| 1. Curve flattening | 2. Line segment rasterization |      3. Sorting       |           4. Painting            |
|:-------------------:|:-----------------------------:|:---------------------:|:--------------------------------:|
|    Bézier curves    |         line segments         |     pixel segments    | sorted pixel segments, old tiles |
|        ⬇️⬇️⬇️       |             ⬇️⬇️⬇️            |         ⬇️⬇️⬇️        |              ⬇️⬇️⬇️              |
|    line segments    |        pixel segments         | sorted pixel segments |      freshly painted tiles       |

## Implementation Highlights ✨

Here are a few implementation highlights that make forma stand out from commonly used vector renderers.

<details>
<summary>Curvature-aware flattening</summary>

All higher cubic Béziers are approximated by quadratic ones, then, in parallel, flattened to line segments according to their curvature. This [technique] was developed by Raph Levien.

[technique]: https://raphlinus.github.io/graphics/curves/2019/12/23/flatten-quadbez.html

</details>

<details>
<summary>Cheap translations and rotations</summary>

Translations and rotations can be rendered without having to re-flatten the curves, all the while maintaining full quality.

</details>

<details>
<summary>Parallel pixel grid intersection</summary>

Line segments are transformed into pixel segments by intersecting them with the pixel grid. We developed a simple method that performs this computation in *O(1)* and which is run in parallel.

</details>

<details>
<summary>Efficient sorting</summary>

We ported [crumsort] to Rust and parallelized it with Rayon, delivering improved performance over its pdqsort implementation for 64-bit random data. Scattering pixel segments with a sort was inspired from Allan MacKinnon's work on [Spinel].

[crumsort]: https://github.com/google/crumsort-rs
[Spinel]: https://cs.opensource.google/fuchsia/fuchsia/+/main:src/graphics/lib/compute/spinel/

</details>

<details>
<summary>Update only the tiles that change (currently CPU-only)</summary>

We implemented a fail-fast per-tile optimizer that tries to skip the painting step entirely. A similar approach could also be tested on the GPU.

</details>

| Animation as it appears on the screen |                             Updated tiles only                             |
|:-------------------------------------:|:--------------------------------------------------------------------------:|
| ![](assets/images/juice.png?raw=true) | ![juice animation updated tiles](assets/images/juice-updated.png?raw=true) |

## Similar Projects

forma draws heavy inspiration from the following projects:

* [Spinel], with a Vulkan 1.2 back-end
* [vello], with a wgpu back-end

[vello]: https://github.com/linebender/vello

## Example

You can use the included `demo` example to render a few examples, one of which is a non-compliant & incomplete SVG renderer:

```sh
cargo run --release -p demo -- svg assets/svgs/paris-30k.svg
```

It renders enormous SVGs at interactive framerates, even on CPU: ([compare to your web browser])

[compare to your web browser]: assets/svgs/paris-30k.svg?raw=true

![window rendering map of Germany](assets/images/paris-30k-rendered.png?raw=true)

## (Currently) Missing Pieces 🧩

Since this project is work-in-progress, breakage in the API, while not drastic, is expected. The performance on the GPU back-end is also expected to improve especially on mobile where performance is known to be poor and where the CPU back-end is currently advised instead.

Other than that:

* Automated layer ordering
* Strokes
* More color spaces for blends & gradients
* Faster GPU sorter
* Use of `f16` for great mobile GPU performance

## Note

This is not an officially supported Google product.
