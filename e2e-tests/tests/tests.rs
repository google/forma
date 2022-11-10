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

mod test_env;

macro_rules! from_env {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            ::std::any::type_name::<T>()
        }

        TestEnv::new(
            type_name_of(f)
                .strip_prefix("tests::")
                .unwrap()
                .strip_suffix("::f")
                .unwrap()
                .replacen("::", "__", 100),
        )
    }};
}

#[cfg(test)]
mod tests {
    use super::test_env::{TestEnv, HEIGHT, PADDING, WIDTH};

    use forma::prelude::*;

    fn triangle() -> Path {
        PathBuilder::new()
            .move_to(Point {
                x: PADDING,
                y: PADDING,
            })
            .line_to(Point {
                x: WIDTH - PADDING,
                y: PADDING,
            })
            .line_to(Point {
                x: WIDTH - PADDING,
                y: HEIGHT - PADDING,
            })
            .build()
    }

    fn custom_square(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Path {
        PathBuilder::new()
            .move_to(Point { x: xmin, y: ymin })
            .line_to(Point { x: xmin, y: ymax })
            .line_to(Point { x: xmax, y: ymax })
            .line_to(Point { x: xmax, y: ymin })
            .build()
    }

    fn square() -> Path {
        custom_square(PADDING, PADDING, WIDTH - PADDING, HEIGHT - PADDING)
    }

    fn inner_square() -> Path {
        custom_square(
            PADDING * 2.0,
            PADDING * 2.0,
            WIDTH - PADDING * 2.0,
            HEIGHT - PADDING * 2.0,
        )
    }

    fn custom_circle(x: f32, y: f32, radius: f32) -> Path {
        let weight = 2.0f32.sqrt() / 2.0;
        PathBuilder::new()
            .move_to(Point::new(x + radius, y))
            .rat_quad_to(
                Point::new(x + radius, y - radius),
                Point::new(x, y - radius),
                weight,
            )
            .rat_quad_to(
                Point::new(x - radius, y - radius),
                Point::new(x - radius, y),
                weight,
            )
            .rat_quad_to(
                Point::new(x - radius, y + radius),
                Point::new(x, y + radius),
                weight,
            )
            .rat_quad_to(
                Point::new(x + radius, y + radius),
                Point::new(x + radius, y),
                weight,
            )
            .build()
    }

    fn circle() -> Path {
        custom_circle(WIDTH * 0.5, HEIGHT * 0.5, WIDTH * 0.5 - PADDING)
    }

    fn inner_circle() -> Path {
        custom_circle(WIDTH * 0.5, HEIGHT * 0.5, WIDTH * 0.5 - PADDING * 2.0)
    }

    fn rainbow_colors(gradient_builder: &mut GradientBuilder) {
        gradient_builder
            .color(Color {
                r: 1.00,
                g: 0.00,
                b: 0.00,
                a: 1.0,
            })
            .color(Color {
                r: 1.00,
                g: 0.32,
                b: 0.00,
                a: 1.0,
            })
            .color(Color {
                r: 0.63,
                g: 0.73,
                b: 0.02,
                a: 1.0,
            })
            .color(Color {
                r: 0.08,
                g: 0.72,
                b: 0.07,
                a: 1.0,
            })
            .color(Color {
                r: 0.05,
                g: 0.70,
                b: 0.69,
                a: 1.0,
            })
            .color(Color {
                r: 0.03,
                g: 0.58,
                b: 0.76,
                a: 1.0,
            })
            .color(Color {
                r: 0.01,
                g: 0.21,
                b: 0.85,
                a: 1.0,
            })
            .color(Color {
                r: 0.11,
                g: 0.01,
                b: 0.89,
                a: 1.0,
            })
            .color(Color {
                r: 0.49,
                g: 0.00,
                b: 0.94,
                a: 1.0,
            })
            .color(Color {
                r: 0.96,
                g: 0.00,
                b: 0.69,
                a: 1.0,
            })
            .color(Color {
                r: 1.00,
                g: 0.00,
                b: 0.00,
                a: 1.0,
            });
    }

    fn vertical_rainbow() -> Gradient {
        let mut gradient_builder = GradientBuilder::new(
            Point { x: PADDING, y: 0.0 },
            Point {
                x: WIDTH - PADDING,
                y: 0.0,
            },
        );
        rainbow_colors(&mut gradient_builder);
        gradient_builder.build().unwrap()
    }

    fn horizontal_rainbow() -> Gradient {
        let mut gradient_builder = GradientBuilder::new(
            Point { x: 0.0, y: PADDING },
            Point {
                x: 0.0,
                y: WIDTH - PADDING,
            },
        );
        rainbow_colors(&mut gradient_builder);
        gradient_builder.build().unwrap()
    }

    fn solid_color_props(color: Color) -> Props {
        Props {
            func: Func::Draw(Style {
                fill: Fill::Solid(color),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn linear_gradient() {
        let test_env = from_env!();
        test_env.test_render(|composition| {
            let mut gradient_builder = GradientBuilder::new(
                Point { x: PADDING, y: 0.0 },
                Point {
                    x: WIDTH - PADDING,
                    y: 0.0,
                },
            );
            gradient_builder
                .color(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 1.0,
                    a: 1.0,
                })
                .color(Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                })
                .color(Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                });
            let props = Props {
                func: Func::Draw(Style {
                    fill: Fill::Gradient(gradient_builder.build().unwrap()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            composition
                .get_mut_or_insert_default(Order::new(1).unwrap())
                .insert(&triangle())
                .set_props(props);
        });
    }

    #[test]
    fn radial_gradient() {
        let test_env = from_env!();
        test_env.test_render(|composition| {
            let mut gradient_builder = GradientBuilder::new(
                Point {
                    x: WIDTH * 0.5,
                    y: HEIGHT * 0.5,
                },
                Point {
                    x: WIDTH - PADDING * 2.0,
                    y: HEIGHT * 0.5,
                },
            );
            gradient_builder
                .r#type(GradientType::Radial)
                .color(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 1.0,
                    a: 1.0,
                })
                .color(Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                })
                .color(Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                });
            let props = Props {
                func: Func::Draw(Style {
                    fill: Fill::Gradient(gradient_builder.build().unwrap()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            composition
                .get_mut_or_insert_default(Order::new(1).unwrap())
                .insert(&circle())
                .set_props(props);
        });
    }

    #[test]
    fn solid_color() {
        let test_env = from_env!();
        let colors = vec![
            (
                Color {
                    r: 0.0,
                    g: 0.0,
                    b: 1.0,
                    a: 1.0,
                },
                "blue",
            ),
            (
                Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.5,
                    a: 1.0,
                },
                "dark_blue",
            ),
            (
                Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                "red",
            ),
            (
                Color {
                    r: 0.5,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                "dark_red",
            ),
            (
                Color {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
                "green",
            ),
            (
                Color {
                    r: 0.0,
                    g: 0.5,
                    b: 0.0,
                    a: 1.0,
                },
                "dark_green",
            ),
            (
                Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.5,
                },
                "transparent_black",
            ),
        ];
        for (color, name) in colors {
            test_env.test_render_param(
                |composition| {
                    composition
                        .get_mut_or_insert_default(Order::new(1).unwrap())
                        .insert(&square())
                        .set_props(solid_color_props(color));
                },
                name,
            );
        }
    }

    #[test]
    fn pixel() {
        // This test is useful when the reasterizer is brocken as it emmits 2 pixel segments.
        from_env!().test_render(|composition| {
            composition
                .get_mut_or_insert_default(Order::new(1).unwrap())
                .insert(&custom_square(
                    PADDING,
                    PADDING,
                    PADDING + 1.0,
                    PADDING + 1.0,
                ))
                .set_props(solid_color_props(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }));
        });
    }

    #[test]
    fn covers() {
        // Draws all compination of pixels offseted by 1/32 on both x and y axis.
        from_env!().test_render(|composition| {
            let layer = composition
                .get_mut_or_insert_default(Order::new(0).unwrap())
                .set_props(solid_color_props(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }));
            for xi in 0..32 {
                for yi in 0..32 {
                    let x0 = xi as f32 * (2.0 + 32.0f32.recip());
                    let y0 = yi as f32 * (2.0 + 32.0f32.recip());
                    layer.insert(&custom_square(x0, y0, x0 + 1.0, y0 + 1.0));
                }
            }
        });
    }

    #[test]
    fn texture() {
        // Draws all compination of pixels offseted by 1/32 on both x and y axis.
        from_env!().test_render(|composition| {
            let image = Image::from_srgba(
                &[
                    [0, 0, 0, 255],
                    [255, 0, 0, 255],
                    [0, 255, 0, 255],
                    [255, 255, 0, 255],
                    [0, 0, 255, 255],
                    [255, 0, 255, 255],
                    [0, 255, 255, 255],
                    [255, 255, 255, 255],
                    [0, 0, 0, 255],
                ],
                3,
                3,
            )
            .unwrap();
            let mut order = 0;
            for xi in 0..8 {
                for yi in 0..8 {
                    let x0 = xi as f32 * 8.0;
                    let y0 = yi as f32 * 8.0;
                    let tx = -x0 - 2.0 + xi as f32 * 4.0f32.recip();
                    let ty = -y0 - 2.0 + yi as f32 * 4.0f32.recip();

                    composition
                        .get_mut_or_insert_default(Order::new(order).unwrap())
                        .insert(&custom_square(x0, y0, x0 + 7.0, y0 + 7.0))
                        .set_props(Props {
                            fill_rule: FillRule::EvenOdd,
                            func: Func::Draw(Style {
                                is_clipped: false,
                                fill: Fill::Texture(Texture {
                                    transform: AffineTransform {
                                        ux: 1.0,
                                        uy: 0.0,
                                        vx: 0.0,
                                        vy: 1.0,
                                        tx,
                                        ty,
                                    },
                                    image: image.clone(),
                                }),
                                blend_mode: BlendMode::Over,
                            }),
                        });
                    order += 1;
                }
            }
        });
    }

    #[test]
    fn blend_modes() {
        let test_env = from_env!();
        let blend_modes = [
            BlendMode::Over,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
            BlendMode::ColorDodge,
            BlendMode::ColorBurn,
            BlendMode::HardLight,
            BlendMode::SoftLight,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Hue,
            BlendMode::Saturation,
            BlendMode::Color,
            BlendMode::Luminosity,
        ];
        for blend_mode in blend_modes {
            test_env.test_render_param(
                |composition| {
                    composition
                        .get_mut_or_insert_default(Order::new(0).unwrap())
                        .insert(&square())
                        .set_props(Props {
                            func: Func::Draw(Style {
                                fill: Fill::Gradient(horizontal_rainbow()),
                                ..Default::default()
                            }),
                            ..Default::default()
                        });

                    composition
                        .get_mut_or_insert_default(Order::new(1).unwrap())
                        .insert(&triangle())
                        .set_props(Props {
                            func: Func::Draw(Style {
                                fill: Fill::Gradient(vertical_rainbow()),
                                blend_mode,
                                ..Default::default()
                            }),
                            ..Default::default()
                        });
                },
                blend_mode,
            );
        }
    }

    #[test]
    fn fill_rules() {
        let test_env = from_env!();
        let fill_rules = [FillRule::EvenOdd, FillRule::NonZero];
        for fill_rule in fill_rules {
            test_env.test_render_param(
                |composition| {
                    let path = PathBuilder::new()
                        .move_to(Point {
                            x: PADDING,
                            y: PADDING,
                        })
                        .line_to(Point {
                            x: WIDTH / 2.0 + PADDING,
                            y: HEIGHT / 2.0 + PADDING,
                        })
                        .line_to(Point {
                            x: WIDTH / 2.0 - PADDING,
                            y: HEIGHT / 2.0 + PADDING,
                        })
                        .line_to(Point {
                            x: WIDTH - PADDING,
                            y: PADDING,
                        })
                        .line_to(Point {
                            x: WIDTH - PADDING,
                            y: HEIGHT - PADDING,
                        })
                        .line_to(Point {
                            x: PADDING,
                            y: HEIGHT - PADDING,
                        })
                        .build();
                    composition
                        .get_mut_or_insert_default(Order::new(0).unwrap())
                        .insert(&path)
                        .set_props(Props {
                            fill_rule,
                            func: Func::Draw(Style {
                                fill: Fill::Solid(Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.8,
                                }),
                                ..Default::default()
                            }),
                        });
                },
                fill_rule,
            );
        }
    }

    #[test]
    fn clipping() {
        let test_env = from_env!();
        test_env.test_render(|composition| {
            // First layer is not clipped.
            composition
                .get_mut_or_insert_default(Order::new(0).unwrap())
                .insert(&square())
                .set_props(solid_color_props(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.7,
                }));

            // Triangular clip shape applies to the next 3 layers ids.
            let props = Props {
                func: Func::Clip(4),
                ..Default::default()
            };
            composition
                .get_mut_or_insert_default(Order::new(1).unwrap())
                .insert(&triangle())
                .set_props(props);

            // The blue square is clipped.
            composition
                .get_mut_or_insert_default(Order::new(2).unwrap())
                .insert(&square())
                .set_props(Props {
                    func: Func::Draw(Style {
                        fill: Fill::Solid(Color {
                            r: 0.5,
                            g: 0.5,
                            b: 1.0,
                            a: 0.7,
                        }),
                        is_clipped: true,
                        ..Default::default()
                    }),
                    ..Default::default()
                });

            // Order No. 3 is intentionnaly left empty to test the clip implementation.

            // The pink circle is immune to clip.
            composition
                .get_mut_or_insert_default(Order::new(4).unwrap())
                .insert(&circle())
                .set_props(Props {
                    func: Func::Draw(Style {
                        fill: Fill::Solid(Color {
                            r: 1.0,
                            g: 0.5,
                            b: 0.5,
                            a: 0.7,
                        }),

                        ..Default::default()
                    }),
                    ..Default::default()
                });

            // Inner square is clipped.
            composition
                .get_mut_or_insert_default(Order::new(5).unwrap())
                .insert(&inner_square())
                .set_props(Props {
                    func: Func::Draw(Style {
                        fill: Fill::Solid(Color {
                            r: 0.5,
                            g: 0.5,
                            b: 1.0,
                            a: 0.6,
                        }),
                        is_clipped: true,
                        ..Default::default()
                    }),
                    ..Default::default()
                });

            // This is not drawn given that `is_clipped: true` and no clipping
            // is active at order 6.
            composition
                .get_mut_or_insert_default(Order::new(6).unwrap())
                .insert(&inner_circle())
                .set_props(Props {
                    func: Func::Draw(Style {
                        fill: Fill::Solid(Color {
                            r: 0.5,
                            g: 1.0,
                            b: 0.5,
                            a: 0.6,
                        }),
                        is_clipped: true,
                        ..Default::default()
                    }),
                    ..Default::default()
                });
        });
    }

    #[test]
    fn clipping2() {
        // This test was introduces to verify that the clipping state is reset between tiles.
        let test_env = from_env!();
        test_env.test_render(|composition| {
            // First layer is not clipped.
            composition
                .get_mut_or_insert_default(Order::new(0).unwrap())
                .insert(&square())
                .set_props(solid_color_props(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.7,
                }));

            // Circular clip shape.
            let props = Props {
                func: Func::Clip(1),
                ..Default::default()
            };
            composition
                .get_mut_or_insert_default(Order::new(1).unwrap())
                .insert(&inner_circle())
                .set_props(props);

            // The blue triangle is clipped.
            composition
                .get_mut_or_insert_default(Order::new(2).unwrap())
                .insert(&triangle())
                .set_props(Props {
                    func: Func::Draw(Style {
                        fill: Fill::Solid(Color {
                            r: 0.5,
                            g: 0.5,
                            b: 1.0,
                            a: 0.7,
                        }),
                        is_clipped: true,
                        ..Default::default()
                    }),
                    ..Default::default()
                });
        });
    }
}
