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

use std::{collections::HashMap, path::Path, time::Duration};

use forma::{prelude::*, styling};
use svg::{
    node::{element::tag, Value},
    parser::Event,
};
use svgtypes::{Color, PathParser, PathSegment, StyleParser, Transform};
use winit::event::VirtualKeyCode;

use crate::{App, Keyboard};

fn reflect(point: Point, against: Point) -> Point {
    Point::new(against.x * 2.0 - point.x, against.y * 2.0 - point.y)
}

struct EllipticalArc {
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    x_axis_rotation: f32,
    angle: f32,
    angle_delta: f32,
}

#[allow(clippy::too_many_arguments)]
fn convert_to_center(
    rx: f32,
    ry: f32,
    x_axis_rotation: f32,
    large_arc: bool,
    sweep: bool,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> Option<EllipticalArc> {
    if (x0 - x1).abs() < f32::EPSILON && (y0 - y1).abs() < f32::EPSILON {
        return None;
    }

    let rx = rx.abs();
    let ry = ry.abs();

    if rx == 0.0 || ry == 0.0 {
        return None;
    }

    let cos_phi = x_axis_rotation.cos();
    let sin_phi = x_axis_rotation.sin();

    let x0 = (x0 * cos_phi + y0 * sin_phi) / rx;
    let y0 = (-x0 * sin_phi + y0 * cos_phi) / ry;

    let x1 = (x1 * cos_phi + y1 * sin_phi) / rx;
    let y1 = (-x1 * sin_phi + y1 * cos_phi) / ry;

    let lx = (x0 - x1) * 0.5;
    let ly = (y0 - y1) * 0.5;

    let mut cx = (x0 + x1) * 0.5;
    let mut cy = (y0 + y1) * 0.5;

    let len_squared = lx * lx + ly * ly;
    if len_squared < 1.0 {
        let mut radicand = ((1.0 - len_squared) / len_squared).sqrt();
        if large_arc != sweep {
            radicand = -radicand;
        }

        cx += -ly * radicand;
        cy += lx * radicand;
    }

    let theta = (y0 - cy).atan2(x0 - cx);
    let mut delta_theta = (y1 - cy).atan2(x1 - cx) - theta;

    let cxs = cx * rx;
    let cys = cy * ry;

    cx = cxs * cos_phi - cys * sin_phi;
    cy = cxs * sin_phi + cys * cos_phi;

    if sweep {
        if delta_theta < 0.0 {
            delta_theta += std::f32::consts::PI * 2.0;
        }
    } else if delta_theta > 0.0 {
        delta_theta -= std::f32::consts::PI * 2.0;
    }

    Some(EllipticalArc {
        cx,
        cy,
        rx,
        ry,
        x_axis_rotation,
        angle: theta,
        angle_delta: delta_theta,
    })
}

fn parse_fill_rule(attrs: &HashMap<String, Value>) -> FillRule {
    if let Some(fill_rule) = attrs.get("fill-rule") {
        if &fill_rule.to_string() == "evenodd" {
            return FillRule::EvenOdd;
        }
    }

    FillRule::NonZero
}

fn parse_color(attrs: &HashMap<String, Value>) -> Option<Color> {
    attrs
        .get("fill")
        .or_else(|| attrs.get("stop-color"))
        .and_then(|fill| fill.parse().ok())
}

fn parse_opacity(attrs: &HashMap<String, Value>) -> Option<f32> {
    attrs
        .get("opacity")
        .or_else(|| attrs.get("stop-opacity"))
        .and_then(|opacity| opacity.parse().ok())
        .or_else(|| {
            attrs
                .get("fill-opacity")
                .and_then(|opacity| opacity.parse().ok())
        })
}

fn parse_blend_mode(attrs: &HashMap<String, Value>) -> Option<BlendMode> {
    attrs.get("style").and_then(|style| {
        StyleParser::from(style.as_ref()).find_map(|pair| {
            pair.ok().and_then(|pair| {
                (pair.0 == "mix-blend-mode").then_some(match pair.1 {
                    "normal" => BlendMode::Over,
                    "multiply" => BlendMode::Multiply,
                    "screen" => BlendMode::Screen,
                    "overlay" => BlendMode::Overlay,
                    "darken" => BlendMode::Darken,
                    "lighten" => BlendMode::Lighten,
                    "color-dodge" => BlendMode::ColorDodge,
                    "color-burn" => BlendMode::ColorBurn,
                    "hard-light" => BlendMode::HardLight,
                    "soft-light" => BlendMode::SoftLight,
                    "difference" => BlendMode::Difference,
                    "exclusion" => BlendMode::Exclusion,
                    "hue" => BlendMode::Hue,
                    "saturation" => BlendMode::Saturation,
                    "color" => BlendMode::Color,
                    "luminosity" => BlendMode::Luminosity,
                    _ => BlendMode::Over,
                })
            })
        })
    })
}

#[derive(Debug, Default)]
struct Group {
    transform: Option<Transform>,
    fill: Option<Color>,
    opacity: Option<f32>,
}

#[derive(Debug)]
pub struct Svg {
    groups: Vec<Group>,
    paths: Vec<(forma::Path, FillRule, Fill, BlendMode)>,
    gradient_builder: Option<(String, GradientBuilder)>,
    gradients: HashMap<String, Gradient>,
    needs_composition: bool,
    x: f32,
    y: f32,
}

impl Svg {
    pub fn new<P: AsRef<Path>>(path: P, scale: f32) -> Self {
        let mut result = Self {
            groups: vec![],
            paths: vec![],
            gradient_builder: None,
            gradients: HashMap::new(),
            needs_composition: true,
            x: 0.0,
            y: 0.0,
        };

        result.open(path);

        let transform = &[scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0];
        for (path, ..) in &mut result.paths {
            *path = path.transform(transform);
        }

        result
    }

    fn group_transform(&self) -> Option<&Transform> {
        self.groups
            .iter()
            .rev()
            .find_map(|group| group.transform.as_ref())
    }

    fn group_fill(&self) -> Option<Color> {
        self.groups.iter().rev().find_map(|group| group.fill)
    }

    fn groups_opacity(&self) -> f32 {
        self.groups
            .iter()
            .filter_map(|group| group.opacity)
            .product()
    }

    fn t(&self, point: Point) -> Point {
        match self.group_transform() {
            None => point,
            Some(t) => {
                let mut x = f64::from(point.x);
                let mut y = f64::from(point.y);
                t.apply_to(&mut x, &mut y);
                Point::new(x as f32, y as f32)
            }
        }
    }

    fn parse_fill(&self, attrs: &HashMap<String, Value>) -> Fill {
        if let Some(gradient) = attrs.get("fill").and_then(|fill| {
            let fill = fill.to_string();

            fill.strip_prefix("url(#")
                .and_then(|fill| fill.strip_suffix(')'))
                .and_then(|id| self.gradients.get(id))
        }) {
            return Fill::Gradient(gradient.clone());
        }

        let color: Option<Color> = parse_color(attrs).or_else(|| self.group_fill());

        let opacity: f32 = parse_opacity(attrs).unwrap_or_else(|| self.groups_opacity());

        let color = color.map_or(
            styling::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            |color| styling::Color {
                a: opacity,
                ..crate::to_linear([color.red, color.green, color.blue])
            },
        );

        Fill::Solid(color)
    }

    fn push_rationals_from_arc(
        &self,
        builder: &mut PathBuilder,
        arc: &EllipticalArc,
        mut end_point: Point,
    ) -> Point {
        let mut angle = arc.angle;
        let mut angle_delta = arc.angle_delta;

        let cos_phi = arc.x_axis_rotation.cos();
        let sin_phi = arc.x_axis_rotation.sin();

        let angle_sweep = std::f32::consts::PI / 2.0;
        let angle_incr = if angle_delta > 0.0 {
            angle_sweep
        } else {
            -angle_sweep
        };

        while angle_delta != 0.0 {
            let theta = angle;
            let sweep = if angle_delta.abs() <= angle_sweep {
                angle_delta
            } else {
                angle_incr
            };

            angle += sweep;
            angle_delta -= sweep;

            let half_sweep = sweep * 0.5;
            let w = half_sweep.cos();

            let mut p1 = Point::new(
                (theta + half_sweep).cos() / w,
                (theta + half_sweep).sin() / w,
            );
            let mut p2 = Point::new((theta + sweep).cos(), (theta + sweep).sin());

            p1.x *= arc.rx;
            p1.y *= arc.ry;
            p2.x *= arc.rx;
            p2.y *= arc.ry;

            let p1 = Point::new(
                (arc.cx + p1.x * cos_phi - p1.y * sin_phi) as f32,
                (arc.cy + p1.x * sin_phi + p1.y * cos_phi) as f32,
            );
            let p2 = Point::new(
                (arc.cx + p2.x * cos_phi - p2.y * sin_phi) as f32,
                (arc.cy + p2.x * sin_phi + p2.y * cos_phi) as f32,
            );

            builder.rat_quad_to(self.t(p1), self.t(p2), w as f32);

            end_point = p2;
        }

        end_point
    }

    pub fn open(&mut self, path: impl AsRef<Path>) {
        for event in svg::open(path).unwrap() {
            match event {
                Event::Tag(tag::Group, tag::Type::Start, attrs) => {
                    self.groups.push(Group {
                        transform: attrs
                            .get("transform")
                            .and_then(|transform| transform.parse().ok()),
                        fill: parse_color(&attrs),
                        opacity: parse_opacity(&attrs),
                    });
                }
                Event::Tag(tag::Group, tag::Type::End, _) => {
                    self.groups.pop();
                }
                Event::Tag(tag::Path, _, attrs) => {
                    if attrs
                        .get("stroke")
                        .filter(|val| val.to_string() != "none")
                        .is_some()
                    {
                        continue;
                    }

                    let mut builder = forma::PathBuilder::new();

                    let data = match attrs.get("d") {
                        Some(data) => data,
                        None => continue,
                    };

                    let mut start_point = None;
                    let mut end_point = Point::new(0.0, 0.0);
                    let mut quad_control_point = None;
                    let mut cubic_control_point = None;

                    let add_diff = |end_point: Point, x, y| {
                        Point::new(end_point.x + x as f32, end_point.y + y as f32)
                    };

                    for segment in PathParser::from(data.to_string().as_str()) {
                        match segment.unwrap() {
                            PathSegment::MoveTo { abs: true, x, y } => {
                                let p = Point::new(x as f32, y as f32);

                                builder.move_to(self.t(p));

                                start_point.take();
                                end_point = p;
                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::MoveTo { abs: false, x, y } => {
                                let p = add_diff(end_point, x, y);

                                builder.move_to(self.t(p));

                                start_point.take();
                                end_point = p;
                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::LineTo { abs: true, x, y } => {
                                let p0 = Point::new(x as f32, y as f32);

                                builder.line_to(self.t(p0));

                                start_point.get_or_insert(end_point);
                                end_point = p0;
                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::LineTo { abs: false, x, y } => {
                                let p0 = add_diff(end_point, x, y);

                                builder.line_to(self.t(p0));

                                start_point.get_or_insert(end_point);
                                end_point = p0;
                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::HorizontalLineTo { abs: true, x } => {
                                let p0 = Point::new(x as f32, end_point.y);

                                builder.line_to(self.t(p0));

                                start_point.get_or_insert(end_point);
                                end_point = p0;
                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::HorizontalLineTo { abs: false, x } => {
                                let p0 = add_diff(end_point, x, 0.0);

                                builder.line_to(self.t(p0));

                                start_point.get_or_insert(end_point);
                                end_point = p0;
                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::VerticalLineTo { abs: true, y } => {
                                let p0 = Point::new(end_point.x, y as f32);

                                builder.line_to(self.t(p0));

                                start_point.get_or_insert(end_point);
                                end_point = p0;
                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::VerticalLineTo { abs: false, y } => {
                                let p0 = add_diff(end_point, 0.0, y);

                                builder.line_to(self.t(p0));

                                start_point.get_or_insert(end_point);
                                end_point = p0;
                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::Quadratic {
                                abs: true,
                                x1,
                                y1,
                                x,
                                y,
                            } => {
                                let p0 = Point::new(x1 as f32, y1 as f32);
                                let p1 = Point::new(x as f32, y as f32);
                                let control_point = p0;

                                builder.quad_to(self.t(control_point), self.t(p1));

                                start_point.get_or_insert(end_point);
                                end_point = p1;
                                quad_control_point = Some(control_point);
                                cubic_control_point = None;
                            }
                            PathSegment::Quadratic {
                                abs: false,
                                x1,
                                y1,
                                x,
                                y,
                            } => {
                                let p0 = add_diff(end_point, x1, y1);
                                let p1 = add_diff(end_point, x, y);
                                let control_point = p0;

                                builder.quad_to(self.t(control_point), self.t(p1));

                                start_point.get_or_insert(end_point);
                                end_point = p1;
                                quad_control_point = Some(control_point);
                                cubic_control_point = None;
                            }
                            PathSegment::CurveTo {
                                abs: true,
                                x1,
                                y1,
                                x2,
                                y2,
                                x,
                                y,
                            } => {
                                let p0 = Point::new(x1 as f32, y1 as f32);
                                let p1 = Point::new(x2 as f32, y2 as f32);
                                let p2 = Point::new(x as f32, y as f32);
                                let control_point = p1;

                                builder.cubic_to(self.t(p0), self.t(control_point), self.t(p2));

                                start_point.get_or_insert(end_point);
                                end_point = p2;
                                quad_control_point = None;
                                cubic_control_point = Some(control_point);
                            }
                            PathSegment::CurveTo {
                                abs: false,
                                x1,
                                y1,
                                x2,
                                y2,
                                x,
                                y,
                            } => {
                                let p0 = add_diff(end_point, x1, y1);
                                let p1 = add_diff(end_point, x2, y2);
                                let p2 = add_diff(end_point, x, y);
                                let control_point = p1;

                                builder.cubic_to(self.t(p0), self.t(control_point), self.t(p2));

                                start_point.get_or_insert(end_point);
                                end_point = p2;
                                quad_control_point = None;
                                cubic_control_point = Some(control_point);
                            }
                            PathSegment::SmoothQuadratic { abs: true, x, y } => {
                                let p1 = Point::new(x as f32, y as f32);
                                let control_point =
                                    reflect(quad_control_point.unwrap_or(end_point), end_point);

                                builder.quad_to(self.t(control_point), self.t(p1));

                                start_point.get_or_insert(end_point);
                                end_point = p1;
                                quad_control_point = Some(control_point);
                                cubic_control_point = None;
                            }
                            PathSegment::SmoothQuadratic { abs: false, x, y } => {
                                let p1 = add_diff(end_point, x, y);
                                let control_point =
                                    reflect(quad_control_point.unwrap_or(end_point), end_point);

                                builder.quad_to(self.t(control_point), self.t(p1));

                                start_point.get_or_insert(end_point);
                                end_point = p1;
                                quad_control_point = Some(control_point);
                                cubic_control_point = None;
                            }
                            PathSegment::SmoothCurveTo {
                                abs: true,
                                x2,
                                y2,
                                x,
                                y,
                            } => {
                                let p1 = Point::new(x2 as f32, y2 as f32);
                                let p2 = Point::new(x as f32, y as f32);
                                let control_point =
                                    reflect(cubic_control_point.unwrap_or(end_point), end_point);

                                builder.cubic_to(self.t(control_point), self.t(p1), self.t(p2));

                                start_point.get_or_insert(end_point);
                                end_point = p2;
                                quad_control_point = None;
                                cubic_control_point = Some(control_point);
                            }
                            PathSegment::SmoothCurveTo {
                                abs: false,
                                x2,
                                y2,
                                x,
                                y,
                            } => {
                                let p1 = add_diff(end_point, x2, y2);
                                let p2 = add_diff(end_point, x, y);
                                let control_point =
                                    reflect(cubic_control_point.unwrap_or(end_point), end_point);

                                builder.cubic_to(self.t(control_point), self.t(p1), self.t(p2));

                                start_point.get_or_insert(end_point);
                                end_point = p2;
                                quad_control_point = None;
                                cubic_control_point = Some(control_point);
                            }
                            PathSegment::EllipticalArc {
                                abs: true,
                                rx,
                                ry,
                                x_axis_rotation,
                                large_arc,
                                sweep,
                                x,
                                y,
                            } => {
                                let arc = convert_to_center(
                                    rx as f32,
                                    ry as f32,
                                    x_axis_rotation as f32,
                                    large_arc,
                                    sweep,
                                    end_point.x as f32,
                                    end_point.y as f32,
                                    x as f32,
                                    y as f32,
                                );

                                if let Some(arc) = arc {
                                    let new_end_point =
                                        self.push_rationals_from_arc(&mut builder, &arc, end_point);

                                    start_point.get_or_insert(end_point);
                                    end_point = new_end_point;
                                }

                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::EllipticalArc {
                                abs: false,
                                rx,
                                ry,
                                x_axis_rotation,
                                large_arc,
                                sweep,
                                x,
                                y,
                            } => {
                                let p0 = add_diff(end_point, x, y);
                                let arc = convert_to_center(
                                    rx as f32,
                                    ry as f32,
                                    x_axis_rotation as f32,
                                    large_arc,
                                    sweep,
                                    end_point.x as f32,
                                    end_point.y as f32,
                                    p0.x as f32,
                                    p0.y as f32,
                                );

                                if let Some(arc) = arc {
                                    let new_end_point =
                                        self.push_rationals_from_arc(&mut builder, &arc, end_point);

                                    start_point.get_or_insert(end_point);
                                    end_point = new_end_point;
                                }

                                quad_control_point = None;
                                cubic_control_point = None;
                            }
                            PathSegment::ClosePath { .. } => {
                                if let Some(start_point) = start_point.take() {
                                    end_point = start_point;
                                    quad_control_point = None;
                                    cubic_control_point = None;
                                }
                            }
                        }
                    }

                    let fill_command = parse_fill_rule(&attrs);

                    let blend_mode = parse_blend_mode(&attrs).unwrap_or_default();

                    self.paths.push((
                        builder.build(),
                        fill_command,
                        self.parse_fill(&attrs),
                        blend_mode,
                    ));
                }
                Event::Tag(tag::Rectangle, tag::Type::Start | tag::Type::Empty, attrs) => {
                    if attrs
                        .get("stroke")
                        .filter(|val| val.to_string() != "none")
                        .is_some()
                    {
                        continue;
                    }

                    let mut builder = forma::PathBuilder::new();

                    let x: f32 = attrs
                        .get("x")
                        .and_then(|opacity| opacity.parse().ok())
                        .unwrap_or_default();
                    let y: f32 = attrs
                        .get("y")
                        .and_then(|opacity| opacity.parse().ok())
                        .unwrap_or_default();
                    let width: f32 = attrs
                        .get("width")
                        .expect("rect missing width")
                        .parse()
                        .unwrap();
                    let height: f32 = attrs
                        .get("height")
                        .expect("rect missing height")
                        .parse()
                        .unwrap();

                    builder.move_to(Point::new(x, y));
                    builder.line_to(Point::new(x, y + height));
                    builder.line_to(Point::new(x + width, y + height));
                    builder.line_to(Point::new(x + width, y));
                    builder.line_to(Point::new(x, y));

                    let fill_command = parse_fill_rule(&attrs);

                    let blend_mode = parse_blend_mode(&attrs).unwrap_or_default();

                    self.paths.push((
                        builder.build(),
                        fill_command,
                        self.parse_fill(&attrs),
                        blend_mode,
                    ));
                }
                Event::Tag(tag::LinearGradient, tag::Type::Start, attrs) => {
                    if !attrs
                        .get("gradientUnits")
                        .map(|value| &**value == "userSpaceOnUse")
                        .unwrap_or_default()
                    {
                        continue;
                    }

                    let id = attrs
                        .get("id")
                        .expect("linearGradient missing id")
                        .to_string();

                    let x1: f32 = attrs
                        .get("x1")
                        .and_then(|opacity| opacity.parse().ok())
                        .expect("linearGradient missing x1");
                    let y1: f32 = attrs
                        .get("y1")
                        .and_then(|opacity| opacity.parse().ok())
                        .expect("linearGradient missing y1");
                    let x2: f32 = attrs
                        .get("x2")
                        .and_then(|opacity| opacity.parse().ok())
                        .expect("linearGradient missing x2");
                    let y2: f32 = attrs
                        .get("y2")
                        .and_then(|opacity| opacity.parse().ok())
                        .expect("linearGradient missing y2");

                    let mut gradient_builder =
                        GradientBuilder::new(Point::new(x1, y1), Point::new(x2, y2));

                    gradient_builder.r#type(GradientType::Linear);

                    self.gradient_builder = Some((id, gradient_builder));
                }
                Event::Tag(tag::LinearGradient, tag::Type::End, _) => {
                    let (id, gradient_builder) = self
                        .gradient_builder
                        .take()
                        .expect("linearGradient missing start tag");

                    self.gradients.insert(
                        id,
                        gradient_builder
                            .build()
                            .expect("linearGradient requires at least 2 stops"),
                    );
                }
                Event::Tag(tag::RadialGradient, tag::Type::Start, attrs) => {
                    if !attrs
                        .get("gradientUnits")
                        .map(|value| &**value == "userSpaceOnUse")
                        .unwrap_or_default()
                    {
                        continue;
                    }

                    let id = attrs
                        .get("id")
                        .expect("radialGradient missing id")
                        .to_string();

                    let cx: f32 = attrs
                        .get("cx")
                        .and_then(|opacity| opacity.parse().ok())
                        .expect("radialGradient missing cx");
                    let cy: f32 = attrs
                        .get("cy")
                        .and_then(|opacity| opacity.parse().ok())
                        .expect("radialGradient missing cy");
                    let r: f32 = attrs
                        .get("r")
                        .and_then(|opacity| opacity.parse().ok())
                        .expect("radialGradient missing r");

                    let mut gradient_builder =
                        GradientBuilder::new(Point::new(cx, cy), Point::new(cx + r, cy));

                    gradient_builder.r#type(GradientType::Radial);

                    self.gradient_builder = Some((id, gradient_builder));
                }
                Event::Tag(tag::RadialGradient, tag::Type::End, _) => {
                    let (id, gradient_builder) = self
                        .gradient_builder
                        .take()
                        .expect("radialGradient missing start tag");

                    self.gradients.insert(
                        id,
                        gradient_builder
                            .build()
                            .expect("radialGradient requires at least 2 stops"),
                    );
                }
                Event::Tag(tag::Stop, _, attrs) => {
                    let fill: Option<Color> = parse_color(&attrs).or_else(|| Some(Color::black()));

                    let opacity: f32 = parse_opacity(&attrs).unwrap_or(1.0);

                    let fill = fill
                        .map(|fill| styling::Color {
                            a: opacity,
                            ..crate::to_linear([fill.red, fill.green, fill.blue])
                        })
                        .expect("stop missing stop-fill");

                    let stop: f32 = attrs
                        .get("offset")
                        .and_then(|stop| {
                            let stop = stop.to_string();
                            stop[..stop.len() - 1].parse().ok()
                        })
                        .expect("stop missing offset");

                    let gradient_builder = &mut self
                        .gradient_builder
                        .as_mut()
                        .expect("stop missing gradient start tag")
                        .1;

                    gradient_builder.color_with_stop(fill, stop / 100.0);
                }
                _ => (),
            }
        }
    }
}

impl App for Svg {
    fn width(&self) -> usize {
        1000
    }

    fn height(&self) -> usize {
        1000
    }

    fn set_width(&mut self, _width: usize) {}

    fn set_height(&mut self, _height: usize) {}

    fn compose(&mut self, composition: &mut Composition, duration: Duration, keyboard: &Keyboard) {
        let x_speed = duration.as_millis() as f32;
        let y_speed = duration.as_millis() as f32;
        if keyboard.is_key_down(VirtualKeyCode::Left) {
            self.x -= x_speed;
            self.needs_composition = true;
        }
        if keyboard.is_key_down(VirtualKeyCode::Right) {
            self.x += x_speed;
            self.needs_composition = true;
        }

        if keyboard.is_key_down(VirtualKeyCode::Up) {
            self.y += y_speed;
            self.needs_composition = true;
        }
        if keyboard.is_key_down(VirtualKeyCode::Down) {
            self.y -= y_speed;
            self.needs_composition = true;
        }
        if !self.needs_composition {
            return;
        }
        let transform = GeomPresTransform::try_from([1.0, 0.0, 0.0, 1.0, -self.x, self.y]).unwrap();

        for (order, (path, fill_rule, fill, blend_mode)) in self.paths.iter().enumerate() {
            let mut layer = composition.create_layer();

            layer
                .insert(path)
                .set_transform(transform)
                .set_props(Props {
                    fill_rule: *fill_rule,
                    func: Func::Draw(Style {
                        fill: fill.clone(),
                        blend_mode: *blend_mode,
                        ..Default::default()
                    }),
                });

            composition.insert(Order::new(order as u32).unwrap(), layer);
        }

        self.needs_composition = false;
    }
}
