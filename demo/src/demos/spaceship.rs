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

use forma::prelude::*;
use nalgebra::Vector2;
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::mem;
use std::time::Duration;
use winit::event::VirtualKeyCode;

use crate::App;
use crate::Keyboard;

// Default size of the squared game area.
const GAME_SIZE: f32 = 1000.0;

/// Restricts the specified point position to the specified box.
fn clamp2d(point: Vector2<f32>, min: f32, max: f32) -> Vector2<f32> {
    Vector2::new(point.x.clamp(min, max), point.y.clamp(min, max))
}

enum ActorType {
    Player,
    Enemy,
}

/// Element displayed on screen and interacting.
pub struct Actor {
    kind: ActorType,
    acc: Vector2<f32>,
    speed: Vector2<f32>,
    pos: Vector2<f32>,
    angle: f32,
    angle_speed: f32,
    max_speed: f32,
    friction: f32,
    layer: Result<Order, Layer>,
    radius: f32,
    alive: bool,
}

impl Actor {
    fn create_enemy(rng: &mut StdRng, composition: &mut Composition) -> Actor {
        let radius = rng.gen_range(5.0f32..10.0).powf(2.0);
        let x = rng.gen_range(0.0..GAME_SIZE);
        let angle = rng.gen_range(-0.2..0.2f32);
        let speed = rng.gen_range(8.0..16.0f32).powf(2.0);
        let color = Color {
            r: rng.gen_range(0.1..0.3f32),
            g: rng.gen_range(0.1..0.3f32),
            b: rng.gen_range(0.1..0.3f32),
            a: 1.0,
        };
        let mut layer = composition.create_layer();
        layer
            .insert(&potatoe_path(0.0, 0.0, radius, rng))
            .set_props(Props {
                fill_rule: FillRule::NonZero,
                func: Func::Draw(Style {
                    fill: Fill::Solid(color),
                    ..Default::default()
                }),
            });
        Actor {
            kind: ActorType::Enemy,
            pos: Vector2::new(x, -0.25 * GAME_SIZE),
            friction: 0.0,
            max_speed: f32::INFINITY,
            speed: Vector2::new(angle.sin(), angle.cos()) * speed,
            acc: Vector2::new(0.0, 0.0),
            angle: 0.0,
            angle_speed: rng.gen_range(-0.8..0.8f32),
            radius,
            layer: Err(layer),
            alive: true,
        }
    }

    fn create_player(composition: &mut Composition) -> Actor {
        let color = Color {
            r: 0.1,
            g: 0.1,
            b: 0.1,
            a: 1.0,
        };
        let mut layer = composition.create_layer();
        layer.insert(&ship_path()).set_props(Props {
            fill_rule: FillRule::NonZero,
            func: Func::Draw(Style {
                fill: Fill::Solid(color),
                ..Default::default()
            }),
        });
        Actor {
            kind: ActorType::Player,
            pos: Vector2::new(0.5 * GAME_SIZE, 0.9 * GAME_SIZE),
            friction: -1.0,
            max_speed: 10.0 * GAME_SIZE,
            speed: Vector2::new(0.0, 0.0),
            acc: Vector2::new(0.0, 0.0),
            angle: 0.0,
            angle_speed: 0.0,
            radius: 60.0,
            layer: Err(layer),
            alive: true,
        }
    }
    /// Applies the interaction logic to a pair of actors.
    fn interact(&mut self, b: &mut Actor) {
        let a = self;
        if !a.alive || !b.alive {
            return;
        }

        if a.overlaps_with(b) {
            // Elastic bounce between the two objects.
            // Unit vector of the axis between the two centers.
            let r_pos_u = (b.pos - a.pos).normalize();
            // Speed of each actor projected on the collision axis.
            let u1 = r_pos_u.dot(&a.speed);
            let u2 = r_pos_u.dot(&b.speed);
            // Do not collide objects moving away.
            if u2 > u1 {
                return;
            }
            // Masses are proportional to the cube of the radius.
            let (m1, m2) = (a.radius.powf(3.0), b.radius.powf(3.0));
            // Speeds after collision on the collision axis.
            let v1 = ((m1 - m2) * u1 + 2.0 * m2 * u2) / (m1 + m2);
            let v2 = ((m2 - m1) * u2 + 2.0 * m1 * u1) / (m1 + m2);
            // Update speeds.
            a.speed += r_pos_u * (-u1 + v1);
            b.speed += r_pos_u * (-u2 + v2);
        }
    }

    /// Executed once per frame to update internal state of the actor.
    fn update(&mut self, keyboard: &Keyboard, delta_t: f32) {
        if !self.alive {
            return;
        }
        if let ActorType::Player = self.kind {
            // Capture keyboard inputs and set avatars acceleration.
            let keyboard_cmd = Vector2::new(
                if keyboard.is_key_down(VirtualKeyCode::Left) {
                    -1.0
                } else {
                    0.0
                } + if keyboard.is_key_down(VirtualKeyCode::Right) {
                    1.0
                } else {
                    0.0
                },
                if keyboard.is_key_down(VirtualKeyCode::Up) {
                    -1.0
                } else {
                    0.0
                } + if keyboard.is_key_down(VirtualKeyCode::Down) {
                    1.0
                } else {
                    0.0
                },
            );
            const THRUST_POWER: f32 = 2000.0;
            self.acc = THRUST_POWER * keyboard_cmd;
            self.pos = clamp2d(self.pos, 0.0, GAME_SIZE);
            self.angle = (-2.0 * self.speed.x / self.max_speed)
                .asin()
                .clamp(-0.2, 0.2);
        }
        // Integrate acceleration, speed and compute position.
        self.acc += self.speed * self.friction;
        let max_speed = self.max_speed;
        self.speed = clamp2d(self.speed + self.acc * delta_t, -max_speed, max_speed);
        self.pos += self.speed * delta_t;
        self.angle += self.angle_speed * delta_t;

        // Kill actors out of the game boundaries.
        self.alive &= self.pos.x > -0.5 * GAME_SIZE
            && self.pos.x < 1.5 * GAME_SIZE
            && self.pos.y > -0.5 * GAME_SIZE
            && self.pos.y < 1.5 * GAME_SIZE;
    }

    /// Returns a transformation the moves and turn the path to the
    /// actor place.
    fn transform(&self) -> GeomPresTransform {
        let (c, s) = (self.angle.cos(), self.angle.sin());
        GeomPresTransform::try_from([c, s, -s, c, self.pos.x, self.pos.y]).unwrap()
    }

    /// Returns true when this actor overlaps with the specified one.
    /// Collision shape is a disc.
    fn overlaps_with(&self, other: &Actor) -> bool {
        let d = self.radius + other.radius;
        let u = self.pos - other.pos;
        u.dot(&u) < d * d
    }
}

/// Monotonic function counting ennemies created since the start of the game.
fn enemy_count(game_time: std::time::Duration) -> i32 {
    let t = game_time.as_secs_f32();
    (0.0005 * t * t + 2.0 * t) as i32
}

pub struct Spaceship {
    height: usize,
    width: usize,
    actors: Vec<Actor>,
    time: Duration,
    rng: StdRng,
}

impl Spaceship {
    pub fn new() -> Spaceship {
        Spaceship {
            height: GAME_SIZE as usize,
            width: GAME_SIZE as usize,
            actors: vec![],
            time: Duration::ZERO,
            rng: StdRng::seed_from_u64(43),
        }
    }
}

impl Spaceship {
    fn update_actors(
        &mut self,
        delta_t: Duration,
        keyboard: &Keyboard,
        composition: &mut Composition,
    ) {
        // Create enemies that popped since the last frame.
        let new_enemies = enemy_count(self.time + delta_t) - enemy_count(self.time);
        for _ in 0..new_enemies {
            let actor = Actor::create_enemy(&mut self.rng, composition);
            self.actors.push(actor);
        }

        // Iterate over all mutable pairs of actors, and apply interactions.
        for a_idx in 0..self.actors.len() {
            let mut b_iter = self.actors[a_idx..].iter_mut();
            let a = b_iter.next().unwrap();
            b_iter.for_each(|b| a.interact(b));
        }

        // Update the position of each actor.
        self.actors
            .iter_mut()
            .for_each(|a| a.update(keyboard, delta_t.as_secs_f32()));
    }

    fn update_composition_and_cleanup_actors(&mut self, composition: &mut Composition) {
        // Update the composition.
        for (order, a) in self.actors.iter_mut().enumerate() {
            if a.alive {
                let order = Order::new(order as u32).unwrap();

                a.layer = match mem::replace(&mut a.layer, Ok(Order::new(0).unwrap())) {
                    Ok(cached_order) => {
                        if cached_order != order {
                            let layer = composition.remove(cached_order).unwrap();
                            composition.insert(order, layer);
                        }

                        Ok(order)
                    }
                    Err(layer) => {
                        composition.insert(order, layer);
                        Ok(order)
                    }
                };

                composition
                    .get_mut(order)
                    .unwrap()
                    .set_transform(a.transform());
            } else if let Ok(order) = a.layer {
                composition.remove(order);
            }
        }

        // Remove actors that are no longer in use.
        self.actors.retain(|a| a.alive);
    }
}

impl Default for Spaceship {
    fn default() -> Self {
        Self::new()
    }
}

impl App for Spaceship {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn set_width(&mut self, width: usize) {
        self.width = width;
    }

    fn set_height(&mut self, height: usize) {
        self.height = height;
    }

    fn compose(&mut self, composition: &mut Composition, elapsed: Duration, keyboard: &Keyboard) {
        // Initialize game : create the player actor on the very first frame.
        if self.actors.is_empty() {
            self.actors.push(Actor::create_player(composition));
        }
        // Run game simulation.
        self.update_actors(elapsed, keyboard, composition);
        self.time += elapsed;
        // Update display.
        self.update_composition_and_cleanup_actors(composition);
    }
}

/// Returns a dented circle path.
fn potatoe_path(x: f32, y: f32, radius: f32, rng: &mut StdRng) -> Path {
    let range = 0.07..1.4;
    PathBuilder::new()
        .move_to(Point::new(x + radius, y))
        .rat_quad_to(
            Point::new(x + radius, y - radius),
            Point::new(x, y - radius),
            rng.gen_range(range.clone()),
        )
        .rat_quad_to(
            Point::new(x - radius, y - radius),
            Point::new(x - radius, y),
            rng.gen_range(range.clone()),
        )
        .rat_quad_to(
            Point::new(x - radius, y + radius),
            Point::new(x, y + radius),
            rng.gen_range(range.clone()),
        )
        .rat_quad_to(
            Point::new(x + radius, y + radius),
            Point::new(x + radius, y),
            rng.gen_range(range),
        )
        .build()
}

/// Return a spaceship path.
fn ship_path() -> Path {
    PathBuilder::new()
        .move_to(Point::new(0.0, 50.0))
        .line_to(Point::new(40.0, 50.0))
        .line_to(Point::new(40.0, 60.0))
        .cubic_to(
            Point::new(47.0, 56.0),
            Point::new(54.0, 57.0),
            Point::new(60.0, 60.0),
        )
        .line_to(Point::new(60.0, 50.0))
        .line_to(Point::new(80.0, 50.0))
        .line_to(Point::new(80.0, 10.0))
        .cubic_to(
            Point::new(67.0, -3.0),
            Point::new(50.0, -13.0),
            Point::new(30.0, -20.0),
        )
        .line_to(Point::new(25.0, -51.0))
        .line_to(Point::new(30.0, -52.0))
        .line_to(Point::new(30.0, -70.0))
        .line_to(Point::new(21.0, -74.0))
        .cubic_to(
            Point::new(17.0, -90.0),
            Point::new(9.0, -102.0),
            Point::new(0.0, -107.0),
        )
        .cubic_to(
            Point::new(-9.0, -102.0),
            Point::new(-17.0, -90.0),
            Point::new(-21.0, -74.0),
        )
        .line_to(Point::new(-30.0, -70.0))
        .line_to(Point::new(-30.0, -52.0))
        .line_to(Point::new(-25.0, -51.0))
        .line_to(Point::new(-30.0, -20.0))
        .cubic_to(
            Point::new(-50.0, -13.0),
            Point::new(-67.0, -3.0),
            Point::new(-80.0, 10.0),
        )
        .line_to(Point::new(-80.0, 50.0))
        .line_to(Point::new(-60.0, 50.0))
        .line_to(Point::new(-60.0, 60.0))
        .cubic_to(
            Point::new(-54.0, 57.0),
            Point::new(-47.0, 56.0),
            Point::new(-40.0, 60.0),
        )
        .line_to(Point::new(-40.0, 50.0))
        .line_to(Point::new(0.0, 50.0))
        .build()
}
