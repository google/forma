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

use std::ops::ControlFlow;

use crate::{
    cpu::painter::{
        layer_workbench::{Context, LayerWorkbenchState, OptimizerTileWriteOp},
        LayerProps,
    },
    styling::{BlendMode, Color, Fill, Func, Style},
};

use super::PassesSharedState;

pub fn skip_fully_covered_layers_pass<'w, 'c, P: LayerProps>(
    workbench: &'w mut LayerWorkbenchState,
    state: &'w mut PassesSharedState,
    context: &'c Context<'_, P>,
) -> ControlFlow<OptimizerTileWriteOp> {
    #[derive(Debug)]
    enum InterestingCover {
        Opaque(Color),
        Incomplete,
    }

    let mut first_interesting_cover = None;
    // If layers were removed, we cannot assume anything because a visible layer
    // might have been removed since last frame.
    let mut visible_layers_are_unchanged = !state.layers_were_removed;
    for (i, &id) in workbench.ids.iter_masked().rev() {
        let props = context.props.get(id);

        if !context.props.is_unchanged(id) {
            visible_layers_are_unchanged = false;
        }

        let is_clipped = || {
            matches!(
                props.func,
                Func::Draw(Style {
                    is_clipped: true,
                    ..
                })
            ) && !state.skip_clipping.contains(&id)
        };

        if is_clipped() || !workbench.layer_is_full(context, id, props.fill_rule) {
            if first_interesting_cover.is_none() {
                first_interesting_cover = Some(InterestingCover::Incomplete);
                // The loop does not break here in order to try to cull some layers that are
                // completely covered.
            }
        } else if let Func::Draw(Style {
            fill: Fill::Solid(color),
            blend_mode: BlendMode::Over,
            ..
        }) = props.func
        {
            if color.a == 1.0 {
                if first_interesting_cover.is_none() {
                    first_interesting_cover = Some(InterestingCover::Opaque(color));
                }

                workbench.ids.skip_until(i);

                break;
            }
        }
    }

    let (i, bottom_color) = match first_interesting_cover {
        // First opaque layer is skipped when blending.
        Some(InterestingCover::Opaque(color)) => {
            // All visible layers are unchanged so we can skip drawing altogether.
            if visible_layers_are_unchanged {
                return ControlFlow::Break(OptimizerTileWriteOp::None);
            }

            (1, color)
        }
        // The clear color is used as a virtual first opqaue layer.
        None => (0, context.clear_color),
        // Visible incomplete cover makes full optimization impossible.
        Some(InterestingCover::Incomplete) => return ControlFlow::Continue(()),
    };

    let color = workbench
        .ids
        .iter_masked()
        .skip(i)
        .try_fold(bottom_color, |dst, (_, &id)| {
            match context.props.get(id).func {
                Func::Draw(Style {
                    fill: Fill::Solid(color),
                    blend_mode,
                    ..
                }) => Some(blend_mode.blend(dst, color)),
                // Fill is not solid.
                _ => None,
            }
        });

    match color {
        Some(color) => ControlFlow::Break(OptimizerTileWriteOp::Solid(color)),
        None => ControlFlow::Continue(()),
    }
}
