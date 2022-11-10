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

use crate::cpu::painter::{
    layer_workbench::{Context, LayerWorkbenchState, OptimizerTileWriteOp},
    LayerProps,
};

use super::PassesSharedState;

pub fn tile_unchanged_pass<'w, 'c, P: LayerProps>(
    workbench: &'w mut LayerWorkbenchState,
    state: &'w mut PassesSharedState,
    context: &'c Context<'_, P>,
) -> ControlFlow<OptimizerTileWriteOp> {
    let clear_color_is_unchanged = context
        .cached_clear_color
        .map(|previous_clear_color| previous_clear_color == context.clear_color)
        .unwrap_or_default();

    let tile_paint = context.cached_tile.as_ref().and_then(|cached_tile| {
        let layers = workbench.ids.len() as u32;
        let previous_layers = cached_tile.update_layer_count(Some(layers));

        let is_unchanged = previous_layers
            .map(|previous_layers| {
                state.layers_were_removed = layers < previous_layers;

                previous_layers == layers
                    && workbench
                        .ids
                        .iter()
                        .all(|&id| context.props.is_unchanged(id))
            })
            .unwrap_or_default();

        (clear_color_is_unchanged && is_unchanged).then_some(OptimizerTileWriteOp::None)
    });

    match tile_paint {
        Some(tile_paint) => ControlFlow::Break(tile_paint),
        None => ControlFlow::Continue(()),
    }
}
