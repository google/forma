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
        layer_workbench::{Context, Index, LayerWorkbenchState, OptimizerTileWriteOp},
        LayerProps,
    },
    styling::{Func, Style},
};

use super::PassesSharedState;

pub fn skip_trivial_clips_pass<'w, 'c, P: LayerProps>(
    workbench: &'w mut LayerWorkbenchState,
    state: &'w mut PassesSharedState,
    context: &'c Context<'_, P>,
) -> ControlFlow<OptimizerTileWriteOp> {
    struct Clip {
        is_full: bool,
        last_layer_id: u32,
        i: Index,
        is_used: bool,
    }

    let mut clip = None;

    for (i, &id) in workbench.ids.iter_masked() {
        let props = context.props.get(id);

        if let Func::Clip(layers) = props.func {
            let is_full = workbench.layer_is_full(context, id, props.fill_rule);

            clip = Some(Clip {
                is_full,
                last_layer_id: id + layers as u32,
                i,
                is_used: false,
            });

            if is_full {
                // Skip full clips.
                workbench.ids.set_mask(i, false);
            }
        }

        if let Func::Draw(Style {
            is_clipped: true, ..
        }) = props.func
        {
            match clip {
                Some(Clip {
                    is_full,
                    last_layer_id,
                    ref mut is_used,
                    ..
                }) if id <= last_layer_id => {
                    if is_full {
                        // Skip clipping when clip is full.
                        state.skip_clipping.insert(id);
                    } else {
                        *is_used = true;
                    }
                }
                _ => {
                    // Skip layer outside of clip.
                    workbench.ids.set_mask(i, false);
                }
            }
        }

        if let Some(Clip {
            last_layer_id,
            i,
            is_used,
            ..
        }) = clip
        {
            if id > last_layer_id {
                clip = None;

                if !is_used {
                    // Remove unused clips.
                    workbench.ids.set_mask(i, false);
                }
            }
        }
    }

    // Clip layer might be last layer.
    if let Some(Clip { i, is_used, .. }) = clip {
        if !is_used {
            // Remove unused clips.
            workbench.ids.set_mask(i, false);
        }
    }

    ControlFlow::Continue(())
}
