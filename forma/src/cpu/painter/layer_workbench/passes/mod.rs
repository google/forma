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

use rustc_hash::FxHashSet;

mod skip_fully_covered_layers;
mod skip_trivial_clips;
mod tile_unchanged;

pub use skip_fully_covered_layers::skip_fully_covered_layers_pass;
pub use skip_trivial_clips::skip_trivial_clips_pass;
pub use tile_unchanged::tile_unchanged_pass;

#[derive(Clone, Debug)]
pub struct PassesSharedState {
    pub skip_clipping: FxHashSet<u32>,
    pub layers_were_removed: bool,
}

impl Default for PassesSharedState {
    fn default() -> Self {
        Self {
            layers_were_removed: true,
            skip_clipping: FxHashSet::default(),
        }
    }
}

impl PassesSharedState {
    pub fn reset(&mut self) {
        self.skip_clipping.clear();
        self.layers_were_removed = true;
    }
}
