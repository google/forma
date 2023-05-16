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

import init, { initThreadPool, context_draw, context_new_circles, context_new_spaceship } from './pkg/web.js';

let context;
let width = 1000;
let height = 1000;

onmessage = function(message) {
  width = message.data.width;
  height = message.data.height;

  if (message.data.elapsed) {
    postMessage(context_draw(context, width, height, message.data.elapsed, message.data.partials, message.data.controls));
  }
}

async function initialize() {
  await init();
  await initThreadPool(navigator.hardwareConcurrency);

  context = context_new_spaceship(width, height);

  postMessage(context_draw(context, width, height, 0, false, 0));
}

initialize();
