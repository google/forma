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

const partials = document.getElementById('partials');

const canvas = document.getElementById('drawing');
const ctx = canvas.getContext('2d');

var controls = 0b0000;

document.addEventListener('keydown', (event) => {
  switch (event.key) {
    case "Up":
    case "ArrowUp":
      controls |= 0b1000;
      break;
    case "Right":
    case "ArrowRight":
      controls |= 0b0100;
      break;
    case "Down":
    case "ArrowDown":
      controls |= 0b0010;
      break;
    case "Left":
    case "ArrowLeft":
      controls |= 0b0001;
      break;
  }
}, false);
document.addEventListener('keyup', (event) => {
  switch (event.key) {
    case "Up":
    case "ArrowUp":
      controls &= 0b0111;
      break;
    case "Right":
    case "ArrowRight":
      controls &= 0b1011;
      break;
    case "Down":
    case "ArrowDown":
      controls &= 0b1101;
      break;
    case "Left":
    case "ArrowLeft":
      controls &= 0b1110;
      break;
  }
}, false);

const worker = new Worker(new URL('./worker.js', import.meta.url), {
  type: 'module'
});
worker.postMessage({
  'width': canvas.width,
  'height': canvas.height,
  'partials': partials.checked,
});

const fps = document.getElementById('fps');

let timings = [];
function pushTiming(timing) {
  timings.push(timing);

  if (timings.length == 50) {
    const sum = timings.reduce((a, b) => a + b, 0);
    const avg = (sum / timings.length) || 0;
    const min = Math.min(...timings);
    const max = Math.max(...timings);

    fps.innerHTML = 'FPS: ' + (1 / avg).toFixed(2) + ' (' + (1 / max).toFixed(2) + '/' + (1 / min).toFixed(2) + ')';

    timings = [];
  }
}

let last_timestamp = 0;
function animation(timestamp) {
  const elapsed = timestamp - last_timestamp;
  last_timestamp = timestamp;

  pushTiming(elapsed / 1000);

  worker.postMessage({
    'elapsed': elapsed,
    'width': canvas.width,
    'height': canvas.height,
    'partials': partials.checked,
    'controls': controls,
  });
}

worker.onmessage = function(message) {
  ctx.putImageData(new ImageData(
    new Uint8ClampedArray(message.data),
    canvas.width,
    canvas.height
  ), 0, 0);

  window.requestAnimationFrame(animation);
}
