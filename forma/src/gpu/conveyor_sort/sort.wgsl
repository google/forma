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

let BLOCK_WIDTH: u32 = {{block_width}}u;
let BLOCK_HEIGHT: u32 = {{block_height}}u;
let BLOCK_LEN: u32 = {{block_len}}u;

struct _u64 {
    lo: u32,
    hi: u32,
}

let MAX = _u64(0xffffffffu, 0xffffffffu);

fn le(x: _u64, y: _u64) -> bool {
    return x.hi < y.hi || x.hi == y.hi && x.lo <= y.lo;
}

fn gt(x: _u64, y: _u64) -> bool {
    return x.hi > y.hi || x.hi == y.hi && x.lo > y.lo;
}

struct Config {
    len: u32,
    len_in_blocks: u32,
    n_way: u32,
}

@group(0) @binding(0) var<storage, read_write> buffer0: array<_u64>;
@group(0) @binding(1) var<storage, read_write> buffer1: array<_u64>;
@group(0) @binding(2) var<uniform> config: Config;
@group(0) @binding(3) var<storage, read_write> offsets: array<u32>;

var<workgroup> block: array<_u64, BLOCK_LEN>;

struct MergeBlock {
    start: u32,
    mid: u32,
    end: u32,
    index: u32,
}

fn getMergeBlock(
    n_way: u32,
    id: u32,
    len_per_id: u32,
) -> MergeBlock {
    let n_way_mask = n_way - 1u;
    let start = (~n_way_mask & id) * len_per_id;
    let index = (n_way_mask & id) * len_per_id;

    let mid = start + (n_way >> 1u) * len_per_id;
    let end = start + n_way * len_per_id;

    return MergeBlock(start, mid, end, index);
}

fn getBuffer0(i: u32) -> _u64 {
    if i < config.len {
        return buffer0[i];
    } else {
        return MAX;
    }
}

fn setBuffer0(i: u32, val: _u64) {
    if i < config.len {
        buffer0[i] = val;
    }
}

fn setBuffer1(i: u32, val: _u64) {
    if i < config.len {
        buffer1[i] = val;
    }
}

fn buffer0ToLocal(
    col: ptr<function, array<_u64, BLOCK_HEIGHT>>,
    offset: u32,
    local_id: u32,
) {
    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        (*col)[i] = getBuffer0(i * BLOCK_WIDTH + offset + local_id);
    }
}

fn localToBuffer0(
    col: ptr<function, array<_u64, BLOCK_HEIGHT>>,
    offset: u32,
    local_id: u32,
) {

    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        setBuffer0(i * BLOCK_WIDTH + offset + local_id, (*col)[i]);
    }
}

fn localToSharedTransposed(
    col: ptr<function, array<_u64, BLOCK_HEIGHT>>,
    local_id: u32,
) {
    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        block[local_id * BLOCK_HEIGHT + i] = (*col)[i];
    }

    workgroupBarrier();
}

fn sharedToBuffer0(offset: u32, local_id: u32) {
    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        let index = i * BLOCK_WIDTH + local_id;
        setBuffer0(index + offset, block[index]);
    }

    workgroupBarrier();
}

fn oddEvenSort(col: ptr<function, array<_u64, BLOCK_HEIGHT>>) {
    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        for (var j = 1u & i; j < BLOCK_HEIGHT - 1u; j += 2u) {
            if gt((*col)[j], (*col)[j + 1u]) {
                let swap = (*col)[j];
                (*col)[j] = (*col)[j + 1u];
                (*col)[j + 1u] = swap;
            }
        }
    }
}

fn findMergeOffsetShared(
    left_start: u32,
    left_len: u32,
    right_start: u32,
    right_len: u32,
    index: u32,
) -> u32 {
    var start = u32(max(0, i32(index) - i32(right_len)));
    var end = min(index, left_len);

    while start < end {
        let mid = (start + end) >> 1u;

        let left = block[left_start + mid];
        let right = block[right_start + index - 1u - mid];

        if le(left, right) {
            start = mid + 1u;
        } else {
            end = mid;
        }
    }

    return start;
}

fn mergeInLocal(
    left_start: u32,
    left_end: u32,
    right_start: u32,
    right_end: u32,
    col: ptr<function, array<_u64, BLOCK_HEIGHT>>,
) {
    var left_i = left_start;
    var right_i = right_start;

    var left = block[left_i];
    var right = block[right_i];

    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        let go_left = (right_i >= right_end)
            || ((left_i < left_end)
            && le(left, right));

        if go_left {
            (*col)[i] = left;

            left_i++;
            left = block[left_i];
        } else {
            (*col)[i] = right;

            right_i++;
            right = block[right_i];
        }
    }

    workgroupBarrier();
}

fn blockSortBlock(block_id: u32, local_id: u32) {
    let offset = block_id * BLOCK_WIDTH * BLOCK_HEIGHT;
    var col: array<_u64, BLOCK_HEIGHT>;

    buffer0ToLocal(&col, offset, local_id);

    oddEvenSort(&col);

    localToSharedTransposed(&col, local_id);

    for (var n_way = 2u; n_way <= BLOCK_WIDTH; n_way <<= 1u) {
        let merge_block = getMergeBlock(n_way, local_id, BLOCK_HEIGHT);

        let shared_offset = findMergeOffsetShared(
            merge_block.start,
            merge_block.mid - merge_block.start,
            merge_block.mid,
            merge_block.end - merge_block.mid,
            merge_block.index,
        );

        mergeInLocal(
            merge_block.start + shared_offset,
            merge_block.mid,
            merge_block.mid + merge_block.index - shared_offset,
            merge_block.end,
            &col,
        );

        localToSharedTransposed(&col, local_id);
    }

    sharedToBuffer0(offset, local_id);
}

fn findMergeOffsetBuffer0(
    left_start: u32,
    left_len: u32,
    right_start: u32,
    right_len: u32,
    index: u32,
) -> u32 {
    var start = u32(max(0, i32(index) - i32(right_len)));
    var end = min(index, left_len);

    while start < end {
        let mid = (start + end) >> 1u;

        let left = getBuffer0(left_start + mid);
        let right = getBuffer0(right_start + index - 1u - mid);

        if le(left, right) {
            start = mid + 1u;
        } else {
            end = mid;
        }
    }

    return start;
}

fn findMergeOffsetBlock(block_id: u32) {
    let merge_block = getMergeBlock(config.n_way, block_id, BLOCK_LEN);

    let mid = min(config.len_in_blocks * BLOCK_LEN, merge_block.mid);
    let end = min(config.len_in_blocks * BLOCK_LEN, merge_block.end);

    let offset = findMergeOffsetBuffer0(
        merge_block.start,
        mid - merge_block.start,
        mid,
        end - mid,
        merge_block.index,
    );

    offsets[block_id] = offset;
}

fn buffer0ToLocal2(
    col: ptr<function, array<_u64, BLOCK_HEIGHT>>,
    offset0: u32,
    len0: u32,
    offset1: u32,
    len1: u32,
    local_id: u32,
) {
    let new_offset1 = offset1 - len0;

    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        let index = i * BLOCK_WIDTH + local_id;

        var block_i: u32;
        if index < len0 {
            block_i = i * BLOCK_WIDTH + offset0 + local_id;
        } else {
            block_i = i * BLOCK_WIDTH + new_offset1 + local_id;
        }

        (*col)[i] = getBuffer0(block_i);
    }
}

fn localToShared(
    col: ptr<function, array<_u64, BLOCK_HEIGHT>>,
    local_id: u32,
) {
    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        block[i * BLOCK_WIDTH + local_id] = (*col)[i];
    }

    workgroupBarrier();
}

fn sharedToBuffer1(offset: u32, local_id: u32) {
    for (var i = 0u; i < BLOCK_HEIGHT; i++) {
        let index = i * BLOCK_WIDTH + local_id;
        setBuffer1(index + offset, block[index]);
    }

    workgroupBarrier();
}

fn mergeBlocksBlock(block_id: u32, local_id: u32) {
    var col: array<_u64, BLOCK_HEIGHT>;

    let merge_block = getMergeBlock(config.n_way, block_id, BLOCK_LEN);
    let n_way_mask = config.n_way - 1u;

    let offset0 = offsets[block_id];
    var offset1: u32;

    if (n_way_mask & block_id) == n_way_mask {
        offset1 = (config.n_way >> 1u) * BLOCK_LEN;
    } else {
        offset1 = offsets[block_id + 1u];
    }

    let delta_offset = offset1 - offset0;
    let left_len = delta_offset;
    let right_len = BLOCK_LEN - delta_offset;

    buffer0ToLocal2(
        &col,
        merge_block.start + offset0,
        left_len,
        merge_block.mid + merge_block.index - offset0,
        right_len,
        local_id,
    );

    localToShared(&col, local_id);

    // Only perform merge if needed and copy otherwise.
    if left_len != 0u && right_len != 0u {
        let i = local_id * BLOCK_HEIGHT;

        let shared_offset = findMergeOffsetShared(
            0u,
            left_len,
            left_len,
            right_len,
            i,
        );

        mergeInLocal(
            shared_offset,
            left_len,
            left_len + i - shared_offset,
            BLOCK_LEN,
            &col,
        );

        localToSharedTransposed(&col, local_id);
    }

    sharedToBuffer1(block_id * BLOCK_LEN, local_id);
}

@compute @workgroup_size({{block_width}})
fn blockSort(
    @builtin(local_invocation_id) local_id_vec: vec3<u32>,
    @builtin(workgroup_id) workgroup_id_vec: vec3<u32>,
    @builtin(num_workgroups) num_workgroups_vec: vec3<u32>,
) {
    let local_id = local_id_vec.x;
    let num_blocks = num_workgroups_vec.x;

    for (
        var block_id = workgroup_id_vec.x;
        block_id < config.len_in_blocks;
        block_id += num_blocks
    ) {
        blockSortBlock(block_id, local_id);
    }
}

@compute @workgroup_size({{block_width}})
fn findMergeOffsets(
    @builtin(global_invocation_id) global_id_vec: vec3<u32>,
) {
    if config.len_in_blocks < (config.n_way >> 1u) {
        return;
    }

    let block_id = global_id_vec.x;

    if block_id < config.len_in_blocks {
        if block_id == 0u {
            offsets[0] = 0u;
        }

        findMergeOffsetBlock(block_id + 1u);
    }
}

@compute @workgroup_size({{block_width}})
fn mergeBlocks(
    @builtin(local_invocation_id) local_id_vec: vec3<u32>,
    @builtin(workgroup_id) workgroup_id_vec: vec3<u32>,
    @builtin(num_workgroups) num_workgroups_vec: vec3<u32>,
) {
    if config.len_in_blocks < (config.n_way >> 1u) {
        return;
    }

    let local_id = local_id_vec.x;
    let num_blocks = num_workgroups_vec.x;

    for (
        var block_id = workgroup_id_vec.x;
        block_id < config.len_in_blocks;
        block_id += num_blocks
    ) {
        mergeBlocksBlock(block_id, local_id);
    }
}
