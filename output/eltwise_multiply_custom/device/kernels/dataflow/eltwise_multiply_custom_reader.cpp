// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/hw/inc/compile_time_args.h"
#include "ttnn/operations/eltwise_multiply_custom/device/kernels/dataflow/common.hpp"

void kernel_main() {
    // Compile-time arguments
    constexpr bool is_dram = tt::tt_metal::get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_tiles_per_core = tt::tt_metal::get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = tt::tt_metal::get_compile_time_arg_val(2);
    constexpr uint32_t start_tile_id = tt::tt_metal::get_compile_time_arg_val(3);
    
    // Runtime arguments
    uint32_t arg_idx = 0;
    const uint32_t input_a_addr = tt::tt_metal::get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_b_addr = tt::tt_metal::get_arg_val<uint32_t>(arg_idx++);
    
    // Circular buffer indices
    constexpr uint32_t cb_input_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_input_b = tt::CBIndex::c_1;
    
    // Tile and data format information
    constexpr uint32_t tile_bytes = get_tile_size(cb_input_a);
    constexpr DataFormat data_format = get_dataformat(cb_input_a);
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_cores>();
    
    // Set up address generators for both input tensors
    const InterleavedAddrGenFast<is_dram> input_a_addrgen = {
        .bank_base_address = input_a_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    
    const InterleavedAddrGenFast<is_dram> input_b_addrgen = {
        .bank_base_address = input_b_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };
    
    uint32_t barrier_count = 0;
    uint32_t current_tile_id = start_tile_id;
    
    // Read tiles for both input tensors
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_per_core; ++tile_idx) {
        // Reserve space in circular buffers for both inputs
        tt::tt_metal::cb_reserve_back(cb_input_a, 1);
        tt::tt_metal::cb_reserve_back(cb_input_b, 1);
        
        // Get write pointers for both circular buffers
        uint32_t input_a_write_ptr = tt::tt_metal::get_write_ptr(cb_input_a);
        uint32_t input_b_write_ptr = tt::tt_metal::get_write_ptr(cb_input_b);
        
        // Read tile from input_a
        tt::tt_metal::noc_async_read_tile(current_tile_id, input_a_addrgen, input_a_write_ptr);
        
        // Read tile from input_b  
        tt::tt_metal::noc_async_read_tile(current_tile_id, input_b_addrgen, input_b_write_ptr);
        
        // Increment barrier count and check if we need to wait
        barrier_count += 2; // We issued 2 async reads
        if (barrier_count >= barrier_threshold) {
            tt::tt_metal::noc_async_read_barrier();
            barrier_count = 0;
        }
        
        // Push tiles to circular buffers
        tt::tt_metal::cb_push_back(cb_input_a, 1);
        tt::tt_metal::cb_push_back(cb_input_b, 1);
        
        // Move to next tile
        current_tile_id++;
    }
    
    // Final barrier to ensure all reads are complete
    tt::tt_metal::noc_async_read_barrier();
}