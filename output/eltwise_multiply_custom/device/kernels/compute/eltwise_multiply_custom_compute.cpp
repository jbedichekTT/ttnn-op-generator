// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"

using namespace ckernel;

void MAIN {
    // Get runtime arguments
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);
    
    // Circular buffer IDs
    constexpr uint32_t cb_id_in0 = 0;   // First input operand
    constexpr uint32_t cb_id_in1 = 1;   // Second input operand  
    constexpr uint32_t cb_id_out = 16;  // Output
    
    // Initialize multiplication operation
    mul_tiles_init(cb_id_in0, cb_id_in1);
    
    // Process tiles
    for (uint32_t i = 0; i < per_core_tile_cnt; ++i) {
        // Wait for input tiles to be available
        cb_wait_front(cb_id_in0, 1);
        cb_wait_front(cb_id_in1, 1);
        
        // Reserve space in output circular buffer
        cb_reserve_back(cb_id_out, 1);
        
        // Acquire destination registers for computation
        acquire_dst();
        
        // Copy input tiles to destination registers
        copy_tile(cb_id_in0, 0, 0);  // Copy tile 0 from cb_id_in0 to dst reg 0
        copy_tile(cb_id_in1, 0, 1);  // Copy tile 0 from cb_id_in1 to dst reg 1
        
        // Perform element-wise multiplication
        mul_tiles(cb_id_in0, cb_id_in1, 0, 0, 0);  // Multiply tiles and store result in dst reg 0
        
        // Pack result and write to output circular buffer
        pack_tile(0, cb_id_out);
        
        // Release destination registers
        release_dst();
        
        // Push result to output and pop consumed inputs
        cb_push_back(cb_id_out, 1);
        cb_pop_front(cb_id_in0, 1);
        cb_pop_front(cb_id_in1, 1);
    }
}