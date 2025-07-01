// device/kernels/dataflow/eltwise_multiply_custom_writer.cpp

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Get compile-time arguments for output tensor configuration
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    
    // Optional arguments for advanced configurations
    uint32_t tiles_per_batch = get_arg_val<uint32_t>(3);  // Number of tiles to write per batch
    
    // If tiles_per_batch is 0, write all tiles in one batch
    if (tiles_per_batch == 0) {
        tiles_per_batch = num_tiles;
    }
    
    // Circular buffer configuration
    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;  // Output circular buffer
    constexpr bool output_is_dram = true;
    
    // Get tile properties
    constexpr uint32_t output_tile_bytes = get_tile_size(cb_id_out);
    constexpr DataFormat output_data_format = get_dataformat(cb_id_out);
    
    // Setup address generator for interleaved output tensor
    const InterleavedAddrGenFast<output_is_dram> output_writer = {
        .bank_base_address = output_addr,
        .page_size = output_tile_bytes,
        .data_format = output_data_format
    };
    
    // Calculate barrier threshold for optimal performance
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<output_tile_bytes, num_cores>();
    uint32_t barrier_count = 0;
    
    // Current tile ID for addressing
    uint32_t current_tile_id = start_tile_id;
    uint32_t tiles_written = 0;
    
    // Main processing loop - write tiles in batches
    while (tiles_written < num_tiles) {
        // Calculate how many tiles to process in this batch
        uint32_t tiles_in_batch = (num_tiles - tiles_written < tiles_per_batch) ? 
                                  (num_tiles - tiles_written) : tiles_per_batch;
        
        // Wait for compute kernel to produce results
        cb_wait_front(cb_id_out, tiles_in_batch);
        
        // Get the L1 read address for the circular buffer
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        
        // Write each tile from circular buffer to DRAM
        for (uint32_t i = 0; i < tiles_in_batch; ++i) {
            // Calculate DRAM address for this tile
            uint64_t output_noc_addr = get_noc_addr(current_tile_id, output_writer);
            
            // Async write tile data from L1 to DRAM
            noc_async_write(l1_read_addr, output_noc_addr, output_tile_bytes);
            
            // Update addresses and counters
            l1_read_addr += output_tile_bytes;
            current_tile_id++;
            
            // Check if we need to flush writes for performance
            if (++barrier_count >= barrier_threshold) {
                noc_async_write_barrier();
                barrier_count = 0;
            }
        }
        
        // Ensure all writes are complete before proceeding
        noc_async_write_barrier();
        barrier_count = 0;
        
        // Release the tiles from circular buffer
        cb_pop_front(cb_id_out, tiles_in_batch);
        
        // Update total tiles written
        tiles_written += tiles_in_batch;
    }
    
    // Final barrier to ensure all writes are committed
    noc_async_write_barrier();
}