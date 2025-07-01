// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include "ttnn/api/ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise_multiply_custom/device/eltwise_multiply_custom_program_factory.hpp"

using namespace tt;
using namespace tt::constants;

namespace ttnn {

namespace operations {

namespace eltwise_multiply_custom {

tt::tt_metal::operation::ProgramWithCallbacks eltwise_multiply_custom_multi_core(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, Tensor& output_tensor) {
    
    tt_metal::Program program{};

    // Get tensor shapes and properties
    const auto& ashape = input_tensor_a.get_padded_shape();
    const auto& bshape = input_tensor_b.get_padded_shape();
    const auto& output_shape = output_tensor.get_padded_shape();

    // Data formats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    // Tile sizes
    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    // Buffers
    tt_metal::Buffer* src0_buffer = input_tensor_a.buffer();
    tt_metal::Buffer* src1_buffer = input_tensor_b.buffer();
    tt_metal::Buffer* dst_buffer = output_tensor.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // Device and core configuration
    tt::tt_metal::IDevice* device = input_tensor_a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    
    // Calculate total number of tiles to process
    uint32_t num_tiles_total = output_tensor.volume() / TILE_HW;
    
    // Split work across cores
    auto [num_cores,
          all_cores,
          core_group_1,
          core_group_2,
          num_tiles_per_core_group_1,
          num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles_total);

    // Buffer addresses
    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    // Circular buffer configuration for input A
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;  // Double buffered
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * in0_single_tile_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    // Circular buffer configuration for input B
    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(num_input_tiles * in1_single_tile_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

    // Circular buffer configuration for output
    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;  // Double buffered
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // Kernel compilation args
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    std::vector<uint32_t> reader_a_compile_time_args = {(uint32_t)src0_is_dram};
    std::vector<uint32_t> reader_b_compile_time_args = {(uint32_t)src1_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)output_cb_index, (uint32_t)dst_is_dram};

    // Create kernels
    auto reader_a_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise_multiply_custom/device/kernels/dataflow/eltwise_multiply_custom_reader.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_a_compile_time_args));

    auto reader_b_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise_multiply_custom/device/kernels/dataflow/eltwise_multiply_custom_reader.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_b_compile_time_args));

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise_multiply_custom/device/kernels/dataflow/eltwise_multiply_custom_writer.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Compute kernel args for group 1
    std::vector<uint32_t> compute_args_group_1 = {
        src0_cb_index,
        src1_cb_index,
        output_cb_index,
        num_tiles_per_core_group_1
    };

    auto compute_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise_multiply_custom/device/kernels/compute/eltwise_multiply_custom_compute.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.compile_args = compute_args_group_1});

    // Create compute kernel for group 2 if needed
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_args_group_2 = {
            src0_cb_index,
            src1_cb_index,
            output_cb_index,
            num_tiles_per_core_group_2
        };

        auto compute_kernel_group_2_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise_multiply_custom/device/kernels/compute/eltwise_multiply_custom_compute.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.compile_args = compute_args_group_2});
    }

    // Set runtime arguments for each core
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / compute_with_storage_grid_size.y, i % compute_with_storage_grid_size.y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        // Reader A runtime args
        tt_metal::SetRuntimeArgs(
            program,
            reader_a_kernel_id,
            core,
            {src0_addr, num_tiles_per_core, num_tiles_written});

        // Reader B runtime args
        tt_metal::SetRuntimeArgs(
            program,
            reader_b_kernel_id,
            core,
            {src1_addr, num_tiles_per_core, num_tiles_written});

        // Writer runtime args
        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {dst_addr, num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    // Runtime arguments callback for dynamic buffer updates
    auto override_runtime_arguments_callback =
        [reader_a_kernel_id, reader_b_kernel_id, writer_kernel_id, num_cores, compute_with_storage_grid_size](
            const void* operation,
            const Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            
            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            for (uint32_t i = 0; i < num_cores; i++) {
                CoreCoord core = {i / compute_with_storage_grid_size.y, i % compute_with_storage_grid_size.y};

                // Update reader A runtime args
                {
                    auto& runtime_args = GetRuntimeArgs(program, reader_a_kernel_id, core);
                    runtime_args[0] = src_buffer_a->address();
                }

                // Update reader B runtime args
                {
                    auto& runtime_args = GetRuntimeArgs(program, reader_b_kernel_id, core);
                    runtime_args[0] = src_buffer_b->address();
                }

                // Update writer runtime args
                {
                    auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[0] = dst_buffer->address();
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace eltwise_multiply_custom

}  // namespace operations

}  // namespace ttnn