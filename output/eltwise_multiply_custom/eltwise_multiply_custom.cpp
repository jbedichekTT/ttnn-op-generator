// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise_multiply_custom/eltwise_multiply_custom.hpp"
#include "ttnn/operations/eltwise_multiply_custom/device/eltwise_multiply_custom_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn {
namespace operations {

ttnn::Tensor EltwiseMultiplyCustomOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DataType>& output_dtype) {
    
    // Input validation
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Input tensor A must be on device");
    TT_FATAL(input_tensor_b.storage_type() == StorageType::DEVICE, "Input tensor B must be on device");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Input tensors must be on the same device");
    
    // Get shapes for validation
    auto shape_a = input_tensor_a.get_padded_shape();
    auto shape_b = input_tensor_b.get_padded_shape();
    
    // Basic shape compatibility check - tensors should have same shape or be broadcastable
    TT_FATAL(
        shape_a == shape_b,
        "Input tensors must have the same shape for element-wise multiplication. Got shapes {} and {}",
        shape_a,
        shape_b);
    
    // Determine output memory config
    auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
    
    // Determine output data type
    auto output_data_type = output_dtype.value_or(input_tensor_a.get_dtype());
    
    // Create the device operation
    return tt::tt_metal::operation::run(
        ttnn::operations::eltwise_multiply_custom::EltwiseMultiplyCustom{
            .memory_config = output_memory_config,
            .dtype = output_data_type
        },
        {input_tensor_a, input_tensor_b},
        {},
        {},
        queue_id)
        .at(0);
}

ttnn::Tensor EltwiseMultiplyCustomOperation::invoke(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DataType>& output_dtype) {
    
    // Use default queue ID
    return invoke(
        ttnn::DefaultQueueId,
        input_tensor_a,
        input_tensor_b,
        memory_config,
        output_dtype);
}

}  // namespace operations
}  // namespace ttnn