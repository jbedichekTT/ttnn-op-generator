// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise_multiply_custom_op.hpp"
#include "eltwise_multiply_custom_program_factory.hpp"

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::eltwise_multiply_custom {

EltwiseMultiplyCustomDeviceOperation::program_factory_t EltwiseMultiplyCustomDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void EltwiseMultiplyCustomDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    
    // Validate input tensors are on device
    TT_FATAL(input_tensor_a.storage_type() == ttnn::StorageType::DEVICE, 
             "EltwiseMultiplyCustom: Input tensor A must be on device");
    TT_FATAL(input_tensor_b.storage_type() == ttnn::StorageType::DEVICE, 
             "EltwiseMultiplyCustom: Input tensor B must be on device");
    
    // Validate input tensors have buffers allocated
    TT_FATAL(input_tensor_a.buffer() != nullptr, 
             "EltwiseMultiplyCustom: Input tensor A must be allocated in buffer on device");
    TT_FATAL(input_tensor_b.buffer() != nullptr, 
             "EltwiseMultiplyCustom: Input tensor B must be allocated in buffer on device");
    
    // Validate tensors are on the same device
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), 
             "EltwiseMultiplyCustom: Input tensors must be on the same device");
    
    // Validate tensor shapes are compatible (same shape for element-wise multiply)
    TT_FATAL(input_tensor_a.get_logical_shape() == input_tensor_b.get_logical_shape(), 
             "EltwiseMultiplyCustom: Input tensors must have the same shape");
    
    // Validate memory layout
    TT_FATAL(input_tensor_a.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
             "EltwiseMultiplyCustom: Input tensor A must use interleaved memory layout");
    TT_FATAL(input_tensor_b.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
             "EltwiseMultiplyCustom: Input tensor B must use interleaved memory layout");
    TT_FATAL(operation_attributes.memory_config.memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
             "EltwiseMultiplyCustom: Output memory config must use interleaved memory layout");
}

void EltwiseMultiplyCustomDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Same validation as cache miss for now
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

EltwiseMultiplyCustomDeviceOperation::shape_return_value_t EltwiseMultiplyCustomDeviceOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    return input_tensor_a.get_logical_shape();
}

EltwiseMultiplyCustomDeviceOperation::tensor_return_value_t EltwiseMultiplyCustomDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    
    if (operation_attributes.output_tensor.has_value()) {
        return operation_attributes.output_tensor.value();
    }

    auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    
    return create_device_tensor(
        output_shape,
        operation_attributes.dtype,
        input_tensor_a.get_layout(),
        input_tensor_a.device(),
        operation_attributes.memory_config);
}

std::tuple<
    EltwiseMultiplyCustomDeviceOperation::operation_attributes_t,
    EltwiseMultiplyCustomDeviceOperation::tensor_args_t>
EltwiseMultiplyCustomDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DataType>& dtype,
    const std::optional<ttnn::Tensor>& output_tensor) {
    
    auto memory_config_val = memory_config.value_or(input_tensor_a.memory_config());
    auto dtype_val = dtype.value_or(input_tensor_a.get_dtype());
    
    return {
        operation_attributes_t{
            memory_config_val,
            dtype_val,
            output_tensor},
        tensor_args_t{
            input_tensor_a,
            input_tensor_b}};
}

}  // namespace ttnn::operations::eltwise_multiply_custom