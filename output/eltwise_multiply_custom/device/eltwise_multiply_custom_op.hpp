// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/device_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn::operations::eltwise_multiply_custom {

struct EltwiseMultiplyCustomDeviceOperation {
    struct operation_attributes_t {
        const ttnn::MemoryConfig memory_config;
        const ttnn::DataType dtype;
        const std::optional<ttnn::Tensor> output_tensor;

        operation_attributes_t(
            const ttnn::MemoryConfig& memory_config,
            const ttnn::DataType& dtype,
            const std::optional<ttnn::Tensor>& output_tensor = std::nullopt) :
            memory_config(memory_config), dtype(dtype), output_tensor(output_tensor) {}
    };

    struct tensor_args_t {
        const ttnn::Tensor& input_tensor_a;
        const ttnn::Tensor& input_tensor_b;

        tensor_args_t(const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b) :
            input_tensor_a(input_tensor_a), input_tensor_b(input_tensor_b) {}
    };

    using shape_return_value_t = ttnn::SimpleShape;
    using tensor_return_value_t = ttnn::Tensor;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::KernelHandle eltwise_multiply_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            const tensor_return_value_t& tensor_return_value,
            cached_program_t& cached_program);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static shape_return_value_t compute_output_shapes(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor_a,
        const ttnn::Tensor& input_tensor_b,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::DataType>& dtype = std::nullopt,
        const std::optional<ttnn::Tensor>& output_tensor = std::nullopt);
};

}  // namespace ttnn::operations::eltwise_multiply_custom