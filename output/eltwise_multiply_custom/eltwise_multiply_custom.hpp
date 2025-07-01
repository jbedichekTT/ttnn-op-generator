// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise_multiply_custom/device/eltwise_multiply_custom_op.hpp"

namespace ttnn {
namespace operations {

struct EltwiseMultiplyCustomOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_a,
        const ttnn::Tensor& input_tensor_b,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::DataType>& output_dtype = std::nullopt);

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor_a,
        const ttnn::Tensor& input_tensor_b,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::DataType>& output_dtype = std::nullopt);
};

}  // namespace operations

constexpr auto eltwise_multiply_custom = ttnn::decorators::register_operation<
    "ttnn::eltwise_multiply_custom",
    ttnn::operations::EltwiseMultiplyCustomOperation>();

}  // namespace ttnn