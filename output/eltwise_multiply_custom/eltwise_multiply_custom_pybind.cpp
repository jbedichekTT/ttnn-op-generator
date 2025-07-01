// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eltwise_multiply_custom_pybind.hpp"

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "eltwise_multiply_custom.hpp"
#include "ttnn-pybind/decorators.hpp"

namespace ttnn::operations::eltwise_multiply_custom {
namespace {

void bind_eltwise_multiply_custom(py::module& module, const char* doc) {
    ttnn::bind_registered_operation(
        module,
        ttnn::eltwise_multiply_custom,
        doc,
        ttnn::pybind_arguments_t{
            py::arg("input_tensor_a"), 
            py::arg("input_tensor_b"), 
            py::arg("memory_config") = std::nullopt});
}

}  // namespace

void py_bind_eltwise_multiply_custom(py::module& module) {
    const auto doc = R"doc(
        Performs element-wise multiplication of two tensors.


        Args:
            input_tensor_a (ttnn.Tensor): the first input tensor.
            input_tensor_b (ttnn.Tensor): the second input tensor.


        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.


        Returns:
            ttnn.Tensor: the output tensor containing the element-wise multiplication result.


        )doc";
    bind_eltwise_multiply_custom(module, doc);
}

}  // namespace ttnn::operations::eltwise_multiply_custom