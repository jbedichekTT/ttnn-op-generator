// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-pybind/pybind_fwd.hpp"

namespace py = pybind11;

namespace ttnn::operations::eltwise_multiply_custom {
void py_bind_eltwise_multiply_custom(py::module& module);
}  // namespace ttnn::operations::eltwise_multiply_custom