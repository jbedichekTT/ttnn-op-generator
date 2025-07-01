Based on the API examples I found, I can now generate the complete program factory header file for the `eltwise_multiply_custom` operation. Here's the implementation:

```hpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/api/ttnn/types.hpp"
#include "ttnn/api/ttnn/tensor/tensor.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::eltwise_multiply_custom::detail {

/**
 * @brief Creates and configures the program for eltwise_multiply_custom operation
 * 
 * This function acts as the program factory, orchestrating the creation of:
 * - Circular buffers for input tensors A, B and output tensor
 * - Reader, writer, and compute kernels
 * - Core range management and parallelization
 * - Runtime argument configuration
 * 
 * @param input_tensor_a First input tensor for element-wise multiplication
 * @param input_tensor_b Second input tensor for element-wise multiplication  
 * @param output_tensor Output tensor to store the result
 * @param device_compute_kernel_config Optional compute kernel configuration
 * @return ProgramWithCallbacks containing the configured program and runtime callback
 */
tt::tt_metal::operation::ProgramWithCallbacks eltwise_multiply_custom_program_factory(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const ttnn::Tensor& output_tensor,
    const std::optional<ttnn::DeviceComputeKernelConfig>& device_compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::eltwise_multiply_custom::detail
```

This header file provides:

1. **Required Includes**: All the validated APIs and headers needed for program factories
2. **Proper Namespace**: Uses the TTNN operations namespace structure
3. **Function Signature**: Matches the expected pattern for program factory functions that return `ProgramWithCallbacks`
4. **Documentation**: Clear description of the function's role as an orchestration engine
5. **Parameters**: 
   - Two input tensors for element-wise multiplication
   - Output tensor for results
   - Optional compute kernel configuration for customization
6. **Return Type**: `ProgramWithCallbacks` which contains both the program and runtime argument callbacks

The function signature follows the established patterns from the examples:
- Takes input/output tensors as parameters
- Includes optional configuration parameters
- Returns `ProgramWithCallbacks` for the operation framework
- Is properly namespaced within the operation's detail namespace

This header declares the main program factory function that will be implemented in the corresponding `.cpp` file, where it will:
- Create circular buffers for tensor data
- Configure reader/writer/compute kernels
- Set up core ranges and parallelization
- Handle runtime argument management
- Create the callback for dynamic runtime argument updates