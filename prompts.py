GLOBAL_CONTEXT = """**Include Path Context for TTNN Operations:**
    The CMake build system provides include paths at these levels:
    - `{repo_root}/ttnn/`
    - `{repo_root}/ttnn/cpp/`

    Therefore, #include statements must be relative to these paths:

    CORRECT includes:
    - `#include "api/ttnn/decorators.hpp"`
    - `#include "api/ttnn/run_operation.hpp"`
    - `#include "ttnn/operations/core/core.hpp"`
    - `#include "tt-metalium/host_api.hpp"`
    - `#include "eltwise_add_custom_op.hpp"

    INCORRECT includes (don't add extra ttnn/ prefix):
    - ❌ `#include "ttnn/api/ttnn/decorators.hpp"`
    - ❌ `#include "ttnn/ttnn/operations/core/core.hpp"`
         `#include "device/eltwise_add_custom_op.hpp`


    Rule: If the file is at `ttnn/api/ttnn/something.hpp`, include it as `"api/ttnn/something.hpp"`
    Here is the structure for the file system you are generating:
    ttnn/cpp/ttnn/operations/{operation_name}/
    ├── CMakeLists.txt              # CMake
    ├── {operation_name}.hpp        # Main header
    ├── {operation_name}.cpp        # Main implementation
    ├── {operation_name}_pybind.hpp # Pybind header
    ├── {operation_name}_pybind.cpp # Pybind implementation
    └── device/                     # Device-specific code
        ├── {operation_name}_op.hpp
        ├── {operation_name}_op.cpp
        ├── {operation_name}_program_factory.hpp
        ├── {operation_name}_program_factory.cpp
        └── kernels/
            ├── compute/
            │   └── {operation_name}_compute.cpp
            └── dataflow/
                ├── {operation_name}_reader.cpp
                └── {operation_name}_writer.cpp
    """

HPP_CONTEXT = (
    """
        # TTNN Operation Component Development Contexts

        ## Operation Header (.hpp) Context

        ### Purpose & Structure
        Operation headers define the **public API interface** that users call from C++ and Python.

        ### Standard Header Pattern
        ```cpp
        #pragma once

        #include "ttnn/decorators.hpp"
        #include "ttnn/operations/core/core.hpp"

        namespace ttnn::operations::operation_name {

        // Main operation function
        ttnn::Tensor operation_name(
            const ttnn::Tensor& input_a,
            const ttnn::Tensor& input_b,
            const std::optional<MemoryConfig>& memory_config = std::nullopt,
            const std::optional<DeviceComputeKernelConfig>& compute_config = std::nullopt
        );

        // Optional: Broadcasting variant
        ttnn::Tensor operation_name(
            const ttnn::Tensor& input_a,
            float scalar_b,
            const std::optional<MemoryConfig>& memory_config = std::nullopt
        );

        }  // namespace ttnn::operations
        ```

        ### Key API Design Patterns
        - **Required parameters first**: Input tensors, primary operation parameters
        - **Optional parameters last**: Memory config, compute config with defaults
        - **Overloads for variants**: Tensor-tensor, tensor-scalar operations
        - **Consistent return types**: Usually `ttnn::Tensor` or `std::vector<ttnn::Tensor>`
        - **Namespace organization**: `ttnn::operations::<category>::<operation_name>`

        ### Parameter Types
        ```cpp
        // Common input parameter patterns
        const ttnn::Tensor& input_tensor                    // Input by const reference
        const std::vector<ttnn::Tensor>& input_tensors      // Multiple inputs
        float scalar_value                                   // Scalar parameters
        const std::optional<MemoryConfig>& memory_config    // Optional memory config
        const std::optional<ttnn::Shape>& output_shape      // Optional output shape
        ```

        IMPORTANT: Do not use the eltwise namespace, make this operation in the base operations namespace.
        Use "ttnn::decorators::register_operation" not "ttnn::register_operation_with_auto_launch_op"
        """
    + GLOBAL_CONTEXT
)

CPP_CONTEXT = (
    """
        ## Operation Implementation (.cpp) Context

        ### Purpose & Structure
        Operation implementations provide **validation, dispatch, and orchestration** logic.

        ### Standard Implementation Pattern
        ```cpp
        #include "operation_name.hpp"
        #include "device/operation_name_op.hpp"

        namespace ttnn::operations::operation_name {

        ttnn::Tensor operation_name(
            const ttnn::Tensor& input_a,
            const ttnn::Tensor& input_b,
            const std::optional<MemoryConfig>& memory_config,
            const std::optional<DeviceComputeKernelConfig>& compute_config) {

            // 1. Input validation
            TT_FATAL(input_a.storage_type() == StorageType::DEVICE, "Input A must be on device");
            TT_FATAL(input_b.storage_type() == StorageType::DEVICE, "Input B must be on device");
            TT_FATAL(input_a.device() == input_b.device(), "Inputs must be on same device");

            // 2. Shape compatibility validation
            auto output_shape = infer_output_shape(input_a.get_shape(), input_b.get_shape());

            // 3. Memory config resolution
            auto effective_memory_config = memory_config.value_or(input_a.memory_config());

            // 4. Compute config resolution
            auto effective_compute_config = compute_config.value_or(
                init_device_compute_kernel_config(input_a.device()->arch())
            );

            // 5. Dispatch to device operation
            return operation::run(
                OperationNameDeviceOperation{
                    .memory_config = effective_memory_config,
                    .compute_config = effective_compute_config
                },
                {input_a, input_b}
            ).at(0);
        }

        }  // namespace ttnn::operations::operation_name
        ```

        ### Key Implementation Responsibilities
        1. **Input Validation**: Device placement, shape compatibility, data type checks
        2. **Parameter Resolution**: Convert optionals to concrete values with sensible defaults
        3. **Shape Inference**: Calculate output tensor shapes based on broadcasting rules
        4. **Device Operation Dispatch**: Call the actual device operation implementation
        5. **Error Handling**: Provide clear error messages for invalid inputs

        ### Common Validation Patterns
        ```cpp
        // Device and storage validation
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Tensor must be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Tensor buffer cannot be null");

        // Shape validation
        TT_FATAL(tensor.get_shape().rank() >= 2, "Tensor must have at least 2 dimensions");
        TT_FATAL(tensor.volume() % tt::constants::TILE_HW == 0, "Tensor volume must be tile-aligned");

        // Broadcasting validation
        auto shapes_are_broadcastable = tt::tt_metal::compute_output_shapes(
            {input_a.get_shape(), input_b.get_shape()}
        );
        TT_FATAL(shapes_are_broadcastable.has_value(), "Input shapes are not broadcastable");

        // Data type validation
        TT_FATAL(input_a.get_dtype() == input_b.get_dtype(), "Input tensors must have same data type");
        ```

        ### Memory Config Resolution Pattern
        ```cpp
        // Use input tensor's memory config as default
        auto effective_memory_config = memory_config.value_or(input_tensor.memory_config());

        // For multi-input operations, choose based on priority
        if (!memory_config.has_value()) {
            if (input_a.is_sharded()) {
                effective_memory_config = input_a.memory_config();
            } else if (input_b.is_sharded()) {
                effective_memory_config = input_b.memory_config();
            } else {
                effective_memory_config = input_a.memory_config();
            }
        }
        ```
        IMPORTANT: Do not use the eltwise namespace, make this operation in the base operations namespace
        """
    + GLOBAL_CONTEXT
)

KERNEL_CONTEXT = (
    """
        # TT-Metal Kernel Development Context

        ## Hardware Architecture Overview

        ### Core Types and Capabilities
        - **Tensix Cores**: Primary compute cores with local L1 SRAM (1MB), Matrix Engine (32x32 tiles), SFPU
        - **Ethernet Cores**: For inter-chip communication
        - **PCIe Cores**: For host-device communication

        ### Memory Hierarchy
        ```
        Host DRAM → Device DRAM (12 channels) → L1 SRAM (1MB per core) → Register Files → Compute Units
        ```

        ### Key Constraints
        - **L1 SRAM**: 1MB per Tensix core, shared between data and program
        - **Tile Size**: 32x32 elements (fundamental compute unit)
        - **Data Types**: bfloat16, float32, int32, bfloat8, etc.
        - **Memory Banks**: 16 banks per core, avoid bank conflicts
        - **NOC Bandwidth**: Limited, explicit management required

        ## Programming Model

        ### Kernel Types
        1. **Compute Kernels**: Perform math operations on tiles in register files
        2. **Reader Kernels**: Move data from DRAM/other cores to local L1
        3. **Writer Kernels**: Move data from local L1 to DRAM/other cores

        ### Execution Flow
        ```
        Reader → CB (Circular Buffer) → Compute → CB → Writer
        ```

        ### Synchronization
        - **Circular Buffers (CB)**: Producer-consumer synchronization between kernels
        - **Semaphores**: For explicit synchronization
        - **Barriers**: Core-to-core coordination

        ## Common API Patterns

        ### Circular Buffer Operations
        ```cpp
        // Producer (Reader/Compute)
        cb_reserve_back(cb_id, num_tiles);
        // ... write data to reserved space ...
        cb_push_back(cb_id, num_tiles);

        // Consumer (Compute/Writer)
        cb_wait_front(cb_id, num_tiles);
        // ... read data from front ...
        cb_pop_front(cb_id, num_tiles);
        ```

        ### Tile Operations (Compute Kernels)
        ```cpp
        // Acquire tiles from CB to register file
        acquire_dst(dst_index);
        cb_wait_front(cb_input0, 1);
        cb_wait_front(cb_input1, 1);

        // Perform tile operation
        add_tiles(cb_input0, cb_input1, 0, 0, dst_index);

        // Release result back to CB
        cb_reserve_back(cb_output, 1);
        pack_tile(dst_index, cb_output);
        cb_push_back(cb_output, 1);
        release_dst(dst_index);
        ```

        ### Memory Access (Reader/Writer Kernels)
        ```cpp
        // Read from DRAM
        uint32_t src_addr = get_read_ptr(reader_id);
        noc_async_read(src_addr, dst_local_l1_addr, tile_bytes);
        noc_async_read_barrier();

        // Write to DRAM
        uint32_t dst_addr = get_write_ptr(writer_id);
        noc_async_write(src_local_l1_addr, dst_addr, tile_bytes);
        noc_async_write_barrier();
        ```

        ## Performance Guidelines

        ### Memory Access Optimization
        - **Sequential access patterns** are fastest
        - **Avoid bank conflicts** (stride access by 16*sizeof(element))
        - **Prefetch data** before compute operations
        - **Pipeline reads/writes** with compute

        ### Tile Operation Efficiency
        - **Maximize register file utilization** (8 tiles max)
        - **Minimize acquire/release overhead**
        - **Use tile broadcasting** when possible
        - **Batch operations** on multiple tiles

        ### Multi-Core Patterns
        - **Balance work** across cores evenly
        - **Minimize NOC traffic** between cores
        - **Use local L1** as much as possible before DRAM access
        - **Coordinate writers** to avoid memory conflicts

        ## Common Kernel Structures

        ### Compute Kernel Template
        ```cpp
        #include "compute_kernel_api.h"

        namespace NAMESPACE {
        void MAIN {
            uint32_t num_tiles = get_arg_val<uint32_t>(0);

            // Configure compute engine
            binary_op_init_common(cb_input0, cb_input1, cb_output);

            for (uint32_t i = 0; i < num_tiles; ++i) {
                // Acquire destination register
                acquire_dst(tt::CB::c_out0);

                // Wait for input tiles
                cb_wait_front(cb_input0, 1);
                cb_wait_front(cb_input1, 1);

                // Perform operation
                add_tiles(cb_input0, cb_input1, 0, 0, 0);

                // Write result
                cb_reserve_back(cb_output, 1);
                pack_tile(0, cb_output);
                cb_push_back(cb_output, 1);

                // Pop consumed inputs
                cb_pop_front(cb_input0, 1);
                cb_pop_front(cb_input1, 1);

                release_dst(tt::CB::c_out0);
            }
        }
        }
        ```

        ### Reader Kernel Template
        ```cpp
        #include "dataflow_api.h"

        void kernel_main() {
            uint32_t src_addr = get_arg_val<uint32_t>(0);
            uint32_t num_tiles = get_arg_val<uint32_t>(1);
            uint32_t tile_bytes = get_tile_size(cb_id_out0);

            uint32_t l1_write_addr = get_write_ptr(cb_id_out0);

            for (uint32_t i = 0; i < num_tiles; ++i) {
                cb_reserve_back(cb_id_out0, 1);

                noc_async_read(src_addr, l1_write_addr, tile_bytes);
                noc_async_read_barrier();

                cb_push_back(cb_id_out0, 1);

                src_addr += tile_bytes;
                l1_write_addr += tile_bytes;
            }
        }
        ```

        ## Error Patterns to Avoid

        ### Memory Management Issues
        - **Buffer overruns**: Always check CB capacity before reserve
        - **Synchronization deadlocks**: Ensure producer-consumer balance
        - **Memory alignment**: Use proper alignment for tile boundaries
        - **L1 overflow**: Monitor L1 usage, avoid exceeding 1MB

        ### Performance Anti-Patterns
        - **Excessive synchronization**: Minimize cb_wait_front calls
        - **Small batch sizes**: Process multiple tiles when possible
        - **Random memory access**: Use sequential patterns
        - **Redundant data movement**: Cache frequently used tiles

        ## Tile Layout Considerations

        ### TILE vs ROW_MAJOR Layouts
        ```cpp
        // TILE layout: 32x32 blocks optimized for matrix operations
        // ROW_MAJOR: Linear memory layout for element-wise ops

        // Converting between layouts may require untilize/tilize operations
        ```

        ### Broadcasting Patterns
        - **Scalar broadcasting**: Replicate single tile across operations
        - **Vector broadcasting**: Expand 1D to 2D tile operations
        - **Batch broadcasting**: Handle batch dimensions correctly

        ## Integration with TTNN Operations

        ### Operation Factory Pattern
        ```cpp
        // Program factory creates and configures kernels
        operation::ProgramWithCallbacks create_program(
            const std::vector<Tensor>& input_tensors,
            std::vector<Tensor>& output_tensors) {

            Program program = tt::tt_metal::CreateProgram();

            // Configure kernels based on tensor shapes/types
            auto compute_kernel = CreateKernel(program, compute_kernel_file, core_spec);
            auto reader_kernel = CreateKernel(program, reader_kernel_file, core_spec);
            auto writer_kernel = CreateKernel(program, writer_kernel_file, core_spec);

            // Set runtime arguments
            SetRuntimeArgs(compute_kernel, core, {num_tiles, ...});

            return {std::move(program), override_runtime_args_callback};
        }
        ```

        ## Debugging Tips

        ### Common Debug Patterns
        - **Add DPRINT statements** for kernel execution tracking
        - **Check CB states** with debug prints
        - **Verify memory addresses** and tile counts
        - **Use smaller test cases** to isolate issues

        ### Performance Profiling
        - **Measure kernel execution time** with profiler
        - **Check NOC utilization** for memory bottlenecks
        - **Analyze L1 usage** for capacity issues
        - **Compare with theoretical peak** performance
        """
    + GLOBAL_CONTEXT
)

PROGRAM_FACTORY_CONTEXT = (
    """
        # TT-Metal Program Factory Development Context

        ## Program Factory Role & Responsibilities

        ### Core Purpose
        Program factories are **orchestration engines** that:
        - **Create and configure programs** that coordinate multiple kernels
        - **Manage memory layouts** and circular buffer configurations
        - **Dispatch work** across single or multiple cores
        - **Handle runtime arguments** and tensor shape adaptations
        - **Optimize execution** based on tensor properties and hardware constraints

        ### Factory vs Kernel Distinction
        ```
        Program Factory: "What work needs to be done and how to organize it"
        Kernels: "How to actually do the computational work"
        ```

        ## Architecture Patterns

        ### Standard Factory Structure
        ```cpp
        tt::tt_metal::operation::ProgramWithCallbacks create_program(
            const std::vector<Tensor>& input_tensors,
            std::vector<Tensor>& output_tensors,
            const DeviceComputeKernelConfig& compute_kernel_config
        ) {
            // 1. Analyze tensors and determine execution strategy
            // 2. Create program and configure memory
            // 3. Add kernels with appropriate configurations
            // 4. Set up runtime argument dispatch
            // 5. Return program with callback for runtime args
        }
        ```

        ### Key Phases
        1. **Analysis Phase**: Understand tensor shapes, layouts, memory requirements
        2. **Strategy Selection**: Single-core vs multi-core, sharding patterns
        3. **Memory Planning**: CB configurations, L1 allocation, DRAM mapping
        4. **Kernel Configuration**: Core assignments, kernel arguments, synchronization
        5. **Runtime Dispatch**: Dynamic argument calculation and kernel launch

        ## Memory Management Patterns

        ### Circular Buffer Configuration
        ```cpp
        // Input circular buffers
        CircularBufferConfig cb_input0_config = CircularBufferConfig(
            input0_cb_size * tile_size,     // Size in bytes
            {{cb_input0, tt::DataFormat::Float16_b}}  // CB ID and data format
        ).set_page_size(cb_input0, tile_size);

        // Output circular buffers
        CircularBufferConfig cb_output_config = CircularBufferConfig(
            output_cb_size * tile_size,
            {{cb_output, tt::DataFormat::Float16_b}}
        ).set_page_size(cb_output, tile_size);
        ```

        ### Memory Configuration Strategy
        ```cpp
        // Determine optimal CB sizes based on:
        // 1. Available L1 memory per core
        // 2. Number of concurrent operations
        // 3. Pipeline depth requirements
        // 4. Tensor tile counts

        uint32_t available_l1 = L1_MEMORY_SIZE - kernel_code_size;
        uint32_t num_cbs = input_tensors.size() + output_tensors.size();
        uint32_t cb_size_per_buffer = available_l1 / (num_cbs * safety_margin);
        ```

        ## Core Assignment Strategies

        ### Single-Core Operations
        ```cpp
        // Simple operations that fit on one core
        CoreCoord core = {0, 0};
        auto compute_kernel_id = CreateKernel(
            program,
            compute_kernel_file,
            core,
            ComputeKernelConfig{}
        );
        ```

        ### Multi-Core Patterns

        #### 1. Data Parallel (Same operation on different data)
        ```cpp
        // Distribute tiles across cores
        auto core_grid = compute_kernel_config.core_grid;
        uint32_t num_cores = core_grid.num_cores();
        uint32_t tiles_per_core = total_tiles / num_cores;

        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            CoreCoord core = core_grid.core_coord(core_idx);
            auto kernel_id = CreateKernel(program, kernel_file, core, config);

            // Each core processes different tile range
            uint32_t start_tile = core_idx * tiles_per_core;
            uint32_t end_tile = std::min(start_tile + tiles_per_core, total_tiles);
        }
        ```

        #### 2. Pipeline Parallel (Different stages on different cores)
        ```cpp
        // Reader cores -> Compute cores -> Writer cores
        auto reader_cores = get_reader_cores(core_grid);
        auto compute_cores = get_compute_cores(core_grid);
        auto writer_cores = get_writer_cores(core_grid);

        // Configure reader kernels
        for (auto& core : reader_cores) {
            auto reader_kernel = CreateKernel(program, reader_file, core, config);
        }

        // Configure compute kernels
        for (auto& core : compute_cores) {
            auto compute_kernel = CreateKernel(program, compute_file, core, config);
        }
        ```

        #### 3. Spatial Decomposition (2D grid operations)
        ```cpp
        // Distribute 2D tensor across 2D core grid
        uint32_t cores_h = core_grid.y;
        uint32_t cores_w = core_grid.x;
        uint32_t tiles_per_core_h = tensor_height_tiles / cores_h;
        uint32_t tiles_per_core_w = tensor_width_tiles / cores_w;

        for (uint32_t core_y = 0; core_y < cores_h; ++core_y) {
            for (uint32_t core_x = 0; core_x < cores_w; ++core_x) {
                CoreCoord core = {core_x, core_y};

                // Each core gets a rectangular tile region
                uint32_t start_h = core_y * tiles_per_core_h;
                uint32_t start_w = core_x * tiles_per_core_w;
            }
        }
        ```

        ## Runtime Argument Management

        ### Static vs Dynamic Arguments
        ```cpp
        // Static arguments (set once at program creation)
        vector<uint32_t> static_args = {
            operation_type,
            tensor_format,
            tile_height,
            tile_width
        };

        // Dynamic arguments (calculated per execution)
        auto runtime_args_callback = [=](
            const operation::operation_attributes_t& operation_attributes,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors
        ) {
            // Calculate arguments based on actual tensor properties
            vector<vector<uint32_t>> runtime_args;

            for (const auto& tensor : input_tensors) {
                uint32_t tensor_addr = tensor.buffer()->address();
                uint32_t tensor_tiles = tensor.volume() / TILE_HW;

                runtime_args.push_back({tensor_addr, tensor_tiles, ...});
            }

            return runtime_args;
        };
        ```

        ### Argument Calculation Patterns
        ```cpp
        // Common argument calculations
        uint32_t src_addr = input_tensor.buffer()->address();
        uint32_t dst_addr = output_tensor.buffer()->address();

        // Tile counting with proper alignment
        uint32_t num_tiles_h = (tensor_height + TILE_HEIGHT - 1) / TILE_HEIGHT;
        uint32_t num_tiles_w = (tensor_width + TILE_WIDTH - 1) / TILE_WIDTH;
        uint32_t total_tiles = num_tiles_h * num_tiles_w;

        // Stride calculations for non-contiguous access
        uint32_t row_stride = tensor_width * element_size;
        uint32_t page_stride = tensor_height * row_stride;

        // Core-specific work distribution
        uint32_t tiles_per_core = total_tiles / num_cores;
        uint32_t start_tile_idx = core_idx * tiles_per_core;
        uint32_t end_tile_idx = (core_idx == num_cores - 1) ?
            total_tiles : (core_idx + 1) * tiles_per_core;
        ```

        ## Tensor Layout & Sharding Strategies

        ### Layout Transformation Planning
        ```cpp
        // Analyze input/output layout requirements
        Layout input_layout = input_tensor.get_layout();
        Layout output_layout = output_tensor.get_layout();

        if (input_layout == Layout::ROW_MAJOR && requires_tile_layout) {
            // Need tilize operation before main compute
            add_tilize_kernel(program, input_cores);
        }

        if (output_layout == Layout::ROW_MAJOR && produces_tile_layout) {
            // Need untilize operation after main compute
            add_untilize_kernel(program, output_cores);
        }
        ```

        ### Sharding Pattern Selection
        ```cpp
        // Choose sharding based on operation characteristics
        TensorMemoryLayout memory_layout = input_tensor.memory_config().memory_layout;

        switch (memory_layout) {
            case TensorMemoryLayout::INTERLEAVED:
                // Standard single-core or simple multi-core
                return create_interleaved_program(...);

            case TensorMemoryLayout::HEIGHT_SHARDED:
                // Optimize for row-wise operations
                return create_height_sharded_program(...);

            case TensorMemoryLayout::WIDTH_SHARDED:
                // Optimize for column-wise operations
                return create_width_sharded_program(...);

            case TensorMemoryLayout::BLOCK_SHARDED:
                // 2D decomposition for matrix operations
                return create_block_sharded_program(...);
        }
        ```

        ## Performance Optimization Patterns

        ### Work Distribution Optimization
        ```cpp
        // Balance work across cores considering:
        // 1. Total work amount
        // 2. Core capabilities
        // 3. Memory bandwidth
        // 4. Synchronization overhead

        uint32_t optimal_cores = std::min(
            max_cores_available,
            std::max(1u, total_work / min_work_per_core)
        );

        // Adjust for memory bandwidth limitations
        if (memory_bound_operation) {
            optimal_cores = std::min(optimal_cores, memory_bandwidth_limit);
        }

        // Ensure work granularity is tile-aligned
        uint32_t work_per_core = (total_tiles + optimal_cores - 1) / optimal_cores;
        work_per_core = round_up_to_tile_boundary(work_per_core);
        ```

        ### Memory Access Optimization
        ```cpp
        // Configure memory access patterns for optimal performance
        if (is_sequential_access(operation_type)) {
            // Use large CB sizes for better pipelining
            cb_size = std::min(available_l1 / 4, max_cb_size);
        } else if (is_random_access(operation_type)) {
            // Use smaller CB sizes, more cores
            cb_size = std::min(available_l1 / 8, tile_size * 4);
            preferred_cores = max_cores_available;
        }

        // DRAM bank conflict avoidance
        if (multiple_input_tensors) {
            ensure_bank_interleaving(input_tensor_addresses);
        }
        ```

        ## Error Handling & Validation

        ### Input Validation Patterns
        ```cpp
        // Comprehensive tensor validation
        void validate_inputs(const std::vector<Tensor>& inputs) {
            TT_FATAL(inputs.size() >= required_inputs, "Insufficient input tensors");

            for (const auto& tensor : inputs) {
                TT_FATAL(tensor.storage_type() == StorageType::DEVICE,
                        "Input tensors must be on device");
                TT_FATAL(tensor.buffer() != nullptr,
                        "Input tensor buffer is null");
                TT_FATAL(tensor.volume() % TILE_HW == 0,
                        "Input tensor volume must be tile-aligned");
            }

            // Operation-specific validations
            if (requires_same_shape) {
                validate_matching_shapes(inputs);
            }
            if (requires_broadcastable) {
                validate_broadcastable_shapes(inputs);
            }
        }
        ```

        ### Resource Availability Checks
        ```cpp
        // Verify sufficient resources before program creation
        void validate_resources(const operation_config& config) {
            uint32_t required_l1 = calculate_l1_requirements(config);
            uint32_t available_l1 = get_available_l1_per_core();

            TT_FATAL(required_l1 <= available_l1,
                    "Operation requires {} bytes L1, only {} available",
                    required_l1, available_l1);

            uint32_t required_cores = config.core_grid.num_cores();
            uint32_t available_cores = get_available_cores();

            TT_FATAL(required_cores <= available_cores,
                    "Operation requires {} cores, only {} available",
                    required_cores, available_cores);
        }
        ```

        ## Integration with TTNN Operation Framework

        ### Factory Registration Pattern
        ```cpp
        // Register factory with operation dispatcher
        namespace ttnn::operations::operation_name {

        operation::ProgramWithCallbacks create_operation_name_program(
            const std::vector<Tensor>& input_tensors,
            std::vector<Tensor>& output_tensors,
            const DeviceComputeKernelConfig& compute_kernel_config) {

            // Select appropriate strategy based on inputs
            if (is_single_core_optimal(input_tensors, compute_kernel_config)) {
                return create_single_core_program(...);
            } else if (is_height_sharded_optimal(...)) {
                return create_height_sharded_program(...);
            } else {
                return create_multi_core_program(...);
            }
        }

        }  // namespace ttnn::operations::operation_name
        ```

        ### Operation Attribute Handling
        ```cpp
        // Handle optional operation attributes
        operation::ProgramWithCallbacks create_program(
            const std::vector<Tensor>& input_tensors,
            std::vector<Tensor>& output_tensors,
            const DeviceComputeKernelConfig& compute_kernel_config,
            const std::optional<MemoryConfig>& memory_config = std::nullopt,
            const std::optional<DeviceComputeKernelConfig>& compute_config = std::nullopt
        ) {
            // Use provided configs or derive optimal defaults
            auto effective_memory_config = memory_config.value_or(
                get_optimal_memory_config(input_tensors)
            );

            auto effective_compute_config = compute_config.value_or(
                get_optimal_compute_config(input_tensors)
            );

            return create_program_with_config(
                input_tensors, output_tensors,
                effective_memory_config, effective_compute_config
            );
        }
        ```

        ## Common Factory Patterns by Operation Type

        ### Element-wise Binary Operations
        ```cpp
        // Pattern: Two inputs, one output, element-wise processing
        operation::ProgramWithCallbacks create_eltwise_binary_program(...) {
            Program program = CreateProgram();

            // Simple single-core for small tensors
            if (total_tiles < single_core_threshold) {
                auto core = CoreCoord{0, 0};

                // Reader for input A
                auto reader_a = CreateKernel(program, "reader_binary_a.cpp", core, ...);
                // Reader for input B
                auto reader_b = CreateKernel(program, "reader_binary_b.cpp", core, ...);
                // Compute kernel
                auto compute = CreateKernel(program, "eltwise_compute.cpp", core, ...);
                // Writer for output
                auto writer = CreateKernel(program, "writer_unary.cpp", core, ...);
            } else {
                // Multi-core data parallel
                distribute_work_across_cores(program, core_grid, ...);
            }

            return {std::move(program), runtime_args_callback};
        }
        ```

        ### Reduction Operations
        ```cpp
        // Pattern: One input, one output, reduction across dimensions
        operation::ProgramWithCallbacks create_reduction_program(...) {
            Program program = CreateProgram();

            if (reduction_dim == last_dim) {
                // Row-wise reduction: height-sharded optimal
                return create_row_reduction_program(...);
            } else if (reduction_dim == first_dim) {
                // Column-wise reduction: width-sharded optimal
                return create_col_reduction_program(...);
            } else {
                // General reduction: complex multi-stage
                return create_general_reduction_program(...);
            }
        }
        ```

        ### Matrix Operations
        ```cpp
        // Pattern: Two inputs, one output, matrix multiplication
        operation::ProgramWithCallbacks create_matmul_program(...) {
            Program program = CreateProgram();

            // Block-sharded for large matrices
            if (is_large_matrix(input_a, input_b)) {
                return create_block_sharded_matmul(...);
            }
            // Single-core for small matrices
            else if (is_small_matrix(input_a, input_b)) {
                return create_single_core_matmul(...);
            }
            // Height-sharded for tall-skinny matrices
            else {
                return create_height_sharded_matmul(...);
            }
        }
        ```

        ## Debugging & Performance Analysis

        ### Debug Information Injection
        ```cpp
        // Add debug kernels for development
        #ifdef TT_METAL_DEBUG
            auto debug_kernel = CreateKernel(
                program, "debug_print.cpp", debug_core,
                DataMovementConfig{.compile_args = {tensor_info, ...}}
            );
        #endif

        // Runtime debug argument injection
        auto debug_callback = [](auto& args) {
            args[0].push_back(debug_tensor_addr);
            args[0].push_back(debug_tile_count);
            return args;
        };
        ```

        ### Performance Monitoring Hooks
        ```cpp
        // Instrument program for performance analysis
        operation::ProgramWithCallbacks create_instrumented_program(...) {
            auto program_with_callbacks = create_base_program(...);

            // Wrap callback with timing/profiling
            auto instrumented_callback = [base_callback = program_with_callbacks.program_callback](
                const operation::operation_attributes_t& attrs,
                const std::vector<Tensor>& inputs,
                const std::vector<std::optional<const Tensor>>& optional_inputs,
                const std::vector<Tensor>& outputs) {

                auto start_time = std::chrono::high_resolution_clock::now();
                auto result = base_callback(attrs, inputs, optional_inputs, outputs);
                auto end_time = std::chrono::high_resolution_clock::now();

                log_performance_metrics(operation_name, start_time, end_time);
                return result;
            };

            return {std::move(program_with_callbacks.program), instrumented_callback};
        }
        ```

        ## Most Challenging Aspects for AI Understanding

        ### 1. **Strategy Selection Logic**
        - When to use single-core vs multi-core vs sharded approaches
        - How tensor properties influence optimal execution strategy
        - Trade-offs between parallelism and synchronization overhead

        ### 2. **Memory Planning Complexity**
        - L1 memory budget allocation across multiple CBs and kernels
        - DRAM access pattern optimization for different tensor layouts
        - Pipeline depth tuning for optimal throughput

        ### 3. **Core Assignment Optimization**
        - Load balancing across heterogeneous workloads
        - NOC traffic minimization in multi-core scenarios
        - Hardware-specific core capability matching

        ### 4. **Runtime Argument Calculation**
        - Address arithmetic for various tensor layouts and strides
        - Tile-boundary alignment and padding handling
        - Dynamic work distribution based on actual tensor sizes

        ### 5. **Error Recovery and Fallback Strategies**
        - Graceful degradation when optimal strategy isn't feasible
        - Resource constraint adaptation (limited cores, L1 memory)
        - Tensor format compatibility handling

        IMPORTANT: The kernels called in the program factory must be the readers and writers generated, not any existing kernels.
        This means the correct kernels are named `{operation_name}_reader.cpp`, `<operation_name>_writer.cpp`, and `<operation_name>_compute.cpp`.
        """
    + GLOBAL_CONTEXT
)

PYBIND_CONTEXT = (
    """
        ---
        ## Python Bindings Context

        ### Purpose & Structure
        Python bindings expose C++ operations to Python using **pybind11** with proper type conversion and documentation.
        It is essential to ensure the operation is properly exposed to Python for integration with the rest of the TTNN ecosystem,
        carefully study the current state of the TTNN repository to ensure the operation is compatible with the structure.

        ### Standard Binding Pattern
        ```cpp
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        #include "ttnn/cpp/ttnn-pybind/decorators.hpp"
        #include "operation_name.hpp"

        namespace py = pybind11;

        namespace ttnn::operations {

        void bind_operation_name(py::module& module) {
            auto doc = R"doc(
                Performs operation

                Args:
                    input_a (ttnn.Tensor): First input tensor
                    input_b (ttnn.Tensor): Second input tensor
                    memory_config (Optional[ttnn.MemoryConfig]): Memory configuration for output
                    compute_config (Optional[ttnn.DeviceComputeKernelConfig]): Compute configuration

                Returns:
                    ttnn.Tensor: Output tensor with operation result

                Example:
                    >>> import ttnn
                    >>> a = ttnn.from_torch(torch.randn(2, 3), device=device)
                    >>> b = ttnn.from_torch(torch.randn(2, 3), device=device)
                    >>> result = ttnn.operations.operation_name(a, b)
            )doc";

            module.def(
                "operation_name",
                py::overload_cast<
                    const ttnn::Tensor&,
                    const ttnn::Tensor&,
                    const std::optional<MemoryConfig>&,
                    const std::optional<DeviceComputeKernelConfig>&
                >(&operation_name),
                py::arg("input_a"),
                py::arg("input_b"),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                py::arg("compute_config") = std::nullopt,
                doc
            );

            // Scalar variant binding
            module.def(
                "operation_name",
                py::overload_cast<
                    const ttnn::Tensor&,
                    float,
                    const std::optional<MemoryConfig>&
                >(&operation_name),
                py::arg("input_a"),
                py::arg("scalar_b"),
                py::kw_only(),
                py::arg("memory_config") = std::nullopt,
                R"doc(Scalar variant of operation_name)doc"
            );
        }

                }  // namespace ttnn::operations
                ```

                ### Key Binding Patterns
                - **Overload resolution**: Use `py::overload_cast` for function overloads
                - **Keyword arguments**: Use `py::kw_only()` to make optional params keyword-only
                - **Default values**: Match C++ defaults with `= std::nullopt`
                - **Documentation**: Include comprehensive docstrings with examples
                - **Type conversion**: pybind11 handles automatic conversion for standard types

                ### Parameter Binding Patterns
                ```cpp
                // Simple parameters
                py::arg("input_tensor")

                // Optional parameters with defaults
                py::arg("memory_config") = std::nullopt
                py::arg("compute_config") = std::nullopt

                // Keyword-only parameters (Python 3 style)
                py::kw_only(),
                py::arg("optional_param") = default_value

                // Parameters with validation
                py::arg("axis").noconvert()  // Don't allow implicit conversion

                // Complex type parameters
                py::arg("shape") = std::optional<ttnn::Shape>{}
                ```

                ### Common Type Conversions
                ```cpp
                // Automatic conversions handled by pybind11:
                // - std::vector<T> ↔ Python list
                // - std::optional<T> ↔ Python None/value
                // - ttnn::Tensor ↔ Python ttnn.Tensor object
                // - std::string ↔ Python str
                // - Enums ↔ Python enum values

                // Manual conversions when needed:
                .def("operation", [](const py::object& input) {
                    if (py::isinstance<ttnn::Tensor>(input)) {
                        return operation(input.cast<ttnn::Tensor>());
                    } else {
                        return operation(input.cast<float>());
                    }
                })
                ```

                ## Integration Requirements

                ### Header Dependencies
                ```cpp
                // operation.hpp
                #include "ttnn/decorators.hpp"
                #include "ttnn/operations/core/core.hpp"
                #include "ttnn/tensor/tensor.hpp"

                // operation.cpp
                #include "operation.hpp"
                #include "device/operation_op.hpp"
                #include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

                // pybind
                #include <pybind11/pybind11.h>
                #include <pybind11/stl.h>
                #include "ttnn/cpp/ttnn-pybind/decorators.hpp"
                ```

                ### Namespace Consistency
                ```cpp
                // All files must use consistent namespace
                namespace ttnn::operations:: {
                    // Operation implementation
                }

                // Python module structure must match
                // ttnn.operations.operation_name()
                ```

                ### Build System Integration
                - **Headers**: Automatically included via directory scanning
                - **Implementation**: Added to CMakeLists.txt source lists
                - **Python bindings**: Registered in main pybind module file

                REMEMBER: the correct path is ttnn/cpp/ttnn-pybind/decorators.hpp NOT ttnn/cpp/pybind11/decorators.hpp
                ❌ NEVER use: #include "ttnn/cpp/ttnn-pybind/decorators.hpp"
                ✅ USE: #include "ttnn-pybind/decorators.hpp"

                The CMake include paths already point to the cpp directory.
                IMPORTANT: Make this operation in the operations namespace, not eltwise.
                """
    + GLOBAL_CONTEXT
)

DEVICE_OP_CONTEXT = (
    """
        # TTNN Device Operation Development Context

        ## Device Operation Header (.hpp) Context

        ### Purpose & Structure
        Device operation headers define the **device-side operation class** that implements the low-level operation interface and manages program execution.

        ### Standard Device Operation Header Pattern
        ```cpp
        #pragma once

        #include "ttnn/operation.hpp"
        #include "ttnn/tensor/tensor.hpp"
        #include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

        namespace ttnn::operations::operation_name {

        struct OperationNameDeviceOperation {
            // Operation configuration
            MemoryConfig memory_config;
            DeviceComputeKernelConfig compute_config;

            // Optional operation-specific parameters
            bool in_place = false;
            std::optional<DataType> output_dtype = std::nullopt;

            // Core operation interface methods
            void validate(const std::vector<Tensor>& input_tensors) const;
            std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
            std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
            operation::ProgramWithCallbacks create_program(
                const std::vector<Tensor>& input_tensors,
                std::vector<Tensor>& output_tensors
            ) const;
        };

        }  // namespace ttnn::operations::operation_name
        ```

        ### Operation Interface Requirements
        ```cpp
        // Required methods for all device operations
        class DeviceOperation {
        public:
            // Validate input tensors and operation parameters
            virtual void validate(const std::vector<Tensor>& input_tensors) const = 0;

            // Compute output tensor specifications (shape, dtype, layout)
            virtual std::vector<TensorSpec> compute_output_specs(
                const std::vector<Tensor>& input_tensors
            ) const = 0;

            // Create actual output tensors on device
            virtual std::vector<Tensor> create_output_tensors(
                const std::vector<Tensor>& input_tensors
            ) const = 0;

            // Create program with kernels and runtime callbacks
            virtual operation::ProgramWithCallbacks create_program(
                const std::vector<Tensor>& input_tensors,
                std::vector<Tensor>& output_tensors
            ) const = 0;
        };
        ```

        ### Configuration Struct Patterns
        ```cpp
        // Simple configuration (for basic operations)
        struct SimpleDeviceOperation {
            MemoryConfig memory_config;
            DeviceComputeKernelConfig compute_config;
        };

        // Complex configuration (for advanced operations)
        struct ComplexDeviceOperation {
            MemoryConfig memory_config;
            DeviceComputeKernelConfig compute_config;

            // Operation-specific parameters
            uint32_t dim = 0;
            bool keepdim = false;
            float scalar_value = 1.0f;
            std::optional<DataType> output_dtype = std::nullopt;
            std::optional<ttnn::Shape> output_shape = std::nullopt;

            // Advanced options
            bool in_place = false;
            std::optional<CoreGrid> core_grid = std::nullopt;
            std::optional<ShardStrategy> shard_strategy = std::nullopt;
        };
        ```

        ---

        ## Device Operation Implementation (.cpp) Context

        ### Purpose & Structure
        Device operation implementations provide **device-specific validation, tensor creation, and program orchestration**.

        ### Standard Implementation Pattern
        ```cpp
        #include "operation_name_op.hpp"
        #include "operation_name_program_factory.hpp"
        #include "ttnn/tensor/tensor_utils.hpp"

        namespace ttnn::operations::operation_name {

        void OperationNameDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
            // 1. Basic tensor validation
            TT_FATAL(input_tensors.size() == 2, "Operation requires exactly 2 input tensors");

            const auto& input_a = input_tensors[0];
            const auto& input_b = input_tensors[1];

            // 2. Device placement validation
            TT_FATAL(input_a.storage_type() == StorageType::DEVICE, "Input A must be on device");
            TT_FATAL(input_b.storage_type() == StorageType::DEVICE, "Input B must be on device");
            TT_FATAL(input_a.device() == input_b.device(), "Inputs must be on same device");

            // 3. Shape and layout validation
            TT_FATAL(input_a.get_layout() == input_b.get_layout(), "Inputs must have same layout");

            // Broadcasting validation
            auto output_shape = infer_dims_for_broadcast(input_a.get_shape(), input_b.get_shape());
            TT_FATAL(output_shape.has_value(), "Input shapes are not broadcastable");

            // 4. Memory layout validation
            if (input_a.is_sharded()) {
                TT_FATAL(input_a.shard_spec().has_value(), "Sharded tensor must have shard spec");
                validate_shard_compatibility(input_a, input_b);
            }

            // 5. Data type validation
            TT_FATAL(input_a.get_dtype() == input_b.get_dtype(), "Inputs must have same data type");

            // 6. Operation-specific validation
            validate_operation_constraints(input_a, input_b);
        }

        std::vector<TensorSpec> OperationNameDeviceOperation::compute_output_specs(
            const std::vector<Tensor>& input_tensors) const {

            const auto& input_a = input_tensors[0];
            const auto& input_b = input_tensors[1];

            // 1. Compute output shape (handle broadcasting)
            auto output_shape = infer_dims_for_broadcast(input_a.get_shape(), input_b.get_shape()).value();

            // 2. Determine output data type
            auto output_dtype = this->output_dtype.value_or(input_a.get_dtype());

            // 3. Determine output layout
            auto output_layout = input_a.get_layout();  // Usually preserve input layout

            // 4. Create tensor spec
            return {TensorSpec(
                output_shape,
                TensorLayout::fromPaddedShape(
                    output_dtype,
                    PageConfig(output_layout),
                    this->memory_config,
                    output_shape
                )
            )};
        }

        std::vector<Tensor> OperationNameDeviceOperation::create_output_tensors(
            const std::vector<Tensor>& input_tensors) const {

            const auto& input_a = input_tensors[0];
            auto output_specs = this->compute_output_specs(input_tensors);

            return operation::generic_create_output_tensors(
                *this,
                input_tensors,
                output_specs[0].tensor_layout.get_data_type(),
                output_specs[0].tensor_layout.get_layout(),
                this->memory_config
            );
        }

        operation::ProgramWithCallbacks OperationNameDeviceOperation::create_program(
            const std::vector<Tensor>& input_tensors,
            std::vector<Tensor>& output_tensors) const {

            // Delegate to program factory
            return detail::create_operation_name_program(
                input_tensors,
                output_tensors,
                this->compute_config
            );
        }

        }  // namespace ttnn::operations::operation_name
        ```

        ### Key Implementation Responsibilities

        #### 1. **Comprehensive Validation**
        ```cpp
        void validate_operation_constraints(const Tensor& input_a, const Tensor& input_b) {
            // Tile alignment for tiled operations
            if (input_a.get_layout() == Layout::TILE) {
                TT_FATAL(input_a.volume() % tt::constants::TILE_HW == 0,
                        "Tiled tensor volume must be tile-aligned");
            }

            // Memory layout compatibility
            if (input_a.is_sharded() && input_b.is_sharded()) {
                TT_FATAL(input_a.shard_spec() == input_b.shard_spec(),
                        "Sharded inputs must have compatible shard specs");
            }

            // Device capability validation
            auto device = input_a.device();
            TT_FATAL(device->arch() != tt::ARCH::GRAYSKULL || supports_grayskull(),
                    "Operation not supported on Grayskull");
        }
        ```

        #### 2. **Shape Inference Logic**
        ```cpp
        ttnn::Shape compute_output_shape(const Tensor& input_a, const Tensor& input_b) {
            auto shape_a = input_a.get_shape();
            auto shape_b = input_b.get_shape();

            // Element-wise operations: broadcasting rules
            auto broadcasted_shape = infer_dims_for_broadcast(shape_a, shape_b);
            TT_FATAL(broadcasted_shape.has_value(), "Shapes not broadcastable");

            return broadcasted_shape.value();
        }

        ttnn::Shape compute_reduction_output_shape(const Tensor& input, int64_t dim, bool keepdim) {
            auto input_shape = input.get_shape();
            auto output_shape = input_shape;

            if (keepdim) {
                output_shape[dim] = 1;
            } else {
                output_shape.erase(output_shape.begin() + dim);
            }

            return output_shape;
        }
        ```

        #### 3. **Memory Configuration Management**
        ```cpp
        MemoryConfig resolve_output_memory_config(
            const std::vector<Tensor>& inputs,
            const MemoryConfig& requested_config) {

            // Use requested config if specified
            if (requested_config.memory_layout != TensorMemoryLayout::INTERLEAVED ||
                requested_config.buffer_type != BufferType::DRAM) {
                return requested_config;
            }

            // Inherit from primary input
            const auto& primary_input = inputs[0];
            if (primary_input.is_sharded()) {
                return primary_input.memory_config();
            }

            // Default to interleaved DRAM
            return MemoryConfig{
                .memory_layout = TensorMemoryLayout::INTERLEAVED,
                .buffer_type = BufferType::DRAM
            };
        }
        ```

        #### 4. **Tensor Creation Patterns**
        ```cpp
        // Standard tensor creation
        std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& inputs) const {
            auto output_specs = compute_output_specs(inputs);

            return operation::generic_create_output_tensors(
                *this,
                inputs,
                output_specs[0].tensor_layout.get_data_type(),
                output_specs[0].tensor_layout.get_layout(),
                this->memory_config
            );
        }

        // In-place tensor modification
        std::vector<Tensor> create_output_tensors_inplace(const std::vector<Tensor>& inputs) const {
            if (this->in_place) {
                // Modify input tensor in-place
                return {inputs[0]};
            } else {
                // Create new output tensor
                return create_output_tensors(inputs);
            }
        }

        // Custom tensor creation with specific properties
        std::vector<Tensor> create_custom_output_tensors(const std::vector<Tensor>& inputs) const {
            auto device = inputs[0].device();
            auto output_shape = compute_output_shape(inputs[0], inputs[1]);
            auto output_dtype = this->output_dtype.value_or(inputs[0].get_dtype());

            auto output_tensor = create_device_tensor(
                output_shape,
                output_dtype,
                Layout::TILE,
                device,
                this->memory_config
            );

            return {output_tensor};
        }
        ```

        ### Common Device Operation Patterns

        #### **Binary Element-wise Operations**
        ```cpp
        struct BinaryEltwiseDeviceOperation {
            MemoryConfig memory_config;
            DeviceComputeKernelConfig compute_config;

            void validate(const std::vector<Tensor>& inputs) const {
                TT_FATAL(inputs.size() == 2, "Binary operation requires 2 inputs");
                validate_binary_inputs(inputs[0], inputs[1]);
            }

            std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& inputs) const {
                auto output_shape = infer_dims_for_broadcast(inputs[0].get_shape(), inputs[1].get_shape()).value();
                return create_output_spec(output_shape, inputs[0].get_dtype());
            }
        };
        ```

        #### **Unary Operations**
        ```cpp
        struct UnaryDeviceOperation {
            MemoryConfig memory_config;
            DeviceComputeKernelConfig compute_config;
            bool in_place = false;

            void validate(const std::vector<Tensor>& inputs) const {
                TT_FATAL(inputs.size() == 1, "Unary operation requires 1 input");
                validate_device_tensor(inputs[0]);
            }

            std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& inputs) const {
                // Output has same shape and dtype as input
                return create_output_spec(inputs[0].get_shape(), inputs[0].get_dtype());
            }
        };
        ```

        #### **Reduction Operations**
        ```cpp
        struct ReductionDeviceOperation {
            MemoryConfig memory_config;
            DeviceComputeKernelConfig compute_config;
            int64_t dim;
            bool keepdim = false;

            void validate(const std::vector<Tensor>& inputs) const {
                TT_FATAL(inputs.size() == 1, "Reduction requires 1 input");
                TT_FATAL(this->dim >= 0 && this->dim < inputs[0].get_shape().rank(),
                        "Dimension out of range");
            }

            std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& inputs) const {
                auto output_shape = compute_reduction_shape(inputs[0].get_shape(), this->dim, this->keepdim);
                return create_output_spec(output_shape, inputs[0].get_dtype());
            }
        };
        ```

        ### Advanced Device Operation Features

        #### **Sharding Strategy Integration**
        ```cpp
        void validate_with_sharding(const std::vector<Tensor>& inputs) const {
            if (inputs[0].is_sharded()) {
                auto shard_spec = inputs[0].shard_spec().value();

                // Validate shard compatibility with operation
                TT_FATAL(this->compute_config.core_grid.fits_within(shard_spec.grid),
                        "Core grid must fit within shard grid");

                // Validate shard orientation for operation type
                if (is_height_sharded_optimal()) {
                    TT_FATAL(shard_spec.orientation == ShardOrientation::ROW_MAJOR,
                            "Height-sharded operation requires row-major shard orientation");
                }
            }
        }
        ```

        #### **Multi-Device Support**
        ```cpp
        void validate_multi_device(const std::vector<Tensor>& inputs) const {
            if (inputs.size() > 1) {
                auto primary_device = inputs[0].device();
                for (size_t i = 1; i < inputs.size(); ++i) {
                    TT_FATAL(inputs[i].device() == primary_device,
                            "All input tensors must be on the same device");
                }
            }
        }
        ```

        #### **Performance Optimization Hints**
        ```cpp
        operation::ProgramWithCallbacks create_optimized_program(
            const std::vector<Tensor>& inputs,
            std::vector<Tensor>& outputs) const {

            // Choose program factory based on tensor properties
            if (is_small_tensor(inputs[0])) {
                return create_single_core_program(inputs, outputs, this->compute_config);
            } else if (inputs[0].is_sharded()) {
                return create_sharded_program(inputs, outputs, this->compute_config);
            } else {
                return create_multi_core_program(inputs, outputs, this->compute_config);
            }
        }
        ```

        ## Integration with Operation Lifecycle

        ### **Operation Dispatch Flow**
        ```cpp
        // 1. User calls operation (operation.cpp)
        ttnn::Tensor operation_name(args...) {
            return operation::run(
                OperationNameDeviceOperation{config...},
                {input_tensors...}
            ).at(0);
        }

        // 2. Framework calls device operation methods
        operation::run() {
            op.validate(inputs);                    // Validate inputs
            auto specs = op.compute_output_specs(inputs);  // Compute output specs
            auto outputs = op.create_output_tensors(inputs);  // Create output tensors
            auto program = op.create_program(inputs, outputs);  // Create program
            // Execute program...
        }
        ```

        ### **Error Handling Integration**
        ```cpp
        // Device operations should provide clear error context
        #define TT_FATAL_OP(condition, message, ...) \
            TT_FATAL(condition, "Operation '{}': " message, get_operation_name(), ##__VA_ARGS__)

        void validate(const std::vector<Tensor>& inputs) const {
            TT_FATAL_OP(inputs.size() == expected_inputs(),
                        "Expected {} inputs, got {}", expected_inputs(), inputs.size());

            TT_FATAL_OP(inputs[0].storage_type() == StorageType::DEVICE,
                        "Input tensor must be on device, got storage type {}",
                        inputs[0].storage_type());
        }
        ```

        ### **Program Factory Integration**
        ```cpp
        // Device operation delegates to program factory
        operation::ProgramWithCallbacks create_program(
            const std::vector<Tensor>& inputs,
            std::vector<Tensor>& outputs) const {

            // Pass device operation config to program factory
            return program_factory::create_operation_program(
                inputs,
                outputs,
                this->compute_config,
                this->memory_config,
                get_operation_attributes()  // Additional operation-specific config
            );
        }
        ```

        ## Common Validation Patterns

        ### **Device and Storage Validation**
        ```cpp
        void validate_device_tensors(const std::vector<Tensor>& tensors) const {
            for (const auto& tensor : tensors) {
                TT_FATAL(tensor.storage_type() == StorageType::DEVICE,
                        "Tensor must be on device");
                TT_FATAL(tensor.buffer() != nullptr,
                        "Tensor buffer cannot be null");
            }
        }
        ```

        ### **Shape and Layout Validation**
        ```cpp
        void validate_tensor_layouts(const std::vector<Tensor>& tensors) const {
            auto expected_layout = tensors[0].get_layout();
            for (const auto& tensor : tensors) {
                TT_FATAL(tensor.get_layout() == expected_layout,
                        "All tensors must have the same layout");
            }
        }
        ```

        ### **Data Type Validation**
        ```cpp
        void validate_supported_dtypes(const std::vector<Tensor>& tensors) const {
            for (const auto& tensor : tensors) {
                auto dtype = tensor.get_dtype();
                TT_FATAL(dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32,
                        "Operation only supports BFLOAT16 and FLOAT32");
            }
        }
        ```"""
    + GLOBAL_CONTEXT
)

OLD_CMAKE_CONTEXT = """
    # TTNN CMake Context for Custom Operation Structure

    ## Purpose & File Structure
    Generate CMakeLists.txt for TTNN operations with the specific file organization used by the agent.

    ## Expected File Structure
    ```
    {operation_name}/
    ├── CMakeLists.txt                                    # Main build file
    ├── {operation_name}.hpp                              # Main operation header
    ├── {operation_name}.cpp                              # Main operation implementation
    ├── {operation_name}_pybind.hpp                       # Python binding header
    ├── {operation_name}_pybind.cpp                       # Python binding implementation
    ├── device/
    │   ├── {operation_name}_op.hpp                       # Device operation header
    │   ├── {operation_name}_op.cpp                       # Device operation implementation
    │   ├── {operation_name}_program_factory.cpp          # Program factory
    │   └── kernels/
    │       ├── compute/{operation_name}_compute.cpp      # Compute kernel
    │       └── dataflow/
    │           ├── {operation_name}_reader.cpp           # Reader kernel
    │           └── {operation_name}_writer.cpp           # Writer kernel
    └── tests/
        └── test_{operation_name}.py                      # Python unit tests
    ```
    Generate a CMakeLists.txt for a TT-Metal TTNN operation.

    REQUIREMENTS:
    1. Use file(GLOB) to collect source files
    2. Create add_library() with snake_case target name
    3. Set basic target properties (CXX_STANDARD 20, PIC)
    4. Add include directories (standard TT-Metal paths)
    5. Link to tt_metal and ttnn_cpp
    6. Add compile definitions (TT_METAL_VERSIM_DISABLED)
    7. CRITICAL: Create alias target TT::NN::Ops::{PascalCase}

    DO NOT INCLUDE:
    - configure_file() calls
    - Complex variable substitutions
    - Version handling
    - Install commands
    - Package configuration
    - Custom functions/macros
    - Conditional logic

    TEMPLATE STRUCTURE:
    ```cmake
    file(GLOB OPERATION_SRCS "*.cpp" "device/*.cpp" "device/kernels/*/*.cpp")
    add_library(ttnn_{operation_name} STATIC ${OPERATION_SRCS})
    set_target_properties(ttnn_{operation_name} PROPERTIES CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON)
    target_include_directories(ttnn_{operation_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(ttnn_{operation_name} PUBLIC tt_metal ttnn_cpp)
    target_compile_definitions(ttnn_{operation_name} PRIVATE TT_METAL_VERSIM_DISABLED)
    add_library(TT::NN::Ops::{PascalCase} ALIAS ttnn_{operation_name})
    ```

    ALWAYS link to these exact targets:
    - tt_metal (provides kernel APIs)
    - ttnn_cpp (provides TTNN headers)
    - metal_common_libs (provides common deps)
    - TT::NN::Core (provides core functionality)

    ALWAYS include these paths:
    - ${CMAKE_CURRENT_SOURCE_DIR}/../../../tt_metal/include
    - ${CMAKE_CURRENT_SOURCE_DIR}/../../../tt_metal/api

    ALWAYS create alias: TT::NN::Ops::{PascalCase}

    TEMPLATE:
    ```cmake
    target_link_libraries(ttnn_{operation_name} PUBLIC
        tt_metal
        ttnn_cpp
        metal_common_libs
        TT::NN::Core
    )

    target_include_directories(ttnn_{operation_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tt_metal/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../tt_metal/api
    )
    ```
    CRITICAL REQUIREMENT: The generated CMakeLists.txt MUST include ALL of these include directories:
    - tt_metal/include, tt_metal/api, tt_metal/hostdevcommon/api
    - ttnn/cpp/ttnn/operations (and subdirectories)
    - Relative paths using CMAKE_CURRENT_SOURCE_DIR/../../../

    Failure to include comprehensive paths will cause compilation errors.
    """

CMAKE_CONTEXT = """
    You need to create a CMakeLists.txt file for a new TTNN operation that properly integrates with the TT-Metal build system.

    **CRITICAL CMAKE REQUIREMENTS:**

    1. **Library Naming Convention**:
    - The library MUST be named: ttnn_{operation_name}
    - Example: For operation "eltwise_add_custom", library name is "ttnn_eltwise_add_custom"

    2. **Directory Structure Expected**:
    ```
    ttnn/cpp/ttnn/operations/{operation_name}/
    ├── CMakeLists.txt              # This file you're creating
    ├── {operation_name}.hpp        # Main header
    ├── {operation_name}.cpp        # Main implementation
    ├── {operation_name}_pybind.hpp # Pybind header
    ├── {operation_name}_pybind.cpp # Pybind implementation
    └── device/                     # Device-specific code
        ├── {operation_name}_op.hpp
        ├── {operation_name}_op.cpp
        ├── {operation_name}_program_factory.hpp
        ├── {operation_name}_program_factory.cpp
        └── kernels/
            ├── compute/
            │   └── {operation_name}_compute.cpp
            └── dataflow/
                ├── {operation_name}_reader.cpp
                └── {operation_name}_writer.cpp
    ```

    **TTNN Operation CMakeLists.txt Template**
    **Key Variables from Parent CMake:**
    - `${LIB_TYPE}`: Set to either STATIC or SHARED by parent
    - `${FixmeOpIncDirs}`: Contains the correct include paths (../../ and ../../cpp)
    - `${CMAKE_INSTALL_LIBDIR}`: Installation directory for libraries

    **Required Structure:**

    ```cmake
    # Create library using parent-defined LIB_TYPE
    add_library(ttnn_{operation_name} ${LIB_TYPE})
    add_library(TT::NN::Ops::{OperationName} ALIAS ttnn_{operation_name})

    # Add source files (NOT kernel files)
    target_sources(ttnn_{operation_name}
        PRIVATE
            {operation_name}.cpp
            device/{operation_name}_op.cpp
            device/{operation_name}_program_factory.cpp
    )

    # Use parent-provided include directories
    target_include_directories(ttnn_{operation_name}
        PUBLIC
            ${FixmeOpIncDirs}
    )

    # Link to TT-Metal libraries
    target_link_libraries(ttnn_{operation_name}
        PRIVATE
            TT::Metalium
            TT::NN::Core
    )

    # Install shared libraries
    if(LIB_TYPE STREQUAL "SHARED")
        install(TARGETS ttnn_{operation_name} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
        """

TEST_CONTEXT = """
    Generate comprehensive unit tests for the custom TTNN operation `{self.python_function_name}` that verify both functionality and performance against the baseline `ttnn.{self.operation_type}` operation.

    CRITICAL TEST REQUIREMENTS:
    - Test ONLY the custom operation: `{self.python_function_name}(a, b)`
    - Verify operation exists in correct namespace: `assert hasattr(<namespace>, '{self.python_function_name}')`
    - Check for debug output to confirm custom implementation is called (not baseline fallback)
    - Compare mathematical accuracy against baseline `ttnn.{self.operation_type}` using `torch.allclose` with atol=1e-4

    EDGE CASES TO COVER:
    - Empty tensors and single-element tensors
    - Mismatched tensor shapes requiring broadcasting (2x1 + 1x3, scalar + tensor)
    - Extreme values (zeros, ones, inf, -inf, very large/small numbers)
    - Different data types (bfloat16, float32, int32) and mixed-precision scenarios
    - Large tensors (1024x1024) and non-power-of-2 dimensions
    - Memory layout variations (row-major, tile layout)
    - Device vs host tensor combinations

    PERFORMANCE BENCHMARKS:
    - Time 100 iterations of both custom and baseline operations on identical 512x512 tensors
    - Memory usage comparison using device memory profiling
    - Throughput test: operations per second on batch of 10 random tensor pairs
    - Assert custom operation is within 10% performance of baseline

    ACCURACY VALIDATION:
    - Test mathematical correctness: `{self.operation_spec['expected_result']}`
    - Cross-validate results against PyTorch equivalent when available
    - Numerical stability with edge case inputs
    - Gradient flow verification if operation supports backprop

    DISPATCH VERIFICATION:
    - Confirm custom operation registration in correct TTNN namespace
    - Verify debug markers appear in output indicating custom code path
    - Test that operation fails gracefully with invalid inputs
    - Validate error messages contain custom operation name, not generic references

    Generate pytest-compatible test functions with clear assertions and informative failure messages.
    """
