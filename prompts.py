TOOL_USE_CONTEXT = '''
    Quick Tool Usage Guide:

    find_api_usages
    When to use: Need to understand how a function/API is used in the codebase

    Finding correct parameters for a function call
    Understanding usage patterns and best practices
    Resolving "no matching function" errors by seeing working examples

    Example triggers: "How is X used?", "Show me examples of Y", "What parameters does Z take?"
    
    parse_and_analyze_code
    When to use: Need to understand the structure of a C++ file

    Before making edits (to understand what's already there)
    Finding functions, classes, includes, namespaces in a file
    Checking if certain elements exist before adding them
    Understanding file organization and dependencies

    Example triggers: "What's in this file?", "Does this file have X?", "Analyze the structure"
    
    apply_targeted_edits
    When to use: Need to modify C++ code programmatically which has already been generated (i.e not during initial generation)

    Adding missing includes
    Inserting new functions or classes
    Modifying existing code elements
    Deleting problematic code
    Making precise edits while preserving file structure

    Example triggers: "Add include for X", "Insert function Y", "Fix the implementation", "Remove Z"
'''

GLOBAL_CONTEXT = """**Include Path Context for TTNN Operations:**
    The CMake build system provides include paths at these levels:
    - `{repo_root}/ttnn/`
    - `{repo_root}/ttnn/cpp/`

    Therefore, #include statements must be relative to these paths:

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
    
    """ + TOOL_USE_CONTEXT

HPP_CONTEXT = (
    """
        # TTNN Operation Component Development Contexts

        ## Operation Header (.hpp) Context

        ### Purpose & Structure
        Operation headers define the **public API interface** that users call from C++ and Python.

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

        IMPORTANT: Do not use the eltwise namespace, make this operation in the base operations namespace
        """
    + GLOBAL_CONTEXT
)

KERNEL_CONTEXT = (
    """
        # TT-Metal Kernel Development Context

        ### Memory Hierarchy
        ```
        Host DRAM → Device DRAM (12 channels) → L1 SRAM (1MB per core) → Register Files → Compute Units
        ```

        ### Kernel Types
        1. **Compute Kernels**: Perform math operations on tiles in register files
        2. **Reader Kernels**: Move data from DRAM/other cores to local L1
        3. **Writer Kernels**: Move data from local L1 to DRAM/other cores

        ### Execution Flow
        ```
        Reader → CB (Circular Buffer) → Compute → CB → Writer
        ```
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

        The CMake include paths already point to the cpp directory.
        IMPORTANT: Make this operation in the operations namespace, not eltwise.
        """ + GLOBAL_CONTEXT
)

DEVICE_OP_CONTEXT = (
    """
        # TTNN Device Operation Development Context

        ### Purpose & Structure
        Device operation headers define the **device-side operation class** that implements the low-level operation interface and manages program execution.        
        """
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
