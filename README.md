# TTNN Operation Generator

AI-powered code generator for custom TTNN operations on TT-Metal hardware. Uses Claude AI to generate complete C++ implementations including device kernels, host code, Python bindings, and build configs.

## Architecture

- **`op_generator.py`** - Main orchestrator, manages generation workflow
- **`persistent_prompt_refiner.py`** - Learning system that improves from build errors  
- **`prompts.py`** - Context templates for each file type
- **`tools.py`** - AI tools for namespace resolution, API search
- **`test_debugger.py`** - Test execution and failure analysis

## Generated Structure

```
ttnn/cpp/ttnn/operations/{operation_name}/
├── CMakeLists.txt
├── {operation_name}.hpp/.cpp              # API interface & implementation
├── {operation_name}_pybind.hpp/.cpp       # Python bindings
└── device/
    ├── {operation_name}_op.hpp/.cpp       # Device operation
    ├── {operation_name}_program_factory.* # Kernel orchestration
    └── kernels/
        ├── compute/*.cpp                  # Compute kernels
        └── dataflow/*.cpp                 # Reader/writer kernels
```

## How It Works

1. **Generate** - Creates all files using specialized prompts
2. **Build** - Attempts compilation with `build_metal.sh`
3. **Fix** - Analyzes errors, uses AI tools to resolve issues
4. **Learn** - Stores fixes in persistent database for future runs
5. **Test** - Optionally generates and debugs unit tests

## Configuration

```python
# op_generator.py
API_KEY = "sk-..."                    # Claude API key
MODEL = "claude-sonnet-4-20250514"   # Model version
BUILD_RETRIES = 10                   # Fix attempts
DEBUG_ONLY = 1                       # Skip generation, only debug
COMPLETE_PARTIAL_OP = 1              # Resume partial operations
```

## AI Tools

- **`resolve_namespace_and_verify`** - Finds correct C++ namespaces
- **`find_api_usages`** - Searches codebase for API examples
- **`extract_symbols_from_files`** - Validates generated interfaces
- **`search_tt_metal_docs`** - Queries Tenstorrent documentation

## Learning Database

Maintains JSON database of compilation fixes:

```json
{
  "global_refinements": {
    "hpp": ["Include patterns..."],
    "cpp": ["API corrections..."]
  },
  "operations": {
    "operation_name": {
      "refinements": {...},
      "attempts": 5,
      "successes": 1
    }
  }
}
```

## Advanced Usage

```bash
# Debug existing operation
DEBUG_ONLY=1 python op_generator.py --operation add

# Complete partial generation  
COMPLETE_PARTIAL_OP=1 python op_generator.py --operation multiply

# Custom operation type
python op_generator.py --operation exp --custom-suffix optimized
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails after retries | Check refinements DB, increase `BUILD_RETRIES` |
| Namespace errors | Verify TT-Metal build, check tool outputs |
| Test timeouts | Usually kernel deadlock, check synchronization |


## Requirements

- TT-Metal + dependencies
- Anthropic API key
