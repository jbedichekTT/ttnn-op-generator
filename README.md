# TTNN Operation Generator - Graph-Based Workflow System

## Overview

This is a refactored version of the TTNN Operation Generator that uses an explicit graph-based workflow system. Instead of hard-coded logic, workflows are defined as directed graphs where nodes represent actions and edges represent transitions.

## Key Benefits

1. **Declarative Workflows**: Define workflows as data structures, not code
2. **Easy Experimentation**: Swap workflows without changing core logic
3. **Visual Debugging**: Generate workflow diagrams
4. **Reusable Components**: Mix and match nodes for custom workflows
5. **Clear Control Flow**: Explicit conditions and loops

## Installation

```bash
# Clone the repository
git clone <your-repo>
cd ttnn_operation_generator

# Install dependencies
pip install -r requirements.txt

# Optional: Install graphviz for visualization
pip install graphviz
# On Ubuntu: sudo apt-get install graphviz
# On Mac: brew install graphviz
```

## Project Structure

```
ttnn_operation_generator/
├── core/                    # Core graph engine
│   ├── node_types.py       # Base classes and types
│   ├── graph_nodes.py      # Node implementations
│   └── workflow_graph.py   # Graph and executor
├── graphs/                  # Pre-built workflows
│   ├── default_graphs.py   # Standard workflows
│   └── custom_graphs.py    # Experimental workflows
├── agents/                  # Agent implementations
│   └── ttnn_agent_refactored.py
├── utils/                   # Utilities
│   └── graph_visualizer.py # Visualization tools
├── example_usage.py        # Usage examples
└── README.md              # This file
```

## Quick Start

### Basic Usage

```python
from agents.ttnn_agent_refactored import TTNNOperationAgent

# Create agent
agent = TTNNOperationAgent(
    operation_type="multiply",
    tt_metal_path="/path/to/tt-metal",
    api_key="your-anthropic-key"  # Or set ANTHROPIC_API_KEY env var
)

# Use default workflow
agent.use_default_workflow()

# Build operation
success = agent.build_operation()
```

### Using Different Pre-built Workflows

```python
# Multi-stage generation with API validation
agent.use_multi_stage_workflow()

# Quick debug (assumes files exist)
agent.use_quick_debug_workflow()

# Complete partial operation
agent.use_partial_completion_workflow()

# Build
success = agent.build_operation()
```

### Creating Custom Workflows

```python
from core.workflow_graph import GraphBuilder
from core.graph_nodes import GenerateFileNode, BuildVerificationNode, SetupNode

# Create custom workflow
builder = GraphBuilder("my_custom_workflow")

# Add nodes
builder.add_node(SetupNode("setup"))
builder.add_node(GenerateFileNode("gen_hpp", file_key="hpp"))
builder.add_node(GenerateFileNode("gen_cpp", file_key="cpp", dependencies=["hpp"]))
builder.add_node(BuildVerificationNode("build"))

# Connect nodes
builder.add_sequence("setup", "gen_hpp", "gen_cpp", "build")

# Set start and end
builder.set_start("setup")
builder.add_end("build")

# Build graph
graph = builder.build()

# Use it
agent.set_workflow_graph(graph)
success = agent.build_operation()
```

### Visualizing Workflows

```python
from utils.graph_visualizer import visualize_graph, create_workflow_documentation

# Visualize any workflow
visualize_graph(agent.workflow_graph, "my_workflow", format="png")

# Generate markdown documentation
doc = create_workflow_documentation(agent.workflow_graph)
with open("workflow_docs.md", "w") as f:
    f.write(doc)
```

## Available Workflows

### 1. Default Workflow
Standard generation with debug loop:
- Generate all files in order
- Build verification
- Debug loop (up to 3 iterations) if build fails
- Targeted fixes based on errors

```python
agent.use_default_workflow()
```

### 2. Multi-Stage Workflow
Enhanced generation with API validation:
- Plan → Validate → Refine → Execute for each file
- Validates APIs exist before using them
- More reliable but slower

```python
agent.use_multi_stage_workflow()
```

### 3. Quick Debug Workflow
For debugging existing operations:
- Assumes files already exist
- Jumps straight to build and debug
- Up to 5 debug iterations

```python
agent.use_quick_debug_workflow()
```

### 4. Partial Completion Workflow
Complete partially generated operations:
- Checks for existing files
- Only generates missing files
- Minimal debug loop

```python
agent.use_partial_completion_workflow()
```

### 5. Test-Driven Workflow
Include test execution in the workflow:
```python
from graphs.default_graphs import create_test_driven_graph

graph = create_test_driven_graph("/path/to/test.py")
agent.set_workflow_graph(graph)
```

## Creating Custom Nodes

```python
from core.node_types import Node, NodeResult, NodeStatus, NodeContext

class MyCustomNode(Node):
    """Custom node implementation."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        # Access the agent
        agent = context.agent
        
        # Access previous node outputs
        prev_result = context.get_node_output("previous_node")
        
        # Access/set global state
        context.set_global("my_key", "my_value")
        value = context.get_global("my_key")
        
        # Do your work
        try:
            # ... your logic here ...
            return NodeResult(
                NodeStatus.SUCCESS,
                {"data": "result"},
                "Success message"
            )
        except Exception as e:
            return NodeResult(
                NodeStatus.FAILURE,
                {"error": str(e)},
                f"Failed: {e}"
            )
```

## Advanced Features

### Conditional Edges

```python
from core.node_types import NodeStatus

# Add conditional edge
builder.add_edge(
    "source_node", 
    "target_node",
    lambda result: result.status == NodeStatus.SUCCESS,
    "on_success"
)
```

### Loop Control

```python
from core.graph_nodes import LoopControlNode

# Add loop with custom exit condition
builder.add_node(
    LoopControlNode(
        "my_loop",
        max_iterations=5,
        exit_condition=lambda ctx: ctx.get_global("done") == True
    )
)
```

### Parallel Execution (Conceptual)

```python
# The experimental graph shows parallel generation
from graphs.custom_graphs import create_experimental_graph

graph = create_experimental_graph()
agent.set_workflow_graph(graph)
```

## Integration with Existing Code

The refactored agent maintains compatibility with the original implementation:

1. **Same file structure**: Files are generated in the same locations
2. **Same API**: Core methods like `build_operation()` work the same
3. **Same dependencies**: Uses the same prompts, tools, and refinement systems

To migrate existing code:

```python
# Old way
agent = TTNNOperationAgent(...)
agent.build_operation()  # Hard-coded workflow

# New way
agent = TTNNOperationAgent(...)
agent.use_default_workflow()  # Explicit workflow
agent.build_operation()
```

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `TT_METAL_PATH`: Path to TT-Metal repository (optional)

## Troubleshooting

### Visualization not working
```bash
# Install graphviz system package
sudo apt-get install graphviz  # Ubuntu
brew install graphviz          # Mac
```

### Import errors
```bash
# Ensure you're in the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### API errors
```bash
# Check API key is set
export ANTHROPIC_API_KEY="your-key-here"
```

## Examples

Run the examples to see different workflows in action:

```bash
# Run all examples
python example_usage.py all

# Run specific example
python example_usage.py multi_stage

# With visualization
python example_usage.py custom --visualize

# Generate documentation for all workflows
python example_usage.py docs
```

## Contributing

To add new workflows:

1. Create new graph definition in `graphs/`
2. Add custom nodes if needed in `core/graph_nodes.py`
3. Add example usage in `example_usage.py`
4. Update documentation

## Future Enhancements

- [ ] True parallel execution with asyncio
- [ ] Workflow persistence and resumption  
- [ ] Dynamic graph modification during execution
- [ ] Integration with CI/CD pipelines
- [ ] Web UI for workflow design
- [ ] Performance metrics and profiling
- [ ] Distributed execution support

## License

[Your License Here]