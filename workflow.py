"""Example usage of the refactored TTNN Operation Agent with graph workflows."""

import argparse
from pathlib import Path

from agents.ttnn_agent import TTNNOperationAgent
from graphs.default_graphs import (
    create_default_graph,
    create_multi_stage_graph,
    create_quick_debug_graph,
    create_partial_completion_graph,
    create_test_driven_graph
)
from graphs.custom_graphs import create_experimental_graph  # If you create custom graphs
from utils.graph_visualizer import visualize_graph, create_workflow_documentation
from utils.interactive_visualizer import InteractiveVisualizer
from core.workflow_graph import GraphBuilder
from core.graph_nodes import (
    GenerateFileNode, BuildVerificationNode, SetupNode
)


def example_basic_usage():
    """Basic usage with default workflow."""
    print("=== Example 1: Basic Usage with Default Workflow ===")
    
    # Create agent
    agent = TTNNOperationAgent(
        operation_type="multiply",
        tt_metal_path="/home/user/tt-metal",
        custom_suffix="custom"
    )
    
    # Use default workflow
    agent.use_default_workflow()
    
    # Build operation
    success = agent.build_operation()
    
    print(f"Operation build {'succeeded' if success else 'failed'}")
    
    # Show how to visualize the logged execution
    if agent.logger.list_executions():
        last_exec = agent.logger.list_executions()[-1]
        exec_log = agent.logger.load_execution_log(last_exec['execution_id'])
        
        if exec_log:
            visualizer = InteractiveVisualizer()
            html_path = visualizer.create_interactive_view(
                agent.workflow_graph,
                exec_log,
                "example_basic_execution"
            )
            print(f"\nInteractive execution view created: {html_path}")
            print("Open in a web browser to explore the execution details")
    
    return success


def example_custom_workflow():
    """Create and use a completely custom workflow."""
    print("\n=== Example 2: Custom Workflow ===")
    
    # Create agent
    agent = TTNNOperationAgent(
        operation_type="add",
        tt_metal_path="/home/user/tt-metal"
    )
    
    # Build a custom workflow using the GraphBuilder
    builder = GraphBuilder("custom_simple")
    
    # Just generate hpp and cpp files, then build
    builder.add_node(SetupNode("setup"))
    builder.add_node(GenerateFileNode("gen_hpp", file_key="hpp", dependencies=[]))
    builder.add_node(GenerateFileNode("gen_cpp", file_key="cpp", dependencies=["hpp"]))
    builder.add_node(BuildVerificationNode("build"))
    
    # Connect nodes
    builder.add_sequence("setup", "gen_hpp", "gen_cpp", "build")
    builder.set_start("setup")
    builder.add_end("build")
    
    # Get the graph
    graph = builder.build()
    
    # Visualize it
    visualize_graph(graph, "custom_workflow")
    
    # Use it
    agent.set_workflow_graph(graph)
    success = agent.build_operation()
    
    return success


def example_multi_stage_workflow():
    """Use the multi-stage workflow with API validation."""
    print("\n=== Example 3: Multi-Stage Workflow ===")
    
    agent = TTNNOperationAgent(
        operation_type="multiply",
        tt_metal_path="/home/user/tt-metal"
    )
    
    # Use multi-stage workflow
    agent.use_multi_stage_workflow()
    
    # Visualize the workflow before execution
    visualize_graph(agent.workflow_graph, "multi_stage_workflow")
    
    # Execute
    success = agent.build_operation()
    
    return success


def example_partial_completion():
    """Complete a partially generated operation."""
    print("\n=== Example 4: Partial Completion ===")
    
    agent = TTNNOperationAgent(
        operation_type="exp",
        tt_metal_path="/home/user/tt-metal"
    )
    
    # Use partial completion workflow
    agent.use_partial_completion_workflow()
    
    # This will check for existing files and only generate missing ones
    success = agent.build_operation()
    
    return success


def example_quick_debug():
    """Debug existing files without regenerating everything."""
    print("\n=== Example 5: Quick Debug ===")
    
    agent = TTNNOperationAgent(
        operation_type="sqrt",
        tt_metal_path="/home/user/tt-metal"
    )
    
    # Use quick debug workflow
    agent.use_quick_debug_workflow()
    
    # This assumes files exist and jumps straight to debugging
    success = agent.build_operation()
    
    return success


def example_test_driven():
    """Use test-driven workflow."""
    print("\n=== Example 6: Test-Driven Development ===")
    
    agent = TTNNOperationAgent(
        operation_type="log",
        tt_metal_path="/home/user/tt-metal",
        run_tests=True
    )
    
    # Create test-driven workflow
    test_path = "/home/user/tt-metal/tests/ttnn/unit_tests/test_log.py"
    graph = create_test_driven_graph(test_path)
    
    agent.set_workflow_graph(graph)
    success = agent.build_operation()
    
    return success


def example_workflow_documentation():
    """Generate documentation for workflows."""
    print("\n=== Example 7: Workflow Documentation ===")
    
    # Document all available workflows
    workflows = {
        "default": create_default_graph(),
        "multi_stage": create_multi_stage_graph(),
        "quick_debug": create_quick_debug_graph(),
        "partial_completion": create_partial_completion_graph(),
    }
    
    for name, graph in workflows.items():
        # Generate visualization
        visualize_graph(graph, f"docs/workflow_{name}")
        
        # Generate markdown documentation
        doc = create_workflow_documentation(graph)
        
        with open(f"docs/workflow_{name}.md", "w") as f:
            f.write(doc)
            
        print(f"Generated documentation for {name} workflow")


def example_conditional_workflow():
    """Create a workflow with complex conditions."""
    print("\n=== Example 8: Conditional Workflow ===")
    
    from core.node_types import NodeStatus
    
    builder = GraphBuilder("conditional")
    
    # Setup
    builder.add_node(SetupNode("setup"))
    
    # Generate core files
    builder.add_node(GenerateFileNode("gen_core", file_key="hpp"))
    builder.add_edge("setup", "gen_core")
    
    # Quick build check
    builder.add_node(BuildVerificationNode("quick_check"))
    builder.add_edge("gen_core", "quick_check")
    
    # Different paths based on result
    builder.add_node(GenerateFileNode("gen_minimal", file_key="cpp", dependencies=["hpp"]))
    builder.add_node(GenerateFileNode("gen_full", file_key="cpp", dependencies=["hpp"]))
    
    # Use minimal generation if hpp compiled fine
    builder.add_edge(
        "quick_check", "gen_minimal",
        lambda r: r.status == NodeStatus.SUCCESS,
        "hpp_ok"
    )
    
    # Use full generation if hpp had issues
    builder.add_edge(
        "quick_check", "gen_full",
        lambda r: r.status == NodeStatus.FAILURE,
        "hpp_failed"
    )
    
    # Final build
    builder.add_node(BuildVerificationNode("final_build"))
    builder.add_edge("gen_minimal", "final_build")
    builder.add_edge("gen_full", "final_build")
    
    builder.set_start("setup")
    builder.add_end("final_build")
    
    graph = builder.build()
    
    # Use it
    agent = TTNNOperationAgent(
        operation_type="conditional_test",
        tt_metal_path="/home/user/tt-metal"
    )
    agent.set_workflow_graph(graph)
    
    return agent.build_operation()


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="TTNN Operation Generator with Graph Workflows"
    )
    
    parser.add_argument(
        "--example",
        choices=[
            "basic", "custom", "multi_stage", "partial", 
            "debug", "test", "docs", "conditional", "all"
        ],
        help="Which example to run"
    )
    
    parser.add_argument(
        "--operation",
        default="multiply",
        help="Operation type to generate"
    )
    
    parser.add_argument(
        "--tt-metal-path",
        default="/home/user/tt-metal",
        help="Path to TT-Metal repository"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate workflow visualizations"
    )
    
    args = parser.parse_args()
    
    # Ensure docs directory exists
    Path("docs").mkdir(exist_ok=True)
    
    # Run examples
    examples = {
        "basic": example_basic_usage,
        "custom": example_custom_workflow,
        "multi_stage": example_multi_stage_workflow,
        "partial": example_partial_completion,
        "debug": example_quick_debug,
        "test": example_test_driven,
        "docs": example_workflow_documentation,
        "conditional": example_conditional_workflow,
    }
    
    if args.example == "all":
        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"Error in {name}: {e}")
    else:
        examples[args.example]()


if __name__ == "__main__":
    main()