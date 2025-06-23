"""Pre-built workflow graphs for common scenarios."""

from ttnn_op_generator.core.workflow_graph import WorkflowGraph, Edge, GraphBuilder
from ttnn_op_generator.core.graph_nodes import (
    GenerateFileNode, BuildVerificationNode, DebugAnalysisNode,
    DebugFixNode, LoopControlNode, SetupNode, CMakeUpdateNode,
    TestExecutionNode, MultiStageSetupNode
)
from ttnn_op_generator.core.node_types import NodeStatus


def create_default_graph() -> WorkflowGraph:
    """Create the default workflow graph that matches original behavior."""
    builder = GraphBuilder("default")
    
    # Setup node
    builder.add_node(SetupNode("setup"))
    
    # File generation nodes
    file_order = [
        ("hpp", []), 
        ("cpp", ["hpp"]), 
        ("op-hpp", ["hpp"]),
        ("op", ["op-hpp", "hpp"]), 
        ("program-factory-hpp", []),
        ("program-factory", ["program-factory-hpp", "op-hpp"]),
        ("reader", ["program-factory-hpp", "op-hpp"]), 
        ("writer", ["program-factory-hpp", "op-hpp"]),
        ("compute", ["program-factory-hpp", "op-hpp"]), 
        ("pybind-hpp", ["hpp"]),
        ("pybind-cpp", ["pybind-hpp", "hpp"]), 
        ("cmake", [])
    ]
    
    # Add file generation nodes
    prev_node = "setup"
    for file_key, deps in file_order:
        node_name = f"generate_{file_key}"
        builder.add_node(
            GenerateFileNode(
                node_name, 
                file_key=file_key, 
                dependencies=deps,
                description=f"Generate {file_key} file"
            )
        )
        builder.add_edge(prev_node, node_name)
        prev_node = node_name
    
    # Initial build verification
    builder.add_node(BuildVerificationNode("initial_build"))
    builder.add_edge("generate_cmake", "initial_build")
    
    # Create end node first
    builder.add_node(SetupNode("end_success", description="Success endpoint"))
    
    # Success path
    builder.add_edge(
        "initial_build", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS,
        "build_success"
    )
    
    # Debug loop for failures
    builder.add_node(
        LoopControlNode(
            "debug_loop", 
            loop_name="debug", 
            max_iterations=3,
            description="Control debug iterations"
        )
    )
    
    builder.add_edge(
        "initial_build", 
        "debug_loop",
        lambda r: r.status == NodeStatus.FAILURE,
        "build_failed"
    )
    
    # Debug analysis
    builder.add_node(
        DebugAnalysisNode(
            "debug_analysis", 
            error_source="initial_build",
            description="Analyze build errors"
        )
    )
    builder.add_edge("debug_loop", "debug_analysis")
    
    # Debug fix
    builder.add_node(
        DebugFixNode(
            "debug_fix", 
            analysis_source="debug_analysis",
            use_targeted_editing=True,
            description="Apply fixes"
        )
    )
    builder.add_edge("debug_analysis", "debug_fix")
    
    # Rebuild after fix
    builder.add_node(BuildVerificationNode("rebuild"))
    builder.add_edge("debug_fix", "rebuild")
    
    # Check rebuild result
    builder.add_edge(
        "rebuild", 
        "debug_loop",
        lambda r: r.status == NodeStatus.FAILURE,
        "still_failing"
    )
    
    builder.add_edge(
        "rebuild", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS,
        "debug_success"
    )
    
    # Mark as end node
    builder.add_end("end_success")
    
    return builder.set_start("setup").build()


def create_multi_stage_graph() -> WorkflowGraph:
    """Create a graph for multi-stage generation with validation."""
    builder = GraphBuilder("multi_stage")
    
    # Setup nodes
    builder.add_node(SetupNode("setup"))
    builder.add_node(MultiStageSetupNode("enable_multistage"))
    builder.add_edge("setup", "enable_multistage")
    
    # File generation with multi-stage
    file_order = [
        ("hpp", []), ("cpp", ["hpp"]), ("op-hpp", ["hpp"]),
        ("op", ["op-hpp", "hpp"]), ("program-factory-hpp", []),
        ("program-factory", ["program-factory-hpp", "op-hpp"]),
        ("reader", ["program-factory-hpp", "op-hpp"]), 
        ("writer", ["program-factory-hpp", "op-hpp"]),
        ("compute", ["program-factory-hpp", "op-hpp"]), 
        ("pybind-hpp", ["hpp"]),
        ("pybind-cpp", ["pybind-hpp", "hpp"]), ("cmake", [])
    ]
    
    prev_node = "enable_multistage"
    for file_key, deps in file_order:
        node_name = f"generate_{file_key}"
        builder.add_node(
            GenerateFileNode(
                node_name, 
                file_key=file_key, 
                dependencies=deps,
                use_multi_stage=True
            )
        )
        builder.add_edge(prev_node, node_name)
        prev_node = node_name
    
    # Build and debug loop (similar to default)
    builder.add_node(BuildVerificationNode("build"))
    builder.add_edge("generate_cmake", "build")
    
    # Create end node first
    builder.add_node(SetupNode("end_success"))
    
    # Success/failure paths
    builder.add_edge(
        "build", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS,
        "success"
    )
    
    # Debug loop
    builder.add_node(LoopControlNode("debug_loop", max_iterations=2))
    builder.add_edge(
        "build", 
        "debug_loop",
        lambda r: r.status == NodeStatus.FAILURE,
        "failed"
    )
    
    builder.add_node(DebugAnalysisNode("analyze", error_source="build"))
    builder.add_edge("debug_loop", "analyze")
    
    builder.add_node(DebugFixNode("fix", analysis_source="analyze"))
    builder.add_edge("analyze", "fix")
    
    builder.add_node(BuildVerificationNode("rebuild"))
    builder.add_edge("fix", "rebuild")
    
    builder.add_edge(
        "rebuild", 
        "debug_loop",
        lambda r: r.status == NodeStatus.FAILURE,
        "retry"
    )
    
    builder.add_edge(
        "rebuild", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS,
        "fixed"
    )
    
    # Mark as end node
    builder.add_end("end_success")
    
    return builder.set_start("setup").build()


def create_quick_debug_graph() -> WorkflowGraph:
    """Create a graph that only runs the debug loop (assumes files exist)."""
    builder = GraphBuilder("quick_debug")
    
    # Start with build check
    builder.add_node(BuildVerificationNode("initial_build"))
    
    # Create end node
    builder.add_node(SetupNode("end_success"))
    
    # If success, we're done
    builder.add_edge(
        "initial_build", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS,
        "already_working"
    )
    
    # Debug loop
    builder.add_node(LoopControlNode("debug_loop", max_iterations=5))
    builder.add_edge(
        "initial_build", 
        "debug_loop",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    # Enhanced debug with more options
    builder.add_node(
        DebugAnalysisNode("analyze", error_source="initial_build")
    )
    builder.add_edge("debug_loop", "analyze")
    
    builder.add_node(
        DebugFixNode(
            "targeted_fix", 
            analysis_source="analyze",
            use_targeted_editing=True
        )
    )
    builder.add_edge("analyze", "targeted_fix")
    
    builder.add_node(BuildVerificationNode("rebuild"))
    builder.add_edge("targeted_fix", "rebuild")
    
    # Loop or exit
    builder.add_edge(
        "rebuild", 
        "debug_loop",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    builder.add_edge(
        "rebuild", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    # Mark as end node
    builder.add_end("end_success")
    
    return builder.set_start("initial_build").build()


def create_partial_completion_graph() -> WorkflowGraph:
    """Create a graph for completing partially generated operations."""
    builder = GraphBuilder("partial_completion")
    
    # Custom setup that checks for existing files
    builder.add_node(
        SetupNode(
            "check_existing",
            check_existing_files=True,
            description="Check for existing files"
        )
    )
    
    # Generate only missing files
    file_order = [
        ("hpp", []), ("cpp", ["hpp"]), ("op-hpp", ["hpp"]),
        ("op", ["op-hpp", "hpp"]), ("program-factory-hpp", []),
        ("program-factory", ["program-factory-hpp", "op-hpp"]),
        ("reader", ["program-factory-hpp", "op-hpp"]), 
        ("writer", ["program-factory-hpp", "op-hpp"]),
        ("compute", ["program-factory-hpp", "op-hpp"]), 
        ("pybind-hpp", ["hpp"]),
        ("pybind-cpp", ["pybind-hpp", "hpp"]), ("cmake", [])
    ]
    
    prev_node = "check_existing"
    for file_key, deps in file_order:
        node_name = f"generate_{file_key}"
        # This node should check if file exists before generating
        builder.add_node(
            GenerateFileNode(
                node_name, 
                file_key=file_key, 
                dependencies=deps,
                skip_if_exists=True
            )
        )
        builder.add_edge(prev_node, node_name)
        prev_node = node_name
    
    # Build and minimal debug
    builder.add_node(BuildVerificationNode("build"))
    builder.add_edge("generate_cmake", "build")
    
    # Create end node
    builder.add_node(SetupNode("end_success"))
    
    builder.add_edge(
        "build", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    # Single debug attempt
    builder.add_node(DebugAnalysisNode("analyze", error_source="build"))
    builder.add_edge(
        "build", 
        "analyze",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    builder.add_node(DebugFixNode("fix", analysis_source="analyze"))
    builder.add_edge("analyze", "fix")
    
    builder.add_node(BuildVerificationNode("final_build"))
    builder.add_edge("fix", "final_build")
    
    builder.add_edge("final_build", "end_success")
    
    # Mark as end node
    builder.add_end("end_success")
    
    return builder.set_start("check_existing").build()


def create_test_driven_graph(test_path: str) -> WorkflowGraph:
    """Create a graph that includes test execution and debugging."""
    builder = GraphBuilder("test_driven")
    
    # Standard setup and generation
    builder.add_node(SetupNode("setup"))
    
    # Simplified generation - could be expanded
    builder.add_node(
        GenerateFileNode("generate_all", file_key="all", dependencies=[])
    )
    builder.add_edge("setup", "generate_all")
    
    # Build
    builder.add_node(BuildVerificationNode("build"))
    builder.add_edge("generate_all", "build")
    
    # Run test
    builder.add_node(
        TestExecutionNode("run_test", test_path=test_path)
    )
    builder.add_edge(
        "build", 
        "run_test",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    # Create end nodes
    builder.add_node(SetupNode("end_success"))
    builder.add_node(SetupNode("end_failure"))
    
    # Test success ends workflow
    builder.add_edge(
        "run_test", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    # Test failure triggers debug
    builder.add_node(LoopControlNode("test_debug_loop", max_iterations=3))
    builder.add_edge(
        "run_test", 
        "test_debug_loop",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    # Debug and fix based on test failure
    builder.add_node(
        DebugAnalysisNode("test_analyze", error_source="run_test")
    )
    builder.add_edge("test_debug_loop", "test_analyze")
    
    builder.add_node(
        DebugFixNode("test_fix", analysis_source="test_analyze")
    )
    builder.add_edge("test_analyze", "test_fix")
    
    # Rebuild and retest
    builder.add_node(BuildVerificationNode("rebuild"))
    builder.add_edge("test_fix", "rebuild")
    
    builder.add_node(TestExecutionNode("retest", test_path=test_path))
    builder.add_edge(
        "rebuild", 
        "retest",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    # Loop or success
    builder.add_edge(
        "retest", 
        "test_debug_loop",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    builder.add_edge(
        "retest", 
        "end_success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    # Build failure handling
    builder.add_edge(
        "build", 
        "end_failure",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    builder.add_edge(
        "rebuild", 
        "test_debug_loop",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    # Mark end nodes
    builder.add_end("end_success")
    builder.add_end("end_failure")
    
    return builder.set_start("setup").build()