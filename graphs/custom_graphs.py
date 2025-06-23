"""Custom workflow graphs for experimental approaches."""

from ttnn_op_generator.core.workflow_graph import WorkflowGraph, GraphBuilder
from ttnn_op_generator.core.graph_nodes import (
    GenerateFileNode, BuildVerificationNode, DebugAnalysisNode,
    DebugFixNode, LoopControlNode, SetupNode, Node
)
from ttnn_op_generator.core.node_types import NodeResult, NodeStatus, NodeContext


class ParallelGenerationNode(Node):
    """Custom node that generates multiple files in parallel (conceptually)."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        file_groups = self.config.get('file_groups', [])
        
        print(f"\n[ParallelGenerationNode] Generating {len(file_groups)} file groups")
        
        success = True
        generated_files = []
        
        for group in file_groups:
            group_name = group['name']
            files = group['files']
            
            print(f"\n[Parallel Group: {group_name}]")
            
            for file_key, deps in files:
                try:
                    # Use the agent's generation logic
                    node = GenerateFileNode(
                        f"temp_{file_key}", 
                        file_key=file_key, 
                        dependencies=deps
                    )
                    result = node.execute(context)
                    
                    if result.status == NodeStatus.SUCCESS:
                        generated_files.append(file_key)
                    else:
                        success = False
                        
                except Exception as e:
                    print(f"  Error generating {file_key}: {e}")
                    success = False
        
        return NodeResult(
            NodeStatus.SUCCESS if success else NodeStatus.FAILURE,
            {"generated_files": generated_files}
        )


class ValidationNode(Node):
    """Custom node for validating generated files before building."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        validation_type = self.config.get('validation_type', 'syntax')
        
        print(f"\n[ValidationNode] Running {validation_type} validation")
        
        issues = []
        
        if validation_type == 'syntax':
            # Check for basic syntax issues
            for file_key, file_data in agent.files.items():
                if file_data['code']:
                    # Simple checks
                    code = file_data['code']
                    
                    # Check for matching braces
                    if code.count('{') != code.count('}'):
                        issues.append(f"{file_key}: Mismatched braces")
                    
                    # Check for includes
                    if file_key.endswith('-hpp') and '#pragma once' not in code:
                        issues.append(f"{file_key}: Missing #pragma once")
                        
        elif validation_type == 'api':
            # Validate API usage
            for file_key, file_data in agent.files.items():
                if file_data['code'] and 'ttnn::' in file_data['code']:
                    # Check for common API patterns
                    code = file_data['code']
                    if 'ttnn::operations::' in code and '#include' not in code:
                        issues.append(f"{file_key}: Using ttnn operations without includes")
        
        if issues:
            print(f"[ValidationNode] Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
            return NodeResult(
                NodeStatus.FAILURE,
                {"issues": issues},
                "Validation failed"
            )
        else:
            print("[ValidationNode] Validation passed")
            return NodeResult(NodeStatus.SUCCESS)


class IncrementalBuildNode(Node):
    """Custom node for incremental building with caching."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        print(f"\n[IncrementalBuildNode] Checking for changes")
        
        # Check what files changed since last build
        changed_files = context.get_global('changed_files', [])
        
        if not changed_files:
            # Full build
            print("[IncrementalBuildNode] No change tracking, doing full build")
            build_node = BuildVerificationNode("temp_build")
            return build_node.execute(context)
        else:
            # Incremental build (simplified - would need ccache/ninja integration)
            print(f"[IncrementalBuildNode] Incremental build for: {changed_files}")
            
            # For now, just do full build
            build_node = BuildVerificationNode("temp_build")
            result = build_node.execute(context)
            
            # Track build result
            if result.status == NodeStatus.SUCCESS:
                context.set_global('last_successful_build', agent.files.copy())
                
            return result


def create_experimental_graph() -> WorkflowGraph:
    """Create an experimental workflow with parallel generation and validation."""
    builder = GraphBuilder("experimental")
    
    # Setup
    builder.add_node(SetupNode("setup"))
    
    # Parallel generation groups
    file_groups = [
        {
            'name': 'headers',
            'files': [
                ('hpp', []),
                ('op-hpp', []),
                ('program-factory-hpp', []),
                ('pybind-hpp', [])
            ]
        },
        {
            'name': 'implementations', 
            'files': [
                ('cpp', ['hpp']),
                ('op', ['op-hpp']),
                ('program-factory', ['program-factory-hpp'])
            ]
        },
        {
            'name': 'kernels',
            'files': [
                ('reader', ['program-factory-hpp']),
                ('writer', ['program-factory-hpp']),
                ('compute', ['program-factory-hpp'])
            ]
        },
        {
            'name': 'bindings',
            'files': [
                ('pybind-cpp', ['pybind-hpp', 'hpp']),
                ('cmake', [])
            ]
        }
    ]
    
    # Generate headers first
    builder.add_node(
        ParallelGenerationNode(
            "gen_headers",
            file_groups=[file_groups[0]]
        )
    )
    builder.add_edge("setup", "gen_headers")
    
    # Validate headers
    builder.add_node(
        ValidationNode("validate_headers", validation_type="syntax")
    )
    builder.add_edge("gen_headers", "validate_headers")
    
    # Generate implementations
    builder.add_node(
        ParallelGenerationNode(
            "gen_impl",
            file_groups=[file_groups[1]]
        )
    )
    builder.add_edge("validate_headers", "gen_impl")
    
    # Generate kernels and bindings in parallel
    builder.add_node(
        ParallelGenerationNode(
            "gen_kernels",
            file_groups=[file_groups[2]]
        )
    )
    builder.add_node(
        ParallelGenerationNode(
            "gen_bindings",
            file_groups=[file_groups[3]]
        )
    )
    
    builder.add_edge("gen_impl", "gen_kernels")
    builder.add_edge("gen_impl", "gen_bindings")
    
    # Validate all code
    builder.add_node(
        ValidationNode("validate_all", validation_type="api")
    )
    builder.add_edge("gen_kernels", "validate_all")
    builder.add_edge("gen_bindings", "validate_all")
    
    # Incremental build
    builder.add_node(IncrementalBuildNode("build"))
    builder.add_edge("validate_all", "build")
    
    # Debug loop if needed
    builder.add_node(LoopControlNode("debug_loop", max_iterations=2))
    builder.add_edge(
        "build", "debug_loop",
        lambda r: r.status == NodeStatus.FAILURE,
        "build_failed"
    )
    
    builder.add_node(DebugAnalysisNode("analyze", error_source="build"))
    builder.add_edge("debug_loop", "analyze")
    
    builder.add_node(DebugFixNode("fix", analysis_source="analyze"))
    builder.add_edge("analyze", "fix")
    
    builder.add_node(IncrementalBuildNode("rebuild"))
    builder.add_edge("fix", "rebuild")
    
    builder.add_edge(
        "rebuild", "debug_loop",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    # Create success node
    builder.add_node(SetupNode("success"))
    
    # Success paths
    builder.add_edge(
        "build", "success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    builder.add_edge(
        "rebuild", "success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    builder.set_start("setup")
    builder.add_end("success")
    
    return builder.build()


def create_iterative_refinement_graph() -> WorkflowGraph:
    """Create a workflow that iteratively refines based on build errors."""
    builder = GraphBuilder("iterative_refinement")
    
    # Initial generation with minimal context
    builder.add_node(SetupNode("setup"))
    
    # Generate core files only
    core_files = ["hpp", "cpp", "cmake"]
    
    prev_node = "setup"
    for file_key in core_files:
        node_name = f"gen_{file_key}"
        builder.add_node(
            GenerateFileNode(node_name, file_key=file_key, dependencies=[])
        )
        builder.add_edge(prev_node, node_name)
        prev_node = node_name
    
    # Build and analyze
    builder.add_node(BuildVerificationNode("initial_build"))
    builder.add_edge(prev_node, "initial_build")
    
    # Refinement loop
    builder.add_node(
        LoopControlNode(
            "refine_loop", 
            max_iterations=5,
            exit_condition=lambda ctx: ctx.get_node_output("build").status == NodeStatus.SUCCESS
        )
    )
    builder.add_edge("initial_build", "refine_loop")
    
    # Analyze what's missing
    builder.add_node(DebugAnalysisNode("analyze_missing", error_source="initial_build"))
    builder.add_edge("refine_loop", "analyze_missing")
    
    # Generate missing components
    builder.add_node(
        GenerateFileNode(
            "gen_missing",
            file_key="dynamic",  # Would be determined by analysis
            dependencies=[]
        )
    )
    builder.add_edge("analyze_missing", "gen_missing")
    
    # Rebuild
    builder.add_node(BuildVerificationNode("build"))
    builder.add_edge("gen_missing", "build")
    
    # Loop back
    builder.add_edge("build", "refine_loop")
    
    # Create success node
    builder.add_node(SetupNode("success"))
    
    # Success
    builder.add_edge(
        "build", "success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    builder.add_end("success")
    
    return builder.set_start("setup").build()


def create_fallback_graph() -> WorkflowGraph:
    """Create a workflow with multiple fallback strategies."""
    builder = GraphBuilder("fallback_strategies")
    
    builder.add_node(SetupNode("setup"))
    
    # Try multi-stage generation first
    builder.add_node(
        GenerateFileNode(
            "try_multistage",
            file_key="all",
            use_multi_stage=True
        )
    )
    builder.add_edge("setup", "try_multistage")
    
    builder.add_node(BuildVerificationNode("build_1"))
    builder.add_edge("try_multistage", "build_1")
    
    # If that fails, try simpler generation
    builder.add_node(
        GenerateFileNode(
            "try_simple",
            file_key="all",
            use_multi_stage=False
        )
    )
    builder.add_edge(
        "build_1", "try_simple",
        lambda r: r.status == NodeStatus.FAILURE,
        "multistage_failed"
    )
    
    builder.add_node(BuildVerificationNode("build_2"))
    builder.add_edge("try_simple", "build_2")
    
    # If that also fails, try with examples
    builder.add_node(
        GenerateFileNode(
            "try_with_examples",
            file_key="all",
            include_examples=True
        )
    )
    builder.add_edge(
        "build_2", "try_with_examples",
        lambda r: r.status == NodeStatus.FAILURE,
        "simple_failed"
    )
    
    builder.add_node(BuildVerificationNode("build_3"))
    builder.add_edge("try_with_examples", "build_3")
    
    # Create end nodes
    builder.add_node(SetupNode("success"))
    builder.add_node(SetupNode("failure"))
    
    # Success paths from any build
    builder.add_edge(
        "build_1", "success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    builder.add_edge(
        "build_2", "success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    builder.add_edge(
        "build_3", "success",
        lambda r: r.status == NodeStatus.SUCCESS
    )
    
    # Final failure
    builder.add_edge(
        "build_3", "failure",
        lambda r: r.status == NodeStatus.FAILURE
    )
    
    builder.set_start("setup")
    builder.add_end("success")
    builder.add_end("failure")
    
    return builder.build()