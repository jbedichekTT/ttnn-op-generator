"""
Configuration-based Workflow Builder
===================================

Builds workflow graphs from YAML configuration files.
"""

from typing import Dict, List, Optional, Any
from ttnn_op_generator.core.workflow_graph import WorkflowGraph, GraphBuilder
from ttnn_op_generator.core.node_types import NodeStatus
from ttnn_op_generator.core.graph_nodes import (
    SetupNode, BuildVerificationNode, DebugAnalysisNode,
    DebugFixNode, LoopControlNode
)
from ttnn_op_generator.config.config_parser import OperationConfig, NodeConfig, FileConfig


class ConfigWorkflowBuilder:
    """Builds workflow graphs from configuration."""
    
    def __init__(self, config: OperationConfig):
        """Initialize with parsed configuration."""
        self.config = config
        self.builder = None
        
    def build_graph(self) -> WorkflowGraph:
        """Build workflow graph from configuration."""
        workflow_name = f"{self.config.name}_workflow"
        self.builder = GraphBuilder(workflow_name)
        
        # Create all nodes
        for node_name, node_config in self.config.workflow.nodes.items():
            self._create_node(node_name, node_config)
            
        # Connect nodes
        for node_name, node_config in self.config.workflow.nodes.items():
            self._connect_node(node_name, node_config)
            
        # Set start node
        self.builder.set_start(self.config.workflow.start)
        
        # Find and mark end nodes
        self._identify_end_nodes()
        
        return self.builder.build()
        
    def _create_node(self, node_name: str, node_config: NodeConfig):
        """Create a node based on its type."""
        node_type = node_config.type
        
        if node_type == 'setup':
            node = SetupNode(
                node_name,
                description=node_config.description
            )
            
        elif node_type == 'generate':
            # Single file generation
            file_keys = node_config.config.get('files', [])
            if file_keys:
                node = ConfigFileGenerationNode(
                    node_name,
                    file_key=file_keys[0],
                    file_config=self.config.files[file_keys[0]],
                    variables=self.config.variables,
                    description=node_config.description
                )
            else:
                raise ValueError(f"Generate node '{node_name}' has no files specified")
                
        elif node_type == 'generate_group':
            # Multiple file generation
            file_keys = node_config.config.get('files', [])
            parallel = node_config.config.get('parallel', False)
            
            node = ConfigFileGroupNode(
                node_name,
                file_keys=file_keys,
                files_config={k: self.config.files[k] for k in file_keys},
                variables=self.config.variables,
                parallel=parallel,
                description=node_config.description
            )
            
        elif node_type == 'build_verification':
            node = BuildVerificationNode(
                node_name,
                description=node_config.description
            )
            
        elif node_type == 'debug_loop':
            node = LoopControlNode(
                node_name,
                loop_name=node_name,
                max_iterations=node_config.config.get('max_attempts', 3),
                description=node_config.description
            )
            
        elif node_type == 'debug_analysis':
            node = DebugAnalysisNode(
                node_name,
                error_source=node_config.config.get('error_source', 'build'),
                description=node_config.description
            )
            
        elif node_type == 'debug_fix':
            node = DebugFixNode(
                node_name,
                analysis_source=node_config.config.get('analysis_source', 'debug_analysis'),
                use_targeted_editing=node_config.config.get('use_targeted_editing', True),
                description=node_config.description
            )
            
        elif node_type == 'end':
            # End nodes are just setup nodes that we'll mark as end
            node = SetupNode(
                node_name,
                description=node_config.description or "End node"
            )
            
        else:
            # Custom node type - create a generic config node
            node = ConfigCustomNode(
                node_name,
                node_type=node_type,
                config=node_config.config,
                description=node_config.description
            )
            
        self.builder.add_node(node)
        
    def _connect_node(self, node_name: str, node_config: NodeConfig):
        """Connect a node to its successors."""
        # Simple next connection
        if node_config.next:
            self.builder.add_edge(node_name, node_config.next)
            
        # Conditional connections
        if node_config.on_success and node_config.on_failure:
            # Success path
            self.builder.add_edge(
                node_name,
                node_config.on_success,
                lambda r: r.status == NodeStatus.SUCCESS,
                "success"
            )
            # Failure path
            self.builder.add_edge(
                node_name,
                node_config.on_failure,
                lambda r: r.status == NodeStatus.FAILURE,
                "failure"
            )
            
    def _identify_end_nodes(self):
        """Identify and mark nodes with no outgoing edges as end nodes."""
        workflow = self.config.workflow
        
        # Find nodes with no outgoing connections
        for node_name in workflow.nodes:
            node_config = workflow.nodes[node_name]
            
            # Check if this node has any outgoing connections
            has_outgoing = (
                node_config.next is not None or
                node_config.on_success is not None or
                node_config.on_failure is not None
            )
            
            # Also check if it's explicitly marked as 'done' or is an 'end' type
            is_end_type = (
                node_config.type == 'end' or
                node_name == 'done' or
                node_config.next == 'done' or
                node_config.on_success == 'done'
            )
            
            if not has_outgoing or is_end_type:
                self.builder.add_end(node_name)


# ==================== Custom Node Implementations ====================

from ttnn_op_generator.core.node_types import Node, NodeResult, NodeContext


class ConfigFileGenerationNode(Node):
    """Node that generates a file based on configuration."""
    
    def __init__(self, name: str, file_key: str, file_config: FileConfig, 
                 variables: Dict[str, str], **kwargs):
        super().__init__(name, **kwargs)
        self.file_key = file_key
        self.file_config = file_config
        self.variables = variables
        
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        # Check if file exists and should be skipped
        if self.file_config.skip_if_exists:
            file_path = agent.output_dir / self.file_config.resolve_path(self.variables)
            if file_path.exists():
                print(f"[ConfigGenerate] Skipping existing file: {file_path}")
                return NodeResult(NodeStatus.SKIP, {"file_key": self.file_key})
                
        print(f"\n[ConfigGenerate] Generating {self.file_key} ({self.file_config.resolve_path(self.variables)})")
        
        # Build dependency context
        dep_context = ""
        for dep_key in self.file_config.dependencies:
            if dep_key in agent.files and agent.files[dep_key]["code"]:
                dep_file = agent.files[dep_key]
                dep_context += f"\n--- Reference: {dep_file['name']} ---\n{dep_file['code']}\n"
                
        # Resolve variables in prompt
        prompt = self.file_config.prompt
        for var, value in self.variables.items():
            prompt = prompt.replace(f"{{{var}}}", str(value))
            
        # Add context if specified
        if self.file_config.context:
            prompt += f"\n\nAdditional Context:\n{self.file_config.context}"
            
        try:
            # Generate the file
            code = agent.generate_with_refined_prompt(prompt, self.file_key, dep_context)
            
            # Update file info in agent
            agent.files[self.file_key] = {
                "name": self.file_config.resolve_path(self.variables),
                "code": code
            }
            agent.save_file(self.file_key, code)
            
            return NodeResult(
                NodeStatus.SUCCESS,
                {"file_key": self.file_key, "file_name": agent.files[self.file_key]['name']}
            )
            
        except Exception as e:
            return NodeResult(
                NodeStatus.FAILURE,
                {"file_key": self.file_key, "error": str(e)},
                f"Failed to generate {self.file_key}: {str(e)}"
            )


class ConfigFileGroupNode(Node):
    """Node that generates multiple files, optionally in parallel."""
    
    def __init__(self, name: str, file_keys: List[str], files_config: Dict[str, FileConfig],
                 variables: Dict[str, str], parallel: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.file_keys = file_keys
        self.files_config = files_config
        self.variables = variables
        self.parallel = parallel
        
    def execute(self, context: NodeContext) -> NodeResult:
        print(f"\n[ConfigGroup] Generating {len(self.file_keys)} files ({'parallel' if self.parallel else 'sequential'})")
        
        results = []
        failed_files = []
        
        for file_key in self.file_keys:
            # Create individual generation node
            gen_node = ConfigFileGenerationNode(
                f"gen_{file_key}",
                file_key=file_key,
                file_config=self.files_config[file_key],
                variables=self.variables
            )
            
            result = gen_node.execute(context)
            results.append((file_key, result))
            
            if result.status == NodeStatus.FAILURE:
                failed_files.append(file_key)
                if not self.parallel:  # Stop on first failure in sequential mode
                    break
                    
        if failed_files:
            return NodeResult(
                NodeStatus.FAILURE,
                {"failed_files": failed_files},
                f"Failed to generate: {', '.join(failed_files)}"
            )
        else:
            return NodeResult(
                NodeStatus.SUCCESS,
                {"generated_files": self.file_keys}
            )


class ConfigCustomNode(Node):
    """Generic node for custom types defined in configuration."""
    
    def __init__(self, name: str, node_type: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, **kwargs)
        self.node_type = node_type
        self.custom_config = config
        
    def execute(self, context: NodeContext) -> NodeResult:
        print(f"\n[CustomNode] Executing {self.node_type} node: {self.name}")
        
        # This is where you'd implement custom node types
        # For now, just succeed
        return NodeResult(
            NodeStatus.SUCCESS,
            {"node_type": self.node_type, "config": self.custom_config}
        )