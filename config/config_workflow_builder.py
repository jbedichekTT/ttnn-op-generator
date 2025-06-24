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
# Add these imports
import importlib
from typing import Callable, Type
from ttnn_op_generator.agents.multi_stage_generator import MultiStageGenerator
from ttnn_op_generator.core.node_types import Node, NodeResult, NodeContext
# ==================== Advanced Node Implementations ====================

class MultiStageSetupNode(Node):
    """Node that enables multi-stage generation."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        print(f"\n[MultiStageSetup] Enabling multi-stage generation")
        
        if hasattr(agent, 'enable_multi_stage_generation'):
            agent.enable_multi_stage_generation()
            return NodeResult(NodeStatus.SUCCESS, {"multi_stage_enabled": True})
        else:
            return NodeResult(
                NodeStatus.FAILURE, 
                {"error": "Multi-stage generation not available"},
                "Agent does not support multi-stage generation"
            )


class MultiStageGenerateNode(Node):
    """Node that uses multi-stage generation for files."""
    
    def __init__(self, name: str, file_keys: List[str], 
                 use_api_validation: bool = True,
                 max_refinement_iterations: int = 3,
                 parallel: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.file_keys = file_keys
        self.use_api_validation = use_api_validation
        self.max_refinement_iterations = max_refinement_iterations
        self.parallel = parallel
        
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        # Ensure multi-stage is enabled
        if not getattr(agent, 'use_multi_stage', False):
            if hasattr(agent, 'enable_multi_stage_generation'):
                agent.enable_multi_stage_generation()
            else:
                return NodeResult(
                    NodeStatus.FAILURE,
                    {"error": "Multi-stage generation not available"}
                )
        
        print(f"\n[MultiStageGenerate] Generating {len(self.file_keys)} files with multi-stage approach")
        
        success_count = 0
        failed_files = []
        generated_files = []
        
        for file_key in self.file_keys:
            try:
                if file_key not in agent.files:
                    failed_files.append(f"{file_key} (not in files config)")
                    continue
                
                # Build dependency context
                file_config = agent.files[file_key]
                dependencies = getattr(file_config, 'dependencies', [])
                
                # Get base prompt - use a generic one if no specific prompt available
                base_prompt = (
                    f"Generate the code for the file `{file_config['name']}` "
                    f"for the `{agent.operation_name}` operation. "
                    f"Use multi-stage generation with API validation."
                )
                
                # Use multi-stage generation
                if hasattr(agent, 'multi_stage_generator') and agent.multi_stage_generator:
                    code = agent.multi_stage_generator.generate_file_multi_stage(
                        file_key, base_prompt, dependencies
                    )
                else:
                    # Fallback to regular generation with refinement
                    code = agent.generate_with_refined_prompt(base_prompt, file_key)
                
                agent.save_file(file_key, code)
                success_count += 1
                generated_files.append(file_key)
                
            except Exception as e:
                print(f"[MultiStageGenerate] Failed to generate {file_key}: {e}")
                failed_files.append(f"{file_key} ({str(e)})")
                
        if failed_files:
            return NodeResult(
                NodeStatus.FAILURE if success_count == 0 else NodeStatus.SUCCESS,
                {
                    "failed_files": failed_files, 
                    "success_count": success_count,
                    "generated_files": generated_files
                },
                f"Generated {success_count}/{len(self.file_keys)} files"
            )
        else:
            return NodeResult(
                NodeStatus.SUCCESS,
                {"generated_files": generated_files, "success_count": success_count}
            )


class CustomValidationNode(Node):
    """Node that applies custom validation rules to generated files."""
    
    def __init__(self, name: str, rules: List[Dict[str, Any]], 
                 fail_on_issues: bool = True, **kwargs):
        super().__init__(name, **kwargs)
        self.rules = rules
        self.fail_on_issues = fail_on_issues
        
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        print(f"\n[CustomValidation] Applying {len(self.rules)} validation rules")
        
        all_issues = []
        files_checked = 0
        
        for file_key, file_data in agent.files.items():
            if file_data.get('code'):
                files_checked += 1
                file_issues = []
                
                for rule in self.rules:
                    if not self._check_rule(file_data['code'], rule):
                        issue = f"{file_key}: {rule.get('description', 'Rule failed')}"
                        file_issues.append(issue)
                        all_issues.append(issue)
                
                if file_issues:
                    print(f"  ✗ {file_key}: {len(file_issues)} issues")
                else:
                    print(f"  ✓ {file_key}: passed")
        
        if all_issues:
            status = NodeStatus.FAILURE if self.fail_on_issues else NodeStatus.SUCCESS
            return NodeResult(
                status,
                {
                    "issues": all_issues,
                    "files_checked": files_checked,
                    "files_with_issues": len(set(issue.split(":")[0] for issue in all_issues))
                },
                f"Found {len(all_issues)} validation issues"
            )
        else:
            return NodeResult(
                NodeStatus.SUCCESS,
                {"files_checked": files_checked, "issues": []}
            )
    
    def _check_rule(self, code: str, rule: Dict[str, Any]) -> bool:
        """Check if code passes a validation rule."""
        rule_type = rule.get('type', 'contains')
        pattern = rule.get('pattern', '')
        
        if rule_type == 'contains':
            return pattern in code
        elif rule_type == 'not_contains':
            return pattern not in code
        elif rule_type == 'regex':
            import re
            return bool(re.search(pattern, code))
        elif rule_type == 'line_count':
            line_count = len(code.split('\n'))
            min_lines = rule.get('min', 0)
            max_lines = rule.get('max', float('inf'))
            return min_lines <= line_count <= max_lines
        elif rule_type == 'function_exists':
            # Simple check for function existence
            return f"def {pattern}" in code or f"{pattern}(" in code
        else:
            print(f"[Warning] Unknown validation rule type: {rule_type}")
            return True


class DynamicNode(Node):
    """Node that can execute arbitrary Python code or call custom handlers."""
    
    def __init__(self, name: str, handler: str, handler_config: Dict[str, Any], **kwargs):
        super().__init__(name, **kwargs)
        self.handler = handler
        self.handler_config = handler_config
        
    def execute(self, context: NodeContext) -> NodeResult:
        try:
            # Try to import and execute the handler
            if '.' in self.handler:
                # Module.function format
                module_name, func_name = self.handler.rsplit('.', 1)
                module = importlib.import_module(module_name)
                handler_func = getattr(module, func_name)
            else:
                # Look for handler in global registry
                handler_func = CUSTOM_NODE_REGISTRY.get(self.handler)
                if not handler_func:
                    return NodeResult(
                        NodeStatus.FAILURE,
                        {"error": f"Handler '{self.handler}' not found"}
                    )
            
            # Execute the handler
            result = handler_func(context, self.handler_config)
            
            # Ensure result is a NodeResult
            if not isinstance(result, NodeResult):
                return NodeResult(NodeStatus.SUCCESS, {"result": result})
            
            return result
            
        except Exception as e:
            return NodeResult(
                NodeStatus.FAILURE,
                {"error": str(e), "handler": self.handler},
                f"Handler execution failed: {str(e)}"
            )


# Global registry for custom node handlers
CUSTOM_NODE_REGISTRY: Dict[str, Callable] = {}

def register_node_handler(name: str):
    """Decorator to register custom node handlers."""
    def decorator(func: Callable):
        CUSTOM_NODE_REGISTRY[name] = func
        return func
    return decorator


# Example custom handlers
@register_node_handler("example_analyzer")
def example_analyzer_handler(context: NodeContext, config: Dict[str, Any]) -> NodeResult:
    """Example custom node handler."""
    agent = context.agent
    analysis_type = config.get('type', 'basic')
    
    if analysis_type == 'file_count':
        file_count = len([f for f in agent.files.values() if f.get('code')])
        return NodeResult(NodeStatus.SUCCESS, {"file_count": file_count})
    
    return NodeResult(NodeStatus.SUCCESS, {"analysis_type": analysis_type})


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
        """Create a node based on its type with support for arbitrary node types."""
        node_type = node_config.type
        
        # Built-in node types
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
            
        # Advanced node types
        elif node_type == 'multi_stage_setup':
            node = MultiStageSetupNode(
                node_name,
                description=node_config.description
            )
            
        elif node_type == 'multi_stage_generate':
            file_keys = node_config.config.get('files', [])
            node = MultiStageGenerateNode(
                node_name,
                file_keys=file_keys,
                use_api_validation=node_config.config.get('api_validation', True),
                max_refinement_iterations=node_config.config.get('max_refinements', 3),
                parallel=node_config.config.get('parallel', False),
                description=node_config.description
            )
            
        elif node_type == 'custom_validation':
            rules = node_config.config.get('rules', [])
            # Also check operation-level validation rules
            if hasattr(self.config, 'settings'):
                rules.extend(self.config.settings.validation_rules)
            
            node = CustomValidationNode(
                node_name,
                rules=rules,
                fail_on_issues=node_config.config.get('fail_on_issues', True),
                description=node_config.description
            )
            
        elif node_type == 'end':
            # End nodes are just setup nodes that we'll mark as end
            node = SetupNode(
                node_name,
                description=node_config.description or "End node"
            )
            
        else:
            # Handle arbitrary custom node types
            node = self._create_custom_node(node_name, node_config)
            
        self.builder.add_node(node)

    def _create_custom_node(self, node_name: str, node_config: NodeConfig) -> Node:
        """Create a custom node type using registered handlers or dynamic execution."""
        node_type = node_config.type
        
        # Check if there's a registered handler in operation settings
        if hasattr(self.config, 'settings') and node_type in self.config.settings.custom_node_handlers:
            handler_class_path = self.config.settings.custom_node_handlers[node_type]
            
            try:
                # Import the handler class
                module_name, class_name = handler_class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                handler_class = getattr(module, class_name)
                
                # Create instance
                return handler_class(node_name, **node_config.config, description=node_config.description)
                
            except Exception as e:
                print(f"[Warning] Failed to load custom handler '{handler_class_path}': {e}")
        
        # Check for handler in config
        handler = node_config.config.get('handler')
        if handler:
            return DynamicNode(
                node_name,
                handler=handler,
                handler_config=node_config.config,
                description=node_config.description
            )
        
        # Fallback to generic custom node
        return ConfigCustomNode(
            node_name,
            node_type=node_type,
            config=node_config.config,
            description=node_config.description
        )
        
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