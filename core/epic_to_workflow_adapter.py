# epic_to_workflow_adapter.py
"""Adapter to convert EpicIR graphs to WorkflowGraph instances."""

from typing import Dict, List, Optional
import networkx as nx
from pathlib import Path

from ttnn_op_generator.front_end.front_end_parser import EpicIR, Opcode
from ttnn_op_generator.core.workflow_graph import WorkflowGraph, GraphBuilder
from ttnn_op_generator.core.graph_nodes import (
    GenerateFileNode, BuildVerificationNode, DebugAnalysisNode,
    DebugFixNode, LoopControlNode, SetupNode
)
from ttnn_op_generator.core.node_types import Node, NodeContext, NodeResult

class EpicToWorkflowAdapter:
    """Converts EpicIR graphs to WorkflowGraph instances."""
    
    def __init__(self):
        self.node_counter = 0
        self.template_context = {}
        self.ro_files = {}
        
    # In epic_to_workflow_adapter.py, ensure the convert method is complete:

    def convert(self, epic_ir: EpicIR) -> WorkflowGraph:
        """Convert an EpicIR graph to a WorkflowGraph."""
        builder = GraphBuilder(name="epic_workflow")
        
        # Track RO dependencies for prompts
        ro_dependencies = {}
        
        # First, find all RO nodes and their targets
        for source, target in epic_ir.graph.edges():
            node_data = epic_ir.graph.nodes[source]
            if node_data['opcode'] == Opcode.READ_ONLY:
                if target not in ro_dependencies:
                    ro_dependencies[target] = []
                ro_dependencies[target].append(node_data['contents'].get('path', ''))
        
        # Create workflow nodes
        node_mapping = {}
        for epic_node_name, epic_node_data in epic_ir.graph.nodes(data=True):
            # Skip RO nodes - they're dependencies, not execution nodes
            if epic_node_data['opcode'] == Opcode.READ_ONLY:
                continue
                
            workflow_node = self._create_workflow_node(
                epic_node_name, 
                epic_node_data['opcode'], 
                epic_node_data['contents'],
                ro_dependencies.get(epic_node_name, [])
            )
            
            if workflow_node:
                builder.add_node(workflow_node)
                node_mapping[epic_node_name] = workflow_node.name
        
        # Create edges (skip edges from RO nodes)
        for source, target in epic_ir.graph.edges():
            source_data = epic_ir.graph.nodes[source]
            if source_data['opcode'] == Opcode.READ_ONLY:
                continue  # Skip edges from RO nodes
                
            if source in node_mapping and target in node_mapping:
                builder.add_edge(node_mapping[source], node_mapping[target])
        
        # Set start node
        if epic_ir.first_node and epic_ir.first_node in node_mapping:
            builder.set_start(node_mapping[epic_ir.first_node])
        
        # Find and mark end nodes (nodes with no outgoing edges)
        all_targets = set()
        all_sources = set()
        for source, target in epic_ir.graph.edges():
            if source in node_mapping:
                all_sources.add(node_mapping[source])
            if target in node_mapping:
                all_targets.add(node_mapping[target])
        
        # Nodes that are sources but not targets are potential end nodes
        for node_name in node_mapping.values():
            outgoing_edges = builder.graph.get_outgoing_edges(node_name)
            if len(outgoing_edges) == 0:
                # This is a leaf node
                builder.add_end(node_name)
        
        return builder.build()
        
    def _create_workflow_node(self, epic_name: str, opcode: Opcode, 
                          contents: dict, ro_deps: List[str] = None) -> Optional[Node]:
        """Create appropriate workflow node based on opcode."""
        
        # DEBUG: Let's see what we're getting
        print(f"[DEBUG _create_workflow_node] Creating node for {opcode.value}")
        print(f"[DEBUG] Contents: {contents}")
        
        if opcode == Opcode.TEMPLATE:
            # Store template path for context
            self.template_context['template_path'] = contents.get('path', '')
            # Return a setup node that loads templates
            self.node_counter += 1
            return TemplateLoaderNode(
                name=f"load_template_{self.node_counter}",
                template_path=contents.get('path', '')  # Pass directly, not as config dict
            )
            
        elif opcode == Opcode.PROMPT:
            # Convert prompt to file generation node
            prompt_text = contents.get('prompt', '')
            self.node_counter += 1
            
            # DEBUG
            print(f"[DEBUG] Creating PromptExecutorNode with prompt length: {len(prompt_text)}")
            
            return PromptExecutorNode(
                name=f"execute_prompt_{self.node_counter}",
                prompt=prompt_text,  # Pass directly as kwargs
                template_context=self.template_context.copy(),
                ro_files=ro_deps or []
            )
        
    def _extract_ro_references(self, prompt_node_name: str) -> List[str]:
        """Extract RO file references for a prompt node."""
        # In the EpicIR, RO nodes have edges TO the prompt node
        # We need to find these in the original graph
        # This would require passing the EpicIR graph reference
        return []


# Custom node implementations for Epic workflow concepts

class TemplateLoaderNode(Node):
    """Load and prepare templates for code generation."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        from ttnn_op_generator.core.node_types import NodeResult, NodeStatus
        
        template_path = self.config.get('template_path', '')  # Now this will work
        print(f"\n[TemplateLoader] Loading templates from: {template_path}")
        
        # Store template information in context
        context.set_global('template_path', template_path)
        
        # Could load actual template files here
        template_files = self._load_templates(template_path)
        context.set_global('templates', template_files)
        
        return NodeResult(NodeStatus.SUCCESS, {"templates_loaded": len(template_files)})

    def _load_templates(self, path: str) -> Dict[str, str]:
        """Load template files from directory."""
        templates = {}
        template_dir = Path(path)
        
        # Text file extensions to load
        text_extensions = {'.hpp', '.cpp', '.h', '.c', '.txt', '.template', '.py', '.cmake'}
        
        if template_dir.exists() and template_dir.is_dir():
            for file_path in template_dir.glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                    try:
                        rel_path = file_path.relative_to(template_dir)
                        templates[str(rel_path)] = file_path.read_text(encoding='utf-8')
                    except UnicodeDecodeError:
                        print(f"[TemplateLoader] Skipping non-text file: {file_path}")
                    except Exception as e:
                        print(f"[TemplateLoader] Error reading {file_path}: {e}")
        
        return templates

class PromptExecutorNode(Node):
    """Execute a prompt to generate code."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        from ttnn_op_generator.core.node_types import NodeResult, NodeStatus
        
        # Now access directly from self.config
        prompt = self.config.get('prompt', '')
        ro_files = self.config.get('ro_files', [])
        
        print(f"\n[PromptExecutor] Executing prompt")
        print(f"[DEBUG] Config keys: {list(self.config.keys())}")
        print(f"[DEBUG] Prompt length: {len(prompt)}")
        print(f"[DEBUG] Prompt preview: {prompt[:200] if prompt else 'EMPTY'}...")
        print(f"RO Dependencies: {ro_files}")
        
        # Check if prompt is empty
        if not prompt.strip():
            return NodeResult(
                NodeStatus.FAILURE,
                {"error": "Empty prompt"},
                "Prompt is empty"
            )
        
        # Read RO files if they exist
        ro_contents = {}
        for ro_path in ro_files:
            try:
                with open(ro_path, 'r') as f:
                    ro_contents[ro_path] = f.read()
            except Exception as e:
                print(f"Warning: Could not read RO file {ro_path}: {e}")
        
        # Parse prompt to determine what files to generate
        files_to_generate = self._parse_prompt_intent(prompt)
        
        # Generate each file
        agent = context.agent
        generated_files = []
        failed_files = []
        
        for file_key in files_to_generate:
            try:
                # Build context including RO files
                full_context = self._build_generation_context(prompt, ro_contents)
                
                # Use agent to generate
                code = agent.generate_with_refined_prompt(
                    full_context, 
                    file_key
                )
                
                agent.save_file(file_key, code)
                generated_files.append(file_key)
            except Exception as e:
                print(f"[PromptExecutor] Error generating {file_key}: {e}")
                failed_files.append(file_key)
        
        if generated_files:
            return NodeResult(
                NodeStatus.SUCCESS, 
                {"files_generated": generated_files, "files_failed": failed_files}
            )
        else:
            return NodeResult(
                NodeStatus.FAILURE,
                {"files_failed": failed_files},
                "Failed to generate any files"
            )
    
    def _parse_prompt_intent(self, prompt: str) -> List[str]:
        """Determine which files the prompt wants to generate."""
        # Simple heuristic - look for keywords
        files = []
        
        prompt_lower = prompt.lower()
        
        # Look for specific file mentions
        if "eltwise_multiply_custom.hpp" in prompt:
            files.append("hpp")
        if "eltwise_multiply_custom.cpp" in prompt:
            files.append("cpp")
        if "device/eltwise_multiply_custom_op.hpp" in prompt:
            files.append("op-hpp")
        if "device/eltwise_multiply_custom_op.cpp" in prompt:
            files.append("op")
        if "program_factory.hpp" in prompt:
            files.append("program-factory-hpp")
        if "program_factory.cpp" in prompt:
            files.append("program-factory")
        if "reader.cpp" in prompt:
            files.append("reader")
        if "writer.cpp" in prompt:
            files.append("writer")
        if "compute.cpp" in prompt:
            files.append("compute")
        if "pybind.hpp" in prompt:
            files.append("pybind-hpp")
        if "pybind.cpp" in prompt:
            files.append("pybind-cpp")
        if "CMakeLists.txt" in prompt:
            files.append("cmake")
            
        # Fallback patterns
        if not files:
            if "header" in prompt_lower and "implementation" in prompt_lower:
                files.extend(["hpp", "cpp"])
            elif "device operation" in prompt_lower:
                files.extend(["op-hpp", "op"])
            elif "program factory" in prompt_lower:
                files.extend(["program-factory-hpp", "program-factory"])
            elif "kernel" in prompt_lower:
                files.extend(["reader", "writer", "compute"])
            elif "python binding" in prompt_lower:
                files.extend(["pybind-hpp", "pybind-cpp"])
            elif "cmake" in prompt_lower:
                files.append("cmake")
            
        print(f"[PromptExecutor] Parsed intent - files to generate: {files}")
        return files
    
    def _build_generation_context(self, prompt: str, ro_contents: Dict[str, str]) -> str:
        """Build full context for generation."""
        context_parts = [prompt]
        
        if ro_contents:
            context_parts.append("\n\nReference Files:")
            for path, content in ro_contents.items():
                context_parts.append(f"\n--- {path} ---")
                context_parts.append(content)
        
        return "\n".join(context_parts)


class DebugLoopNode(Node):
    """Implements a debug loop that retries on build failures."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        from ttnn_op_generator.core.node_types import NodeResult, NodeStatus
        
        # This node sets up the loop context
        # The actual loop logic is handled by graph edges
        max_iterations = self.config.get('max_iterations', 3)
        
        loop_name = "debug_loop"
        current = context.iteration_counts.get(loop_name, 0)
        
        if current >= max_iterations:
            return NodeResult(
                NodeStatus.SUCCESS,
                {"should_exit": True},
                "Max iterations reached"
            )
        
        return NodeResult(
            NodeStatus.SUCCESS,
            {"should_continue": True}
        )


class CompletionNode(Node):
    """Marks successful completion of the workflow."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        from ttnn_op_generator.core.node_types import NodeResult, NodeStatus
        
        print("\n[Completion] Workflow completed successfully!")
        
        # Could do cleanup, reporting, etc.
        agent = context.agent
        
        return NodeResult(
            NodeStatus.SUCCESS,
            {"operation_name": agent.operation_name}
        )


class MultiStageEnableNode(Node):
    """Enable multi-stage generation in the agent."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        from ttnn_op_generator.core.node_types import NodeResult, NodeStatus
        
        agent = context.agent
        if hasattr(agent, 'enable_multi_stage_generation'):
            agent.enable_multi_stage_generation()
            print("[MultiStageEnable] Multi-stage generation enabled")
        
        return NodeResult(NodeStatus.SUCCESS)