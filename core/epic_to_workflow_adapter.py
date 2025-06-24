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
                ro_dependencies.get(epic_node_name, [])  # Pass RO deps
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
        
        return builder.build()
    
    def _create_workflow_node(self, epic_name: str, opcode: Opcode, 
                              contents: dict) -> Optional[Node]:
        """Create appropriate workflow node based on opcode."""
        
        if opcode == Opcode.TEMPLATE:
            # Store template path for context
            self.template_context['template_path'] = contents.get('path', '')
            # Return a setup node that loads templates
            return TemplateLoaderNode(
                name=f"load_template_{self.node_counter}",
                config={'template_path': contents.get('path', '')}
            )
            
        if opcode == Opcode.PROMPT:
            # Convert prompt to file generation node
            prompt_text = contents.get('prompt', '')
            self.node_counter += 1
            return PromptExecutorNode(
                name=f"execute_prompt_{self.node_counter}",
                config={
                    'prompt': prompt_text,
                    'template_context': self.template_context.copy(),
                    'ro_files': ro_deps or []
                }
            )
            
        elif opcode == Opcode.RUN:
            # Map to build verification
            return BuildVerificationNode(
                name=f"build_verify_{self.node_counter}",
                config={}
            )
            
        elif opcode == Opcode.DEBUG_LOOP:
            # Don't create edges from this node - it's a marker
            self.node_counter += 1
            return DebugLoopNode(
                name=f"debug_loop_{self.node_counter}",
                config={'max_iterations': 3}
            )
            
        elif opcode == Opcode.EXIT:
            # Create completion node
            return CompletionNode(
                name=f"completion_{self.node_counter}",
                config={}
            )
            
        elif opcode == Opcode.READ_ONLY:
            # Store RO file reference
            self.ro_files[epic_name] = contents.get('path', '')
            # RO nodes don't translate directly - they're dependencies
            return None
        
        elif opcode == Opcode.COMMAND:
            command = contents.get('command', '')
            if command == 'enable_multi_stage':
                return MultiStageEnableNode(
                    name=f"enable_multi_stage_{self.node_counter}",
                    config={}
                )

        self.node_counter += 1
        return None
    
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
        
        template_path = self.config.get('template_path', '')
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
        
        return template


class PromptExecutorNode(Node):
    """Execute a prompt to generate code."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        from ttnn_op_generator.core.node_types import NodeResult, NodeStatus
        
        prompt = self.config.get('prompt', '')
        ro_files = self.config.get('ro_files', [])
        
        print(f"\n[PromptExecutor] Executing prompt")
        print(f"RO Dependencies: {ro_files}")
        
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
        for file_key in files_to_generate:
            # Build context including RO files
            full_context = self._build_generation_context(prompt, ro_contents)
            
            # Use agent to generate
            code = agent.generate_with_refined_prompt(
                full_context, 
                file_key
            )
            
            agent.save_file(file_key, code)
        
        return NodeResult(
            NodeStatus.SUCCESS, 
            {"files_generated": files_to_generate}
        )
    
    def _parse_prompt_intent(self, prompt: str) -> List[str]:
        """Determine which files the prompt wants to generate."""
        # Simple heuristic - look for keywords
        files = []
        
        prompt_lower = prompt.lower()
        if "test" in prompt_lower:
            files.extend(["test_cpp", "test_py"])
        elif "program" in prompt_lower or "add two numbers" in prompt_lower:
            # Likely wants the full operation
            files.extend(["hpp", "cpp", "op-hpp", "op", "compute"])
        else:
            # Default to main files
            files.extend(["hpp", "cpp"])
            
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