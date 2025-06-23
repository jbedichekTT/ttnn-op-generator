"""Concrete node implementations for TTNN operation generation."""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from ttnn_op_generator.core.node_types import Node, NodeResult, NodeStatus, NodeContext

class GenerateFileNode(Node):
    """Node for generating a specific file."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        file_key = self.config['file_key']
        dependencies = self.config.get('dependencies', [])
        
        agent = context.agent
        
        print(f"\n[GenerateFileNode] Generating {file_key} ({agent.files[file_key]['name']})")
        
        # Build dependency context
        dep_context = ""
        for dep_key in dependencies:
            if dep_key in agent.files and agent.files[dep_key]["code"]:
                dep_name = agent.files[dep_key]["name"]
                dep_context += f"\n--- Reference File: {dep_name} ---\n{agent.files[dep_key]['code']}\n"
        
        # Get appropriate context based on file type
        base_context = self._get_file_context(file_key)
        
        base_prompt = (
            f"Generate the code for the file `{agent.files[file_key]['name']}` "
            f"for the `{agent.operation_name}` operation. "
            f"Use the provided context and reference files to ensure consistency.\n\n{base_context}"
        )
        
        try:
            # Use multi-stage generation if enabled
            if agent.use_multi_stage and hasattr(agent, 'multi_stage_generator') and agent.multi_stage_generator:
                code = agent.multi_stage_generator.generate_file_multi_stage(
                    file_key, base_prompt, dependencies
                )
            else:
                code = agent.generate_with_refined_prompt(base_prompt, file_key, dep_context)
            
            agent.save_file(file_key, code)
            return NodeResult(
                NodeStatus.SUCCESS, 
                {"file_key": file_key, "file_name": agent.files[file_key]['name']}
            )
            
        except Exception as e:
            return NodeResult(
                NodeStatus.FAILURE, 
                {"file_key": file_key, "error": str(e)},
                f"Failed to generate {file_key}: {str(e)}"
            )
    
    def _get_file_context(self, file_key: str) -> str:
        """Get the appropriate context for a file type."""
        
        # Get base prompt from agent's prompt system
        from ..prompts import ( # Corrected import path
            HPP_CONTEXT, CPP_CONTEXT, DEVICE_OP_CONTEXT,
            PROGRAM_FACTORY_CONTEXT, KERNEL_CONTEXT,
            PYBIND_CONTEXT, CMAKE_CONTEXT
        )
        
        context_map = {
            "hpp": HPP_CONTEXT, 
            "cpp": CPP_CONTEXT,
            "op-hpp": DEVICE_OP_CONTEXT, 
            "op": DEVICE_OP_CONTEXT,
            "program-factory-hpp": PROGRAM_FACTORY_CONTEXT,
            "program-factory": PROGRAM_FACTORY_CONTEXT,
            "reader": KERNEL_CONTEXT, 
            "writer": KERNEL_CONTEXT,
            "compute": KERNEL_CONTEXT, 
            "pybind-hpp": PYBIND_CONTEXT,
            "pybind-cpp": PYBIND_CONTEXT, 
            "cmake": CMAKE_CONTEXT,
        }
        
        return context_map.get(file_key, "")



class BuildVerificationNode(Node):
    """Node for running build verification."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        print(f"\n[BuildVerificationNode] Running build verification")
        
        success, error_output = agent.run_build_verification()
        
        if success:
            print("[BuildVerificationNode] Build successful!")
            if hasattr(agent, 'prompt_refiner'):
                agent.prompt_refiner.mark_build_success()
            return NodeResult(NodeStatus.SUCCESS, {"build_output": "Build successful"})
        else:
            print("[BuildVerificationNode] Build failed")
            
            # Clean and analyze errors
            cleaned_errors = agent.clean_build_errors(error_output)
            critical_errors = agent.extract_critical_errors(error_output)
            
            return NodeResult(
                NodeStatus.FAILURE, 
                {
                    "raw_errors": error_output,
                    "cleaned_errors": cleaned_errors,
                    "critical_errors": critical_errors
                },
                "Build failed with errors"
            )


class DebugAnalysisNode(Node):
    """Node for analyzing build errors and identifying files to fix."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        print(f"\n[DebugAnalysisNode] Analyzing build errors")
        
        # Get error data from previous build node
        error_source = self.config.get('error_source', 'build')
        build_result = context.get_node_output(error_source)
        
        if not build_result or build_result.status == NodeStatus.SUCCESS:
            return NodeResult(NodeStatus.SKIP, message="No errors to analyze")
            
        cleaned_errors = build_result.data.get('cleaned_errors', '')
        critical_errors = build_result.data.get('critical_errors', {})
        
        # Identify files to fix
        files_to_fix = agent._identify_files_to_fix(cleaned_errors, critical_errors)
        
        if not files_to_fix:
            return NodeResult(
                NodeStatus.FAILURE, 
                {"cleaned_errors": cleaned_errors, "critical_errors": critical_errors},
                "Could not identify files to fix"
            )
        
        print(f"[DebugAnalysisNode] Identified files to fix: {files_to_fix}")
        
        return NodeResult(
            NodeStatus.SUCCESS, 
            {
                "files_to_fix": files_to_fix,
                "cleaned_errors": cleaned_errors,
                "critical_errors": critical_errors
            }
        )


class DebugFixNode(Node):
    """Node for generating fixes for identified files."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        print(f"\n[DebugFixNode] Applying fixes")
        
        # Get analysis data
        analysis_source = self.config.get('analysis_source', 'debug_analysis')
        analysis = context.get_node_output(analysis_source)
        
        if not analysis or analysis.status != NodeStatus.SUCCESS:
            return NodeResult(NodeStatus.SKIP, message="No analysis data available")
            
        files_to_fix = analysis.data.get('files_to_fix', [])
        cleaned_errors = analysis.data.get('cleaned_errors', '')
        critical_errors = analysis.data.get('critical_errors', {})
        
        fixes_applied = False
        fixed_files = []
        
        # Try targeted editing first if configured
        if self.config.get('use_targeted_editing', True) and hasattr(agent, '_use_targeted_editing'):
            for file_key in files_to_fix:
                if file_key in agent.files and agent.files[file_key]["code"]:
                    print(f"[DebugFixNode] Attempting targeted edit for {file_key}")
                    if agent._use_targeted_editing(file_key, cleaned_errors):
                        fixes_applied = True
                        fixed_files.append(file_key)
        
        # Fall back to full regeneration for remaining files
        remaining_files = [f for f in files_to_fix if f not in fixed_files]
        if remaining_files:
            print(f"[DebugFixNode] Regenerating files: {remaining_files}")
            
            for file_key in remaining_files:
                fix_prompt = agent._create_single_file_debugging_prompt(
                    file_key, cleaned_errors, critical_errors
                )
                
                response_text = agent.generate_with_refined_prompt(fix_prompt, file_key)
                code = agent.parse_response(response_text)
                
                if code:
                    agent.save_file(file_key, code)
                    fixes_applied = True
                    fixed_files.append(file_key)
        
        return NodeResult(
            NodeStatus.SUCCESS if fixes_applied else NodeStatus.FAILURE,
            {"fixes_applied": fixes_applied, "fixed_files": fixed_files}
        )


class ConditionalNode(Node):
    """Node that evaluates a condition and determines next path."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        # Evaluate condition based on context
        condition_fn = self.config.get('condition')
        
        if condition_fn:
            try:
                result = condition_fn(context)
                return NodeResult(
                    NodeStatus.SUCCESS, 
                    {"condition_result": result}
                )
            except Exception as e:
                return NodeResult(
                    NodeStatus.FAILURE,
                    {"error": str(e)},
                    f"Condition evaluation failed: {str(e)}"
                )
        
        return NodeResult(NodeStatus.SUCCESS, {"condition_result": True})


class LoopControlNode(Node):
    """Node that manages loop iterations."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        loop_name = self.config.get('loop_name', self.name)
        max_iterations = self.config.get('max_iterations', 3)
        
        # Initialize or increment iteration count
        current = context.iteration_counts.get(loop_name, 0)
        context.iteration_counts[loop_name] = current + 1
        
        print(f"\n[LoopControlNode] {loop_name} iteration {current + 1}/{max_iterations}")
        
        # Check if should continue
        should_continue = current < max_iterations
        
        # Check additional exit condition if provided
        exit_condition = self.config.get('exit_condition')
        if exit_condition:
            try:
                if exit_condition(context):
                    should_continue = False
                    print(f"[LoopControlNode] Exit condition met")
            except Exception as e:
                print(f"[LoopControlNode] Error checking exit condition: {e}")
        
        return NodeResult(
            NodeStatus.SUCCESS,
            {
                "should_continue": should_continue, 
                "iteration": current + 1,
                "max_iterations": max_iterations
            }
        )


class SetupNode(Node):
    """Node for initial setup operations."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        print(f"\n[SetupNode] Setting up for operation: {agent.operation_name}")
        
        # Verify TT-Metal setup
        if not agent.verify_tt_metal_setup():
            return NodeResult(
                NodeStatus.FAILURE,
                message="TT-Metal setup verification failed"
            )
        
        # Clean/create output directory
        if agent.output_dir.exists():
            shutil.rmtree(agent.output_dir)
        agent.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize any other required state
        context.set_global('setup_complete', True)
        
        return NodeResult(NodeStatus.SUCCESS)


class CMakeUpdateNode(Node):
    """Node for updating CMake files."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        if not self.config.get('enabled', False):
            return NodeResult(NodeStatus.SKIP, message="CMake update disabled")
        
        cmake_path = self.config.get('cmake_path', "/home/user/tt-metal/ttnn/CMakeLists.txt")
        
        print(f"\n[CMakeUpdateNode] Updating CMake at {cmake_path}")
        
        try:
            success = agent.add_operation_to_cmake(cmake_path, agent.operation_type)
            
            if success:
                return NodeResult(NodeStatus.SUCCESS, {"cmake_path": cmake_path})
            else:
                return NodeResult(
                    NodeStatus.FAILURE,
                    message="Failed to update CMake file"
                )
        except Exception as e:
            return NodeResult(
                NodeStatus.FAILURE,
                {"error": str(e)},
                f"CMake update failed: {str(e)}"
            )


class TestExecutionNode(Node):
    """Node for running tests."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        test_path = self.config.get('test_path')
        if not test_path:
            return NodeResult(NodeStatus.SKIP, message="No test path specified")
        
        if not agent.run_tests:
            return NodeResult(NodeStatus.SKIP, message="Test execution disabled")
        
        print(f"\n[TestExecutionNode] Running test: {test_path}")
        
        if hasattr(agent, 'run_and_debug_test'):
            success = agent.run_and_debug_test(test_path)
            
            return NodeResult(
                NodeStatus.SUCCESS if success else NodeStatus.FAILURE,
                {"test_path": test_path}
            )
        else:
            return NodeResult(
                NodeStatus.SKIP,
                message="Test execution not available"
            )


class MultiStageSetupNode(Node):
    """Node for enabling multi-stage generation."""
    
    def execute(self, context: NodeContext) -> NodeResult:
        agent = context.agent
        
        print(f"\n[MultiStageSetupNode] Enabling multi-stage generation")
        
        if hasattr(agent, 'enable_multi_stage_generation'):
            agent.enable_multi_stage_generation()
            return NodeResult(NodeStatus.SUCCESS)
        else:
            return NodeResult(
                NodeStatus.SKIP,
                message="Multi-stage generation not available"
            )