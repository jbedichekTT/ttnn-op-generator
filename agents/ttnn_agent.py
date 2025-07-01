"""
Refactored TTNN Operation Agent for Graph-Based Workflow System
==============================================================

This agent bridges the legacy operation generation logic with the new
graph-based workflow system, providing all the methods expected by the
workflow nodes while maintaining compatibility with existing components.
"""

import os
import re
import json
import time
import shutil
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

# Import workflow components
from ttnn_op_generator.core.workflow_graph import WorkflowGraph, GraphExecutor
from ttnn_op_generator.core.execution_logger import ExecutionLogger
from ttnn_op_generator.graphs.default_graphs import (
    create_default_graph,
    create_multi_stage_graph,
    create_quick_debug_graph,
    create_partial_completion_graph,
    create_test_driven_graph
)

# Import existing components from the legacy system
from ttnn_op_generator.prompts import *
from ttnn_op_generator.tools.tools import AVAILABLE_TOOLS, TOOL_EXECUTORS
from ttnn_op_generator.refinement.persistent_prompt_refiner import PersistentPromptRefiner

# Configuration (from legacy)
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
BUILD_RETRIES = 3
MAX_TOKENS = 8192
THINKING_TOKENS = 1024
TOOL_USE_DELAY = 1


class TTNNOperationAgent:
    """
    Refactored agent for generating TTNN operations using graph-based workflows.
    
    This agent provides all the methods expected by the workflow nodes while
    leveraging the graph execution system for orchestration.
    """
    
    def __init__(
        self,
        operation_type: str = "add",
        tt_metal_path: str = "/path/to/tt-metal",
        api_key: str = "",
        build_retries: int = BUILD_RETRIES,
        run_tests: bool = True,
        custom_suffix: str = "custom",
        refinement_db_path: str = "ttnn_refinements_db.json",
        enable_logging: bool = True,
        log_dir: str = "workflow_logs"
    ):
        """Initialize the agent with operation configuration."""
        # API configuration
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("API key must be provided or set in ANTHROPIC_API_KEY environment variable")
            
        self.model = MODEL
        
        # Operation configuration
        self.operation_type = operation_type
        self.custom_suffix = custom_suffix
        self.operation_name = f"eltwise_{operation_type}_{custom_suffix}"
        self.operation_class_name = f"Eltwise{operation_type.title()}{custom_suffix.title()}"
        self.python_function_name = f"eltwise_{operation_type}_{custom_suffix}"
        
        # Paths
        self.tt_metal_path = Path(tt_metal_path)
        self.ttnn_ops_path = self.tt_metal_path / "ttnn/cpp/ttnn/operations/"
        self.output_dir = self.ttnn_ops_path / self.operation_name
        self.build_dir = self.tt_metal_path / "build"
        
        # Configuration
        self.build_retries = build_retries
        self.run_tests = run_tests
        self.max_tokens = MAX_TOKENS
        self.iteration = 0
        self.generation_attempt = 0
        
        # File structure
        self.files = {
            "hpp": {"name": f"{self.operation_name}.hpp", "code": ""},
            "cpp": {"name": f"{self.operation_name}.cpp", "code": ""},
            "op-hpp": {"name": f"device/{self.operation_name}_op.hpp", "code": ""},
            "op": {"name": f"device/{self.operation_name}_op.cpp", "code": ""},
            "program-factory-hpp": {"name": f"device/{self.operation_name}_program_factory.hpp", "code": ""},
            "program-factory": {"name": f"device/{self.operation_name}_program_factory.cpp", "code": ""},
            "reader": {"name": f"device/kernels/dataflow/{self.operation_name}_reader.cpp", "code": ""},
            "writer": {"name": f"device/kernels/dataflow/{self.operation_name}_writer.cpp", "code": ""},
            "compute": {"name": f"device/kernels/compute/{self.operation_name}_compute.cpp", "code": ""},
            "pybind-hpp": {"name": f"{self.operation_name}_pybind.hpp", "code": ""},
            "pybind-cpp": {"name": f"{self.operation_name}_pybind.cpp", "code": ""},
            "cmake": {"name": "CMakeLists.txt", "code": ""},
        }
        
        # Components
        self.prompt_refiner = PersistentPromptRefiner(
            self.operation_name, 
            db_path=refinement_db_path,
            load_from_db=True
        )
        
        # Multi-stage generation support
        self.multi_stage_generator = None
        self.use_multi_stage = False
        
        # Workflow graph (to be set)
        self.workflow_graph = None
        
        # Execution logger
        self.logger = ExecutionLogger(log_dir=log_dir, enabled=enable_logging)
        
        # Set environment variables for tools
        os.environ["TT_METAL_PATH"] = str(self.tt_metal_path)
        os.environ["TTNN_OUTPUT_DIR"] = str(self.output_dir)
        
    # ==================== Workflow Management ====================
    
    def use_default_workflow(self):
        """Use the default workflow graph."""
        self.workflow_graph = create_default_graph()
        print(f"[Agent] Using default workflow: {self.workflow_graph.name}")
        
    def use_multi_stage_workflow(self):
        """Use the multi-stage generation workflow."""
        self.workflow_graph = create_multi_stage_graph()
        print(f"[Agent] Using multi-stage workflow: {self.workflow_graph.name}")
        
    def use_quick_debug_workflow(self):
        """Use the quick debug workflow (assumes files exist)."""
        self.workflow_graph = create_quick_debug_graph()
        print(f"[Agent] Using quick debug workflow: {self.workflow_graph.name}")
        
    def use_partial_completion_workflow(self):
        """Use the partial completion workflow."""
        self.workflow_graph = create_partial_completion_graph()
        print(f"[Agent] Using partial completion workflow: {self.workflow_graph.name}")
        
    def use_test_driven_workflow(self, test_path: str):
        """Use the test-driven workflow."""
        self.workflow_graph = create_test_driven_graph(test_path)
        print(f"[Agent] Using test-driven workflow: {self.workflow_graph.name}")
        
    def set_workflow_graph(self, graph: WorkflowGraph):
        """Set a custom workflow graph."""
        self.workflow_graph = graph
        print(f"[Agent] Using custom workflow: {graph.name}")
        
    def build_operation(self) -> bool:
        """
        Execute the workflow to build the operation.
        
        This is the main entry point that runs the configured workflow graph.
        """
        if not self.workflow_graph:
            print("[Agent] No workflow set, using default")
            self.use_default_workflow()
            
        print(f"\n[Agent] Starting workflow execution for: {self.operation_name}")
        
        # Create executor with logging
        from ttnn_op_generator.core.node_types import NodeContext
        
        executor = GraphExecutor(self.workflow_graph, logger=self.logger)
        context = NodeContext(agent=self)
        
        # Execute the workflow
        success = executor.execute(context)
        
        return success
        
    # ==================== Core Methods Expected by Nodes ====================
    
    def verify_tt_metal_setup(self) -> bool:
        """Verify TT-Metal repository is properly set up."""
        if not self.tt_metal_path.exists():
            print(f"[Error] TT-Metal path not found: {self.tt_metal_path}")
            return False
            
        build_script = self.tt_metal_path / "build_metal.sh"
        if not build_script.exists():
            print(f"[Error] build_metal.sh not found at: {build_script}")
            return False
            
        if not os.access(build_script, os.X_OK):
            print(f"[Warning] build_metal.sh is not executable, attempting to fix...")
            try:
                os.chmod(build_script, 0o755)
            except Exception as e:
                print(f"[Error] Could not make build_metal.sh executable: {e}")
                return False
                
        print("[Setup] TT-Metal setup verified successfully")
        return True
        
    def run_build_verification(self) -> Tuple[bool, str]:
        """Run TT-Metal build to verify the generated operation compiles."""
        print(f"[Build] Running build verification for {self.operation_name}")
        
        original_dir = os.getcwd()
        os.chdir(self.tt_metal_path)
        
        try:
            build_script = "./build_metal.sh"
            env = os.environ.copy()
            env["TT_METAL_HOME"] = str(self.tt_metal_path)
            
            result = subprocess.run(
                [build_script],
                capture_output=True,
                text=True,
                timeout=1200,
                env=env,
                shell=False
            )
            
            if result.returncode == 0:
                print(f"[Build] Success! {self.operation_name} compiled successfully.")
                return True, "Build successful"
            else:
                print(f"[Build] Failed with return code: {result.returncode}")
                return False, result.stdout + "\n" + result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Build timeout - build_metal.sh took too long (>20 minutes)"
        except Exception as e:
            return False, f"Build failed with exception: {str(e)}"
        finally:
            os.chdir(original_dir)
            
    def save_file(self, file_key: str, code: str):
        """Save generated file to the appropriate location."""
        file_config = self.files[file_key]
        full_path = self.output_dir / file_config["name"]
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with full_path.open("w") as f:
            f.write(code)
            
        print(f"  ✓ Saved: {file_config['name']}")
        self.files[file_key]["code"] = code
        
    def clean_build_errors(self, raw_error_output: str) -> str:
        """Clean and extract only relevant compilation errors from build output."""
        lines = raw_error_output.split("\n")
        
        # Patterns to identify lines we want to keep
        error_patterns = [
            r"error:",
            r"fatal error:",
            r"FAILED:",
            r"undefined reference",
            r"no matching function",
            r"redefinition of",
            r"previous definition",
            r"file not found",
            r"cannot find",
            r"CMake Error",
            r"ninja: build stopped",
        ]
        
        # Patterns to skip
        skip_patterns = [
            r"^INFO:",
            r"^-- ",
            r"^\[.*\] Building",
            r"^\[.*\] Re-checking",
            r"^CMake.*:",
            r"^CPM:",
            r"^CCACHE_",
            r"Configuring done",
            r"Generating done",
            r"Build files have been written",
        ]
        
        cleaned_errors = []
        error_context = []
        in_error_block = False
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Skip known non-error patterns
            if any(re.search(pattern, line) for pattern in skip_patterns):
                continue
                
            # Check if this line contains an error
            is_error_line = any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns)
            
            if is_error_line:
                in_error_block = True
                if error_context:
                    cleaned_errors.extend(error_context[-3:])
                    error_context = []
                cleaned_errors.append(line)
            elif in_error_block:
                if line.strip().startswith("^") or "|" in line[:10]:
                    cleaned_errors.append(line)
                elif len(cleaned_errors) > 0 and len(cleaned_errors[-1]) > 0:
                    cleaned_errors.append(line)
                    in_error_block = False
            else:
                error_context.append(line)
                if len(error_context) > 5:
                    error_context.pop(0)
                    
        # Format cleaned output
        return self._format_cleaned_errors(cleaned_errors)
        
    def extract_critical_errors(self, raw_error_output: str) -> Dict[str, List[str]]:
        """Extract critical errors and organize them by error type."""
        lines = raw_error_output.split("\n")
        errors_by_type = {
            "file_not_found": [],
            "undefined_reference": [],
            "redefinition": [],
            "no_matching_function": [],
            "other": [],
        }
        
        for line in lines:
            if "file not found" in line:
                match = re.search(r"'([^']+)'\s+file not found", line)
                if match:
                    errors_by_type["file_not_found"].append(match.group(1))
            elif "undefined reference" in line:
                match = re.search(r"undefined reference to\s+[`']([^'`]+)", line)
                if match:
                    errors_by_type["undefined_reference"].append(match.group(1))
            elif "redefinition of" in line:
                match = re.search(r"redefinition of\s+'([^']+)'", line)
                if match:
                    errors_by_type["redefinition"].append(match.group(1))
            elif "no matching function" in line:
                match = re.search(r"no matching function for call to\s+'([^']+)'", line)
                if match:
                    errors_by_type["no_matching_function"].append(match.group(1))
            elif "error:" in line and not any(
                x in line for x in ["file not found", "undefined reference", "redefinition"]
            ):
                errors_by_type["other"].append(line.strip())
                
        # Remove empty categories
        return {k: v for k, v in errors_by_type.items() if v}
        
    # ==================== Code Generation Methods ====================
    
    def generate_with_refined_prompt(self, base_prompt: str, file_key: str, context: str = "") -> str:
        """Generate code using a prompt that's been refined based on previous errors."""
        self.generation_attempt += 1
        
        # Apply refinements from previous attempts
        refined_prompt = self.prompt_refiner.apply_refinements_to_prompt(base_prompt, file_key)
        
        # Add context if provided
        if context:
            refined_prompt = f"{refined_prompt}\n\n{context}"
            
        print(f"[Generation] Attempt {self.generation_attempt} for {file_key}")
        
        messages = [{"role": "user", "content": refined_prompt}]
        response_text = self.get_generation_with_tools(messages)
        code = self.parse_response(response_text)
        
        return code
        
    def get_generation_with_tools(self, messages: List[Dict]) -> str:
        """Handle the full tool use workflow with Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        system_prompt = f"""You are an expert in Tenstorrent's TT-Metal SDK. You are generating a custom TTNN operation '{self.operation_name}'.
        Your goal is to generate code which compiles and correctly defines the desired operation.  You will be provided with a list of includes and APIs
        to use in the file, you must use those includes and APIs, and no others.  """
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
            "tools": AVAILABLE_TOOLS,
        }
        
        print("[API] Sending request to Claude...")
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()
        
        # Handle tool use
        if response_data.get("stop_reason") == "tool_use":
            time.sleep(TOOL_USE_DELAY)
            messages.append({"role": "assistant", "content": response_data["content"]})
            return self.handle_tool_use(response_data, messages, system_prompt, headers)
            
        # Return direct text response
        return "".join(
            block["text"] for block in response_data.get("content", [])
            if block.get("type") == "text"
        )
        
    def handle_tool_use(self, response_data: Dict, messages: List[Dict], 
                       system_prompt: str, headers: Dict) -> str:
        """Execute tools requested by the model and send results back."""
        print("[Tool Use] Model has requested to run tools")
        
        tool_use_blocks = [
            block for block in response_data["content"]
            if block["type"] == "tool_use"
        ]
        
        tool_results_content = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block["name"]
            tool_id = tool_block["id"]
            tool_input = tool_block["input"]
            
            print(f"  -> Calling tool '{tool_name}'")
            
            if tool_name in TOOL_EXECUTORS:
                result = TOOL_EXECUTORS[tool_name](**tool_input)
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": str(result)
                })
                
        # Send tool results back
        messages.append({"role": "user", "content": tool_results_content})
        time.sleep(TOOL_USE_DELAY)
        
        print("[API] Sending tool results back to Claude...")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
            "tools": AVAILABLE_TOOLS,
        }
        
        final_response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        final_response.raise_for_status()
        final_response_data = final_response.json()
        
        # Handle recursive tool use
        if final_response_data.get("stop_reason") == "tool_use":
            messages.append({"role": "assistant", "content": final_response_data["content"]})
            return self.handle_tool_use(final_response_data, messages, system_prompt, headers)
            
        return "".join(
            block["text"] for block in final_response_data.get("content", [])
            if block.get("type") == "text"
        )
        
    def parse_response(self, response_text: str) -> str:
        """Extract code from Claude's response."""
        pattern = r"```(?:\w+)?\n([\s\S]*?)\n```"
        matches = re.findall(pattern, response_text)
        
        if not matches:
            print(f"[Parsing Error] No code block found in response")
            return ""
            
        return matches[0].strip()
        
    # ==================== Debug Methods ====================
    
    def _identify_files_to_fix(self, cleaned_errors: str, critical_errors: Dict[str, List[str]]) -> List[str]:
        """Identify which files need to be fixed based on build errors."""
        # Build error summary for LLM
        error_summary = ""
        if critical_errors:
            error_summary = "\n**Critical Error Types Found:**\n"
            for error_type, errors in critical_errors.items():
                error_summary += f"- {error_type.replace('_', ' ').title()}: {len(errors)} occurrences\n"
                
        identification_prompt = f"""Analyze these build errors and identify which files need to be fixed.

        Build Errors:
        {cleaned_errors}

        {error_summary}

        Available files:
        {chr(10).join(f"- {key}: {data['name']}" for key, data in self.files.items())}

        Return ONLY a JSON list of file keys that need to be fixed:
        ```json
        ["cpp", "hpp", "op-hpp"]
        ```"""

        try:
            messages = [{"role": "user", "content": identification_prompt}]
            response_text = self.get_generation_with_tools(messages)
            
            # Extract JSON
            json_match = re.search(r"```json\n([\s\S]*?)\n```", response_text)
            if json_match:
                file_list = json.loads(json_match.group(1))
                return [f for f in file_list if f in self.files]
        except Exception as e:
            print(f"[Debug] Error identifying files: {e}")
            
        # Fallback: identify based on error patterns
        files_to_fix = set()
        
        for error_type, errors in critical_errors.items():
            if error_type == "file_not_found":
                # Include path issues usually in headers
                files_to_fix.update(["hpp", "op-hpp", "pybind-hpp"])
            elif error_type == "undefined_reference":
                # Implementation issues
                files_to_fix.update(["cpp", "op", "program-factory"])
            elif error_type == "no_matching_function":
                # API usage issues
                for key in self.files:
                    if any(err in cleaned_errors for err in errors):
                        files_to_fix.add(key)
                        
        return list(files_to_fix)
        
    def _create_single_file_debugging_prompt(self, file_key: str, cleaned_errors: str, 
                                           critical_errors: Dict[str, List[str]]) -> str:
        """Create a debugging prompt for regenerating a single file."""
        file_info = self.files[file_key]
        current_code = file_info["code"]
        
        # Get context from related files
        context_map = {
            "cpp": ["hpp"],
            "hpp": [],
            "op": ["op-hpp", "hpp"],
            "op-hpp": ["hpp"],
            "program-factory": ["program-factory-hpp", "hpp"],
            "program-factory-hpp": ["hpp"],
            "reader": ["program-factory-hpp", "op-hpp"],
            "writer": ["program-factory-hpp", "op-hpp"],
            "compute": ["program-factory-hpp", "op-hpp"],
            "pybind-cpp": ["pybind-hpp", "hpp"],
            "pybind-hpp": ["hpp"],
            "cmake": [],
        }
        
        context_files = []
        relevant_files = context_map.get(file_key, [])
        
        for ctx_key in relevant_files:
            if ctx_key in self.files and self.files[ctx_key]["code"]:
                context_files.append(
                    f"\n--- {self.files[ctx_key]['name']} (for reference) ---\n"
                    f"```cpp\n{self.files[ctx_key]['code']}\n```"
                )
                
        context_section = "\n".join(context_files)
        
        return f"""Fix the file '{file_info['name']}' for operation '{self.operation_name}' based on these build errors.

            Build Errors:
            {cleaned_errors}

            Current Content of {file_info['name']}:
            ```cpp
            {current_code}
            Reference Files:
            {context_section}
            Generate the complete, corrected version of {file_info['name']}.
            The fix should address all compilation errors while maintaining compatibility with other files.
            Provide the corrected code in a code block."""
    
    def _use_targeted_editing(self, file_key: str, cleaned_errors: str) -> bool:
        """Use tree-sitter based targeted editing for fixes."""
        try:
            from ttnn_op_generator.tools.tree_sitter_editor import TreeSitterEditor, CodeEdit
            from ttnn_op_generator.tools.tree_sitter_tool import parse_file
            
            editor = TreeSitterEditor()
            file_path = self.output_dir / self.files[file_key]["name"]
            
            if not file_path.exists():
                return False
                
            # Create targeted edit prompt
            edit_prompt = f"""Analyze these compilation errors and suggest TARGETED EDITS to fix them.
                File: {self.files[file_key]['name']}
                Errors: {cleaned_errors}
                Provide specific edit commands like:

                INSERT_INCLUDE: #include "missing_header.hpp"
                DELETE_FUNCTION: function_name
                MODIFY_FUNCTION: function_name

                cppnew implementation
                ```"""

            messages = [{"role": "user", "content": edit_prompt}]
            response = self.get_generation_with_tools(messages)
            
            # Parse and apply edits
            edits = self._parse_edit_commands(response)
            
            if edits:
                result = editor.apply_edits(file_path, edits)
                if result.success:
                    self.files[file_key]["code"] = result.new_content
                    print(f"[Targeted Edit] Applied {len(edits)} edits to {file_key}")
                    return True
                        
        except ImportError:
            print("[Warning] Tree-sitter editor not available, falling back to regeneration")
        except Exception as e:
            print(f"[Targeted Edit] Error: {e}")
            
        return False
        
    def _parse_edit_commands(self, response: str) -> List:
        """Parse edit commands from LLM response."""
        from ttnn_op_generator.tools.tree_sitter_editor import CodeEdit
        
        edits = []
        lines = response.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('INSERT_INCLUDE:'):
                include = line.split(':', 1)[1].strip()
                edits.append(CodeEdit(
                    operation='insert',
                    target_type='include',
                    target_name=include,
                    content=include
                ))
            elif line.startswith('DELETE_FUNCTION:'):
                func_name = line.split(':', 1)[1].strip()
                edits.append(CodeEdit(
                    operation='delete',
                    target_type='function',
                    target_name=func_name
                ))
            elif line.startswith('MODIFY_FUNCTION:'):
                func_name = line.split(':', 1)[1].strip()
                # Get code block
                i += 1
                if i < len(lines) and lines[i].strip() == '```cpp':
                    code_lines = []
                    i += 1
                    while i < len(lines) and lines[i].strip() != '```':
                        code_lines.append(lines[i])
                        i += 1
                    
                    code = '\n'.join(code_lines)
                    edits.append(CodeEdit(
                        operation='modify',
                        target_type='function',
                        target_name=func_name,
                        content=code
                    ))
                    
            i += 1
            
        return edits
        
    # ==================== Multi-Stage Support ====================
    
    def enable_multi_stage_generation(self):
        """Enable multi-stage generation for this agent."""
        if self.multi_stage_generator is None:
            try:
                # Import the multi-stage generator if available
                from ttnn_op_generator.agents.multi_stage_generator import MultiStageGenerator
                self.multi_stage_generator = MultiStageGenerator(self)
                self.use_multi_stage = True
                print("[Agent] Multi-stage generation enabled")
            except ImportError:
                print("[Warning] Multi-stage generator not available")
                self.use_multi_stage = False
                
    # ==================== Test Support ====================
    
    def enable_multi_stage_prompt_generation(self):
        """Enable multi-stage generation for this agent."""
        if self.multi_stage_generator is None:
            try:
                # Import the multi-stage generator if available
                from ttnn_op_generator.agents.multi_stage_prompt_generator import MultiStagePromptGenerator
                self.multi_stage_generator = MultiStagePromptGenerator(self)
                self.use_multi_stage = True
                print("[Agent] Multi-stage prompt generation enabled")
            except ImportError:
                print("[Warning] Multi-stage prompt generator not available")
                self.use_multi_stage = False

    def run_and_debug_test(self, test_path: str) -> bool:
        """Run and debug a test (placeholder for test debugger integration)."""
        print(f"[Test] Running test: {test_path}")
        
        # This would integrate with TestDebugger if available
        # For now, just run the test as a subprocess
        try:
            result = subprocess.run(
                ["python", test_path],
                capture_output=True,
                text=True,
                cwd=self.tt_metal_path
            )
            
            if result.returncode == 0:
                print("[Test] Test passed!")
                return True
            else:
                print(f"[Test] Test failed:\n{result.stderr}")
                return False
                
        except Exception as e:
            print(f"[Test] Error running test: {e}")
            return False
            
    # ==================== CMake Support ====================
    
    def add_operation_to_cmake(self, cmake_path: str, operation_name: str) -> bool:
        """Add the operation to the main CMake file."""
        try:
            with open(cmake_path, "r") as f:
                content = f.read()
                
            # Add pybind source
            pybind_line = (
                f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/cpp/ttnn/operations/"
                f"{self.operation_name}/{self.operation_name}_${{PY_BINDING}}.cpp"
            )
            pybind_end = content.find(")\n\nset(CCL_EXPERIMENTAL_TTNN_SRCS_PYBIND")
            if pybind_end != -1:
                content = content[:pybind_end] + f"\n{pybind_line}" + content[pybind_end:]
                
            # Add subdirectory
            subdirectory_line = f"add_subdirectory(cpp/ttnn/operations/{self.operation_name})"
            last_subdirectory = content.rfind("add_subdirectory(cpp/ttnn/operations/")
            if last_subdirectory != -1:
                line_end = content.find("\n", last_subdirectory)
                content = content[:line_end] + f"\n{subdirectory_line}" + content[line_end:]
                
            # Add library link
            lib_name = "ttnn_" + self.operation_name
            ttnncpp_link_start = content.find("target_link_libraries(\n    ttnncpp\n    PRIVATE")
            if ttnncpp_link_start != -1:
                private_section_end = content.find(")", ttnncpp_link_start)
                if private_section_end != -1:
                    content = content[:private_section_end] + f"\n        {lib_name}" + content[private_section_end:]
                    
            with open(cmake_path, "w") as f:
                f.write(content)
                
            print(f"[CMake] Successfully added {self.operation_name} to CMake")
            return True
            
        except Exception as e:
            print(f"[CMake] Error modifying CMake file: {e}")
            return False
            
    # ==================== Helper Methods ====================
    
    def _format_cleaned_errors(self, cleaned_errors: List[str]) -> str:
        """Format cleaned errors for better readability."""
        # Group errors by file
        file_errors = {}
        current_file = None
        
        for line in cleaned_errors:
            # Match file path with line number
            file_match = re.match(r"(/[^:]+\.(?:cpp|hpp|h)):(\d+):(\d+):\s*(.*)", line)
            if file_match:
                filepath = file_match.group(1)
                line_num = file_match.group(2)
                col_num = file_match.group(3)
                error_msg = file_match.group(4)
                
                filename = os.path.basename(filepath)
                current_file = filename
                
                if filename not in file_errors:
                    file_errors[filename] = []
                file_errors[filename].append(f"Line {line_num}: {error_msg}")
            elif current_file and line.strip():
                file_errors[current_file].append(f"  {line.strip()}")
                
        # Format output
        output = []
        if file_errors:
            output.append("=== Compilation Errors by File ===\n")
            for filename, errors in file_errors.items():
                output.append(f"\n{filename}:")
                for error in errors:
                    output.append(f"  {error}")
                    
        # Add any FAILED commands
        output.append("\n\n=== Failed Build Commands ===")
        for line in cleaned_errors:
            if line.startswith("FAILED:"):
                output.append(line)
                
        return "\n".join(output) if output else "\n".join(cleaned_errors)