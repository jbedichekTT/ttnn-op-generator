import requests
import json
import os
import time
import re
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass
from prompts import *
from tools import *
from persistent_prompt_refiner import PersistentPromptRefiner
from test_debugger import TestDebugger
from tree_sitter_editor import CodeEdit


# Configuration
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
BUILD_RETRIES = 3
MAX_TOKENS = 8192
THINKING_TOKENS = 3096
API_KEY = ""
DEBUG_ONLY = 1
REFINEMENT_ITERS = 4
TOOL_USE_DELAY = 0  # Delay between tool use and tool result, to avoid hitting the API limit too quickly
COMPLETE_PARTIAL_OP = 0
USE_REFINMENT_DB = 0
ADD_TO_CMAKE = 0

class TTNNOperationAgent:
    """Agent for generating, building, and debugging TTNN operations."""

    def __init__(
        self,
        operation_type: str = "add",
        tt_metal_path: str = "/path/to/tt-metal",
        build_retries: int = BUILD_RETRIES,
        run_tests: bool = True,
        custom_suffix: str = "custom",
        refinement_db_path: str = "ttnn_refinements_db.json",
    ):
        self.api_key = API_KEY
        self.model = MODEL
        self.operation_type = operation_type
        self.custom_suffix = custom_suffix
        self.operation_name = f"eltwise_{operation_type}_{custom_suffix}"
        self.operation_class_name = f"Eltwise{operation_type.title()}{custom_suffix.title()}"
        self.python_function_name = f"eltwise_{operation_type}_{custom_suffix}"

        self.tt_metal_path = Path(tt_metal_path)
        self.ttnn_ops_path = self.tt_metal_path / "ttnn/cpp/ttnn/operations/"
        self.output_dir = self.ttnn_ops_path / self.operation_name
        self.iteration = 0
        self.build_retries = build_retries
        self.run_tests = run_tests
        self.build_dir = self.tt_metal_path / "build"
        self.max_tokens = MAX_TOKENS

        self.files = {
            "cpp": {"name": f"{self.operation_name}.cpp", "code": ""},
            "hpp": {"name": f"{self.operation_name}.hpp", "code": ""},
            "pybind-cpp": {"name": f"{self.operation_name}_pybind.cpp", "code": ""},
            "pybind-hpp": {"name": f"{self.operation_name}_pybind.hpp", "code": ""},
            "cmake": {"name": "CMakeLists.txt", "code": ""},
            "program-factory": {"name": f"device/{self.operation_name}_program_factory.cpp", "code": ""},
            "program-factory-hpp": {"name": f"device/{self.operation_name}_program_factory.hpp", "code": ""},
            "reader": {"name": f"device/kernels/dataflow/{self.operation_name}_reader.cpp", "code": ""},
            "writer": {"name": f"device/kernels/dataflow/{self.operation_name}_writer.cpp", "code": ""},
            "compute": {"name": f"device/kernels/compute/{self.operation_name}_compute.cpp", "code": ""},
            "op": {"name": f"device/{self.operation_name}_op.cpp", "code": ""},
            "op-hpp": {"name": f"device/{self.operation_name}_op.hpp", "code": ""},
        }

        self.prompt_refiner = PersistentPromptRefiner(self.operation_name, db_path=refinement_db_path, load_from_db=USE_REFINMENT_DB)
        self.generation_attempt = 0

        self.test_debugger = TestDebugger(self.tt_metal_path, self.operation_name)
        self.test_debug_iterations = 5

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
        return True

    def run_build_verification(self) -> Tuple[bool, str]:
        """Run TT-Metal build using build_metal.sh to verify the generated operation compiles."""
        print(f"[Stage: Build Verification] Building {self.operation_name} with build_metal.sh")
        original_dir = os.getcwd()
        os.chdir(self.tt_metal_path)
        try:
            build_script = "./build_metal.sh"
            env = os.environ.copy()
            env["TT_METAL_HOME"] = str(self.tt_metal_path)
            result = subprocess.run([build_script], capture_output=True, text=True, timeout=1200, env=env, shell=False)
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

    def parse_response(self, response_text: str) -> str:
        """Extract code from Claude's response."""
        pattern = r"```(?:\w+)?\n([\s\S]*?)\n```"
        matches = re.findall(pattern, response_text)
        if not matches:
            print(f"[Parsing Error] No code block found in response: {response_text}")
        return matches[0].strip() if matches else ""

    def save_file(self, file_key: str, code: str):
        """Save generated file to the appropriate location."""
        file_config = self.files[file_key]
        full_path = self.output_dir / file_config["name"]
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with full_path.open("w") as f:
            f.write(code)
        print(f"  ✓ Saved: {file_config['name']}")
        self.files[file_key]["code"] = code

    def add_operation_to_cmake(self, cmake_path: str, operation_name: str) -> bool:
        """Add a new operation to the main TTNN CMakeLists.txt"""
        try:
            with open(cmake_path, "r") as f:
                content = f.read()
            pybind_line = f"    ${{CMAKE_CURRENT_SOURCE_DIR}}/cpp/ttnn/operations/{self.operation_name}/{self.operation_name}_${{PY_BINDING}}.cpp"
            pybind_end = content.find(")\n\nset(CCL_EXPERIMENTAL_TTNN_SRCS_PYBIND")
            if pybind_end != -1:
                content = content[:pybind_end] + f"\n{pybind_line}" + content[pybind_end:]
            subdirectory_line = f"add_subdirectory(cpp/ttnn/operations/{self.operation_name})"
            last_subdirectory = content.rfind("add_subdirectory(cpp/ttnn/operations/")
            if last_subdirectory != -1:
                line_end = content.find("\n", last_subdirectory)
                content = content[:line_end] + f"\n{subdirectory_line}" + content[line_end:]
            lib_name = "ttnn_" + self.operation_name
            ttnncpp_link_start = content.find("target_link_libraries(\n    ttnncpp\n    PRIVATE")
            if ttnncpp_link_start != -1:
                private_section_end = content.find(")", ttnncpp_link_start)
                if private_section_end != -1:
                    content = content[:private_section_end] + f"\n        {lib_name}" + content[private_section_end:]
            with open(cmake_path, "w") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to modify CMake file: {e}")
            return False

    def _create_comprehensive_fix_prompt(self, file_key: str, current_content: str, error_output: str) -> str:
        """Create a comprehensive prompt for fixing a specific file based on build errors."""
        file_config = self.files[file_key]
        context_files = []
        for other_key, other_config in self.files.items():
            if other_key != file_key and (p := self.output_dir / other_config["name"]).exists():
                context_files.append(f"\n--- {other_config['name']} (for reference) ---\n{p.read_text()[:1000]}...")
        context_section = "".join(context_files[:2])
        return f"""CRITICAL BUILD ERROR FIXING for {self.operation_name}
            You must fix the file '{file_config['name']}'.
            BUILD ERRORS TO FIX: {error_output}
            CURRENT PROBLEMATIC FILE CONTENT: {current_content}
            REFERENCE CONTEXT FROM OTHER FILES: {context_section}
            REQUIREMENTS:
            1. Fix all compilation errors shown in the build output.
            2. Maintain compatibility with TT-Metal/TTNN APIs and conventions.
            3. Preserve the unique naming: {self.operation_name}.
            CRITICAL: The generated code must be complete, compilable, and enclosed in ```.
            Generate the corrected {file_config['name']} file:"""

    def get_generation_with_tools(self, messages: List[Dict]) -> str:
        """
        Handles the full tool use workflow, including enabling extended thinking.
        This is the primary method for interacting with the Claude API.
        """
        # Set environment variables for tools to access the repository path and output directory
        os.environ["TT_METAL_PATH"] = str(self.tt_metal_path)
        os.environ["TTNN_OUTPUT_DIR"] = str(self.output_dir)

        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

        system_prompt = """
        You are an expert in Tenstorrent's TT-Metal SDK. You are generating a CUSTOM TTNN operation '{self.operation_name}'.
        Your goal is to generate code which compiles and correctly defines the desired operation.  If you wish to view an example operation,
        you can call tools on this one: /home/user/tt-metal/ttnn/cpp/ttnn/operations/matmul.  It contains these files:
        
        matmul.cpp
        matmul.hpp
        device/matmul_op.hpp
        device/matmul_op.cpp
        device/matmul_op_multi_core_program_factory.cpp
        device/kernels/compute/bmm.cpp
        device/kernels/dataflow/reader_bmm_tile_layout.cpp
        device/kernels/dataflow/writer_bmm_tile_layout.cpp

        Use this as a reference on how to use APIs, don't copy it but instead analyze and understand it.  
        Don't try and read the eltwise operations as examples, it won't be useful.
        """
                

        # The payload for the API call, including the tools and thinking parameter
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
            "tools": AVAILABLE_TOOLS,  # Assumes AVAILABLE_TOOLS is imported from tools.py
            "thinking": {"type": "enabled", "budget_tokens": THINKING_TOKENS},
        }

        print("[API Call] Sending request to Claude with thinking enabled...")
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()

        # If the model decides to use a tool, hand off to the tool handler
        if response_data.get("stop_reason") == "tool_use":
            time.sleep(TOOL_USE_DELAY)
            messages.append({"role": "assistant", "content": response_data["content"]})
            return self.handle_tool_use(response_data, messages, system_prompt, headers)

        # Otherwise, return the direct text response
        return "".join(block["text"] for block in response_data.get("content", []) if block.get("type") == "text")

    def handle_tool_use(self, response_data: Dict, messages: List[Dict], system_prompt: str, headers: Dict) -> str:
        """
        Executes tools requested by the model and sends the results back.
        This function can be called recursively if the model requests tools multiple times.
        """

        print("[Tool Use] Model has requested to run a tool.")
        tool_use_blocks = [block for block in response_data["content"] if block["type"] == "tool_use"]

        tool_results_content = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block["name"]
            tool_id = tool_block["id"]
            tool_input = tool_block["input"]

            print(f"  -> Calling tool '{tool_name}' with input: {tool_input}")
            # Assumes TOOL_EXECUTORS is imported from tools.py
            if tool_name in TOOL_EXECUTORS:
                result = TOOL_EXECUTORS[tool_name](**tool_input)
                tool_results_content.append({"type": "tool_result", "tool_use_id": tool_id, "content": str(result)})

        # Append the tool results to the conversation history
        messages.append({"role": "user", "content": tool_results_content})
        time.sleep(TOOL_USE_DELAY)
        print("[API Call] Sending tool results back to Claude...")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages,
            "tools": AVAILABLE_TOOLS,
            "thinking": {"type": "enabled", "budget_tokens": THINKING_TOKENS},
        }
        time.sleep(TOOL_USE_DELAY)
        final_response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        final_response.raise_for_status()
        final_response_data = final_response.json()

        # Handle cases where the model wants to use another tool after getting results
        if final_response_data.get("stop_reason") == "tool_use":
            messages.append({"role": "assistant", "content": final_response_data["content"]})
            return self.handle_tool_use(final_response_data, messages, system_prompt, headers)

        # Return the final text response
        return "".join(block["text"] for block in final_response_data.get("content", []) if block.get("type") == "text")

    def apply_code_fixes_from_llm_response(self, response_text: str):
        """
        Parses the LLM's debugging response to find and save corrected code blocks.
        Handles multiple common formats that LLMs use to present code fixes.
        """
        found_fix = False

        # Pattern 1: "File: filename" followed by code block
        pattern1 = re.compile(
            r"(?:File|file):\s*([\w\./\\]+)\s*\n```(?:\w+)?\n([\s\S]*?)\n```", re.MULTILINE | re.IGNORECASE
        )

        # Pattern 2: "filename:" followed by code block
        pattern2 = re.compile(r"`?([\w\./\\]+)`?\s*:\s*\n```(?:\w+)?\n([\s\S]*?)\n```", re.MULTILINE)

        # Pattern 3: "Here is the corrected code for `filename`:" format
        pattern3 = re.compile(
            r"(?:Here is the )?(?:corrected|fixed|updated)?\s*(?:code for|file:?)\s*`?([\w\./\\]+)`?\s*:?\s*\n```(?:\w+)?\n([\s\S]*?)\n```",
            re.MULTILINE | re.IGNORECASE,
        )

        # Pattern 4: Code block with filename in header
        pattern4 = re.compile(r"```(?:\w+)?\s+(?:file:)?\s*([\w\./\\]+)\n([\s\S]*?)\n```", re.MULTILINE | re.IGNORECASE)

        # Try all patterns
        patterns = [pattern1, pattern2, pattern3, pattern4]

        for pattern in patterns:
            for match in pattern.finditer(response_text):
                filename = match.group(1).strip()
                code = match.group(2).strip()

                # Clean up the filename (remove any backticks or quotes)
                filename = filename.strip("`\"'")

                # Debug output
                print(f"[Debug] Found code fix for file: {filename}")
                print(f"[Debug] Code preview: {code[:100]}...")

                # Find the corresponding file key in our self.files dictionary
                file_matched = False
                for key, data in self.files.items():
                    # Check if the filename matches the expected file name
                    # Handle both full paths and just filenames
                    if (
                        data["name"] == filename
                        or data["name"].endswith(filename)
                        or filename.endswith(data["name"])
                        or os.path.basename(data["name"]) == os.path.basename(filename)
                    ):
                        print(f"[Debug] Matched to file key '{key}': {data['name']}")
                        self.save_file(key, code)
                        found_fix = True
                        file_matched = True
                        break

                if not file_matched:
                    print(f"[Debug] Warning: Could not match '{filename}' to any known file")
                    # Try to be more flexible with matching
                    for key, data in self.files.items():
                        # More aggressive matching - check if any part of the path matches
                        if any(part in filename for part in data["name"].split("/")):
                            print(f"[Debug] Fuzzy matched to file key '{key}': {data['name']}")
                            self.save_file(key, code)
                            found_fix = True
                            break

        if not found_fix:
            print("[Debug] No specific file fixes were found in the LLM response.")
            # Additional debug: show what formats we found in the response
            print("[Debug] Response preview for debugging:")
            lines = response_text.split("\n")
            for i, line in enumerate(lines):
                print(line)  # TODO: remove
                if any(keyword in line.lower() for keyword in ["file:", "corrected", "fixed", "```"]):
                    print(f"  Line {i}: {line[:100]}")

        return found_fix

    def clean_build_errors(self, raw_error_output: str) -> str:
        """
        Clean and extract only the relevant compilation errors from build output.
        Removes CMake configuration, build system messages, and other noise.

        Args:
            raw_error_output: Raw output from the build process

        Returns:
            Cleaned error output containing only relevant compilation errors
        """
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

        # Patterns to skip (configuration and status messages)
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
            # Skip empty lines and configuration output
            if not line.strip():
                continue

            # Skip known non-error patterns
            if any(re.search(pattern, line) for pattern in skip_patterns):
                continue

            # Check if this line contains an error
            is_error_line = any(re.search(pattern, line, re.IGNORECASE) for pattern in error_patterns)

            if is_error_line:
                in_error_block = True
                # If we have accumulated context, add it
                if error_context:
                    cleaned_errors.extend(error_context[-3:])  # Keep last 3 lines of context
                    error_context = []
                cleaned_errors.append(line)
            elif in_error_block:
                # Continue capturing lines after an error (for multi-line errors)
                if line.strip().startswith("^") or "|" in line[:10]:  # Error indicators
                    cleaned_errors.append(line)
                elif len(cleaned_errors) > 0 and len(cleaned_errors[-1]) > 0:
                    # Add one more line after error for context, then stop
                    cleaned_errors.append(line)
                    in_error_block = False
            else:
                # Accumulate potential context lines
                error_context.append(line)
                if len(error_context) > 5:
                    error_context.pop(0)

        # Extract specific error information
        error_summary = []

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

                # Extract just the filename for readability
                filename = os.path.basename(filepath)
                current_file = filename

                if filename not in file_errors:
                    file_errors[filename] = []
                file_errors[filename].append(f"Line {line_num}: {error_msg}")
            elif current_file and line.strip():
                # Add additional error details to the current file
                file_errors[current_file].append(f"  {line.strip()}")

        # Format the cleaned output
        if file_errors:
            error_summary.append("=== Compilation Errors by File ===\n")
            for filename, errors in file_errors.items():
                error_summary.append(f"\n{filename}:")
                for error in errors:
                    error_summary.append(f"  {error}")

        # Add any FAILED commands for context
        error_summary.append("\n\n=== Failed Build Commands ===")
        for line in cleaned_errors:
            if line.startswith("FAILED:"):
                error_summary.append(line)

        # If we didn't extract any specific errors, include the cleaned output
        if not file_errors:
            error_summary = cleaned_errors

        return "\n".join(error_summary)

    def extract_critical_errors(self, raw_error_output: str) -> Dict[str, List[str]]:
        """
        Extract critical errors and organize them by error type.
        Returns a dictionary mapping error types to lists of specific errors.
        """
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

    def _create_tool_based_debugging_prompt(self, cleaned_errors: str, critical_errors: Dict[str, List[str]]) -> str:
        """
        Creates a detailed prompt that instructs the LLM to use tools for debugging.
        Now uses cleaned error output for better clarity.
        """
        file_context = ""
        for key, data in self.files.items():
            if data["code"]:  # Only include files that have been generated
                file_context += f"\n--- Current content of `{data['name']}` ---\n```{key}\n{data['code']}...\n```\n"

        # Build error type summary
        error_summary = ""
        if critical_errors:
            error_summary = "\n**Critical Error Types Found:**\n"
            for error_type, errors in critical_errors.items():
                error_summary += f"- {error_type.replace('_', ' ').title()}: {len(errors)} occurrences\n"
                if error_type == "file_not_found":
                    error_summary += f"  Missing files: {', '.join(errors[:5])}\n"
                elif error_type == "undefined_reference":
                    error_summary += f"  Undefined symbols: {', '.join(errors[:5])}\n"

        return f"""We are trying to build the C++ operation '{self.operation_name}' but it failed.

        Here are the compilation errors (cleaned and organized):
        <build_errors>
        {cleaned_errors}
        </build_errors>

        {error_summary}

        Here is the current state of the generated files:
        {file_context}

        Your task is to act as a master C++ debugger for the TT-Metal framework.

        APPROACH:
        1. Analyze the compilation errors carefully. Focus on the root causes, not symptoms.
        2. For each error type, formulate a hypothesis about what's wrong.
        3. Use the available tools strategically:
        - For 'file not found' errors: Use `find_file_in_repository` to locate the correct paths
        - For 'undefined reference' errors: Use `extract_symbols_from_file` to check function signatures
        - For header issues: Search for the correct include paths in the TT-Metal structure

        4. Based on your investigation, provide complete, corrected code for ONLY the files that need changes.

        CRITICAL FORMAT REQUIREMENTS:
        You MUST present each corrected file using EXACTLY this format (this is parsed by regex):

        File: exact/path/to/filename.cpp
        ```cpp
        // Complete corrected code here
        // Must be the FULL file content, not snippets
        ```

        File: another/file/that/needs/fixing.hpp
        ```cpp
        // Complete corrected code for this file
        ```

        EXAMPLES OF CORRECT FORMAT:
        File: {self.operation_name}.cpp
        ```cpp
        #include "{self.operation_name}.hpp"
        // ... rest of the complete file
        ```

        File: device/{self.operation_name}_op.hpp
        ```cpp
        #pragma once
        // ... complete header file
        ```

        IMPORTANT RULES:
        - Use EXACTLY "File: " (with space and colon) followed by the filename
        - The filename must match EXACTLY one of these: {', '.join(data['name'] for data in self.files.values())}
        - Put the language identifier (cpp) directly after the opening triple backticks
        - Each file must be complete - not excerpts or snippets
        - Only include files that need to be changed to fix the errors
        - The kernel files (reader, writer, compute) use different headers than host code
        - Device-side code has different include paths than host-side code

        Focus on fixing the ROOT CAUSE of the errors. Now analyze and provide the corrected files."""

    def build_operation(self) -> bool:
        """
        Generates all files for the operation using a coordinated, tool-powered workflow.
        After generation, it proceeds to the debugging loop with prompt refinement.
        """
        print(f"\n[Workflow Start] Generating operation: {self.operation_name}")
        if not self.verify_tt_metal_setup():
            return False

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # First attempt - generate all files
        success = self._generate_all_files_with_refinement()

        if not success:
            # If initial generation had issues, start the refined generation loop
            for attempt in range(REFINEMENT_ITERS):  # Allow up to 3 full regeneration attempts
                print(f"\n[Regeneration Attempt {attempt + 1}] Using refined prompts based on error analysis")

                # Build and analyze errors
                build_success, error_output = self.run_build_verification()
                if build_success:
                    print("[Success] Build succeeded with refined prompts!")
                    return True

                # Analyze errors and refine prompts
                current_file_contents = {key: data["code"] for key, data in self.files.items() if data["code"]}
                self.prompt_refiner.analyze_errors_and_refine_prompts(
                    error_output, current_file_contents, self.api_key, self.model
                )

                # Show refinement summary
                print(self.prompt_refiner.get_refinement_summary())

                # Regenerate files with refined prompts
                success = self._generate_all_files_with_refinement()

        # Final debug loop if needed
        if not success:
            success, _ = self.debug_build_loop()

        return success

    def _generate_all_files_with_refinement(self) -> bool:
        """Generate all files using refined prompts."""
        file_gen_order = [
            ("hpp", HPP_CONTEXT, []),
            ("cpp", CPP_CONTEXT, ["hpp"]),
            ("op-hpp", DEVICE_OP_CONTEXT, ["hpp"]),
            ("op", DEVICE_OP_CONTEXT, ["op-hpp"]),
            ("program-factory-hpp", PROGRAM_FACTORY_CONTEXT, []),
            ("program-factory", PROGRAM_FACTORY_CONTEXT, ["program-factory-hpp"]),
            ("reader", KERNEL_CONTEXT, ["program-factory"]),
            ("writer", KERNEL_CONTEXT, ["program-factory"]),
            ("compute", KERNEL_CONTEXT, ["program-factory"]),
            ("pybind-hpp", PYBIND_CONTEXT, ["hpp"]),
            ("pybind-cpp", PYBIND_CONTEXT, ["pybind-hpp", "hpp"]),
            ("cmake", CMAKE_CONTEXT, []),
        ]

        for key, context, deps in file_gen_order:
            dep_context = ""
            for dep_key in deps:
                if self.files[dep_key]["code"]:
                    dep_name = self.files[dep_key]["name"]
                    dep_context += f"\n--- Reference File: {dep_name} ---\n{self.files[dep_key]['code']}\n"

            base_prompt = (
                f"Generate the code for the file `{self.files[key]['name']}` "
                f"for the `{self.operation_name}` operation. "
                f"Use the provided context and reference files to ensure consistency.\n\n{context}"
            )

            print(f"\n--- Generating {key} ({self.files[key]['name']}) with refinements ---")
            code = self.generate_with_refined_prompt(base_prompt, key, dep_context)
            self.save_file(key, code)

        # Try initial build
        success, _ = self.run_build_verification()
        return success

    def complete_partial_operation(self) -> bool:
        """Complete a partially generated operation by only creating missing files."""
        print(f"\n[Partial Completion] Checking for existing files in {self.output_dir}")

        file_gen_order = [
            ("hpp", HPP_CONTEXT, []),
            ("cpp", CPP_CONTEXT, ["hpp"]),
            ("op-hpp", DEVICE_OP_CONTEXT, ["hpp"]),
            ("op", DEVICE_OP_CONTEXT, ["op-hpp"]),
            ("program-factory-hpp", PROGRAM_FACTORY_CONTEXT, []),
            ("program-factory", PROGRAM_FACTORY_CONTEXT, ["program-factory-hpp"]),
            ("reader", KERNEL_CONTEXT, ["program-factory"]),
            ("writer", KERNEL_CONTEXT, ["program-factory"]),
            ("compute", KERNEL_CONTEXT, ["program-factory"]),
            ("pybind-hpp", PYBIND_CONTEXT, ["hpp"]),
            ("pybind-cpp", PYBIND_CONTEXT, ["pybind-hpp", "hpp"]),
            ("cmake", CMAKE_CONTEXT, []),
        ]

        # Load existing files into memory
        existing_count = 0
        for key, file_info in self.files.items():
            file_path = self.output_dir / file_info["name"]
            if file_path.exists():
                self.files[key]["code"] = file_path.read_text()
                existing_count += 1
                print(f"  ✓ Found existing: {file_info['name']}")

        print(
            f"[Partial Completion] Found {existing_count} existing files, generating {len(self.files) - existing_count} missing files"
        )

        # Generate only missing files
        for key, context, deps in file_gen_order:
            if self.files[key]["code"]:  # Skip if already exists
                continue

            # Build context from dependencies (both existing and newly generated)
            dep_context = ""
            for dep_key in deps:
                if self.files[dep_key]["code"]:
                    dep_name = self.files[dep_key]["name"]
                    dep_context += f"\n--- Reference File: {dep_name} ---\n{self.files[dep_key]['code']}\n"

            base_prompt = (
                f"Generate the code for the file `{self.files[key]['name']}` "
                f"for the `{self.operation_name}` operation. "
                f"Use the provided context and reference files to ensure consistency.\n\n{context}"
            )

            print(f"\n--- Generating missing file: {key} ({self.files[key]['name']}) ---")
            code = self.generate_with_refined_prompt(base_prompt, key, dep_context)
            self.save_file(key, code)

        # Verify the complete operation builds
        success, _ = self.run_build_verification()
        if success:
            self.prompt_refiner.mark_build_success()
        return success

    def generate_with_refined_prompt(self, base_prompt: str, file_key: str, context: str = "") -> str:
        """Generate code using a prompt that's been refined based on previous errors."""
        self.generation_attempt += 1

        # Apply refinements from previous attempts
        refined_prompt = self.prompt_refiner.apply_refinements_to_prompt(base_prompt, file_key)

        # Add context if provided
        if context:
            refined_prompt = f"{refined_prompt}\n\n{context}"

        print(f"[Generation Attempt {self.generation_attempt}] Using refined prompt for {file_key}")

        messages = [{"role": "user", "content": refined_prompt}]
        response_text = self.get_generation_with_tools(messages)
        code = self.parse_response(response_text)

        return code

    def _create_test_debugging_prompt(
        self, test_path: str, error_analysis: Dict[str, Any], stdout: str, stderr: str
    ) -> str:
        """Create comprehensive debugging prompt for test failures"""

        # Read the test file
        test_code = Path(test_path).read_text()

        # Get current code state
        code_context = ""
        for key, data in self.files.items():
            if data["code"]:
                code_context += f"\n--- {data['name']} ---\n```cpp\n{data['code']}...\n```\n"

        # Build error summary
        error_summary = f"""
        Error Type: {error_analysis['error_type']}
        Last Successful Operation: {error_analysis.get('last_successful_operation', 'Unknown')}

        Stack Trace:
        {chr(10).join(error_analysis['stack_trace'][:10])}

        Device/Kernel Errors:
        {chr(10).join(error_analysis['kernel_errors'][:5])}

        Tensor/Shape Errors:
        {chr(10).join(error_analysis['tensor_errors'][:5])}
        """
        print(f"Error Summary: {error_summary}")

        # Error-specific guidance
        debugging_hints = self._get_debugging_hints(error_analysis["error_type"])

        return f"""The test for operation '{self.operation_name}' is failing.

        TEST CODE:
        ```python
        {test_code}
        TEST OUTPUT:
        <stdout>
        {stdout}
        </stdout>
        <stderr>
        {stderr}
        </stderr>
        ERROR ANALYSIS:
        {error_summary}
        CURRENT IMPLEMENTATION:
        {code_context}
        DEBUGGING APPROACH FOR {error_analysis['error_type'].upper()} ERROR:
        {debugging_hints}
        Your task:

        Analyze why the test is failing based on the error output
        Use available tools to investigate:

        For undefined symbols: use extract_symbols_from_file
        For includes: use find_file_in_repository
        For API usage: use find_api_usage


        Identify the root cause (common issues):

        Incorrect tensor memory layout or shape handling
        Missing device synchronization
        Kernel not properly configured
        Buffer allocation issues
        Incorrect kernel arguments


        Provide complete corrected code for files that need changes

        CRITICAL: For timeout/hang errors, focus on:

        Infinite loops in kernels
        Missing synchronization barriers
        Deadlocks in parallel execution
        Incorrect work distribution

        IMPORTANT RULES:
        - Use EXACTLY "File: " (with space and colon) followed by the filename
        - The filename must match EXACTLY one of these: {', '.join(data['name'] for data in self.files.values())}
        - Put the language identifier (cpp) directly after the opening triple backticks
        - Each file must be complete - not excerpts or snippets
        - Only include files that need to be changed to fix the errors
        """

    def run_and_debug_test(self, test_path: str) -> bool:
        """Run test with debugging loop similar to build debugging"""
        print(f"\n[Test Debug Loop] Starting test execution and debugging")

        for iteration in range(self.test_debug_iterations):
            print(f"\n[Test Debug Iteration {iteration + 1}/{self.test_debug_iterations}]")

            # Build metal
            success, _ = self.debug_build_loop()
            # Run the test
            success, stdout, stderr = self.test_debugger.run_test_with_capture(test_path)

            if success:
                print("[Success] Test passed!")
                return True

            # Analyze the failure
            error_analysis = self.test_debugger.extract_test_errors(stdout, stderr)

            print(f"\n[Test Error Analysis]")
            print(f"Error Type: {error_analysis['error_type']}")
            if error_analysis["last_successful_operation"]:
                print(f"Last Successful Operation: {error_analysis['last_successful_operation']}")

            # Create debugging prompt
            debug_prompt = self._create_test_debugging_prompt(test_path, error_analysis, stdout, stderr)

            # Get fixes from LLM
            messages = [{"role": "user", "content": debug_prompt}]
            print("\n[Debug] Asking LLM to analyze test failure and propose fixes...")
            response_text = self.get_generation_with_tools(messages)

            # Apply fixes
            fixes_applied = self.apply_code_fixes_from_llm_response(response_text)

            if not fixes_applied:
                print("[Warning] No fixes were applied, trying alternative approach")

        print(f"[Test Debug Failed] Could not fix test after {self.test_debug_iterations} attempts")
        return False

    def _get_debugging_hints(self, error_type: str) -> str:
        """Get specific debugging hints based on error type"""
        hints = {
            "timeout": """

        Check for infinite loops in kernel code (while loops without proper exit conditions)
        Verify work distribution doesn't exceed tensor dimensions
        Ensure all kernels complete and synchronize properly
        Check if waiting for non-existent data (reader waiting for data that writer never produces)
        Verify circular buffer sizes and semaphore usage
        """,
            "segfault": """
        Check all pointer dereferences and array accesses
        Verify tensor buffer allocations match access patterns
        Ensure tensor strides and shapes are correctly calculated
        Check for buffer overruns in kernel code
        """,
            "assertion": """
        Review all TT_ASSERT and assert statements
        Check tensor shape compatibility
        Verify preconditions for operations
        Ensure valid parameter ranges
        """,
            "device": """
        Check kernel compilation and loading
        Verify device memory allocations
        Ensure proper work group sizes
        Check for device-specific limitations
        """,
            "runtime": """
        Check API usage matches TT-Metal conventions
        Verify all required parameters are set
        Ensure proper initialization sequence
        Check for missing implementations
        """,
        }
        return hints.get(error_type, "Analyze the error carefully and check common failure points.")

    def _identify_files_to_fix(self, cleaned_errors: str, critical_errors: Dict[str, List[str]]) -> List[str]:
        """
        Use LLM to identify which files need to be regenerated based on build errors.
        Returns a list of file keys that need fixing.
        """
        # Build error summary
        error_summary = ""
        if critical_errors:
            error_summary = "\n**Critical Error Types Found:**\n"
            for error_type, errors in critical_errors.items():
                error_summary += f"- {error_type.replace('_', ' ').title()}: {len(errors)} occurrences\n"
                if error_type == "file_not_found":
                    error_summary += f"  Missing files: {', '.join(errors[:5])}\n"
                elif error_type == "undefined_reference":
                    error_summary += f"  Undefined symbols: {', '.join(errors[:5])}\n"

        identification_prompt = f"""Analyze these build errors for operation '{self.operation_name}' and identify which files need to be regenerated.

                Build Errors:
                <errors>
                {cleaned_errors}
                </errors>

                {error_summary}

                Available files in the project:
                {chr(10).join(f"- {key}: {data['name']}" for key, data in self.files.items())}

                Your task:
                1. Analyze the errors carefully
                2. Identify which specific files are causing the errors
                3. Return ONLY the file keys that need to be regenerated

                For example:
                - If there are undefined references to functions that should be in the .cpp file, include "cpp"
                - If header guards are wrong or declarations are missing, include "hpp"
                - If device operations have wrong signatures, include "op" or "op-hpp"
                - If kernel functions are missing or wrong, include "reader", "writer", or "compute"

                Return your answer as a JSON list of file keys, like:
                ```json
                ["cpp", "hpp", "op-hpp"]
                Only include files that actually need changes to fix the errors. Be precise."""
        try:
            messages = [{"role": "user", "content": identification_prompt}]
            response_text = self.get_generation_with_tools(messages)

            # Extract JSON list from response
            import json

            json_match = re.search(r"```json\n([\s\S]*?)\n```", response_text)
            if json_match:
                file_list = json.loads(json_match.group(1))
                # Validate that these are actual file keys
                valid_files = [f for f in file_list if f in self.files]
                return valid_files
            else:
                print(f"[Debug] No JSON list found in response: {response_text}")
        except Exception as e:
            print(f"[Debug] Error identifying files to fix: {e}")
            return []
        
    def _create_single_file_debugging_prompt(
        self, file_key: str, cleaned_errors: str, critical_errors: Dict[str, List[str]]
    ) -> str:
        """
        Create a debugging prompt for regenerating a single specific file.
        """
        file_info = self.files[file_key]
        current_code = file_info["code"]
        # Get context from related files
        context_files = []

        # Define which files are most relevant for context based on the file being fixed
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

        relevant_files = context_map.get(file_key, [])
        for ctx_key in relevant_files:
            if ctx_key in self.files and self.files[ctx_key]["code"]:
                context_files.append(
                    f"\n--- {self.files[ctx_key]['name']} (for reference) ---\n"
                    f"```cpp\n{self.files[ctx_key]['code']}\n```"
                )

        context_section = "\n".join(context_files)

        # Build error summary specific to this file
        error_summary = ""
        if critical_errors:
            error_summary = "\n**Relevant Error Types:**\n"
            for error_type, errors in critical_errors.items():
                if errors:
                    error_summary += f"- {error_type.replace('_', ' ').title()}: {', '.join(errors[:3])}\n"

        return f"""Fix the file '{file_info['name']}' for operation '{self.operation_name}' based on these build errors.
            Build Errors:
            <errors>
            {cleaned_errors}
            </errors>
            {error_summary}
            Current Content of {file_info['name']}:
            cpp{current_code}
            Reference Files:
            {context_section}
            Your task:

            Generate the complete, corrected version of {file_info['name']}

            CRITICAL FORMAT REQUIREMENTS:
            You MUST present each corrected file using EXACTLY this format (this is parsed by regex):

            File: exact/path/to/filename.cpp
            ```cpp
            // Complete corrected code here
            // Must be the FULL file content, not snippets
            ```

            IMPORTANT RULES:
            - Use EXACTLY "File: " (with space and colon) followed by the filename
            - The filename must match EXACTLY one of these: {', '.join(data['name'] for data in self.files.values())}
            - Put the language identifier (cpp) directly after the opening triple backticks
            - Each file must be complete - not excerpts or snippets
            - Only include the specified file

            Focus on fixing the ROOT CAUSE of the errors.  Use the tree-sitter tool extensively to ensure that the generated files have no compilation errors.
            """

    def _use_targeted_editing(self, file_key: str, error_output: str) -> bool:
        """
        Use tree-sitter based targeted editing instead of full regeneration
        """
        from tree_sitter_editor import TreeSitterEditor, CodeEdit
        
        editor = TreeSitterEditor()
        file_path = self.output_dir / self.files[file_key]["name"]
        
        if not file_path.exists():
            return False
        
        # Parse current file
        tree_id = parse_file(file_path)
        
        # Create prompt for targeted edits
        edit_prompt = self._create_targeted_edit_prompt(file_key, error_output, tree_id)
        
        # Get LLM to suggest edits
        messages = [{"role": "user", "content": edit_prompt}]
        response = self.get_generation_with_tools(messages)
        
        # Parse edits from response
        edits = self._parse_edit_commands(response)
        
        if edits:
            # Apply edits
            result = editor.apply_edits(file_path, edits)
            if result.success:
                self.files[file_key]["code"] = result.new_content
                print(f"[Targeted Edit] Successfully applied {len(edits)} edits to {file_key}")
                return True
        
        return False

    def _create_targeted_edit_prompt(self, file_key: str, error_output: str, tree_id: str) -> str:
        """
        Create prompt for targeted editing using tree-sitter
        """
        # Get current file structure
        structure_query = """
        [
        (function_definition
            declarator: (function_declarator
            declarator: (identifier) @fn_name)) @fn
        (class_specifier
            name: (type_identifier) @class_name) @class
        (namespace_definition
            name: (namespace_identifier) @ns_name) @ns
        (preproc_include) @include
        ]
        """
        
        structure = query(tree_id, structure_query)
        
        structure_summary = "Current file structure:\n"
        for item in structure:
            if '@fn_name' in item['name']:
                structure_summary += f"- Function: {item['text']}\n"
            elif '@class_name' in item['name']:
                structure_summary += f"- Class: {item['text']}\n"
            elif '@ns_name' in item['name']:
                structure_summary += f"- Namespace: {item['text']}\n"
            elif '@include' in item['name']:
                structure_summary += f"- Include: {item['text'].strip()}\n"
        
        return f"""Analyze these compilation errors and suggest TARGETED EDITS to fix them.
            You have access to tree-sitter to parse and edit the code surgically.

            File: {self.files[file_key]['name']}
            {structure_summary}

            Compilation Errors:
            {error_output}

            Instead of regenerating the entire file, provide SPECIFIC EDIT COMMANDS:

            AVAILABLE EDIT OPERATIONS:
            1. INSERT_INCLUDE: Add a missing include
            Example: INSERT_INCLUDE: #include "ttnn/tensor/tensor.hpp"

            2. INSERT_FUNCTION: Add a new function
            Example: INSERT_FUNCTION inside:ttnn
            ```cpp
            void my_function() {{
                // implementation
            }}

            DELETE_FUNCTION: Remove a function
            Example: DELETE_FUNCTION: function_name
            MODIFY_FUNCTION: Change an existing function
            Example: MODIFY_FUNCTION: function_name
            cppReturnType function_name(new_params) {{
                // new implementation
            }}

            INSERT_MEMBER: Add member to a class
            Example: INSERT_MEMBER: ClassName
            cppint new_member_;


            Analyze the errors and provide ONLY the necessary edits to fix them.
            Each edit should be on a new line starting with the operation name.
            """
    
    def _parse_edit_commands(self, response: str) -> List[CodeEdit]:
        """
        Parse edit commands from LLM response
        """
        from tree_sitter_editor import CodeEdit
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
                
            elif line.startswith('INSERT_FUNCTION'):
                # Parse location hint
                parts = line.split()
                location_hint = None
                if len(parts) > 1 and parts[1].startswith('inside:'):
                    location_hint = parts[1]
                
                # Get code block
                i += 1
                if i < len(lines) and lines[i].strip() == '```cpp':
                    code_lines = []
                    i += 1
                    while i < len(lines) and lines[i].strip() != '```':
                        code_lines.append(lines[i])
                        i += 1
                    
                    code = '\n'.join(code_lines)
                    # Extract function name from code
                    import re
                    name_match = re.search(r'\b(\w+)\s*\(', code)
                    func_name = name_match.group(1) if name_match else "unknown"
                    
                    edits.append(CodeEdit(
                        operation='insert',
                        target_type='function',
                        target_name=func_name,
                        content=code,
                        location_hint=location_hint
                    ))
                    
            elif line.startswith('DELETE_FUNCTION:'):
                func_name = line.split(':', 1)[1].strip()
                edits.append(CodeEdit(
                    operation='delete',
                    target_type='function',
                    target_name=func_name,
                    content=None
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

    def debug_build_loop(self) -> Tuple[bool, str]:
        """
        Enhanced debug loop that tries targeted editing before full regeneration
        """
        print(f"\n[Debug Loop] Starting intelligent build debugging...")
        
        for i in range(self.build_retries):
            self.iteration = i + 1
            print(f"\n[Debug Iteration {self.iteration}/{self.build_retries}]")
            
            success, raw_error_output = self.run_build_verification()
            if success:
                print(f"[Success] Build succeeded after {self.iteration} iterations!")
                self.prompt_refiner.mark_build_success()
                return True, "Build successful"
            

            print(f'[Debug] Raw error output: {raw_error_output}')
            # Clean errors
            cleaned_errors = self.clean_build_errors(raw_error_output)
            critical_errors = self.extract_critical_errors(raw_error_output)
            
            # Identify problematic files
            files_to_fix = self._identify_files_to_fix(cleaned_errors, critical_errors)
            
            if not files_to_fix:
                print("[Debug] Could not identify which files to fix")
                continue
            
            print(f"[Debug] Files identified for fixes: {files_to_fix}")
            
            # Try targeted editing first
            targeted_success = False
            for file_key in files_to_fix:
                if self.files[file_key]["code"]:  # File exists
                    print(f"\n[Debug] Attempting targeted edits for {file_key}")
                    if self._use_targeted_editing(file_key, cleaned_errors):
                        targeted_success = True
            
            if targeted_success:
                print("[Debug] Applied targeted edits, retrying build...")
                continue
            
            # Fall back to full regeneration if targeted editing didn't work
            print("[Debug] Targeted editing unsuccessful, falling back to full regeneration")
            
            for file_key in files_to_fix:
                print(f"\n[Debug] Regenerating file: {file_key}")
                fix_prompt = self._create_single_file_debugging_prompt(
                    file_key, cleaned_errors, critical_errors
                )
                
                messages = [{"role": "user", "content": fix_prompt}]
                response_text = self.generate_with_refined_prompt(fix_prompt, file_key)
                
                code = self.parse_response(response_text)
                if code:
                    self.save_file(file_key, code)
        
        return False, "Build debugging failed"

def main():
    """Example usage of the agent"""
    import argparse

    parser = argparse.ArgumentParser(description="TTNN eltwise operation generator")
    parser.add_argument("--operation", default="multiply", help="Type of eltwise operation")
    parser.add_argument(
        "--tt-metal-path", required=False, default="/home/user/tt-metal", help="Path to TT-Metal repository"
    )
    parser.add_argument("--custom-suffix", default="custom", help="Suffix for operation names")
    args = parser.parse_args()

    agent = TTNNOperationAgent(
        operation_type=args.operation, tt_metal_path=args.tt_metal_path, custom_suffix=args.custom_suffix
    )
    if ADD_TO_CMAKE: # Should only need to be called once per operation
        agent.add_operation_to_cmake("/home/user/tt-metal/ttnn/CMakeLists.txt", args.operation)
    if COMPLETE_PARTIAL_OP:
        success = agent.complete_partial_operation()
        return 0
    elif DEBUG_ONLY:
        success, _ = agent.debug_build_loop()
        #success = agent.run_and_debug_test("/home/user/tt-metal/tests/ttnn/unit_tests/test_exp_add.py")
        # agent.add_operation_to_cmake("/home/user/tt-metal/ttnn/CMakeLists.txt", args.operation)
    else:
        success = agent.build_operation()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
