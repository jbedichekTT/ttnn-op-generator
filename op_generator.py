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


# Configuration
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
BUILD_RETRIES = 10
MAX_TOKENS = 8192
THINKING_TOKENS = 3096
API_KEY = ""
DEBUG_ONLY = 1
REFINEMENT_ITERS = 4
TOOL_USE_DELAY = 40  # Delay between tool use and tool result, to avoid hitting the API limit too quickly
COMPLETE_PARTIAL_OP = 0


@dataclass
class SymbolDependency:
    """Track symbol dependencies between files"""

    symbol: str
    defined_in: str
    used_in: List[str]
    symbol_type: str  # 'function', 'class', 'struct', 'namespace'


@dataclass
class ExtractedPattern:
    """Pattern extracted from existing code"""

    pattern_type: str  # 'api', 'structure', 'include', 'namespace'
    pattern: str
    example: str
    file_type: str


class CodeAnalyzer:
    """Analyzes existing TT-Metal code to extract patterns"""

    def __init__(self, tt_metal_path: Path):
        self.tt_metal_path = tt_metal_path
        self.patterns = {}
        self.discovered_apis = {}

    def analyze_existing_operation(self, operation_name: str) -> Dict[str, List[ExtractedPattern]]:
        """Analyze an existing operation to extract patterns"""
        print(f"[Analysis] Analyzing existing operation: {operation_name}")
        patterns = {"header": [], "implementation": [], "device_op": [], "pybind": [], "cmake": []}
        ops_path = self.tt_metal_path / "ttnn/cpp/ttnn/operations"
        for category in ["eltwise", "unary", "binary", "core"]:
            op_dir = ops_path / category / operation_name
            if op_dir.exists():
                patterns.update(self._analyze_operation_directory(op_dir, operation_name))
                break
        return patterns

    def _analyze_operation_directory(self, op_dir: Path, op_name: str) -> Dict[str, List[ExtractedPattern]]:
        """Analyze all files in an operation directory"""
        patterns = {}
        header_file = op_dir / f"{op_name}.hpp"
        if header_file.exists():
            patterns["header"] = self._extract_header_patterns(header_file)
        impl_file = op_dir / f"{op_name}.cpp"
        if impl_file.exists():
            patterns["implementation"] = self._extract_implementation_patterns(impl_file)
        device_op_file = op_dir / "device" / f"{op_name}_op.hpp"
        if device_op_file.exists():
            patterns["device_op"] = self._extract_device_op_patterns(device_op_file)
        pybind_file = op_dir / f"{op_name}_pybind.cpp"
        if pybind_file.exists():
            patterns["pybind"] = self._extract_pybind_patterns(pybind_file)
        return patterns

    def _extract_header_patterns(self, file_path: Path) -> List[ExtractedPattern]:
        patterns = []
        content = file_path.read_text()
        namespace_match = re.search(r"namespace\s+([\w:]+)\s*{", content)
        if namespace_match:
            patterns.append(
                ExtractedPattern(
                    pattern_type="namespace",
                    pattern=namespace_match.group(1),
                    example=namespace_match.group(0),
                    file_type="header",
                )
            )
        func_pattern = r"([\w:]+)\s+(\w+)\s*\([^)]*\)\s*;"
        for match in re.finditer(func_pattern, content):
            patterns.append(
                ExtractedPattern(
                    pattern_type="function_declaration",
                    pattern=f"{match.group(1)} {match.group(2)}(...)",
                    example=match.group(0),
                    file_type="header",
                )
            )
        include_pattern = r'#include\s+[<"]([^>"]+)[>"]'
        includes = set()
        for match in re.finditer(include_pattern, content):
            includes.add(match.group(1))
        if includes:
            patterns.append(
                ExtractedPattern(
                    pattern_type="includes",
                    pattern="common_includes",
                    example="\n".join([f'#include "{inc}"' for inc in list(includes)[:5]]),
                    file_type="header",
                )
            )
        return patterns

    def _extract_implementation_patterns(self, file_path: Path) -> List[ExtractedPattern]:
        patterns = []
        content = file_path.read_text()
        dispatch_pattern = r"operation::run\s*\([^)]+\)"
        if re.search(dispatch_pattern, content):
            match = re.search(dispatch_pattern, content)
            patterns.append(
                ExtractedPattern(
                    pattern_type="dispatch",
                    pattern="operation::run",
                    example=match.group(0),
                    file_type="implementation",
                )
            )
        validation_pattern = r"TT_FATAL\s*\([^;]+\);"
        validations = [match.group(0) for match in re.finditer(validation_pattern, content)]
        if validations:
            patterns.append(
                ExtractedPattern(
                    pattern_type="validation", pattern="TT_FATAL", example=validations[0], file_type="implementation"
                )
            )
        return patterns

    def _extract_device_op_patterns(self, file_path: Path) -> List[ExtractedPattern]:
        patterns = []
        content = file_path.read_text()
        struct_pattern = r"struct\s+(\w+)\s*{([^}]+)}"
        match = re.search(struct_pattern, content, re.DOTALL)
        if match:
            patterns.append(
                ExtractedPattern(
                    pattern_type="device_op_struct",
                    pattern=f"struct {match.group(1)}",
                    example=match.group(0)[:200] + "...",
                    file_type="device_op",
                )
            )
        method_pattern = (
            r"(void|std::vector<[\w<>]+>)\s+(validate|compute_output_specs|create_output_tensors|create_program)\s*\("
        )
        for match in re.finditer(method_pattern, content):
            patterns.append(
                ExtractedPattern(
                    pattern_type="device_op_method",
                    pattern=f"{match.group(1)} {match.group(2)}",
                    example=match.group(0),
                    file_type="device_op",
                )
            )
        return patterns

    def _extract_pybind_patterns(self, file_path: Path) -> List[ExtractedPattern]:
        patterns = []
        content = file_path.read_text()
        bind_pattern = r"module\.def\s*\([^;]+\);"
        for match in re.finditer(bind_pattern, content, re.DOTALL):
            patterns.append(
                ExtractedPattern(
                    pattern_type="pybind_def",
                    pattern="module.def",
                    example=match.group(0)[:150] + "...",
                    file_type="pybind",
                )
            )
            break
        return patterns

    def discover_ttnn_apis(self) -> Dict[str, List[str]]:
        print("[Discovery] Scanning for TTNN APIs...")
        apis = {"registration_macros": [], "operation_functions": [], "tensor_types": [], "utility_functions": []}
        headers_to_scan = [
            "ttnn/decorators.hpp",
            "ttnn/operation.hpp",
            "ttnn/tensor/tensor.hpp",
            "ttnn/operations/core/core.hpp",
        ]
        for header in headers_to_scan:
            header_path = self.tt_metal_path / header
            if header_path.exists():
                content = header_path.read_text()
                macro_pattern = r"#define\s+(REGISTER_\w+|register_\w+)"
                for match in re.finditer(macro_pattern, content):
                    apis["registration_macros"].append(match.group(1))
                func_pattern = r"(template\s*<[^>]+>\s*)?([\w:]+)\s+(register_\w+|create_\w+|run)\s*\("
                for match in re.finditer(func_pattern, content):
                    apis["operation_functions"].append(f"{match.group(3)}")
        return apis


class SymbolTracker:
    """Tracks symbol dependencies across generated files"""

    def __init__(self):
        self.symbols: Dict[str, SymbolDependency] = {}
        self.file_symbols: Dict[str, Set[str]] = {}

    def add_symbol_definition(self, symbol: str, file: str, symbol_type: str):
        if symbol not in self.symbols:
            self.symbols[symbol] = SymbolDependency(symbol=symbol, defined_in=file, used_in=[], symbol_type=symbol_type)
        else:
            self.symbols[symbol].defined_in = file
        if file not in self.file_symbols:
            self.file_symbols[file] = set()
        self.file_symbols[file].add(symbol)

    def add_symbol_usage(self, symbol: str, file: str):
        if symbol not in self.symbols:
            self.symbols[symbol] = SymbolDependency(
                symbol=symbol, defined_in="<undefined>", used_in=[file], symbol_type="unknown"
            )
        elif file not in self.symbols[symbol].used_in:
            self.symbols[symbol].used_in.append(file)

    def validate_symbols(self) -> List[str]:
        return [
            f"Symbol '{symbol}' used in {dep.used_in} but never defined"
            for symbol, dep in self.symbols.items()
            if dep.defined_in == "<undefined>"
        ]

    def extract_symbols_from_code(self, code: str, file_type: str) -> Tuple[Set[str], Set[str]]:
        defined, used = set(), set()
        if file_type == "header":
            func_pattern = r"(?:constexpr\s+)?(?:inline\s+)?(?:auto|[\w:]+)\s+(\w+)\s*\([^)]*\)"
            for match in re.finditer(func_pattern, code):
                if not match.group(1).startswith("_"):
                    defined.add(match.group(1))
            struct_pattern = r"(?:struct|class)\s+(\w+)"
            for match in re.finditer(struct_pattern, code):
                defined.add(match.group(1))
        elif file_type == "implementation":
            func_impl_pattern = r"[\w:]+\s+(?:[\w:]+::)?(\w+)\s*\([^)]*\)\s*{"
            for match in re.finditer(func_impl_pattern, code):
                defined.add(match.group(1))
            call_pattern = r"(\w+)\s*\("
            for match in re.finditer(call_pattern, code):
                if match.group(1) not in ["if", "while", "for", "switch", "return"]:
                    used.add(match.group(1))
        return defined, used


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

        self.analyzer = CodeAnalyzer(self.tt_metal_path)
        self.symbol_tracker = SymbolTracker()
        self.discovered_patterns = {}
        self.discovered_apis = {}
        self.generated_code = {}

        self.prompt_refiner = PersistentPromptRefiner(self.operation_name, db_path=refinement_db_path)
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

    def get_generation(self, prompt, max_tokens=MAX_TOKENS, is_fix=False) -> str:
        """Call Claude API to generate code."""
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        system_prompt = (
            f"You are an expert in Tenstorrent's TT-Metal SDK, generating a CUSTOM TTNN operation '{self.operation_name}'.\n"
            f"Use EXACT naming: operation '{self.operation_name}', Python function '{self.python_function_name}'.\n"
        )
        if is_fix:
            final_prompt = f"COMPILATION ERROR FIXES NEEDED:\n{prompt}\n\nGenerate the corrected code."
        else:
            final_prompt = prompt
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": final_prompt}],
        }
        stage = "Fixing" if is_fix else "Generating"
        print(f"[{stage}] Calling LLM for '{self.operation_name}' (Iteration {self.iteration})")
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json().get("content", [{}])[0].get("text", "")
        except Exception as e:
            print(f"[Error] API call failed: {e}")
            return ""

    def parse_response(self, response_text: str) -> str:
        """Extract code from Claude's response."""
        pattern = r"```(?:\w+)?\n([\s\S]*?)\n```"
        matches = re.findall(pattern, response_text)
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

    def combine_context(self, files):
        """Combine context from multiple files into a single string for generation."""
        output = files[0]
        for file in files[1:]:
            output += f"\n--- Make your work compatible with this following material, this is the {file['name']} implementation that you need ---\n{file['code']}"
        return output

    def generate_prompt(self, context, file) -> str:
        """Generate the base prompt emphasizing unique operation generation."""
        return f"""Generate a complete and UNIQUE TTNN eltwise operation for '{self.operation_type}' named '{self.operation_name}'.
        - Python function: ttnn.operations.{self.python_function_name}(a, b)
        - C++ class: {self.operation_class_name}
        CRITICAL: Must build successfully and use UNIQUE naming. Add debug markers to verify custom code execution.
        CONTEXT: {context}
        Generate the code for the file: {file['name']}
        MAKE SURE THE GENERATED CODE IS ENCLOSED IN ```, AS IN: ``` <code> ```
        """

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

    def analyze_ground_truth(self):
        """Analyze existing operations to understand patterns"""
        print("\n[Phase 0] Ground Truth Analysis")
        reference_ops = ["add", "multiply", "subtract"]
        for op in reference_ops:
            patterns = self.analyzer.analyze_existing_operation(op)
            if patterns and any(patterns.values()):
                self.discovered_patterns[op] = patterns
                print(f"✓ Analyzed {op} operation")
        self.discovered_apis = self.analyzer.discover_ttnn_apis()
        print(f"✓ Discovered {sum(len(v) for v in self.discovered_apis.values())} TTNN APIs")
        self._build_dynamic_contexts()

    def _build_dynamic_contexts(self):
        """Build contexts from discovered patterns"""
        print("\n[Context Building] Creating dynamic contexts from analysis")
        self.dynamic_context = {
            "header_examples": [],
            "implementation_examples": [],
            "device_op_examples": [],
            "pybind_examples": [],
        }
        for op, patterns in self.discovered_patterns.items():
            for file_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if file_type == "header":
                        self.dynamic_context["header_examples"].append(pattern.example)
                    elif file_type == "implementation":
                        self.dynamic_context["implementation_examples"].append(pattern.example)
                    elif file_type == "device_op":
                        self.dynamic_context["device_op_examples"].append(pattern.example)
                    elif file_type == "pybind":
                        self.dynamic_context["pybind_examples"].append(pattern.example)

    def generate_with_validation(self, prompt: str, file_key: str) -> str:
        """Generate code with immediate validation"""
        enhanced_prompt = self._enhance_prompt_with_discoveries(prompt, file_key)
        response = self.get_generation(enhanced_prompt)
        code = self.parse_response(response)
        self.generated_code[file_key] = code
        return code

    def _enhance_prompt_with_discoveries(self, prompt: str, file_key: str) -> str:
        """Enhance prompt with discovered patterns"""
        enhancement = "\n\nBASED ON ANALYSIS OF EXISTING OPERATIONS:\n"
        file_type = "header" if file_key in ["hpp", "op-hpp"] else "implementation"
        if file_type in self.dynamic_context:
            examples = self.dynamic_context.get(f"{file_type}_examples", [])
            if examples:
                enhancement += f"Example patterns:\n"
                for i, example in enumerate(examples[:2]):
                    enhancement += f"Example {i+1}:\n```cpp\n{example}\n```\n"
        return prompt + enhancement

    def _process_build_errors_for_debugging(self, raw_error_output: str) -> str:
        """Process raw build output to extract and format errors for LLM analysis."""
        lines = raw_error_output.split("\n")
        cpp_errors = self._extract_cpp_compilation_errors(lines)
        cmake_errors = self._extract_cmake_errors(lines)
        all_error_blocks = cpp_errors + cmake_errors
        if not all_error_blocks:
            return raw_error_output[-4000:]
        return "\n\n=== ERROR BLOCK ===\n\n".join(all_error_blocks)

    def _extract_cpp_compilation_errors(self, lines: List[str]) -> List[str]:
        """Extract C++ compilation errors with file:line:column format"""
        error_blocks = []
        i = 0
        while i < len(lines):
            line = lines[i]
            cpp_error_pattern = r"(/[^:]+):(\d+):(\d+):\s+(error|fatal error):"
            match = re.search(cpp_error_pattern, line)
            if match:
                error_block = []
                context_start = max(0, i - 3)
                error_block.extend(l for l in lines[context_start:i] if l.strip())
                error_block.append(line)
                i += 1
                while i < len(lines) and (
                    not re.search(cpp_error_pattern, lines[i]) and not lines[i].strip().startswith("ninja:")
                ):
                    if lines[i].strip():
                        error_block.append(lines[i])
                    i += 1
                error_blocks.append("\n".join(error_block))
            else:
                i += 1
        return error_blocks

    def _extract_cmake_errors(self, lines: List[str]) -> List[str]:
        """Extract CMake errors with CMake Error format"""
        error_blocks = []
        for i, line in enumerate(lines):
            if "cmake error" in line.lower() or line.strip().startswith("CMake Error"):
                error_blocks.append("\n".join(lines[i : min(len(lines), i + 20)]))
        return error_blocks

    def _identify_problematic_files(self, error_output: str) -> List[str]:
        """Use LLM to identify which files need to be regenerated."""
        print("[Debug Analysis] Using LLM to identify problematic files...")
        analysis_prompt = f"""You are analyzing build errors for '{self.operation_name}'.
            BUILD ERRORS: {error_output}
            AVAILABLE FILES: {', '.join(self.files.keys())}
            Analyze the errors and determine which files need to be regenerated.
            Return ONLY the file keys that need to be regenerated, enclosed in ``` like this:
            ```
            cpp
            hpp
            ```
            Return only the relevant files, no other text."""
        try:
            response = self.get_generation(analysis_prompt, max_tokens=1024)
            file_list_str = self.parse_response(response)
            if not file_list_str:
                return []
            identified_files = [line.strip() for line in file_list_str.split("\n") if line.strip() in self.files]
            print(f"[Debug Analysis] LLM identified files: {identified_files}")
            return identified_files
        except Exception as e:
            print(f"[Debug Analysis Error] Failed to get LLM analysis: {e}")
            return []

    def _regenerate_problematic_files(self, file_keys: List[str], error_output: str) -> bool:
        """Regenerate specific files with error context to fix build issues."""
        print(f"[Debug Regeneration] Regenerating {len(file_keys)} files with error context")
        regenerated_count = 0
        for file_key in file_keys:
            if file_key in self.files:
                if self._regenerate_single_file_with_errors(file_key, error_output):
                    regenerated_count += 1
        return regenerated_count > 0

    def _regenerate_single_file_with_errors(self, file_key: str, error_output: str) -> bool:
        """Regenerate a single file using error context and original file content."""
        try:
            file_config = self.files[file_key]
            file_path = self.output_dir / file_config["name"]
            current_content = file_path.read_text() if file_path.exists() else ""
            fix_prompt = self._create_comprehensive_fix_prompt(file_key, current_content, error_output)
            fixed_code = self.parse_response(self.get_generation(fix_prompt, is_fix=True))
            if not fixed_code:
                return False
            self.save_file(file_key, fixed_code)
            return True
        except Exception as e:
            print(f"[Debug] Error regenerating {file_key}: {e}")
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

        system_prompt = (
            f"You are an expert in Tenstorrent's TT-Metal SDK. You are generating a CUSTOM TTNN operation '{self.operation_name}'.\n"
            "Your primary goal is to generate correct, complete, and buildable code.\n"
            # "While writing an #include directive for a file, you may use the 'find_file_in_repository' tool to verify its exact path.\n"
            "To resolve undefined function errors, use 'extract_symbols_from_file' on the relevant header to get the correct signature.\n"
            "Use 'find_api_usage' to find real usage examples of a specific API function. Use resolve_namespace_and_verify to verify namespace usage and include paths, \n"
            "it should be called for every import and namespace. Use 'search_tt_metal_docs' to understand the purpose/function of a given API. \n"
        )

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

    '''
    def debug_build_loop(self) -> Tuple[bool, str]:
        """
        Runs the build-and-fix loop, using tools to diagnose and fix build errors.
        Also refines prompts based on errors for future use.
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

            # Clean the error output
            cleaned_errors = self.clean_build_errors(raw_error_output)
            critical_errors = self.extract_critical_errors(raw_error_output)

            # Refine prompts based on these errors for future use
            current_file_contents = {
                key: data['code'] for key, data in self.files.items() if data['code']
            }
            self.prompt_refiner.analyze_errors_and_refine_prompts(
                cleaned_errors, current_file_contents, self.api_key, self.model
            )
            print("[Prompt Refiner]" + self.prompt_refiner.get_refinement_summary())

            print(f"\n[Debug] Cleaned error summary:")
            print("="*60)
            print(cleaned_errors)
            print("="*60)

            # Create fix prompt
            fix_prompt = self._create_tool_based_debugging_prompt(cleaned_errors, critical_errors)
            messages = [{"role": "user", "content": fix_prompt}]

            print("\n[Debug] Asking LLM to analyze build errors and propose fixes...")
            response_text = self.get_generation_with_tools(messages)

            # Apply the fixes
            self.apply_code_fixes_from_llm_response(response_text)

        print(f"[Debug Failed] Could not fix the build after {self.build_retries} attempts.")
        print("\n[Final Refinement Summary]")
        print(self.prompt_refiner.get_refinement_summary())
        return False, "Build debugging failed"
    '''

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

    def debug_build_loop(self) -> Tuple[bool, str]:
        """
        Runs the build-and-fix loop, using tools to diagnose and fix build errors.
        Also refines prompts based on errors for future use.
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

            # Clean the error output
            cleaned_errors = self.clean_build_errors(raw_error_output)
            critical_errors = self.extract_critical_errors(raw_error_output)

            # Refine prompts based on these errors for future use
            current_file_contents = {key: data["code"] for key, data in self.files.items() if data["code"]}
            self.prompt_refiner.analyze_errors_and_refine_prompts(
                cleaned_errors, current_file_contents, self.api_key, self.model
            )
            print("[Prompt Refiner]" + self.prompt_refiner.get_refinement_summary())

            print(f"\n[Debug] Cleaned error summary:")
            print("=" * 60)
            print(cleaned_errors)
            print("=" * 60)

            # First, identify which files need to be fixed
            files_to_fix = self._identify_files_to_fix(cleaned_errors, critical_errors)

            if not files_to_fix:
                print("[Debug] Could not identify which files to fix")
                continue

            print(f"[Debug] Files identified for regeneration: {files_to_fix}")

            # Fix each file one by one
            for file_key in files_to_fix:
                print(f"\n[Debug] Regenerating file: {file_key} ({self.files[file_key]['name']})")

                # Create file-specific debugging prompt
                fix_prompt = self._create_single_file_debugging_prompt(file_key, cleaned_errors, critical_errors)

                messages = [{"role": "user", "content": fix_prompt}]
                print(f"[Debug] Getting fix for {file_key}...")
                response_text = self.generate_with_refined_prompt(fix_prompt, file_key)

                # Extract and apply the fix for this specific file
                code = self.parse_response(response_text)
                if code:
                    self.save_file(file_key, code)
                    print(f"[Debug] Applied fix for {file_key}")
                else:
                    print(f"[Debug] Failed to extract code for {file_key}")

        print(f"[Debug Failed] Could not fix the build after {self.build_retries} attempts.")
        print("\n[Final Refinement Summary]")
        print(self.prompt_refiner.get_refinement_summary())
        return False, "Build debugging failed"

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
        except Exception as e:
            print(f"[Debug] Error identifying files to fix: {e}")

        # Fallback to the old method if JSON extraction fails
        return self._identify_problematic_files(cleaned_errors)

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

            Requirements:

            Fix all compilation errors related to this file
            Preserve the unique naming: {self.operation_name}
            Ensure the code integrates properly with the reference files shown above

            Generate the complete corrected code for {file_info['name']}.

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

            Focus on fixing the ROOT CAUSE of the errors. Now analyze and provide the corrected files.
            """

    def _create_tool_based_debugging_prompt(self, cleaned_errors: str, critical_errors: Dict[str, List[str]]) -> str:
        """
        This method is now deprecated in favor of file-by-file debugging.
        Kept for backward compatibility if needed.
        """
        # Call the new identification method
        files_to_fix = self._identify_files_to_fix(cleaned_errors, critical_errors)
        # Return a simple message since we're not using this approach anymore
        return f"Files identified for fixing: {files_to_fix}. Use file-by-file debugging instead."


def main():
    """Example usage of the agent"""
    import argparse

    parser = argparse.ArgumentParser(description="TTNN eltwise operation generator")
    parser.add_argument("--operation", default="add", help="Type of eltwise operation")
    parser.add_argument(
        "--tt-metal-path", required=False, default="/home/user/tt-metal", help="Path to TT-Metal repository"
    )
    parser.add_argument("--custom-suffix", default="custom", help="Suffix for operation names")
    args = parser.parse_args()

    agent = TTNNOperationAgent(
        operation_type=args.operation, tt_metal_path=args.tt_metal_path, custom_suffix=args.custom_suffix
    )
    if COMPLETE_PARTIAL_OP:
        success = agent.complete_partial_operation()
        return 0
    elif DEBUG_ONLY:
        # success, _ = agent.debug_build_loop()
        success = agent.run_and_debug_test("/home/user/tt-metal/tests/ttnn/unit_tests/test_exp_add.py")
        # agent.add_operation_to_cmake("/home/user/tt-metal/ttnn/CMakeLists.txt", args.operation)
    else:
        success = agent.build_operation()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
