import os
import re
from pathlib import Path
import time
from typing import Union, List, Dict, Tuple, Optional, Any
import tempfile
import subprocess
import json
import urllib
import os
import re
import tempfile
import subprocess
import json
import urllib
from ttnn_op_generator.tools.include_tool import validate_includes_for_file 
from ttnn_op_generator.tools.tree_sitter_tool import parse_file, query, has_errors


# --- External/Missing Dependencies (keep in try-except if not available) ---
try:
    from tree_sitter_editor import TreeSitterEditor, CodeEdit
except ImportError:
    print("Warning: `tree_sitter_editor` not found. Targeted editing features may be limited.")
    TreeSitterEditor = None # Assign None to prevent NameError later
    CodeEdit = None # Assign None to prevent NameError later

# --- Tool Implementations ---
def wait_for_usage_limits():
    time.sleep(1)  # This is to prevent the API from receiving too many requests per minute.


def find_files_in_repository(filenames: Union[str, List[str]]) -> str:
    """
    Finds one or more files in the tt-metal repository.
    Can accept either a single filename string or a list of filenames.
    """
    # Convert single filename to list for uniform processing
    if isinstance(filenames, str):
        filenames = [filenames]

    search_path_str = os.environ.get("TT_METAL_PATH")
    if not search_path_str:
        return "Error: TT_METAL_PATH environment variable is not set."

    search_path = Path(search_path_str)
    print(f"[Tool Call] Searching for {len(filenames)} file(s) in '{search_path}'...")
    
    results = {}

    # Build a set of all files for efficient lookup
    all_files = {}
    for root, _, files in os.walk(search_path):
        for file in files:
            if file not in all_files:
                all_files[file] = []
            all_files[file].append(Path(root) / file)

    # Search for each requested file
    for filename in filenames:
        if filename in all_files:
            # If multiple matches, return all paths
            paths = []
            for found_path in all_files[filename]:
                relative_path = str(found_path.relative_to(search_path))
                paths.append(relative_path)

            if len(paths) == 1:
                results[filename] = f"Found at: {paths[0]}"
            else:
                results[filename] = f"Found {len(paths)} matches:\n" + "\n".join(f"  - {p}" for p in paths[:5])
                if len(paths) > 5:
                    results[filename] += f"\n  ... and {len(paths) - 5} more"
        else:
            results[filename] = "Not found"

    # Format results
    output = f"File search results:\n"
    output += "=" * 80 + "\n"
    for filename, result in results.items():
        output += f"\n{filename}:\n{result}\n"

    print(f"[Tool Result] Found {sum(1 for r in results.values() if r != 'Not found')} of {len(filenames)} files")
    return output


def extract_symbols_from_files(filepaths: Union[str, List[str]]) -> str:
    """
    Extracts C++ symbols from one or more files.
    Can accept either a single filepath string or a list of filepaths.
    """
    # Convert single filepath to list for uniform processing
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    output_dir_str = os.environ.get("TTNN_OUTPUT_DIR")
    if not output_dir_str:
        return "Error: TTNN_OUTPUT_DIR environment variable is not set."

    print(f"[Tool Call] Extracting symbols from {len(filepaths)} file(s)...")
    results = {}

    # Regex patterns
    func_pattern = re.compile(r"^\s*([\w::<>,\s]+?)\s+([\w_:]+)\s*\(([^)]*)\)\s*(const)?\s*(?:;|{)", re.MULTILINE)
    class_struct_pattern = re.compile(r"^\s*(?:class|struct)\s+([\w_]+)", re.MULTILINE)

    for filepath in filepaths:
        full_path = Path(output_dir_str) / filepath

        if not full_path.exists():
            results[filepath] = f"Error: File does not exist"
            continue

        try:
            content = full_path.read_text()

            # Extract functions
            functions = [
                f"{match.group(1).strip()} {match.group(2).strip()}({match.group(3).strip()})"
                + (" const" if match.group(4) else "")
                for match in func_pattern.finditer(content)
            ]

            # Extract classes/structs
            classes_and_structs = [match.group(1) for match in class_struct_pattern.finditer(content)]

            results[filepath] = {"classes_structs": classes_and_structs, "functions": functions}
        except Exception as e:
            results[filepath] = f"Error reading file: {str(e)}"

    # Format results
    output = f"Symbol extraction results for {len(filepaths)} file(s):\n"
    output += "=" * 80 + "\n"

    for filepath, result in results.items():
        output += f"\n=== {filepath} ===\n"
        if isinstance(result, str):  # Error case
            output += result + "\n"
        else:
            output += f"Classes/Structs: {result['classes_structs'] or ['None']}\n"
            output += f"Functions ({len(result['functions'])}):\n"
            for func in result["functions"]:
                output += f"  - {func}\n"

    print(f"[Tool Result] Extracted symbols from {len(results)} files")
    print(f"[extract_symbols DEBUG] {output}")
    return output


def read_ttnn_example_files(filenames: Union[str, List[str]], summary_only: bool = False) -> str:
    """
    Reads one or more example files from the TTNN examples directory.
    Can accept either a single filename string or a list of filenames.
    If summary_only=True, returns just key patterns instead of full content.
    """
    # Convert single filename to list for uniform processing
    if isinstance(filenames, str):
        filenames = [filenames]

    tt_metal_path = os.environ.get("TT_METAL_PATH")
    if not tt_metal_path:
        return "Error: TT_METAL_PATH environment variable is not set."

    examples_path = Path(tt_metal_path) / "ttnn/cpp/ttnn/operations/examples"
    print(f"[Tool Call] Reading {len(filenames)} example file(s)...")

    # Build file index for efficient lookup
    file_index = {}
    for root, _, files in os.walk(examples_path):
        for f in files:
            if f.endswith((".hpp", ".cpp")):
                file_index[f] = Path(root) / f

    results = {}

    for filename in filenames:
        if filename in file_index:
            file_path = file_index[filename]
            try:
                content = file_path.read_text()

                if summary_only:
                    # Extract key patterns instead of full content
                    summary = {
                        "size": len(content),
                        "includes": re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)[:5],
                        "namespaces": re.findall(r"namespace\s+(\w+)", content),
                        "classes": re.findall(r"(?:class|struct)\s+(\w+)", content),
                        "key_functions": re.findall(r"(?:void|auto|Tensor)\s+(\w+)\s*\(", content)[:10],
                    }
                    results[filename] = summary
                else:
                    results[filename] = content

            except Exception as e:
                results[filename] = f"Error reading file: {str(e)}"
        else:
            results[filename] = "File not found in examples directory"

    # Format results
    if summary_only:
        output = f"Summary of {len(filenames)} example file(s):\n"
        output += "=" * 80 + "\n"
        for filename, result in results.items():
            output += f"\n=== {filename} ===\n"
            if isinstance(result, str):
                output += result + "\n"
            else:
                output += f"Size: {result['size']} bytes\n"
                output += f"Key includes: {', '.join(result['includes'])}\n"
                output += f"Namespaces: {', '.join(result['namespaces'])}\n"
                output += f"Classes/Structs: {', '.join(result['classes'])}\n"
                output += f"Key functions: {', '.join(result['key_functions'])}\n"
    else:
        output = f"Contents of {len(filenames)} example file(s):\n"
        output += "=" * 80 + "\n"
        for filename, content in results.items():
            output += f"\n=== {filename} ===\n"
            output += str(content) + "\n"
            output += "=" * 80 + "\n"

    print(f"[Tool Result] Read {len(results)} files")
    return output


def find_api_usages(function_names: Union[str, List[str]], max_examples_per_function: int = 3) -> str:
    """
    Searches for usage examples of one or more API functions.
    Can accept either a single function name string or a list of function names.
    """
    # Convert single function name to list for uniform processing
    if isinstance(function_names, str):
        function_names = [function_names]

    tt_metal_path = os.environ.get("TT_METAL_PATH")
    if not tt_metal_path:
        return "Error: TT_METAL_PATH environment variable is not set."

    search_path = Path(tt_metal_path)
    print(f"[Tool Call] Searching for usage examples of {len(function_names)} function(s)...")
    print(f"[Tool Call] Searching for functions")
    # Directories to search
    search_dirs = ["ttnn/cpp/ttnn/operations", "tt_metal/impl", "tt_metal/common", "tests/ttnn"]

    results = {}
    context_lines = 5

    for function_name in function_names:
        examples = []

        # Create regex patterns for this function
        patterns = [
            re.compile(rf"\b{re.escape(function_name)}\s*\("),
            re.compile(rf"\.\s*{re.escape(function_name)}\s*\("),
            re.compile(rf"::\s*{re.escape(function_name)}\s*\("),
        ]

        for search_dir in search_dirs:
            if len(examples) >= max_examples_per_function:
                break

            dir_path = search_path / search_dir
            if not dir_path.exists():
                continue

            for root, _, files in os.walk(dir_path):
                if len(examples) >= max_examples_per_function:
                    break

                for file in files:
                    if len(examples) >= max_examples_per_function:
                        break

                    if not file.endswith((".cpp", ".hpp", ".h")):
                        continue

                    file_path = Path(root) / file
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()

                        for i, line in enumerate(lines):
                            if any(pattern.search(line) for pattern in patterns):
                                start = max(0, i - context_lines)
                                end = min(len(lines), i + context_lines + 1)

                                # Find complete statement
                                while end < len(lines) and ";" not in lines[i:end]:
                                    end = min(len(lines), end + 1)

                                context = "".join(lines[start:end])
                                rel_path = file_path.relative_to(search_path)

                                examples.append({"file": str(rel_path), "line": i + 1, "context": context})
                                break
                    except Exception:
                        continue

        results[function_name] = examples

    # Format results
    output = f"API usage search results for {len(function_names)} function(s):\n"
    output += "=" * 80 + "\n"

    for function_name, examples in results.items():
        output += f"\n=== {function_name} ===\n"
        if not examples:
            output += "No usage examples found\n"
        else:
            output += f"Found {len(examples)} example(s):\n\n"
            for i, example in enumerate(examples, 1):
                output += f"Example {i} - {example['file']}:{example['line']}\n"
                output += "-" * 60 + "\n"
                output += example["context"]
                output += "\n"

    total_examples = sum(len(examples) for examples in results.values())
    print(f"[Tool Result] Found {total_examples} total usage examples")
    return output


def resolve_namespace_and_verify(
    symbols: Union[str, List[str]], context: str = "", verify_compilation: bool = True
) -> str:
    """
    Resolves the correct namespace and include path for symbols, and optionally
    verifies by compiling a test program.

    Args:
        symbols: Single symbol or list of symbols to resolve (e.g., 'run', 'CreateCircularBuffer')
        context: Optional code context to help identify the correct usage
        verify_compilation: Whether to verify with actual compilation
    """
    wait_for_usage_limits()

    if isinstance(symbols, str):
        symbols = [symbols]

    tt_metal_path = Path(os.environ.get("TT_METAL_PATH", "/home/user/tt-metal"))

    print(f"[Tool Call] Resolving namespaces for {len(symbols)} symbol(s)...")

    results = {}

    for symbol in symbols:
        print(f"\n[Namespace Resolution] Analyzing '{symbol}'...")

        # Step 1: Find all occurrences of the symbol
        occurrences = find_symbol_occurrences(symbol, tt_metal_path)

        # Step 2: Analyze occurrences to extract namespace patterns
        namespace_info = analyze_namespace_patterns(occurrences, symbol)

        # Step 3: Determine most likely correct usage
        best_match = determine_best_match(namespace_info, context, symbol)

        # Step 4: Optionally verify with compilation
        if verify_compilation and best_match and best_match.get("namespace"):
            compilation_result = verify_with_compilation(
                symbol, best_match["namespace"], best_match.get("include_files", []), tt_metal_path
            )
            best_match["compilation_verified"] = compilation_result["success"]
            best_match["compilation_output"] = compilation_result.get("output", "")

        results[symbol] = best_match

    # Format results
    return format_namespace_results(results)


def find_symbol_occurrences(symbol: str, search_path: Path) -> List[Dict]:
    """Find all occurrences of a symbol in the codebase"""
    occurrences = []

    # Directories to prioritize
    search_dirs = ["ttnn/cpp/ttnn", "tt_metal/api", "tt_metal/impl", "tt_metal/common", "ttnn/operations"]

    # Use grep for fast searching
    for search_dir in search_dirs:
        dir_path = search_path / search_dir
        if not dir_path.exists():
            continue

        try:
            # Search for the symbol with context
            cmd = [
                "grep",
                "-r",
                "-n",
                "-B2",
                "-A2",
                f"\\b{re.escape(symbol)}\\b",
                str(dir_path),
                "--include=*.hpp",
                "--include=*.h",
                "--include=*.cpp",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout:
                # Parse grep output
                current_file = None
                for line in result.stdout.split("\n"):
                    if not line:
                        continue

                    # Check if this is a file:line match
                    match = re.match(r"^([^:]+):(\d+)[:-](.*)$", line)
                    if match:
                        filepath = match.group(1)
                        line_num = match.group(2)
                        content = match.group(3)

                        # Extract namespace and usage pattern
                        namespace_pattern = extract_namespace_from_line(content, symbol)

                        occurrences.append(
                            {
                                "file": filepath,
                                "line": line_num,
                                "content": content.strip(),
                                "namespace_pattern": namespace_pattern,
                                "file_type": "header" if filepath.endswith((".hpp", ".h")) else "source",
                            }
                        )

        except subprocess.TimeoutExpired:
            print(f"[Warning] Search timed out in {search_dir}")
            continue

    return occurrences


def extract_namespace_from_line(line: str, symbol: str) -> Optional[str]:
    """Extract namespace pattern from a line of code"""
    # Patterns to look for
    patterns = [
        # namespace::namespace::symbol
        rf"((?:\w+::)+){re.escape(symbol)}",
        # using namespace::symbol
        rf"using\s+((?:\w+::)+){re.escape(symbol)}",
        # namespace { ... symbol ... }
        rf"namespace\s+(\w+)\s*{{[^}}]*{re.escape(symbol)}",
        # template<> struct namespace::symbol
        rf"(?:template\s*<[^>]*>\s*)?(?:struct|class)\s+((?:\w+::)*){re.escape(symbol)}",
    ]

    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            namespace = match.group(1)
            if namespace and namespace.endswith("::"):
                namespace = namespace[:-2]  # Remove trailing ::
            return namespace

    # Check if symbol appears without namespace (might be in current namespace)
    if re.search(rf"\b{re.escape(symbol)}\b", line):
        # Look for namespace declaration in context
        ns_match = re.search(r"namespace\s+(\w+)", line)
        if ns_match:
            return ns_match.group(1)
        return ""  # No namespace (global or current namespace)

    return None


def analyze_namespace_patterns(occurrences: List[Dict], symbol: str) -> Dict:
    """Analyze occurrences to identify namespace patterns"""
    namespace_stats = {}
    include_files = {}
    usage_examples = {}

    for occurrence in occurrences:
        namespace = occurrence.get("namespace_pattern")
        if namespace is not None:
            # Track namespace frequency
            full_ns = f"{namespace}::{symbol}" if namespace else symbol
            if full_ns not in namespace_stats:
                namespace_stats[full_ns] = {"count": 0, "files": set(), "examples": []}

            namespace_stats[full_ns]["count"] += 1
            namespace_stats[full_ns]["files"].add(occurrence["file"])
            namespace_stats[full_ns]["examples"].append(occurrence["content"])

            # Extract include files
            if occurrence["file_type"] == "header":
                if full_ns not in include_files:
                    include_files[full_ns] = set()

                # Get relative include path
                include_path = extract_include_path(occurrence["file"])
                if include_path:
                    include_files[full_ns].add(include_path)

    # Convert sets to lists for JSON serialization
    for ns in namespace_stats:
        namespace_stats[ns]["files"] = list(namespace_stats[ns]["files"])

    for ns in include_files:
        include_files[ns] = list(include_files[ns])

    return {"namespace_stats": namespace_stats, "include_files": include_files}


def extract_include_path(filepath: str) -> Optional[str]:
    """Extract the include path from a file path"""
    # Common include directory patterns
    include_patterns = [r".*/ttnn/(.*)", r".*/tt_metal/(.*)", r".*/tt_eager/(.*)", r".*/tt_lib/(.*)"]

    for pattern in include_patterns:
        match = re.search(pattern, filepath)
        if match:
            return match.group(1)

    # If no pattern matches, try to get relative path from common roots
    path_parts = Path(filepath).parts
    for i, part in enumerate(path_parts):
        if part in ["ttnn", "tt_metal", "tt_eager"]:
            return "/".join(path_parts[i:])

    return None


def determine_best_match(namespace_info: Dict, context: str, symbol: str) -> Dict:
    """Determine the most likely correct namespace based on analysis"""
    namespace_stats = namespace_info["namespace_stats"]
    include_files = namespace_info["include_files"]

    if not namespace_stats:
        return {"symbol": symbol, "found": False, "message": f"Symbol '{symbol}' not found in codebase"}

    # Score each namespace option
    scores = {}
    for full_ns, stats in namespace_stats.items():
        score = 0

        # Frequency score
        score += stats["count"] * 10

        # File diversity score
        score += len(stats["files"]) * 5

        # Context relevance score
        if context:
            # Check if namespace appears in context
            ns_part = full_ns.replace(f"::{symbol}", "")
            if ns_part and ns_part in context:
                score += 50

            # Check if any example matches context pattern
            for example in stats["examples"][:3]:
                if any(word in example for word in context.split() if len(word) > 3):
                    score += 10

        # Prefer commonly used namespaces
        common_namespaces = ["ttnn", "tt::tt_metal", "tt::operations"]
        ns_part = full_ns.replace(f"::{symbol}", "")
        if ns_part in common_namespaces:
            score += 20

        scores[full_ns] = score

    # Get best match
    best_ns = max(scores.keys(), key=lambda k: scores[k])
    best_stats = namespace_stats[best_ns]

    # Extract namespace part
    namespace_part = best_ns.replace(f"::{symbol}", "")

    result = {
        "symbol": symbol,
        "found": True,
        "namespace": namespace_part,
        "full_qualified": best_ns,
        "include_files": include_files.get(best_ns, []),
        "confidence": "high" if scores[best_ns] > 100 else "medium" if scores[best_ns] > 50 else "low",
        "occurrences": best_stats["count"],
        "examples": best_stats["examples"][:3],  # Top 3 examples
        "alternatives": [],
    }

    # Add alternatives
    sorted_options = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for alt_ns, alt_score in sorted_options[1:4]:  # Next 3 best options
        if alt_score > 20:  # Only include reasonable alternatives
            alt_namespace = alt_ns.replace(f"::{symbol}", "")
            result["alternatives"].append(
                {"namespace": alt_namespace, "full_qualified": alt_ns, "include_files": include_files.get(alt_ns, [])}
            )

    return result


def verify_with_compilation(symbol: str, namespace: str, include_files: List[str], tt_metal_path: Path) -> Dict:
    """Verify namespace usage by compiling a test program"""

    # Create a minimal test program
    test_code = generate_test_program(symbol, namespace, include_files)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
        f.write(test_code)
        test_file = f.name

    try:
        # Compile command similar to TT-Metal build
        compile_cmd = [
            "g++",
            "-std=c++17",
            "-c",  # Compile only, don't link
            f"-I{tt_metal_path}",
            f"-I{tt_metal_path}/tt_metal",
            f"-I{tt_metal_path}/ttnn",
            f"-I{tt_metal_path}/tt_metal/api",
            f"-I{tt_metal_path}/tt_metal/impl",
            "-fsyntax-only",  # Syntax check only
            test_file,
        ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=5)

        return {
            "success": result.returncode == 0,
            "output": result.stderr if result.returncode != 0 else "Compilation successful",
            "test_code": test_code,
        }

    except Exception as e:
        return {"success": False, "output": f"Compilation test failed: {str(e)}", "test_code": test_code}
    finally:
        # Clean up
        os.unlink(test_file)


def generate_test_program(symbol: str, namespace: str, include_files: List[str]) -> str:
    """Generate a minimal test program to verify namespace usage"""

    # Common includes that might be needed
    common_includes = ["ttnn/operations/eltwise/binary/binary.hpp", "tt_metal/tt_metal.hpp", "ttnn/ttnn.hpp"]

    # Build include list
    includes = []
    for inc in include_files[:3]:  # Limit to avoid too many includes
        includes.append(f'#include "{inc}"')

    # Add common includes if not already present
    for inc in common_includes:
        if not any(inc in existing for existing in includes):
            includes.append(f'// #include "{inc}"  // Uncomment if needed')

    # Generate test code
    full_qualified = f"{namespace}::{symbol}" if namespace else symbol

    test_code = f"""// Auto-generated test to verify namespace resolution
    {chr(10).join(includes)}

    // Test: Verify that {full_qualified} is accessible
    void test_function() {{
        // Just reference the symbol to verify it exists and is accessible
        using function_ptr = decltype(&{full_qualified});
        (void)function_ptr();  // Suppress unused variable warning
    }}

    // Alternative test for different symbol types
    namespace test {{
        // If it's a type, this will work
        // using test_type = {full_qualified};

        // If it's a template, this might work
        // template<typename T>
        // using test_template = {full_qualified}<T>;
    }}
    """

    return test_code


def format_namespace_results(results: Dict[str, Dict]) -> str:
    """Format namespace resolution results"""
    output = "Namespace Resolution Results:\n"
    output += "=" * 80 + "\n"

    for symbol, info in results.items():
        output += f"\n=== Symbol: {symbol} ===\n"

        if not info.get("found", False):
            output += f"❌ {info.get('message', 'Symbol not found')}\n"
            continue

        output += f"✓ Found with {info['occurrences']} occurrences\n"
        output += f"Confidence: {info['confidence']}\n\n"

        output += f"Correct Usage:\n"
        output += f"  Full Qualified: {info['full_qualified']}\n"
        output += f"  Namespace: {info['namespace'] or '(global)'}\n"

        if info.get("include_files"):
            output += f"\nRequired Includes:\n"
            for inc in info["include_files"][:3]:
                output += f'  #include "{inc}"\n'

        if info.get("compilation_verified") is not None:
            output += f"\nCompilation Test: {'✓ PASSED' if info['compilation_verified'] else '✗ FAILED'}\n"
            if not info["compilation_verified"] and info.get("compilation_output"):
                output += f"Error: {info['compilation_output'][:200]}...\n"

        if info.get("examples"):
            output += f"\nUsage Examples:\n"
            for i, example in enumerate(info["examples"][:2], 1):
                output += f"  {i}. {example}\n"

        if info.get("alternatives"):
            output += f"\nAlternative Namespaces (if above doesn't work):\n"
            for alt in info["alternatives"]:
                output += f"  - {alt['full_qualified']}"
                if alt.get("include_files"):
                    output += f" (include: {alt['include_files'][0]})"
                output += "\n"
    print(f"[Namespace DEBUG] namespace output: {output}")
    return output


# Specialized function for common TT-Metal namespace issues
def check_common_namespace_issues(error_output: str) -> str:
    """
    Analyzes common namespace errors in TT-Metal and provides specific fixes
    """
    wait_for_usage_limits()

    print("[Tool Call] Analyzing namespace errors...")

    issues = []

    # Pattern 1: no matching function for call to 'namespace::function'
    no_match_pattern = r"no matching function for call to '([^']+)'"
    matches = re.findall(no_match_pattern, error_output)

    for match in matches:
        # Extract the function call
        parts = match.split("::")
        if len(parts) > 1:
            symbol = parts[-1].split("(")[0]  # Remove parameters
            attempted_namespace = "::".join(parts[:-1])

            # Resolve correct namespace
            result = resolve_namespace_and_verify([symbol], verify_compilation=False)
            issues.append(
                {"type": "wrong_namespace", "symbol": symbol, "attempted": attempted_namespace, "resolution": result}
            )

    # Pattern 2: 'namespace' has not been declared
    undeclared_pattern = r"'(\w+)' has not been declared"
    matches = re.findall(undeclared_pattern, error_output)

    for match in matches:
        result = resolve_namespace_and_verify([match], verify_compilation=False)
        issues.append({"type": "undeclared", "symbol": match, "resolution": result})

    # Format results
    output = "Common Namespace Issues Found:\n"
    output += "=" * 80 + "\n"

    for issue in issues:
        output += f"\n{issue['type'].upper()}: {issue['symbol']}\n"
        if issue["type"] == "wrong_namespace":
            output += f"  Attempted: {issue['attempted']}::{issue['symbol']}\n"
        output += issue["resolution"]
        output += "\n"

    return output


def search_tt_metal_docs(api_names: Union[str, List[str]], include_examples: bool = True) -> str:
    """
    Searches the Tenstorrent documentation for API information.

    Args:
        api_names: Single API name or list of API names to search for
        include_examples: Whether to include usage examples in the results

    Returns:
        Formatted string with API descriptions and signatures
    """
    # Convert single API name to list for uniform processing
    if isinstance(api_names, str):
        api_names = [api_names]

    print(f"[Tool Call] Searching Tenstorrent docs for {len(api_names)} API(s)...")

    results = {}
    base_url = "https://docs.tenstorrent.com"

    for api_name in api_names:
        try:
            # First, try to search the docs
            search_results = search_docs_site(api_name, base_url)

            if search_results:
                # Extract API information from the most relevant result
                api_info = extract_api_info(search_results[0], api_name, include_examples)
                print(f"[RETRIVED API INFO] {api_info}")
                results[api_name] = api_info
            else:
                # If no search results, try direct URL patterns
                api_info = try_direct_urls(api_name, base_url, include_examples)
                results[api_name] = (
                    api_info if api_info else {"found": False, "message": "API not found in documentation"}
                )

        except Exception as e:
            results[api_name] = {"found": False, "message": f"Error searching docs: {str(e)}"}

    # Format results
    print(f"[DEBUG] Docs tools raw results: {results}")
    return format_doc_results(results)


def search_docs_site(api_name: str, base_url: str) -> List[Dict]:
    """
    Search the documentation site for the API.
    This is a simplified version - in reality, you might need to handle JavaScript-rendered content.
    """
    # Common documentation URL patterns for TT-Metal
    search_patterns = [
        f"/tt-metal/latest/search.html?q={urllib.parse.quote(api_name)}",
        f"/api/metal/{urllib.parse.quote(api_name)}.html",
        f"/tt-metal/latest/api/{urllib.parse.quote(api_name)}.html",
    ]

    results = []

    for pattern in search_patterns:
        try:
            url = base_url + pattern
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    content = response.read().decode("utf-8")
                    # Extract relevant information from the page
                    if api_name.lower() in content.lower():
                        results.append({"url": url, "content": content})
                        break
        except:
            continue

    return results


def extract_api_info(search_result: Dict, api_name: str, include_examples: bool) -> Dict:
    """
    Extract API signature and description from documentation page content.
    """
    content = search_result["content"]
    url = search_result["url"]

    info = {
        "found": True,
        "url": url,
        "signature": None,
        "description": None,
        "parameters": [],
        "return_type": None,
        "examples": [],
    }

    # Extract function signature (looking for code blocks with the API name)
    # Pattern for C++ function signatures
    sig_pattern = rf"<pre[^>]*>.*?{re.escape(api_name)}\s*\([^)]*\).*?</pre>"
    sig_match = re.search(sig_pattern, content, re.IGNORECASE | re.DOTALL)

    if not sig_match:
        # Try alternative patterns
        sig_pattern = rf"<code[^>]*>.*?{re.escape(api_name)}\s*\([^)]*\).*?</code>"
        sig_match = re.search(sig_pattern, content, re.IGNORECASE | re.DOTALL)

    if sig_match:
        # Clean up the signature
        signature = re.sub(r"<[^>]+>", "", sig_match.group(0))
        signature = signature.strip()
        info["signature"] = signature

        # Try to extract return type
        return_match = re.match(r"^\s*(\w+(?:\s*<[^>]+>)?)\s+", signature)
        if return_match:
            info["return_type"] = return_match.group(1)

    # Extract description (look for paragraphs near the API name)
    desc_pattern = rf"<p[^>]*>([^<]*{re.escape(api_name)}[^<]*)</p>"
    desc_match = re.search(desc_pattern, content, re.IGNORECASE)

    if desc_match:
        info["description"] = desc_match.group(1).strip()
    else:
        # Try to find any descriptive text near the API
        context_pattern = rf"(\w.*?{re.escape(api_name)}.*?\w)"
        context_matches = re.findall(context_pattern, content, re.IGNORECASE)
        if context_matches:
            info["description"] = context_matches[0][:200] + "..."

    # Extract parameters if possible
    param_section = re.search(r"Parameters?:?(.*?)(?:Returns?:|Example:|$)", content, re.IGNORECASE | re.DOTALL)
    if param_section:
        param_text = param_section.group(1)
        # Simple parameter extraction
        param_pattern = r"(\w+)\s*[-–:]\s*([^,\n]+)"
        params = re.findall(param_pattern, param_text)
        info["parameters"] = [{"name": p[0], "description": p[1].strip()} for p in params[:5]]

    # Extract examples if requested
    if include_examples:
        example_pattern = r"<pre[^>]*>(.*?{re.escape(api_name)}.*?)</pre>"
        example_matches = re.findall(example_pattern, content, re.IGNORECASE | re.DOTALL)
        if example_matches:
            info["examples"] = [re.sub(r"<[^>]+>", "", ex).strip() for ex in example_matches[:2]]

    return info


def try_direct_urls(api_name: str, base_url: str, include_examples: bool) -> Dict:
    """
    Try common direct URL patterns for API documentation.
    """
    # Common API documentation paths
    url_patterns = [
        f"/tt-metal/latest/api.html#{api_name}",
        f"/api/{api_name}.html",
        f"/reference/{api_name}",
    ]

    for pattern in url_patterns:
        try:
            url = base_url + pattern
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    content = response.read().decode("utf-8")
                    return extract_api_info({"url": url, "content": content}, api_name, include_examples)
        except:
            continue

    return None


def format_doc_results(results: Dict[str, Dict]) -> str:
    """
    Format the documentation search results.
    """
    output = "Tenstorrent Documentation Search Results:\n"
    output += "=" * 80 + "\n"

    for api_name, info in results.items():
        output += f"\n=== {api_name} ===\n"

        if not info.get("found", False):
            output += f"❌ {info.get('message', 'Not found in documentation')}\n"
            continue

        output += f"✓ Found in documentation\n"
        if info.get("url"):
            output += f"URL: {info['url']}\n"

        if info.get("signature"):
            output += f"\nSignature:\n{info['signature']}\n"

        if info.get("return_type"):
            output += f"\nReturn Type: {info['return_type']}\n"

        if info.get("description"):
            output += f"\nDescription:\n{info['description']}\n"

        if info.get("parameters"):
            output += f"\nParameters:\n"
            for param in info["parameters"]:
                output += f"  - {param['name']}: {param['description']}\n"

        if info.get("examples"):
            output += f"\nExamples:\n"
            for i, example in enumerate(info["examples"], 1):
                output += f"Example {i}:\n```cpp\n{example}\n```\n"

    return output

def parse_and_analyze_code(file_path: str) -> Dict[str, Any]:
    """
    Parse a C++ file with tree-sitter and return its structure.
    
    This implementation works with the correct tree-sitter API where
    query.captures() returns a dictionary of {capture_name: [nodes]}.
    """
    
    tree_id = parse_file(file_path)
    
    # Initialize the structure
    structure = {
        "has_errors": has_errors(tree_id),
        "functions": [],
        "classes": [],
        "includes": [],
        "namespaces": [],
        "structs": [],
        "global_variables": [],
        "typedefs": [],
        "macros": []
    }
    
    # 1. Extract functions
    func_query = """
    (function_definition
        type: (_)? @return_type
        declarator: (function_declarator
            declarator: (identifier) @fn_name
            parameters: (parameter_list) @params)
        body: (compound_statement)? @body
    ) @fn_def
    
    (declaration
        type: (_)? @return_type
        declarator: (function_declarator
            declarator: (identifier) @fn_name
            parameters: (parameter_list) @params)
    ) @fn_decl
    """
    
    # Run the query - returns list of dicts with name, text, byte_range
    func_results = query(tree_id, func_query)
    
    # Group captures by their parent node (function definition/declaration)
    # We'll use byte ranges to determine nesting
    fn_groups = []
    
    # First, identify all function definitions and declarations
    for result in func_results:
        if result['name'] in ['fn_def', 'fn_decl']:
            fn_groups.append({
                'type': result['name'],
                'range': result['byte_range'],
                'is_definition': result['name'] == 'fn_def',
                'captures': {result['name']: result}
            })
    
    # Then, assign other captures to their parent function
    for result in func_results:
        if result['name'] not in ['fn_def', 'fn_decl']:
            # Find which function this capture belongs to
            capture_start, capture_end = result['byte_range']
            
            for group in fn_groups:
                group_start, group_end = group['range']
                if capture_start >= group_start and capture_end <= group_end:
                    group['captures'][result['name']] = result
                    break
    
    # Convert groups to function info
    for group in fn_groups:
        captures = group['captures']
        fn_info = {
            "name": captures.get('fn_name', {}).get('text', 'unknown'),
            "byte_range": group['range'],
            "is_definition": group['is_definition'],
            "return_type": captures.get('return_type', {}).get('text'),
            "parameters": captures.get('params', {}).get('text'),
            "full_range": group['range']
        }
        structure["functions"].append(fn_info)
    
    # 2. Extract classes
    class_query = """
    (class_specifier
        name: (type_identifier) @class_name
        body: (field_declaration_list) @class_body
    ) @class_spec
    """
    
    class_results = query(tree_id, class_query)
    
    # Group by class
    class_groups = []
    for result in class_results:
        if result['name'] == 'class_spec':
            class_groups.append({
                'range': result['byte_range'],
                'captures': {result['name']: result}
            })
    
    # Assign captures to classes
    for result in class_results:
        if result['name'] != 'class_spec':
            capture_start, capture_end = result['byte_range']
            
            for group in class_groups:
                group_start, group_end = group['range']
                if capture_start >= group_start and capture_end <= group_end:
                    group['captures'][result['name']] = result
                    break
    
    # Convert to class info
    for group in class_groups:
        captures = group['captures']
        class_info = {
            "name": captures.get('class_name', {}).get('text', 'unknown'),
            "type": "class",
            "byte_range": group['range'],
            "members": [],
            "methods": []
        }
        structure["classes"].append(class_info)
    
    # 3. Extract structs  
    struct_query = """
    (struct_specifier
        name: (type_identifier) @struct_name
        body: (field_declaration_list) @struct_body
    ) @struct_spec
    """
    
    struct_results = query(tree_id, struct_query)
    
    # Group by struct
    struct_groups = []
    for result in struct_results:
        if result['name'] == 'struct_spec':
            struct_groups.append({
                'range': result['byte_range'],
                'captures': {result['name']: result}
            })
    
    # Assign captures to structs
    for result in struct_results:
        if result['name'] != 'struct_spec':
            capture_start, capture_end = result['byte_range']
            
            for group in struct_groups:
                group_start, group_end = group['range']
                if capture_start >= group_start and capture_end <= group_end:
                    group['captures'][result['name']] = result
                    break
    
    # Convert to struct info
    for group in struct_groups:
        captures = group['captures']
        struct_info = {
            "name": captures.get('struct_name', {}).get('text', 'unknown'),
            "type": "struct",
            "byte_range": group['range'],
            "members": [],
            "methods": []
        }
        structure["structs"].append(struct_info)
    
    # 4. Extract includes
    include_query = """
    (preproc_include
        path: (string_literal) @include_path_str
    ) @include
    
    (preproc_include
        path: (system_lib_string) @include_path_sys
    ) @include
    """
    
    include_results = query(tree_id, include_query)
    
    # Group by include directive
    include_groups = []
    for result in include_results:
        if result['name'] == 'include':
            include_groups.append({
                'range': result['byte_range'],
                'captures': {result['name']: result}
            })
    
    # Assign paths to includes
    for result in include_results:
        if result['name'] in ['include_path_str', 'include_path_sys']:
            capture_start, capture_end = result['byte_range']
            
            for group in include_groups:
                group_start, group_end = group['range']
                if capture_start >= group_start and capture_end <= group_end:
                    group['captures']['path'] = result
                    group['captures']['is_system'] = result['name'] == 'include_path_sys'
                    break
    
    # Convert to include info
    for group in include_groups:
        captures = group['captures']
        if 'path' in captures:
            path_text = captures['path']['text']
            clean_path = path_text.strip('"<>')
            structure["includes"].append({
                "path": clean_path,
                "raw": path_text,
                "byte_range": group['range'],
                "is_system": captures.get('is_system', False)
            })
    
    # 5. Extract namespaces
    namespace_query = """
    (namespace_definition
        name: (namespace_identifier) @ns_name
        body: (declaration_list) @ns_body
    ) @namespace
    """
    
    namespace_results = query(tree_id, namespace_query)
    
    # Group by namespace
    namespace_groups = []
    for result in namespace_results:
        if result['name'] == 'namespace':
            namespace_groups.append({
                'range': result['byte_range'],
                'captures': {result['name']: result}
            })
    
    # Assign captures to namespaces
    for result in namespace_results:
        if result['name'] != 'namespace':
            capture_start, capture_end = result['byte_range']
            
            for group in namespace_groups:
                group_start, group_end = group['range']
                if capture_start >= group_start and capture_end <= group_end:
                    group['captures'][result['name']] = result
                    break
    
    # Convert to namespace info
    for group in namespace_groups:
        captures = group['captures']
        ns_info = {
            "name": captures.get('ns_name', {}).get('text', 'anonymous'),
            "byte_range": group['range'],
            "body_range": captures.get('ns_body', {}).get('byte_range')
        }
        structure["namespaces"].append(ns_info)
    
    # 6. Extract typedefs
    typedef_query = """
    (type_definition
        type: (_) @original_type
        declarator: (type_identifier) @alias
    ) @typedef
    """
    
    typedef_results = query(tree_id, typedef_query)
    
    # Group by typedef
    typedef_groups = []
    for result in typedef_results:
        if result['name'] == 'typedef':
            typedef_groups.append({
                'range': result['byte_range'],
                'captures': {result['name']: result}
            })
    
    # Assign captures to typedefs
    for result in typedef_results:
        if result['name'] != 'typedef':
            capture_start, capture_end = result['byte_range']
            
            for group in typedef_groups:
                group_start, group_end = group['range']
                if capture_start >= group_start and capture_end <= group_end:
                    group['captures'][result['name']] = result
                    break
    
    # Convert to typedef info
    for group in typedef_groups:
        captures = group['captures']
        typedef_info = {
            "alias": captures.get('alias', {}).get('text', 'unknown'),
            "original_type": captures.get('original_type', {}).get('text'),
            "byte_range": group['range']
        }
        structure["typedefs"].append(typedef_info)
    
    # 7. Extract macros
    macro_query = """
    (preproc_def
        name: (identifier) @macro_name
        value: (preproc_arg) @macro_value
    ) @macro
    """
    
    macro_results = query(tree_id, macro_query)
    
    # Group by macro
    macro_groups = []
    for result in macro_results:
        if result['name'] == 'macro':
            macro_groups.append({
                'range': result['byte_range'],
                'captures': {result['name']: result}
            })
    
    # Assign captures to macros
    for result in macro_results:
        if result['name'] != 'macro':
            capture_start, capture_end = result['byte_range']
            
            for group in macro_groups:
                group_start, group_end = group['range']
                if capture_start >= group_start and capture_end <= group_end:
                    group['captures'][result['name']] = result
                    break
    
    # Convert to macro info
    for group in macro_groups:
        captures = group['captures']
        macro_info = {
            "name": captures.get('macro_name', {}).get('text', 'unknown'),
            "value": captures.get('macro_value', {}).get('text', '').strip() if 'macro_value' in captures else None,
            "byte_range": group['range']
        }
        structure["macros"].append(macro_info)
    
    # 8. Extract global variables (simplified - declarations at file scope)
    var_query = """
    (declaration
        type: (_) @var_type
        declarator: (identifier) @var_name
    ) @var_decl
    """
    
    # This is a simplified approach - in reality, we'd need to check
    # that these declarations are at file scope, not inside functions/classes
    
    # Add summary statistics
    structure["summary"] = {
        "total_functions": len(structure["functions"]),
        "function_definitions": sum(1 for f in structure["functions"] if f.get("is_definition", False)),
        "function_declarations": sum(1 for f in structure["functions"] if not f.get("is_definition", False)),
        "total_classes": len(structure["classes"]),
        "total_structs": len(structure["structs"]),
        "total_includes": len(structure["includes"]),
        "system_includes": sum(1 for i in structure["includes"] if i.get("is_system", False)),
        "local_includes": sum(1 for i in structure["includes"] if not i.get("is_system", False)),
        "total_namespaces": len(structure["namespaces"]),
        "total_global_vars": len(structure["global_variables"]),
        "total_typedefs": len(structure["typedefs"]),
        "total_macros": len(structure["macros"])
    }
    
    print(f"[Tree-sitter Debug] Structure summary: {structure['summary']}")
    return structure
      
def apply_targeted_edits(file_path: str, edits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply a list of targeted edits to a file
    """
    from tree_sitter_editor import TreeSitterEditor, CodeEdit
    
    editor = TreeSitterEditor()
    
    # Convert dict edits to CodeEdit objects
    edit_objects = []
    for edit in edits:
        edit_objects.append(CodeEdit(
            operation=edit["operation"],
            target_type=edit["target_type"],
            target_name=edit["target_name"],
            content=edit.get("content"),
            location_hint=edit.get("location_hint")
        ))
    
    result = editor.apply_edits(file_path, edit_objects)
    
    return {
        "success": result.success,
        "message": result.message,
        "new_content": result.new_content
    }

# Add these new tools to AVAILABLE_TOOLS:
"""
{
        "name": "read_ttnn_example_files",
        "description": "Reads one or more example files from TTNN examples. Accepts either a single filename or a list. Can return full contents or just a summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filenames": {
                    "type": ["string", "array"],
                    "description": "Single filename string or list of example filenames, e.g., 'example.hpp' or ['example.hpp', 'example.cpp']",
                    "items": {"type": "string"} if "array" else None
                },
                "summary_only": {
                    "type": "boolean",
                    "description": "If true, returns key patterns/summary instead of full content (useful for many files)",
                    "default": False
                }
            },
            "required": ["filenames"]
        }
    },

{
        "name": "check_common_namespace_issues",
        "description": "Analyzes build error output to identify and resolve common namespace issues in TT-Metal. Automatically extracts problematic symbols and suggests corrections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "error_output": {
                    "type": "string",
                    "description": "The build error output containing namespace-related errors"
                }
            },
            "required": ["error_output"]
        }
    }
 {
        "name": "extract_symbols_from_files",
        "description": "Extracts symbols (i.e classes, functions, and structs) from one or more C++ files in the output directory. Accepts either a single filepath or a list of filepaths. Returns all symbols in one call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": ["string", "array"],
                    "description": "Single filepath string or list of filepaths relative to output directory, e.g., 'op.hpp' or ['op.hpp', 'device/op_device.hpp']",
                    "items": {"type": "string"} if "array" else None,
                }
            },
            "required": ["filepaths"],
        },
    },
{
        "name": "resolve_namespace_and_verify",
        "description": "Resolves the correct namespace and include path for C++ symbols by searching the codebase. Can optionally verify with compilation. Essential for fixing 'no matching function' and namespace errors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": ["string", "array"],
                    "description": "Single symbol or list of symbols to resolve, e.g., 'run' or ['run', 'CreateCircularBuffer', 'tt_metal_device']",
                    "items": {"type": "string"} if "array" else None,
                },
                "context": {
                    "type": "string",
                    "description": "Optional code context to help identify correct usage",
                    "default": "",
                },
                "verify_compilation": {
                    "type": "boolean",
                    "description": "Whether to verify namespace with test compilation (default: true)",
                    "default": True,
                },
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "search_tt_metal_docs",
        "description": "Searches Tenstorrent documentation for API signatures and usage. Helps fix errors related to understanding the functionality of APIs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "api_names": {
                    "type": ["string", "array"],
                    "description": "API name(s) to search, e.g., 'CreateProgram' or ['CreateProgram', 'EnqueueWriteBuffer']",
                    "items": {"type": "string"},
                },
                "include_examples": {
                    "type": "boolean",
                    "description": "Include usage examples (default: true)",
                    "default": True,
                },
            },
            "required": ["api_names"],
        },
    },
"""
# --- Updated Tool Definitions for the API ---

AVAILABLE_TOOLS = [
        {
        "name": "find_files_in_repository",
        "description": "Searches for one or more files in the TT-Metal repository. Accepts either a single filename or a list of filenames. Returns the relative paths for all requested files in one call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filenames": {
                    "type": ["string", "array"],
                    "description": "Single filename string or list of filenames to search for, e.g., 'tensor.hpp' or ['tensor.hpp', 'device.hpp', 'program.hpp']",
                    "items": {"type": "string"} if "array" else None
                }
            },
            "required": ["filenames"]
        }
    },
        {
        "name": "find_api_usages",
        "description": "Searches for usage examples of one or more API functions. Accepts either a single function name or a list. Returns examples for all requested functions in one call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "function_names": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ],
                    "description": "Single function name or list of API functions to find, e.g., 'create_program' or ['create_program', 'CreateCircularBuffer', 'run_operation']"
                },
                "max_examples_per_function": {
                    "type": "integer",
                    "description": "Maximum examples to return per function (default: 3)",
                    "default": 3
                }
            },
            "required": ["function_names"]
        }
        },
    {
        "name": "parse_and_analyze_code",
        "description": "Parse a C++ file and analyze its tree-sitter structure",
        "input_schema": {  # Changed from "parameters" to "input_schema"
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the C++ file to analyze"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "apply_targeted_edits",
        "description": "Apply targeted edits to a C++ file",
        "input_schema": {  # Changed from "parameters" to "input_schema"
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "edits": {
                    "type": "array",
                    "description": "List of edit operations",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string", "description": "Type of edit: insert, delete, replace, modify"},
                            "target_type": {"type": "string", "description": "Type of target: function, class, include, namespace, member"},
                            "target_name": {"type": "string", "description": "Name of the target to edit"},
                            "content": {"type": "string", "description": "New content for insert/replace/modify operations"},
                            "location_hint": {"type": "string", "description": "Where to place the edit, e.g., 'inside:namespace_name'"}
                        },
                        "required": ["operation", "target_type", "target_name"]
                    }
                }
            },
            "required": ["file_path", "edits"]
        }
    },
    {
    "name": "validate_includes_for_file",
    "description": "Validates C++ include paths by attempting to compile them in the context of the target file. Returns corrected paths and recommendations. Essential for fixing include errors.",
    "input_schema": {
        "type": "object",
        "properties": {
            "include_paths": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}}
                ],
                "description": "Single include path or list of include paths to validate, e.g., 'ttnn/tensor.hpp' or ['tt_metal/device.hpp', 'ttnn/operation.hpp']"
            },
            "target_file_path": {
                "type": "string",
                "description": "Path of the target file relative to operation directory, e.g., 'eltwise_multiply_custom.cpp' or 'device/eltwise_multiply_custom_op.hpp'"
            }
        },
        "required": ["include_paths", "target_file_path"]
    },
    }
]

# Add to TOOL_EXECUTORS:

# --- Tool Executor Mapping ---

TOOL_EXECUTORS = {
    "find_files_in_repository": find_files_in_repository,
    #"extract_symbols_from_files": extract_symbols_from_files,
    # "read_ttnn_example_files": read_ttnn_example_files,
    "find_api_usages": find_api_usages,
    "parse_and_analyze_code": parse_and_analyze_code,
    "apply_targeted_edits": apply_targeted_edits,
    "validate_includes_for_file": validate_includes_for_file
    #"resolve_namespace_and_verify": resolve_namespace_and_verify,
    #"search_tt_metal_docs": search_tt_metal_docs
    # "check_common_namespace_issues": check_common_namespace_issues
}
