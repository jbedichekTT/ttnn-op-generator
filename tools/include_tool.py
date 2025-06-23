import os
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Union

def validate_includes_for_file(include_paths: Union[str, List[str]], target_file_path: str) -> str:
    """
    Validates include paths by attempting to compile them in the context of the target file.
    
    This tool creates a minimal C++ file with the specified includes and attempts to compile it
    in the same directory context as the target file to verify the includes are correct.
    
    Args:
        include_paths: Single include path or list of include paths to validate
        target_file_path: Path of the target file (relative to output directory) where these includes will be used
                         e.g., "eltwise_multiply_custom.cpp" or "device/eltwise_multiply_custom_op.hpp"
    
    Returns:
        Formatted validation results with corrections
    """
    # Convert single include to list
    if isinstance(include_paths, str):
        include_paths = [include_paths]
    
    tt_metal_path = Path(os.environ.get("TT_METAL_PATH", "/home/user/tt-metal"))
    output_dir = Path(os.environ.get("TTNN_OUTPUT_DIR", ""))
    
    if not output_dir:
        return "Error: TTNN_OUTPUT_DIR environment variable not set"
    
    # Determine the directory of the target file
    target_file = Path(target_file_path)
    target_dir = output_dir / target_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Include Validation] Validating {len(include_paths)} includes for {target_file_path}")
    print(f"[Include Validation] Target directory: {target_dir}")
    
    results = []
    
    for include_path in include_paths:
        result = {
            "include": include_path,
            "valid": False,
            "corrected_path": None,
            "error": None,
            "recommendation": None
        }
        
        # Skip system includes that we know are available
        system_includes = ["Python.h", "pybind11/pybind11.h", "pybind11/stl.h", 
                          "cstdint", "cmath", "vector", "memory", "string", "optional"]
        if include_path in system_includes or include_path.startswith("pybind11/"):
            result["valid"] = True
            result["corrected_path"] = include_path
            result["recommendation"] = "System include - use angle brackets: <>"
            results.append(result)
            continue
        
        # Create a test file to compile
        test_content = f"""#include "{include_path}"
        // Test compilation for include validation
        int main() {{ return 0; }}
        """
        
        # Write test file in the target directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', dir=target_dir, delete=False) as f:
            f.write(test_content)
            test_file = f.name
        
        try:
            # Try to compile with various include paths
            compile_attempts = []
            
            # Attempt 1: Try as-is with standard include paths
            cmd = [
                "g++",
                "-std=c++17",
                "-c",
                f"-I{tt_metal_path}",
                f"-I{tt_metal_path}/ttnn",
                f"-I{tt_metal_path}/tt_metal",
                f"-I{tt_metal_path}/ttnn/cpp",
                f"-I{target_dir}",  # Include the target directory itself
                "-fsyntax-only",
                test_file
            ]
            
            result_1 = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            compile_attempts.append(("standard paths", result_1))
            
            if result_1.returncode == 0:
                result["valid"] = True
                result["corrected_path"] = include_path
                results.append(result)
                print(f'[INCLUDE TOOL DEBUG] validated include: {result["corrected_path"]}')
                continue
            
            # Attempt 2: Check if it's a local file (same directory)
            local_file = target_dir / include_path
            if local_file.exists() or (target_dir / Path(include_path).name).exists():
                # Try with just the filename
                filename = Path(include_path).name
                test_content_local = f"""#include "{filename}"
                int main() {{ return 0; }}
                """
                with open(test_file, 'w') as f:
                    f.write(test_content_local)
                
                result_2 = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result_2.returncode == 0:
                    result["valid"] = True
                    result["corrected_path"] = filename
                    result["recommendation"] = "Use local include (filename only) for files in same directory"
                    results.append(result)
                    continue
            
            # Attempt 3: Search for the file in the repository
            filename = Path(include_path).name
            
            print(f'[INCLUDE TOOL DEBUG] Searching for {filename} in tt-metal')
            found_files = find_files_in_repository(filename, tt_metal_path)
            
            if found_files:
                # Try each found file
                for found_path in found_files[:3]:  # Limit to first 3 matches
                    try:
                        print(f'[INCLUDE TOOL DEBUG] Trying to compile {found_path}')
                        rel_path = Path(found_path).relative_to(tt_metal_path)
                        test_content_found = f"""#include "{rel_path}"
                        int main() {{ return 0; }}
                        """
                        with open(test_file, 'w') as f:
                            f.write(test_content_found)
                        
                        result_3 = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                        if result_3.returncode == 0:
                            print(f'[INCLUDE TOOL DEBUG] Include {rel_path} compiles successfully.')
                            result["valid"] = True
                            result["corrected_path"] = str(rel_path)
                            result["recommendation"] = f"File found at: {rel_path}"
                            break
                    except:
                        continue
            
            # If still not found, try common corrections
            corrections = {
                "tt_metal/operation.hpp": "ttnn/decorators.hpp",
                "ttnn/operation.hpp": "ttnn/decorators.hpp",
                "tt_metal/program.hpp": "tt_metal/impl/dispatch/program.hpp",
                "tt_metal/device.hpp": "tt_metal/impl/dispatch/device.hpp",
            }
            
            if include_path in corrections:
                corrected = corrections[include_path]
                test_content_corrected = f"""#include "{corrected}"
                int main() {{ return 0; }}
                """
                with open(test_file, 'w') as f:
                    f.write(test_content_corrected)
                
                result_4 = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result_4.returncode == 0:
                    result["valid"] = True
                    result["corrected_path"] = corrected
                    result["recommendation"] = f"Known correction: use {corrected}"
                    results.append(result)
                    continue
            
            # If all attempts failed, provide error details
            result["error"] = result_1.stderr.split('\n')[0] if result_1.stderr else "File not found"
            results.append(result)
            
        except Exception as e:
            result["error"] = str(e)
            results.append(result)
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    # Format results
    output = format_validation_results(results, target_file_path)
    return output

def find_files_in_repository(filename: str, search_path: Path) -> List[str]:
    """Find all occurrences of a file in the repository"""
    found_files = []
    
    # Limit search to relevant directories
    search_dirs = ["ttnn", "tt_metal", "tt_eager"]
    
    for search_dir in search_dirs:
        dir_path = search_path / search_dir
        if dir_path.exists():
            try:
                # Use find command for efficiency
                cmd = ["find", str(dir_path), "-name", filename, "-type", "f"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout:
                    found_files.extend(result.stdout.strip().split('\n'))
            except:
                # Fallback to Python glob if find fails
                for path in dir_path.rglob(filename):
                    if path.is_file():
                        found_files.append(str(path))
    
    return found_files

def format_validation_results(results: List[Dict], target_file_path: str) -> str:
    """Format validation results for the model"""
    output = []
    output.append(f"Include Validation Results for {target_file_path}:")
    output.append("=" * 80)
    
    # Summary
    valid_count = sum(1 for r in results if r["valid"])
    total_count = len(results)
    output.append(f"\nSummary: {valid_count}/{total_count} includes are valid\n")
    
    # Group results by validity
    valid_includes = [r for r in results if r["valid"]]
    invalid_includes = [r for r in results if not r["valid"]]
    
    if valid_includes:
        output.append("✓ VALID INCLUDES:")
        for result in valid_includes:
            output.append(f"\n  Original: {result['include']}")
            if result['corrected_path'] != result['include']:
                output.append(f"  → Use: {result['corrected_path']}")
            if result.get('recommendation'):
                output.append(f"  Note: {result['recommendation']}")
    
    if invalid_includes:
        output.append("\n\n✗ INVALID INCLUDES:")
        for result in invalid_includes:
            output.append(f"\n  Original: {result['include']}")
            output.append(f"  Error: {result['error']}")
            if result.get('recommendation'):
                output.append(f"  Suggestion: {result['recommendation']}")
    
    # Provide corrected include list
    output.append("\n\nCORRECTED INCLUDE LIST:")
    output.append("Use these includes in your file:\n")
    
    for result in results:
        if result["valid"]:
            include = result["corrected_path"]
            # Determine if it should use angle brackets
            if any(include.startswith(p) or include in ["Python.h", "pybind11/pybind11.h", "pybind11/stl.h"] 
                   for p in ["pybind11/", "std"]):
                output.append(f'#include <{include}>')
            else:
                output.append(f'#include "{include}"')
        else:
            output.append(f'// INVALID: #include "{result["include"]}" - {result["error"]}')
    output = "\n".join(output)
    print(f"[INCLUDE TOOL DEBUG] final tool output: {output}")
    return output


