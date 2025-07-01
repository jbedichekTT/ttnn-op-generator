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
    with the same include paths that the TT-Metal build system uses.
    
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
                          "cstdint", "cmath", "vector", "memory", "string", "optional",
                          "functional", "variant", "fmt/format.h"]
        
        # Check if it's a system include
        is_system = False
        for sys_inc in system_includes:
            if include_path == sys_inc or include_path.startswith(sys_inc.split('/')[0] + '/'):
                is_system = True
                break
                
        if is_system:
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
        
        # Write test file in /tmp to avoid path issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(test_content)
            test_file = f.name
        
        try:
            # Build the include paths that match TT-Metal's CMake configuration
            # Based on the CMake files, these are the include directories:
            include_dirs = [
                str(tt_metal_path),                              # Root
                str(tt_metal_path / "ttnn" / "cpp"),            # For ttnn/ includes
                str(tt_metal_path / "tt_metal" / "include"),     # TT-Metal includes
                str(tt_metal_path / "tt_metal"),                 # TT-Metal root
                str(tt_metal_path / "tt_eager"),                 # TT eager includes
                str(tt_metal_path / "tt_metal" / "third_party" / "umd"),  # UMD
            ]
            
            # Build compile command with all include directories
            cmd = ["g++", "-std=c++17", "-c", "-fsyntax-only"]
            for inc_dir in include_dirs:
                cmd.extend(["-I", inc_dir])
            cmd.append(test_file)
            
            # Try to compile
            result_1 = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result_1.returncode == 0:
                result["valid"] = True
                result["corrected_path"] = include_path
                print(f'[Include Validation] ✓ {include_path} is valid')
                results.append(result)
                continue
            
            # If compilation failed, try to find the correct path
            print(f'[Include Validation] ✗ {include_path} failed to compile')
            
            # Extract just the filename
            filename = Path(include_path).name
            
            # Search for the file in the repository
            print(f'[Include Validation] Searching for {filename} in tt-metal')
            found_files = find_files_in_repository(filename, tt_metal_path)
            
            if found_files:
                # Try each found file to see which one compiles
                for found_path in found_files[:5]:  # Limit to first 5 matches
                    try:
                        # Get the include path relative to one of our include directories
                        found_path_obj = Path(found_path)
                        
                        # Try different relative paths
                        possible_includes = []
                        
                        # If it's under ttnn/cpp/ttnn, the include should be "ttnn/..."
                        if "ttnn/cpp/ttnn" in str(found_path):
                            rel_to_cpp = found_path_obj.relative_to(tt_metal_path / "ttnn" / "cpp")
                            possible_includes.append(str(rel_to_cpp))
                        
                        # If it's under tt_metal, try relative to tt_metal
                        if "tt_metal" in str(found_path):
                            try:
                                rel_to_metal = found_path_obj.relative_to(tt_metal_path / "tt_metal")
                                possible_includes.append(f"tt_metal/{rel_to_metal}")
                            except:
                                pass
                        
                        # Try relative to root
                        try:
                            rel_to_root = found_path_obj.relative_to(tt_metal_path)
                            possible_includes.append(str(rel_to_root))
                        except:
                            pass
                        
                        # Test each possible include
                        for test_include in possible_includes:
                            test_content_found = f"""#include "{test_include}"
int main() {{ return 0; }}
"""
                            with open(test_file, 'w') as f:
                                f.write(test_content_found)
                            
                            result_test = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                            if result_test.returncode == 0:
                                print(f'[Include Validation] ✓ Found working include: {test_include}')
                                result["valid"] = True
                                result["corrected_path"] = test_include
                                result["recommendation"] = f"File found at: {found_path}"
                                break
                        
                        if result["valid"]:
                            break
                            
                    except Exception as e:
                        print(f'[Include Validation] Error testing {found_path}: {e}')
                        continue
            
            # If still not found, check common corrections
            corrections = {
                "ttnn/decorators.hpp": "ttnn/decorators.hpp",  # This should exist
                "ttnn/run_operation.hpp": "ttnn/run_operation.hpp",  # This should exist
                "ttnn/tensor/tensor.hpp": "ttnn/tensor/tensor.hpp",  # This should exist
                "tt_metal/operation.hpp": "ttnn/decorators.hpp",  # Redirect to new location
                "ttnn/operation.hpp": "ttnn/decorators.hpp",  # Redirect to new location
                "tt_metal/common/assert.hpp": "tt_metal/common/assert.hpp",
            }
            
            if include_path in corrections and not result["valid"]:
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
            if not result["valid"]:
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
                    files = result.stdout.strip().split('\n')
                    found_files.extend([f for f in files if f])  # Filter empty strings
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
                   for p in ["pybind11/", "std", "fmt/"]) or "/" not in include:
                output.append(f'#include <{include}>')
            else:
                output.append(f'#include "{include}"')
        else:
            output.append(f'// INVALID: #include "{result["include"]}" - {result["error"]}')
    
    output = "\n".join(output)
    print(f"[Include Validation] Complete")
    return output