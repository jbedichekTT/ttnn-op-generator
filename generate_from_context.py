#!/usr/bin/env python3
"""
Generate code files from context files created by generate_context.py

This script reads the context files and executes the generation stage
to produce the actual code files.

Enhanced features:
- Support for @FILE references to include previously generated files
- Configurable generation sequence to handle dependencies
"""

import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.ttnn_agent import TTNNOperationAgent


class CodeFromContextGenerator:
    """Generate code files from pre-generated context files."""
    
    def __init__(self, operation_name: str, context_dir: str = "context", 
                 output_dir: str = "output", tt_metal_path: str = "/home/user/tt-metal"):
        """Initialize the generator."""
        self.operation_name = operation_name
        self.context_dir = Path(context_dir) / operation_name
        self.output_dir = Path(output_dir) / operation_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract operation type from operation_name (e.g., "eltwise_multiply_custom" -> "multiply")
        parts = operation_name.split('_')
        if len(parts) >= 3 and parts[0] == "eltwise":
            self.operation_type = parts[1]
            self.custom_suffix = parts[2] if len(parts) > 2 else "custom"
        else:
            # Fallback
            self.operation_type = "multiply"
            self.custom_suffix = "custom"
        
        # Create agent
        self.agent = TTNNOperationAgent(
            operation_type=self.operation_type,
            tt_metal_path=tt_metal_path,
            custom_suffix=self.custom_suffix
        )
        
        # Default generation sequence (can be overridden)
        self.generation_sequence = [
            "hpp",                    # Main header first
            "cpp",                    # Implementation
            "op-hpp",                 # Device operation header
            "op",                     # Device operation implementation
            "program-factory-hpp",    # Program factory header
            "program-factory",        # Program factory implementation
            "reader",                 # Kernel files
            "writer",
            "compute",
            "pybind-hpp",            # Python bindings
            "pybind-cpp",
            "cmake"                  # Build configuration
        ]
        
        # Track generated files for @FILE references
        self.generated_files = {}

        self.file_structure = {
            "hpp": f"{self.operation_name}.hpp",
            "cpp": f"{self.operation_name}.cpp",
            "op-hpp": f"device/{self.operation_name}_op.hpp",
            "op": f"device/{self.operation_name}_op.cpp",
            "program-factory-hpp": f"device/{self.operation_name}_program_factory.hpp",
            "program-factory": f"device/{self.operation_name}_program_factory.cpp",
            "reader": f"device/kernels/dataflow/{self.operation_name}_reader.cpp",
            "writer": f"device/kernels/dataflow/{self.operation_name}_writer.cpp",
            "compute": f"device/kernels/compute/{self.operation_name}_compute.cpp",
            "pybind-hpp": f"{self.operation_name}_pybind.hpp",
            "pybind-cpp": f"{self.operation_name}_pybind.cpp",
            "cmake": "CMakeLists.txt",
    }
        
    def set_generation_sequence(self, sequence: List[str]):
        """Set a custom generation sequence."""
        self.generation_sequence = sequence
        
    def list_context_files(self) -> Dict[str, Path]:
        """List all context files organized by file key."""
        if not self.context_dir.exists():
            raise FileNotFoundError(f"Context directory not found: {self.context_dir}")
            
        context_files = {}
        
        # Map context filename patterns to file keys
        pattern_to_key = {
            f"{self.operation_name}.cpp": "cpp",
            f"{self.operation_name}.hpp": "hpp",
            f"{self.operation_name}_op.cpp": "op",
            f"{self.operation_name}_op.hpp": "op-hpp",
            f"{self.operation_name}_program_factory.cpp": "program-factory",
            f"{self.operation_name}_program_factory.hpp": "program-factory-hpp",
            f"{self.operation_name}_reader.cpp": "reader",
            f"{self.operation_name}_writer.cpp": "writer",
            f"{self.operation_name}_compute.cpp": "compute",
            f"{self.operation_name}_pybind.cpp": "pybind-cpp",
            f"{self.operation_name}_pybind.hpp": "pybind-hpp",
            "CMakeLists.txt": "cmake",
        }
        
        for file in self.context_dir.glob("*.txt"):
            if any(skip in file.name for skip in ["_ERROR.txt", "00_operation_summary.txt"]):
                continue
                
            # Extract the base filename from context file
            # Context files are named like: "cpp_eltwise_multiply_custom.cpp.txt"
            # or "eltwise_multiply_custom.cpp.txt"
            
            filename = file.stem  # Remove .txt
            
            # Try direct pattern matching first
            for pattern, key in pattern_to_key.items():
                if pattern in filename:
                    context_files[key] = file
                    break
                    
        return context_files
        
    def extract_prompt_from_context(self, context_file: Path) -> Optional[str]:
        """Extract the final generation prompt from a context file."""
        with open(context_file, 'r') as f:
            content = f.read()
            
        # Look for the FINAL GENERATION PROMPT section
        prompt_marker = "FINAL GENERATION PROMPT"
        marker_pos = content.find(prompt_marker)
        
        if marker_pos == -1:
            print(f"Warning: No final prompt found in {context_file.name}")
            return None
            
        # Find the start of the actual prompt (after the separator line)
        prompt_start = content.find("\n\n", marker_pos)
        if prompt_start == -1:
            return None
            
        # Extract everything after the marker section
        prompt = content[prompt_start:].strip()
        
        # Process @FILE references
        prompt = self._process_file_references(prompt)
        
        return prompt
        
    def _process_file_references(self, prompt: str) -> str:
        """Process @FILE references in the prompt, replacing them with file contents."""
        file_pattern = r'@FILE\s+([^\s\n]+)'
        
        def replace_file_reference(match):
            file_ref = match.group(1)
            print(f"[File Reference] Found reference: {file_ref}")
            # Try to match against file keys first
            if file_ref in self.file_structure:
                file_path = self.output_dir / self.file_structure[file_ref]
            else:
                # Try as a direct path
                file_path = Path(file_ref)
                if not file_path.is_absolute():
                    file_path = self.output_dir / file_path
            
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    print(f"[File Reference] Loaded {file_path} ({len(content)} bytes)")
                    
                    return f"""// Contents of {file_path.name}:
                    ```cpp
                    {content}
                    ```"""
                except Exception as e:
                    print(f"[File Reference] Error reading {file_path}: {e}")
                    return f"// ERROR: Could not read {file_ref}: {e}"
            
            print(f"[File Reference] Warning: File not found: {file_ref}")
            return f"// ERROR: File not found: {file_ref}"
        
        # Replace all @FILE references
        processed_prompt = re.sub(file_pattern, replace_file_reference, prompt)
        #print(f"[File Reference] Processed prompt: {processed_prompt}")
        return processed_prompt
        
    def extract_file_info_from_context(self, context_file: Path) -> Dict[str, str]:
        """Extract file information from context file."""
        info = {
            "file_key": None,
            "file_name": None,
            "context_file": context_file.name
        }
        
        stem = context_file.stem  # Remove .txt
        
        # Use the same pattern matching as list_context_files
        pattern_to_key = {
            f"{self.operation_name}.cpp": "cpp",
            f"{self.operation_name}.hpp": "hpp",
            f"{self.operation_name}_op.cpp": "op",
            f"{self.operation_name}_op.hpp": "op-hpp",
            f"{self.operation_name}_program_factory.cpp": "program-factory",
            f"{self.operation_name}_program_factory.hpp": "program-factory-hpp",
            f"{self.operation_name}_reader.cpp": "reader",
            f"{self.operation_name}_writer.cpp": "writer",
            f"{self.operation_name}_compute.cpp": "compute",
            f"{self.operation_name}_pybind.cpp": "pybind-cpp",
            f"{self.operation_name}_pybind.hpp": "pybind-hpp",
            "CMakeLists.txt": "cmake",
        }
        
        # Match using the same logic as list_context_files
        for pattern, key in pattern_to_key.items():
            if pattern in stem:
                info["file_key"] = key
                break
        
        # Set the correct filename from our file structure mapping
        if info["file_key"] and info["file_key"] in self.file_structure:
            info["file_name"] = self.file_structure[info["file_key"]]
        else:
            # Fallback
            info["file_name"] = stem
        
        return info
        
    def generate_code_from_prompt(self, prompt: str) -> str:
        """Generate code using the agent with the given prompt."""
        print("[Code Generation] Calling agent...")
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.agent.get_generation_with_tools(messages)
            
            if response is None:
                print("[Code Generation] Warning: Agent returned None")
                return "// ERROR: Agent returned no response\n"
                
            # Extract code from response
            code = self._extract_code_from_response(response)
            
            return code
            
        except Exception as e:
            print(f"[Code Generation] Error: {str(e)}")
            return f"// ERROR: {str(e)}\n"
            
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from agent response."""
        # Try to extract from code blocks
        code_patterns = [
            r'```(?:cpp|c\+\+|cuda|python|cmake)\n([\s\S]*?)\n```',
            r'```\n([\s\S]*?)\n```'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
                
        # If no code blocks, return the whole response
        return response
        
    def determine_output_path(self, file_info: Dict[str, str]) -> Path:
        """Determine the output path for a generated file."""
        file_key = file_info.get("file_key")
        
        # Use the file structure mapping
        if file_key and file_key in self.file_structure:
            file_path = self.file_structure[file_key]
            full_path = self.output_dir / file_path
            
            # Ensure parent directories exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            return full_path
        else:
            # Fallback to original behavior
            file_name = file_info.get("file_name", "unknown.txt")
            return self.output_dir / file_name
            
    def generate_single_file(self, context_file: Path) -> bool:
        """Generate a single code file from its context."""
        print(f"\n{'='*60}")
        print(f"Processing: {context_file.name}")
        print(f"{'='*60}")
        
        # Extract file info
        file_info = self.extract_file_info_from_context(context_file)
        print(f"File: {file_info['file_name']}")
        print(f"Key: {file_info['file_key']}")
        
        # Extract prompt
        prompt = self.extract_prompt_from_context(context_file)
        if not prompt:
            print("✗ Failed to extract prompt from context file")
            return False
            
        print(f"Prompt length: {len(prompt)} characters")
        
        # Generate code
        code = self.generate_code_from_prompt(prompt)
        
        if code.startswith("// ERROR:"):
            print("✗ Code generation failed")
            return False
            

        # Determine output path
        output_path = self.determine_output_path(file_info)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write code to file
        with open(output_path, 'w') as f:
            f.write(code)
            
        print(f"✓ Generated: {output_path}")
        print(f"  Size: {len(code)} bytes")
        
        # Track generated file for @FILE references
        self.generated_files[file_info['file_name']] = str(output_path)
        self.generated_files[file_info['file_key']] = str(output_path)
        self.generated_files[output_path.name] = str(output_path)
        
        return True
        
    def generate_all_files(self):
        """Generate all code files from context files in the specified sequence."""
        print(f"\nGenerating code for operation: {self.operation_name}")
        print(f"Context directory: {self.context_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # List context files
        context_files = self.list_context_files()
        print(f"\nFound {len(context_files)} context files")
        
        # Generate files in sequence
        print(f"\nGeneration sequence: {' -> '.join(self.generation_sequence)}")
        
        success_count = 0
        generated_in_sequence = []
        
        # First, generate files in the specified sequence
        for file_key in self.generation_sequence:
            if file_key in context_files:
                print(f"\n[Sequence {len(generated_in_sequence)+1}/{len(self.generation_sequence)}] Generating {file_key}")
                if self.generate_single_file(context_files[file_key]):
                    success_count += 1
                generated_in_sequence.append(file_key)
        
        # Then, generate any remaining files not in the sequence
        remaining_keys = set(context_files.keys()) - set(generated_in_sequence)
        if remaining_keys:
            print(f"\nGenerating remaining files not in sequence: {remaining_keys}")
            for file_key in sorted(remaining_keys):
                if self.generate_single_file(context_files[file_key]):
                    success_count += 1
                
        print(f"\n{'='*60}")
        print(f"Generation complete: {success_count}/{len(context_files)} files generated")
        print(f"Output written to: {self.output_dir}")
        print(f"{'='*60}")
        
        # Write generation summary
        self._write_generation_summary(list(context_files.values()), success_count)
        
    def _write_generation_summary(self, context_files: List[Path], success_count: int):
        """Write a summary of the generation process."""
        summary_file = self.output_dir / "generation_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Code Generation Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Operation: {self.operation_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Success: {success_count}/{len(context_files)} files\n\n")
            
            f.write(f"Generation Sequence:\n")
            f.write(f"{'-'*40}\n")
            for i, key in enumerate(self.generation_sequence, 1):
                f.write(f"{i:2}. {key}\n")
            f.write(f"\n")
            
            f.write(f"Files Generated:\n")
            f.write(f"{'-'*40}\n")
            
            for context_file in context_files:
                file_info = self.extract_file_info_from_context(context_file)
                output_path = self.determine_output_path(file_info)
                
                if output_path.exists():
                    size = output_path.stat().st_size
                    f.write(f"✓ {file_info['file_name']:40} {size:8} bytes\n")
                else:
                    f.write(f"✗ {file_info['file_name']:40} FAILED\n")

    def complete_partial_generation(self):
        """Complete a partially generated operation by detecting existing files and resuming."""
        print(f"\nCompleting partial generation for operation: {self.operation_name}")
        print(f"Context directory: {self.context_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Check which files already exist
        existing_files = self._detect_existing_files()
        
        # List available context files
        context_files = self.list_context_files()
        
        # Determine which files need to be generated
        files_to_generate = []
        files_already_exist = []
        
        for file_key in self.generation_sequence:
            if file_key not in context_files:
                continue
                
            if file_key in existing_files:
                files_already_exist.append(file_key)
            else:
                files_to_generate.append(file_key)
        
        # Also check for files not in the sequence
        remaining_keys = set(context_files.keys()) - set(self.generation_sequence)
        for file_key in sorted(remaining_keys):
            if file_key not in existing_files:
                files_to_generate.append(file_key)
            else:
                files_already_exist.append(file_key)
        
        # Report status
        print(f"\nGeneration Status:")
        print(f"  Total context files: {len(context_files)}")
        print(f"  Already generated: {len(files_already_exist)}")
        print(f"  Need to generate: {len(files_to_generate)}")
        
        if files_already_exist:
            print(f"\nExisting files (will skip):")
            for key in files_already_exist:
                if key in self.file_structure:
                    print(f"  ✓ {key}: {self.file_structure[key]}")
        
        if not files_to_generate:
            print("\nAll files already generated! Nothing to do.")
            return
        
        print(f"\nFiles to generate:")
        for key in files_to_generate:
            if key in self.file_structure:
                print(f"  ○ {key}: {self.file_structure[key]}")
        
        # Populate generated_files with existing files for @FILE references
        for file_key in files_already_exist:
            if file_key in self.file_structure:
                file_path = self.output_dir / self.file_structure[file_key]
                if file_path.exists():
                    self.generated_files[self.file_structure[file_key]] = str(file_path)
                    self.generated_files[file_key] = str(file_path)
                    self.generated_files[file_path.name] = str(file_path)
        
        # Generate missing files
        print(f"\nResuming generation...")
        success_count = 0
        
        for i, file_key in enumerate(files_to_generate, 1):
            if file_key in context_files:
                print(f"\n[{i}/{len(files_to_generate)}] Generating {file_key}")
                if self.generate_single_file(context_files[file_key]):
                    success_count += 1
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"Partial generation complete!")
        print(f"  Previously generated: {len(files_already_exist)}")
        print(f"  Newly generated: {success_count}/{len(files_to_generate)}")
        print(f"  Total: {len(files_already_exist) + success_count}/{len(context_files)}")
        print(f"{'='*60}")
        
        # Write updated summary
        self._write_generation_summary(list(context_files.values()), 
                                    len(files_already_exist) + success_count)

    def _detect_existing_files(self) -> Dict[str, Path]:
        """Detect which files have already been generated."""
        existing_files = {}
        
        for file_key, file_path_str in self.file_structure.items():
            full_path = self.output_dir / file_path_str
            if full_path.exists():
                # Check if file has content (not empty)
                if full_path.stat().st_size > 0:
                    existing_files[file_key] = full_path
        
        return existing_files

    def regenerate_file(self, file_key: str, force: bool = False) -> bool:
        """Regenerate a specific file, optionally forcing regeneration even if it exists."""
        context_files = self.list_context_files()
        
        if file_key not in context_files:
            print(f"Error: No context file found for '{file_key}'")
            return False
        
        if file_key in self.file_structure:
            file_path = self.output_dir / self.file_structure[file_key]
            if file_path.exists() and not force:
                response = input(f"File {file_path} already exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Skipping regeneration.")
                    return False
        
        print(f"\nRegenerating {file_key}...")
        return self.generate_single_file(context_files[file_key])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate code files from context files"
    )
    
    parser.add_argument(
        "--operation_name",
        help="Operation name (e.g., eltwise_multiply_custom)",
        default='eltwise_multiply_custom'
    )
    
    parser.add_argument(
        "--context-dir",
        default="context",
        help="Base directory containing context files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for generated code"
    )
    
    parser.add_argument(
        "--file",
        help="Generate only a specific file (e.g., hpp_eltwise_multiply_custom.hpp.txt)"
    )
    
    parser.add_argument(
        "--tt-metal-path",
        default="/home/user/tt-metal",
        help="Path to TT-Metal repository"
    )
    
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    
    parser.add_argument(
        "--sequence",
        nargs="+",
        help="Custom generation sequence (e.g., --sequence hpp cpp op-hpp op)"
    )
    
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Continue partial generation (e.g., after an interrupted run)"
    )
    args = parser.parse_args()
    
    # Create generator
    generator = CodeFromContextGenerator(
        operation_name=args.operation_name,
        context_dir=args.context_dir,
        output_dir=args.output_dir,
        tt_metal_path=args.tt_metal_path
    )
    
    # Set custom sequence if provided
    if args.sequence:
        generator.set_generation_sequence(args.sequence)
    
    if args.partial:
        generator.complete_partial_generation()
        return 0

    try:
        if args.file:
            # Generate single file
            context_file = generator.context_dir / args.file
            if not context_file.exists():
                print(f"Error: Context file not found: {context_file}")
                sys.exit(1)
                
            success = generator.generate_single_file(context_file)
            sys.exit(0 if success else 1)
        else:
            # Generate all files
            generator.generate_all_files()
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()