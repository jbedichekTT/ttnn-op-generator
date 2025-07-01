#!/usr/bin/env python3
"""
Generate context and prompts for TTNN operation generation.

This script uses the MultiStagePromptGenerator to analyze and plan
the generation of a TTNN operation, exporting the prompts that would
be used for each file to the context/ folder.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.ttnn_agent import TTNNOperationAgent
from agents.multi_stage_prompt_generator import MultiStagePromptGenerator
from prompts import (
    HPP_CONTEXT, CPP_CONTEXT, DEVICE_OP_CONTEXT,
    PROGRAM_FACTORY_CONTEXT, KERNEL_CONTEXT, 
    PYBIND_CONTEXT, CMAKE_CONTEXT
)


class ContextGenerator:
    """Generate context files for TTNN operation generation."""
    
    def __init__(self, operation_type: str, tt_metal_path: str, 
                 custom_suffix: str = "custom", output_dir: str = "context",
                 database_path: str = "include_api_database.json"):
        """Initialize the context generator."""
        self.operation_type = operation_type
        self.operation_name = f"eltwise_{operation_type}_{custom_suffix}"
        self.output_dir = Path(output_dir) / self.operation_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        
        # Create agent
        self.agent = TTNNOperationAgent(
            operation_type=operation_type,
            tt_metal_path=tt_metal_path,
            custom_suffix=custom_suffix
        )
        
        # Check if database exists
        if not Path(database_path).exists():
            print(f"Warning: API database not found at {database_path}")
            print("Context generation will proceed without API validation.")
            print("Run build_api_database.py first for better results.")
        
        # Set up multi-stage generator with database path
        self.generator = MultiStagePromptGenerator(self.agent, database_path=database_path)
        self.generator.context_dir = self.output_dir  # Override context directory
        self.generator.operation_name = self.operation_name  # Set operation name
        
        # File generation order
        self.file_order = [
            ("hpp", [], "header file"),
            ("cpp", ["hpp"], "implementation file"),
            ("op-hpp", ["hpp"], "device operation header"),
            ("op", ["op-hpp", "hpp"], "device operation implementation"),
            ("program-factory-hpp", ["hpp"], "program factory header"),
            ("program-factory", ["program-factory-hpp", "op-hpp"], "program factory implementation"),
            ("reader", ["program-factory-hpp", "op-hpp"], "reader kernel"),
            ("writer", ["program-factory-hpp", "op-hpp"], "writer kernel"),
            ("compute", ["program-factory-hpp", "op-hpp"], "compute kernel"),
            ("pybind-hpp", ["hpp"], "Python binding header"),
            ("pybind-cpp", ["pybind-hpp", "hpp"], "Python binding implementation"),
            ("cmake", [], "CMake configuration"),
        ]
        
        # Context mapping
        self.context_map = {
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
        
    def generate_all_contexts(self):
        """Generate context files for all components of the operation."""
        print(f"\n{'='*80}")
        print(f"Generating Context for Operation: {self.operation_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"API Database: {self.database_path}")
        print(f"{'='*80}\n")
        
        # Write operation summary
        self._write_operation_summary()
        
        # Generate context for each file
        for file_key, dependencies, description in self.file_order:
            print(f"\nGenerating context for {description}: {self.agent.files[file_key]['name']}")
            
            try:
                self._generate_file_context(file_key, dependencies)
            except Exception as e:
                print(f"  ✗ Error generating context: {str(e)}")
                # Write error details to a file
                self._write_error_file(file_key, e)
                continue
                
        print(f"\n{'='*80}")
        print(f"Context generation complete!")
        print(f"Files written to: {self.output_dir}")
        print(f"{'='*80}\n")
        
    def _generate_file_context(self, file_key: str, dependencies: List[str]):
        """Generate context for a single file."""
        # Get base context
        context = self.context_map.get(file_key, "")
        base_prompt = (
            f"Generate the code for the file `{self.agent.files[file_key]['name']}` "
            f"for the `{self.operation_name}` operation.\n\n{context}"
        )
        
        # Initialize plan to None
        plan = None
        
        try:
            # Run planning stage
            print("  - Running planning stage...")
            plan = self.generator._planning_stage(file_key, base_prompt, dependencies)
            
            if plan is None:
                raise ValueError("Planning stage returned None")
            
            # Run validation stage
            print("  - Running validation stage...")
            validation_success = self.generator._validation_stage(plan)
            
            # If validation failed, run refinement
            if not validation_success:
                print("  - Running refinement stage...")
            #    plan = self.generator._refinement_stage(plan, base_prompt)
                # Re-validate after refinement
            #    validation_success = self.generator._validation_stage(plan)
            
            # Build final prompt
            print("  - Building final prompt...")
            final_prompt = self.generator._build_execution_prompt(plan, base_prompt)
            
            # Write context file
            self._write_context_file(file_key, plan, final_prompt)
            
        except Exception as e:
            print(f"  ✗ Error in generation pipeline: {str(e)}")
            # Try to write what we have
            if plan:
                final_prompt = f"ERROR: {str(e)}\n\n{base_prompt}"
                self._write_context_file(file_key, plan, final_prompt)
            raise
        
    def _write_context_file(self, file_key: str, plan, final_prompt: str):
        """Write a context file with plan and prompt information."""
        filename = f"{file_key}_{self.agent.files[file_key]['name'].replace('/', '_')}.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            # Handle case where plan might be incomplete
            if hasattr(plan, 'required_apis'):
                # Required APIs
                f.write(f"Required APIs ({len(plan.required_apis)}):\n")
                for api in plan.required_apis:
                    f.write(f"  - {api.namespace}::{api.name}\n")
                    if hasattr(api, 'validated') and api.validated:
                        f.write(f"    ✓ Validated in: {getattr(api, 'include_path', 'unknown')}\n")
                        if hasattr(api, 'signature') and api.signature:
                            f.write(f"    Signature: {api.signature}\n")
                    else:
                        f.write(f"    ✗ NOT FOUND\n")
                        if hasattr(api, 'validation_error') and api.validation_error:
                            f.write(f"    Error: {api.validation_error}\n")
                    f.write("\n")
            else:
                f.write("No APIs identified (planning may have failed)\n\n")
            
            # Required Includes
            if hasattr(plan, 'required_includes'):
                f.write(f"\nRequired Includes ({len(plan.required_includes)}):\n")
                for include in plan.required_includes:
                    if isinstance(include, dict):
                        f.write(f"  - {include.get('path', include)}\n")
                        if 'reason' in include:
                            f.write(f"    Reason: {include['reason']}\n")
                        if 'expected_apis' in include:
                            f.write(f"    Expected: {', '.join(include['expected_apis'])}\n")
                    else:
                        f.write(f"  - {include}\n")
            
            # Namespaces
            if hasattr(plan, 'namespace_imports'):
                f.write(f"\nNamespace Imports:\n")
                for ns in plan.namespace_imports:
                    f.write(f"  - using namespace {ns};\n")
                    
            # Key Patterns (if available)
            if hasattr(plan, 'key_patterns') and plan.key_patterns:
                f.write(f"\nKey Patterns to Follow:\n")
                for pattern in plan.key_patterns:
                    f.write(f"  - {pattern}\n")
                    
            # Common Mistakes (if available)
            if hasattr(plan, 'common_mistakes') and plan.common_mistakes:
                f.write(f"\nCommon Mistakes to Avoid:\n")
                for mistake in plan.common_mistakes:
                    f.write(f"  - {mistake}\n")
            
            # Final Prompt
            f.write(f"\n\n{'='*80}\n")
            f.write("FINAL GENERATION PROMPT\n")
            f.write("=" * 80 + "\n\n")
            f.write(final_prompt)
            
        print(f"  ✓ Written to: {filename}")
        
    def _write_error_file(self, file_key: str, error: Exception):
        """Write an error file with details about what went wrong."""
        filename = f"{file_key}_ERROR.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"ERROR GENERATING CONTEXT FOR: {self.agent.files[file_key]['name']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Error Type: {type(error).__name__}\n")
            f.write(f"Error Message: {str(error)}\n\n")
            
            # Write traceback
            import traceback
            f.write("Traceback:\n")
            f.write("-" * 40 + "\n")
            traceback.print_exc(file=f)
            
        print(f"  ✓ Error details written to: {filename}")
        
    def _write_operation_summary(self):
        """Write an operation summary file."""
        summary_file = self.output_dir / "00_operation_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"TTNN Operation Generation Context\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Operation Type: {self.operation_type}\n")
            f.write(f"Operation Name: {self.operation_name}\n")
            f.write(f"Class Name: {self.agent.operation_class_name}\n")
            f.write(f"Python Function: {self.agent.python_function_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"API Database: {self.database_path}\n\n")
            
            f.write("Files to Generate:\n")
            f.write("-" * 40 + "\n")
            for file_key, deps, desc in self.file_order:
                f.write(f"{file_key:20} -> {self.agent.files[file_key]['name']}\n")
                if deps:
                    f.write(f"{'':20}    Dependencies: {', '.join(deps)}\n")
                    
            f.write("\n\nFile Generation Order:\n")
            f.write("-" * 40 + "\n")
            for i, (file_key, _, desc) in enumerate(self.file_order, 1):
                f.write(f"{i:2}. {file_key:20} ({desc})\n")
                
            f.write("\n\nKey Requirements:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Use modern TTNN APIs (ttnn namespace)\n")
            f.write("2. Use ttnn::decorators::register_operation\n")
            f.write("3. Inherit from DeviceOperation<YourOp>\n")
            f.write("4. Use operation::ProgramWithCallbacks\n")
            f.write("5. Include proper headers for all types\n")
            f.write("6. Follow const& conventions for inputs\n")
            f.write("7. Support optional output memory config\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate context and prompts for TTNN operation generation"
    )
    
    parser.add_argument(
        "--operation",
        default="multiply",
        help="Operation type (e.g., multiply, add, subtract)"
    )
    
    parser.add_argument(
        "--suffix",
        default="custom",
        help="Custom suffix for the operation name"
    )
    
    parser.add_argument(
        "--tt-metal-path",
        default="/home/user/tt-metal",
        help="Path to TT-Metal repository"
    )
    
    parser.add_argument(
        "--output-dir",
        default="context",
        help="Output directory for context files"
    )
    
    parser.add_argument(
        "--database-path",
        default="include_api_database.json",
        help="Path to API database JSON file"
    )
    
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Run without agent calls (for testing)"
    )

    parser.add_argument(
        "--file",
        help="Generate context for a single file (e.g., hpp, cpp, etc.)"
    )

    parser.add_argument(
        "--dependencies",
        nargs="+",
        help="List of dependencies for the file (e.g., hpp, hpp, cpp, etc.)"
    )    
    
    args = parser.parse_args()
    
    # Set API key if provided
    #if args.api_key:
    #    os.environ["ANTHROPIC_API_KEY"] = args.api_key
    
    # Check API key (unless running without agent)
    #if not args.no_agent and not os.environ.get("ANTHROPIC_API_KEY"):
    #    print("Error: ANTHROPIC_API_KEY environment variable not set!")
    #    print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    #    print("Or pass it with: --api-key 'your-key-here'")
    #    print("Or run with --no-agent for testing without API calls")
    #    sys.exit(1)
    
    # Create and run generator
    generator = ContextGenerator(
        operation_type=args.operation,
        tt_metal_path=args.tt_metal_path,
        custom_suffix=args.suffix,
        output_dir=args.output_dir,
        database_path=args.database_path
    )
    
    try:
        if args.file:
            generator._generate_file_context(args.file, None)
        else:
            generator.generate_all_contexts()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()