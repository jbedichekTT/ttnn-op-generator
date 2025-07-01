#!/usr/bin/env python3
"""
Programmatic Include Debugger
=============================

Analyzes C++ files using tree-sitter to identify all API usage and automatically
determines the correct includes using the API database.
"""

import os
import re
import json
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any

# Import the database query tools
from ttnn_op_generator.tools.database_query_tool import (
    find_header_for_api_tool, 
    search_apis_tool,
    get_apis_from_header_tool
)

# Import the API extractor (tree-sitter based)
from ttnn_op_generator.tools.api_extractor import extract_apis_from_header


class IncludeDebugger:
    """
    Analyzes C++ files to identify missing/incorrect includes and generates
    corrections using the API database and tree-sitter parsing.
    """
    
    def __init__(self, api_key: str = "",
                 database_path: str = "include_api_database.json",
                 tt_metal_path: str = None):
        self.api_key = api_key
        self.database_path = database_path
        self.tt_metal_path = tt_metal_path or os.environ.get("TT_METAL_PATH", "/home/user/tt-metal")
        
    def debug_file_includes(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point: Debug includes for a C++ file using tree-sitter parsing.
        
        Args:
            file_path: Path to the C++ file
            content: Optional file content (if not provided, will read from file)
            
        Returns:
            Dictionary containing analysis results
        """
        # Read file content if not provided
        if content is None:
            with open(file_path, 'r') as f:
                content = f.read()
        
        print(f"\n[Include Debugger] Analyzing {file_path}")
        
        # Extract current includes
        current_includes = self._extract_current_includes(content)
        print(f"[Include Debugger] Found {len(current_includes)} current includes")
        
        # Use tree-sitter to extract API usage from the file
        print("[Include Debugger] Extracting API usage using tree-sitter...")
        
        # For .cpp files, we need to extract API usage, not declarations
        file_path_obj = Path(file_path)
        
        if file_path_obj.suffix in ['.cpp', '.cc', '.cxx']:
            # Source file - extract API usage
            apis_in_file = self._extract_api_usage_from_file(str(file_path_obj.absolute()))
        else:
            # Header file - use the existing extractor
            if file_path_obj.exists():
                # For files outside TT-Metal, we need a different approach
                # Just use the usage extractor for now
                apis_in_file = self._extract_api_usage_from_file(str(file_path_obj.absolute()))
            else:
                # Try as relative to TT-Metal
                apis_in_file = extract_apis_from_header(file_path, self.tt_metal_path)
        
        if 'error' in apis_in_file:
            print(f"[Include Debugger] Error extracting APIs: {apis_in_file['error']}")
            return {
                'error': apis_in_file['error'],
                'current_includes': list(current_includes)
            }
        
        # Convert the extracted APIs to a format for finding headers
        used_apis = self._convert_extracted_apis(apis_in_file)
        total_apis = sum(len(apis) for apis in used_apis.values())
        print(f"[Include Debugger] Found {total_apis} API usages")
        
        # Find required headers for each API using the database
        required_headers = self.find_required_headers_from_apis(apis_in_file)
        
        # Determine what needs to be fixed
        all_required_headers = set()
        for header_info in required_headers.values():
            if isinstance(header_info, str):
                all_required_headers.add(header_info)
            elif isinstance(header_info, dict) and 'defined_in' in header_info:
                all_required_headers.add(header_info['defined_in'])
            elif isinstance(header_info, list):
                for item in header_info:
                    if isinstance(item, dict) and 'header' in item:
                        all_required_headers.add(item['header'])
        
        # Analyze what's missing
        missing_includes = all_required_headers - current_includes
        
        # Find unnecessary includes by checking what each include provides
        unnecessary_includes = self._find_unnecessary_includes(
            current_includes, apis_in_file, all_required_headers
        )
        
        # Generate the fix prompt
        fix_prompt = self._generate_fix_prompt(
            file_path, content, current_includes, required_headers,
            missing_includes, unnecessary_includes, apis_in_file
        )
        
        return {
            'current_includes': list(current_includes),
            'used_apis': used_apis,
            'apis_in_file': apis_in_file,
            'required_headers': required_headers,
            'missing_includes': list(missing_includes),
            'unnecessary_includes': list(unnecessary_includes),
            'fix_prompt': fix_prompt
        }

    def apply_include_fixes(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the file and apply include fixes using Claude's text editor.
        
        Returns:
            Dictionary with the API response and fixed content
        """
        # First, debug the file
        debug_info = self.debug_file_includes(file_path, content)
        
        if 'error' in debug_info:
            return debug_info
        
        if not debug_info['missing_includes'] and not debug_info['unnecessary_includes']:
            print("[Include Debugger] No include fixes needed!")
            return {
                'success': True,
                'message': 'No include fixes needed',
                'content': content or open(file_path, 'r').read()
            }
        
        # Call Claude with the text editor tool
        response = self._call_claude_with_editor(
            debug_info['fix_prompt'],
            content or open(file_path, 'r').read()
        )
        
        return response
    
    def find_required_headers_from_apis(self, apis_in_file):
        """Find which headers are required based on API usage."""
        required_headers = set()
        
        for api_type, api_list in apis_in_file.items():
            for api_name in api_list:
                if api_type == "functions":
                    print(f"[Include Debugger] Looking up function: {api_name}")
                    result = find_header_for_api_tool(api_name, 'functions', self.database_path)
                    
                    # Check if API was found - the key check is "error" NOT in result
                    if isinstance(result, dict) and "error" not in result:
                        # API was found!
                        header = result["defined_in"]
                        print(f"[Include Debugger] Function {api_name} found in {header}")
                        required_headers.add(header)
                    else:
                        # API not found
                        print(f"[Include Debugger] Function {api_name} not found in database")
                        if isinstance(result, dict):
                            if "partial_matches" in result:
                                print(f"  Partial matches: {result['partial_matches'][:3]}")
                            elif "hint" in result:
                                print(f"  Hint: {result['hint']}")
                
                elif api_type == "types":
                    # Handle types (classes, structs, etc.)
                    print(f"[Include Debugger] Looking up type: {api_name}")
                    
                    # Try different type categories
                    for type_category in ["classes", "structs", "enums", "typedefs"]:
                        result = find_header_for_api_tool(api_name, type_category, self.database_path)
                        if isinstance(result, dict) and "error" not in result:
                            header = result["defined_in"]
                            print(f"[Include Debugger] Type {api_name} ({type_category}) found in {header}")
                            required_headers.add(header)
                            break
                    else:
                        print(f"[Include Debugger] Type {api_name} not found in database")
                        
        return required_headers 

    def _convert_extracted_apis(self, apis_in_file: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Convert tree-sitter extracted APIs to a simpler format for display."""
        used_apis = {}
        
        # Functions - extract just the function names
        functions = set()
        for func in apis_in_file.get('functions', []):
            func_name = func.split('(')[0].strip()
            if ' ' in func_name:
                func_name = func_name.split()[-1]
            functions.add(func_name)
        used_apis['functions'] = functions
        
        # Template functions
        used_apis['template_functions'] = set(apis_in_file.get('template_functions', []))
        
        # Types (classes, structs, enums)
        types = set()
        types.update(apis_in_file.get('classes', []))
        types.update(apis_in_file.get('structs', []))
        types.update(apis_in_file.get('enums', []))
        used_apis['types'] = types
        
        # Other declarations
        used_apis['typedefs'] = set(apis_in_file.get('typedefs', []))
        used_apis['usings'] = set(apis_in_file.get('usings', []))
        used_apis['constants'] = set(apis_in_file.get('constants', []))
        
        return used_apis
    
    def _extract_current_includes(self, content: str) -> Set[str]:
        """Extract current include statements from the file."""
        includes = set()
        
        # Match both #include <...> and #include "..."
        include_pattern = re.compile(r'#include\s*[<"]([^>"]+)[>"]')
        
        for match in include_pattern.finditer(content):
            include_path = match.group(1)
            includes.add(include_path)
        
        return includes
    
    def _find_unnecessary_includes(self, current_includes: Set[str], 
                                 apis_in_file: Dict[str, List[str]],
                                 required_headers: Set[str]) -> Set[str]:
        """
        Find includes that aren't providing any used APIs.
        
        This is a conservative check - we only mark as unnecessary if we're
        confident the header isn't being used.
        """
        unnecessary = set()
        
        # Skip checking certain types of includes
        skip_patterns = [
            'iostream', 'vector', 'string', 'map', 'set',  # STL
            'cstdint', 'cstring', 'cmath', 'algorithm',    # C standard
            'memory', 'utility', 'functional', 'tuple',     # Modern C++
        ]
        
        for include in current_includes:
            # Skip system/standard headers
            if any(pattern in include for pattern in skip_patterns):
                continue
            
            # Skip if it's in our required set
            if include in required_headers:
                continue
            
            # Skip local project headers (relative paths)
            if include.startswith('./') or include.startswith('../'):
                continue
            
            # Check if this header provides any APIs we're using
            apis_provided = get_apis_from_header_tool(include, self.database_path)
            
            if 'error' in apis_provided:
                # Can't determine, so don't mark as unnecessary
                continue
            
            # Check if any of the APIs from this header are used
            header_is_used = False
            
            if 'api_summary' in apis_provided:
                for api_type, api_info in apis_provided['api_summary'].items():
                    if not isinstance(api_info, dict) or 'items' not in api_info:
                        continue
                    
                    # Get the corresponding APIs from our file
                    file_apis = apis_in_file.get(api_type, [])
                    
                    # Check for overlap
                    for provided_api in api_info['items']:
                        # For functions, need to compare just names
                        if api_type == 'functions':
                            provided_name = provided_api.split('(')[0].strip()
                            if ' ' in provided_name:
                                provided_name = provided_name.split()[-1]
                            
                            for used_func in file_apis:
                                used_name = used_func.split('(')[0].strip()
                                if ' ' in used_name:
                                    used_name = used_name.split()[-1]
                                
                                if provided_name == used_name:
                                    header_is_used = True
                                    break
                        else:
                            # For other types, direct comparison
                            if provided_api in file_apis:
                                header_is_used = True
                                break
                    
                    if header_is_used:
                        break
            
            if not header_is_used:
                unnecessary.add(include)
        
        return unnecessary
    
    def _generate_fix_prompt(self, file_path: str, content: str,
                           current_includes: Set[str],
                           required_headers: Dict[str, Any],
                           missing_includes: Set[str],
                           unnecessary_includes: Set[str],
                           apis_in_file: Dict[str, List[str]]) -> str:
        """Generate a prompt for Claude to fix the includes."""
        
        # Build a clear summary of what needs to be fixed
        fixes_needed = []
        
        if missing_includes:
            fixes_needed.append("ADD these missing includes:")
            for include in sorted(missing_includes):
                fixes_needed.append(f'  #include "{include}"')
        
        if unnecessary_includes:
            fixes_needed.append("\nREMOVE these unnecessary includes:")
            for include in sorted(unnecessary_includes):
                fixes_needed.append(f'  #include "{include}"  # Not used by any API in this file')
        
        # Build API to header mapping for context
        api_mappings = []
        for api_key, header_info in required_headers.items():
            api_type, api_name = api_key.split('::', 1)
            
            if isinstance(header_info, dict):
                header = header_info.get('defined_in', header_info.get('include_path', 'unknown'))
                api_mappings.append(f"  {api_name} ({api_type}) -> {header}")
            elif isinstance(header_info, str):
                api_mappings.append(f"  {api_name} ({api_type}) -> {header_info}")
        
        # Show what APIs were found in the file
        api_summary = []
        for api_type, api_list in apis_in_file.items():
            if api_list:
                api_summary.append(f"  {api_type}: {len(api_list)} items")
        
        prompt = f"""Fix the #include statements in this C++ file: {file_path}

        Based on tree-sitter analysis, these are the required changes:

        {chr(10).join(fixes_needed)}

        APIs found in this file:
        {chr(10).join(api_summary)}

        Required headers for these APIs:
        {chr(10).join(api_mappings[:20])}  # Limit to first 20 for readability
        {f"... and {len(api_mappings) - 20} more" if len(api_mappings) > 20 else ""}

        Current includes in the file:
        {chr(10).join(f'  {i+1}. #include "{inc}"' for i, inc in enumerate(sorted(current_includes)))}

        Instructions:
        1. Update ONLY the #include section at the top of the file
        2. Do not modify any other code
        3. Keep includes organized:
        - System/standard headers first (using < >)
        - TT-Metal/TTNN headers next (using " ")
        - Local/relative headers last (using " ")
        4. Make the minimal necessary changes to fix compilation

        Use the str_replace_based_edit_tool to make these changes."""
        
        return prompt
    
    def _call_claude_with_editor(self, prompt: str, file_content: str) -> Dict[str, Any]:
        """
        Call Claude API with the text editor tool to fix includes.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # System prompt for include fixing
        system_prompt = """You are an expert C++ developer fixing include statements in TT-Metal/TTNN code.
        You will use the text editor tool to make precise changes to fix compilation issues.
        Only modify #include statements - do not change any other code.
        The text editor tool allows you to replace text sections precisely."""
        
        # Prepare the message with file content
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\nCurrent file content:\n```cpp\n{file_content}\n```"
            }
        ]
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": messages,
            "tools": [
                {
                    "type": "text_editor_20250429",
                    "name": "str_replace_based_edit_tool"
                }
            ]
        }
        
        print("[Include Debugger] Calling Claude to fix includes...")
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Process the response
            content_blocks = response_data.get("content", [])
            
            # Look for the tool use
            for block in content_blocks:
                if block.get("type") == "tool_use" and block.get("name") == "str_replace_based_edit_tool":
                    # Extract the tool result
                    tool_input = block.get("input", {})
                    
                    # The actual implementation would depend on how the tool returns the edited content
                    # For now, we'll return the response structure
                    return {
                        'success': True,
                        'response': response_data,
                        'tool_input': tool_input,
                        'message': 'Applied include fixes'
                    }
            
            # If no tool was used, check if there's a text response
            text_content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
            
            return {
                'success': False,
                'response': response_data,
                'text_content': text_content,
                'message': 'Claude did not use the text editor tool as expected'
            }
            
        except requests.exceptions.HTTPError as e:
            return {
                'success': False,
                'error': str(e),
                'response_text': e.response.text if hasattr(e, 'response') else None,
                'message': f'HTTP Error calling Claude API: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Error calling Claude API: {str(e)}'
            }

    def _extract_api_usage_from_file(self, file_path: str) -> Dict[str, List[str]]:
        """Extract API usage (not declarations) from a C++ source file."""
        from ttnn_op_generator.tools.tree_sitter_tool import parse_file, query
        
        print(f"[Include Debugger] Extracting API usage from: {file_path}")
        
        # Parse with tree-sitter
        tree_id = parse_file(file_path)
        
        api_usage = {
            "function_calls": [],
            "types_used": [],
            "namespaces": [],
            "macros": []
        }
        
        # Extract function CALLS (not declarations)
        call_query = """
        (call_expression
            function: (identifier) @func_name
        ) @call
        
        (call_expression
            function: (qualified_identifier) @qualified_func
        ) @qualified_call
        
        (call_expression
            function: (field_expression
                field: (field_identifier) @method_name)
        ) @method_call
        """
        
        call_results = query(tree_id, call_query)
        for result in call_results:
            if result['name'] in ['func_name', 'qualified_func', 'method_name']:
                api_usage['function_calls'].append(result['text'])
        
        # Extract type usage (not declarations)
        type_usage_query = """
        (declaration
            type: (type_identifier) @type_name
        ) @type_use
        
        (parameter_declaration
            type: (type_identifier) @param_type
        ) @param_type_use
        
        (class_specifier
            name: (type_identifier) @base_class
        ) @inheritance
        """
        
        type_results = query(tree_id, type_usage_query)
        for result in type_results:
            if result['name'] in ['type_name', 'param_type', 'base_class']:
                api_usage['types_used'].append(result['text'])
        
        # Extract namespace usage
        namespace_query = """
        (qualified_identifier
            scope: (namespace_identifier) @namespace
        ) @ns_use
        """
        
        ns_results = query(tree_id, namespace_query)
        for result in ns_results:
            if result['name'] == 'namespace':
                api_usage['namespaces'].append(result['text'])
        
        # Remove duplicates
        for key in api_usage:
            api_usage[key] = list(set(api_usage[key]))
        
        print(f"[Include Debugger] Found {len(api_usage['function_calls'])} function calls, "
            f"{len(api_usage['types_used'])} types used")
        
        # Convert to format expected by the rest of the code
        return {
            "functions": api_usage['function_calls'],
            "classes": api_usage['types_used'],  # Types include classes
            "structs": [],  # Will be in types_used
            "enums": [],  # Will be in types_used
            "typedefs": [],
            "namespaces": api_usage['namespaces'],
            "template_functions": [],
            "usings": [],
            "constants": api_usage.get('macros', [])
        }


def main():
    """Command line interface for the include debugger."""
    parser = argparse.ArgumentParser(
        description="Debug and fix C++ include statements using tree-sitter and Claude AI"
    )
    
    parser.add_argument(
        "--file-path",
        help="Path to the C++ file to analyze"
    )
    
    #parser.add_argument(
    #    "--api-key",
    #    required=True,
    #    help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    #    default=os.environ.get("ANTHROPIC_API_KEY")
    #)
    
    parser.add_argument(
        "--database",
        default="home/user/tt-metal/ttnn_op_generator/include_api_database.json",
        help="Path to the API database JSON file (default: include_api_database.json)"
    )
    
    parser.add_argument(
        "--tt-metal-path",
        default=os.environ.get("TT_METAL_PATH"),
        help="Path to TT-Metal repository (default: from TT_METAL_PATH env var)"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes using Claude (default: just analyze)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for fixed content (default: overwrite input file when using --fix)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}")
        return 1
    
    #if not args.api_key:
    #    print("Error: API key required. Use --api-key or set ANTHROPIC_API_KEY")
    #    return 1
    
    # Create debugger
    debugger = IncludeDebugger(
        database_path=args.database,
        tt_metal_path=args.tt_metal_path
    )
    
    # Run analysis or fix
    if args.fix:
        print(f"Analyzing and fixing includes in: {args.file_path}")
        result = debugger.apply_include_fixes(args.file_path)
        
        if result.get('success'):
            print(f"\n✓ Success: {result.get('message', 'Fixes applied')}")
            
            # Save the fixed content if provided
            if 'content' in result:
                output_path = args.output or args.file_path
                with open(output_path, 'w') as f:
                    f.write(result['content'])
                print(f"✓ Saved fixed file to: {output_path}")
        else:
            print(f"\n✗ Failed: {result.get('message', 'Unknown error')}")
            if args.verbose and 'error' in result:
                print(f"Error details: {result['error']}")
            return 1
    else:
        print(f"Analyzing includes in: {args.file_path}")
        result = debugger.debug_file_includes(args.file_path)
        
        if 'error' in result:
            print(f"\n✗ Error: {result['error']}")
            return 1
        
        # Display results
        print(f"\n=== Include Analysis Results ===")
        print(f"Current includes: {len(result['current_includes'])}")
        print(f"Missing includes: {len(result['missing_includes'])}")
        print(f"Unnecessary includes: {len(result['unnecessary_includes'])}")
        
        if args.verbose:
            print("\n--- Current Includes ---")
            for inc in sorted(result['current_includes']):
                print(f"  #include \"{inc}\"")
            
            print("\n--- APIs Found (by tree-sitter) ---")
            for api_type, count in result['used_apis'].items():
                if count:
                    print(f"  {api_type}: {len(count)} items")
            
            if result['missing_includes']:
                print("\n--- Missing Includes ---")
                for inc in sorted(result['missing_includes']):
                    print(f"  #include \"{inc}\"")
            
            if result['unnecessary_includes']:
                print("\n--- Potentially Unnecessary Includes ---")
                for inc in sorted(result['unnecessary_includes']):
                    print(f"  #include \"{inc}\"")
        
        # Always show the summary
        if result['missing_includes'] or result['unnecessary_includes']:
            print(f"\nUse --fix to automatically fix these issues")
        else:
            print(f"\n✓ No include issues found!")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())