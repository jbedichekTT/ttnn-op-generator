#!/usr/bin/env python3
"""
Robust C++ to Header file generator using tree-sitter for accurate parsing.

This tool extracts all public declarations from a .cpp file to generate
a complete .hpp file.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

# Try to use tree-sitter if available
try:
    from tree_sitter_tool import parse_file, query
    USE_TREE_SITTER = True
except ImportError:
    USE_TREE_SITTER = False
    print("Warning: tree-sitter not available, using regex-based parsing")


@dataclass
class Declaration:
    """Represents a C++ declaration."""
    type: str  # 'function', 'class', 'struct', 'enum', 'typedef', 'using'
    name: str
    signature: str
    namespace: List[str]
    requires_definition: bool = False


class CppToHppGenerator:
    """Generate header files from C++ implementation files."""
    
    def __init__(self, cpp_path: Path):
        self.cpp_path = cpp_path
        self.cpp_content = cpp_path.read_text()
        self.includes: Set[str] = set()
        self.declarations: List[Declaration] = []
        self.namespace_stack: List[str] = []
        
    def generate_hpp(self) -> str:
        """Generate the header file content."""
        if USE_TREE_SITTER:
            self._parse_with_tree_sitter()
        else:
            self._parse_with_regex()
            
        return self._build_header()
        
    def _parse_with_tree_sitter(self):
        """Use tree-sitter for accurate parsing."""
        tree_id = parse_file(str(self.cpp_path))
        
        # Extract includes
        include_query = """
        (preproc_include
            path: (_) @include_path
        ) @include
        """
        
        include_results = query(tree_id, include_query)
        for result in include_results:
            if result['name'] == 'include':
                include_text = result['text']
                # Skip .cpp includes and implementation details
                if not any(skip in include_text for skip in ['.cpp"', '.cc"', '/detail/', '/impl/']):
                    self.includes.add(include_text)
        
        # Extract function definitions
        function_query = """
        (function_definition
            type: (_) @return_type
            declarator: (function_declarator
                declarator: (identifier) @func_name
                parameters: (parameter_list) @params
            )
        ) @func_def
        """
        
        func_results = query(tree_id, function_query)
        
        # Group results by function
        func_groups = {}
        for result in func_results:
            if result['name'] == 'func_def':
                start = result['byte_range'][0]
                func_groups[start] = {
                    'definition': result['text'],
                    'parts': {}
                }
        
        # Associate parts with functions
        for result in func_results:
            if result['name'] != 'func_def':
                for start in func_groups:
                    end = func_groups[start]['definition'].find('{')
                    if end == -1:
                        end = len(func_groups[start]['definition'])
                    
                    # Check if this result is within the function signature
                    if result['byte_range'][0] >= start and result['byte_range'][0] < start + end:
                        func_groups[start]['parts'][result['name']] = result['text']
        
        # Build function declarations
        for func_data in func_groups.values():
            parts = func_data['parts']
            if 'func_name' in parts:
                # Extract signature (everything before the opening brace)
                full_def = func_data['definition']
                sig_end = full_def.find('{')
                if sig_end != -1:
                    signature = full_def[:sig_end].strip()
                    if not signature.endswith(';'):
                        signature += ';'
                    
                    # Determine namespace
                    namespace = self._find_namespace_context(full_def)
                    
                    self.declarations.append(Declaration(
                        type='function',
                        name=parts['func_name'],
                        signature=signature,
                        namespace=namespace
                    ))
        
        # Extract class/struct definitions
        class_query = """
        [
            (class_specifier
                name: (type_identifier) @class_name
                body: (field_declaration_list) @class_body
            ) @class_def
            (struct_specifier
                name: (type_identifier) @struct_name
                body: (field_declaration_list) @struct_body
            ) @struct_def
        ]
        """
        
        class_results = query(tree_id, class_query)
        
        for result in class_results:
            if result['name'] in ['class_def', 'struct_def']:
                # For now, create forward declaration
                # In a more sophisticated version, we'd check if the full definition is needed
                type_kind = 'class' if result['name'] == 'class_def' else 'struct'
                
                # Find the name
                full_text = result['text']
                match = re.match(rf'{type_kind}\s+(\w+)', full_text)
                if match:
                    name = match.group(1)
                    self.declarations.append(Declaration(
                        type=type_kind,
                        name=name,
                        signature=f"{type_kind} {name};",
                        namespace=self._find_namespace_context(full_text)
                    ))
        
    def _parse_with_regex(self):
        """Fallback regex-based parsing."""
        lines = self.cpp_content.split('\n')
        
        current_namespace = []
        brace_depth = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Track braces for namespace depth
            brace_depth += line.count('{') - line.count('}')
            
            # Handle includes
            if line.startswith('#include'):
                if not any(skip in line for skip in ['.cpp"', '.cc"', '/detail/', '/impl/']):
                    self.includes.add(line)
            
            # Handle namespace
            namespace_match = re.match(r'namespace\s+(\w+)\s*\{', line)
            if namespace_match:
                current_namespace.append(namespace_match.group(1))
            
            # Handle namespace closing
            if '}' in line and current_namespace and brace_depth < len(current_namespace):
                current_namespace.pop()
            
            # Handle function definitions
            # Look for pattern: ReturnType functionName(...) {
            func_pattern = r'^([\w\s\*\&:<>,]+)\s+(\w+)\s*\(([^{]*)\)\s*(const)?\s*(noexcept)?\s*\{'
            
            # Build complete function signature (may span multiple lines)
            full_line = line
            j = i
            while j < len(lines) - 1 and '{' not in full_line and not line.endswith(';'):
                j += 1
                full_line += ' ' + lines[j].strip()
            
            func_match = re.match(func_pattern, full_line)
            if func_match:
                return_type = func_match.group(1).strip()
                func_name = func_match.group(2)
                params = func_match.group(3).strip()
                const_qual = func_match.group(4) or ''
                noexcept_qual = func_match.group(5) or ''
                
                # Build declaration
                signature = f"{return_type} {func_name}({params})"
                if const_qual:
                    signature += f" {const_qual}"
                if noexcept_qual:
                    signature += f" {noexcept_qual}"
                signature += ";"
                
                self.declarations.append(Declaration(
                    type='function',
                    name=func_name,
                    signature=signature,
                    namespace=list(current_namespace)
                ))
                
                i = j
            
            # Handle struct/class definitions
            struct_pattern = r'^(class|struct)\s+(\w+)\s*(\{|$)'
            struct_match = re.match(struct_pattern, line)
            if struct_match:
                keyword = struct_match.group(1)
                name = struct_match.group(2)
                
                # For now, just forward declare
                self.declarations.append(Declaration(
                    type=keyword,
                    name=name,
                    signature=f"{keyword} {name};",
                    namespace=list(current_namespace)
                ))
            
            # Handle using declarations and typedefs
            if line.startswith('using ') and '=' in line and line.endswith(';'):
                self.declarations.append(Declaration(
                    type='using',
                    name='',
                    signature=line,
                    namespace=list(current_namespace)
                ))
            elif line.startswith('typedef ') and line.endswith(';'):
                self.declarations.append(Declaration(
                    type='typedef',
                    name='',
                    signature=line,
                    namespace=list(current_namespace)
                ))
            
            i += 1
            
    def _find_namespace_context(self, text: str) -> List[str]:
        """Try to determine namespace context from surrounding text."""
        # This is a simplified version - in reality, we'd need to track position
        # in the file and maintain namespace state
        namespaces = []
        
        # Look for namespace declarations before this text in the file
        pos = self.cpp_content.find(text)
        if pos != -1:
            before_text = self.cpp_content[:pos]
            
            # Count namespace openings and closings
            namespace_stack = []
            for match in re.finditer(r'namespace\s+(\w+)\s*\{', before_text):
                namespace_stack.append(match.group(1))
            
            # Simplified: assume each } closes a namespace
            close_count = before_text.count('}')
            open_count = len(namespace_stack)
            
            # Take the namespaces that are still open
            if open_count > close_count:
                namespaces = namespace_stack[-(open_count - close_count):]
                
        return namespaces
        
    def _build_header(self) -> str:
        """Build the final header file content."""
        lines = []
        
        # Header guard
        guard_name = self.cpp_path.stem.upper() + "_HPP"
        lines.append("#pragma once")
        lines.append("")
        
        # Add includes
        if self.includes:
            lines.extend(sorted(self.includes))
            lines.append("")
        
        # Group declarations by namespace
        namespace_groups: Dict[tuple, List[Declaration]] = {}
        for decl in self.declarations:
            ns_key = tuple(decl.namespace)
            if ns_key not in namespace_groups:
                namespace_groups[ns_key] = []
            namespace_groups[ns_key].append(decl)
        
        # Output declarations grouped by namespace
        for ns_tuple, declarations in namespace_groups.items():
            # Open namespaces
            for ns in ns_tuple:
                lines.append(f"namespace {ns} {{")
            
            if ns_tuple:
                lines.append("")
            
            # Add declarations
            for decl in declarations:
                lines.append(decl.signature)
                
            if ns_tuple:
                lines.append("")
            
            # Close namespaces
            for ns in reversed(ns_tuple):
                lines.append(f"}}  // namespace {ns}")
        
        return '\n'.join(lines)


def generate_hpp_for_ttnn_operation(cpp_path: Path) -> str:
    """
    Specialized generator for TTNN operations that follow standard patterns.
    """
    cpp_content = cpp_path.read_text()
    
    # Extract operation name from filename
    # e.g., "eltwise_multiply_custom.cpp" -> "eltwise_multiply_custom"
    op_name = cpp_path.stem
    
    # Common includes for TTNN operations
    includes = []
    for line in cpp_content.split('\n'):
        if line.startswith('#include'):
            # Filter out implementation-only includes
            if not any(skip in line for skip in ['.cpp"', '.cc"', '/detail/', 'program_factory.hpp']):
                includes.append(line)
    
    # For TTNN operations, we know the pattern
    # Look for the main operation function
    func_pattern = rf'(Tensor)\s+{op_name}\s*\(([^{{]+)\)'
    match = re.search(func_pattern, cpp_content, re.DOTALL)
    
    if match:
        params = match.group(2).strip()
        # Clean up parameters (remove newlines, extra spaces)
        params = re.sub(r'\s+', ' ', params)
        
        function_signature = f"Tensor {op_name}({params});"
    else:
        # Fallback signature
        function_signature = f"""Tensor {op_name}(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt);"""
    
    # Build the header
    hpp_content = f"""#pragma once

{chr(10).join(includes)}

namespace ttnn {{

{function_signature}

}}  // namespace ttnn
"""
    
    return hpp_content


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate header file from C++ implementation"
    )
    parser.add_argument("--cpp-file", type=Path, help="Input C++ file")
    parser.add_argument("-o", "--output", type=Path, help="Output header file")
    parser.add_argument("--ttnn", action="store_true", 
                       help="Use TTNN operation specific patterns")
    
    args = parser.parse_args()
    
    if not args.cpp_file.exists():
        print(f"Error: File not found: {args.cpp_file}")
        return 1
    
    # Generate header
    if args.ttnn:
        hpp_content = generate_hpp_for_ttnn_operation(args.cpp_file)
    else:
        generator = CppToHppGenerator(args.cpp_file)
        hpp_content = generator.generate_hpp()
    
    # Output
    if args.output:
        output_path = args.output
    else:
        output_path = args.cpp_file.with_suffix('.hpp')
    
    output_path.write_text(hpp_content)
    print(f"Generated header: {output_path}")
    
    # Also print for review
    print("\nGenerated content:")
    print("=" * 80)
    print(hpp_content)
    
    return 0


if __name__ == "__main__":
    exit(main())