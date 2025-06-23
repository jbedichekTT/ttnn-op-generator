"""
Tree-sitter based targeted code editor for C++ files
-----------------------------------------------------
Provides high-level editing operations that maintain code structure
and formatting while making surgical changes to C++ source files.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from ttnn_op_generator.tools.tree_sitter_tool import parse_file, query, replace_span, has_errors

@dataclass
class CodeEdit:
    """Represents a single edit operation"""
    operation: str  # 'insert', 'delete', 'replace', 'modify'
    target_type: str  # 'function', 'class', 'include', 'namespace', 'member'
    target_name: str
    content: Optional[str] = None
    location_hint: Optional[str] = None  # 'after:X', 'before:Y', 'inside:Z'
    
@dataclass
class EditResult:
    """Result of an edit operation"""
    success: bool
    new_content: str
    message: str
    tree_id: str

class TreeSitterEditor:
    """High-level C++ code editor using tree-sitter"""
    
    def __init__(self):
        self.cached_trees = {}
        
    def apply_edits(self, file_path: Union[str, Path], edits: List[CodeEdit]) -> EditResult:
        """Apply a sequence of edits to a file"""
        tree_id = parse_file(file_path)
        
        for edit in edits:
            if edit.operation == 'insert':
                tree_id = self._insert_code(tree_id, edit)
            elif edit.operation == 'delete':
                tree_id = self._delete_code(tree_id, edit)
            elif edit.operation == 'replace':
                tree_id = self._replace_code(tree_id, edit)
            elif edit.operation == 'modify':
                tree_id = self._modify_code(tree_id, edit)
                
        # Get final content
        final_content = self._get_tree_content(tree_id)
        
        return EditResult(
            success=not has_errors(tree_id),
            new_content=final_content,
            message="Edits applied successfully" if not has_errors(tree_id) else "Syntax errors detected",
            tree_id=tree_id
        )
    
    def _insert_code(self, tree_id: str, edit: CodeEdit) -> str:
        """Insert code at appropriate location"""
        if edit.target_type == 'include':
            return self._insert_include(tree_id, edit.content)
        elif edit.target_type == 'function':
            return self._insert_function(tree_id, edit.content, edit.location_hint)
        elif edit.target_type == 'class':
            return self._insert_class(tree_id, edit.content, edit.location_hint)
        elif edit.target_type == 'member':
            return self._insert_class_member(tree_id, edit.target_name, edit.content)
        return tree_id
    
    def _insert_include(self, tree_id: str, include_line: str) -> str:
        """Insert include at the appropriate position"""
        # Query for existing includes
        include_query = """
        (translation_unit
          (preproc_include) @include
        )
        """
        includes = query(tree_id, include_query)
        
        if includes:
            # Insert after last include
            last_include = includes[-1]
            insert_pos = last_include['byte_range'][1]
            return replace_span(tree_id, insert_pos, insert_pos, f"\n{include_line}")
        else:
            # Insert at beginning of file
            return replace_span(tree_id, 0, 0, f"{include_line}\n\n")
    
    def _insert_function(self, tree_id: str, function_code: str, location_hint: str) -> str:
        """Insert function at appropriate location"""
        if location_hint and location_hint.startswith("inside:"):
            # Insert inside a namespace or class
            container = location_hint.split(":")[1]
            container_query = f"""
            [
              (namespace_definition
                name: (namespace_identifier) @ns_name
                body: (declaration_list) @ns_body)
              (class_specifier
                name: (type_identifier) @class_name
                body: (field_declaration_list) @class_body)
            ]
            """
            
            results = query(tree_id, container_query)
            for i in range(0, len(results), 2):
                if i+1 < len(results) and results[i]['text'] == container:
                    # Found the container, insert at end of its body
                    body = results[i+1]
                    # Insert before the closing brace
                    insert_pos = body['byte_range'][1] - 1
                    return replace_span(tree_id, insert_pos, insert_pos, f"\n{function_code}\n")
        
        # Default: insert at end of file
        current_content = self._get_tree_content(tree_id)
        return replace_span(tree_id, len(current_content.encode()), len(current_content.encode()), f"\n{function_code}\n")
    
    def _delete_code(self, tree_id: str, edit: CodeEdit) -> str:
        """Delete specified code element"""
        if edit.target_type == 'function':
            query_str = """
            (function_definition
              declarator: (function_declarator
                declarator: (identifier) @fn_name)) @fn
            """
        elif edit.target_type == 'class':
            query_str = """
            (class_specifier
              name: (type_identifier) @class_name) @class
            """
        elif edit.target_type == 'include':
            query_str = """
            (preproc_include
              path: [
                (string_literal) @path
                (system_lib_string) @path
              ]) @include
            """
        else:
            return tree_id
            
        results = query(tree_id, query_str)
        
        # Find the target
        for i in range(0, len(results), 2):
            if i+1 < len(results):
                name_capture = results[i]
                element_capture = results[i+1]
                
                if edit.target_type == 'include':
                    # Check if include path matches
                    if edit.target_name in name_capture['text']:
                        return replace_span(tree_id, element_capture['byte_range'][0], 
                                         element_capture['byte_range'][1] + 1, "")  # +1 for newline
                elif name_capture['text'] == edit.target_name:
                    # Delete the element
                    return replace_span(tree_id, element_capture['byte_range'][0], 
                                     element_capture['byte_range'][1], "")
        
        return tree_id
    
    def _modify_code(self, tree_id: str, edit: CodeEdit) -> str:
        """Modify existing code element"""
        if edit.target_type == 'function':
            # Find function and replace its signature or body
            query_str = """
            (function_definition
              declarator: (function_declarator
                declarator: (identifier) @fn_name)
              body: (compound_statement) @body) @fn
            """
            results = query(tree_id, query_str)
            
            for i in range(0, len(results), 3):
                if i+2 < len(results) and results[i]['text'] == edit.target_name:
                    fn_capture = results[i+2]
                    # Replace entire function
                    return replace_span(tree_id, fn_capture['byte_range'][0],
                                     fn_capture['byte_range'][1], edit.content)
        
        return tree_id
    
    def _get_tree_content(self, tree_id: str) -> str:
        """Get the current content of a tree"""
        # Import the new function from tree_sitter_tool
        from tree_sitter_tool import get_tree_content
        return get_tree_content(tree_id)
    
    def analyze_for_fixes(self, tree_id: str, error_info: Dict) -> List[CodeEdit]:
        """Analyze tree and errors to suggest edits"""
        edits = []
        
        # Check for missing includes
        if 'undefined_reference' in error_info:
            for symbol in error_info['undefined_reference']:
                # Suggest adding include
                possible_header = self._guess_header_for_symbol(symbol)
                if possible_header:
                    edits.append(CodeEdit(
                        operation='insert',
                        target_type='include',
                        target_name=possible_header,
                        content=f'#include "{possible_header}"'
                    ))
        
        # Check for missing function implementations
        if 'undefined_function' in error_info:
            for func in error_info['undefined_function']:
                # Check if declaration exists but not definition
                decl_query = f"""
                (declaration
                  declarator: (function_declarator
                    declarator: (identifier) @fn_name))
                """
                results = query(tree_id, decl_query)
                
                for result in results:
                    if result['text'] == func:
                        # Declaration exists, need implementation
                        edits.append(CodeEdit(
                            operation='insert',
                            target_type='function',
                            target_name=func,
                            content=f"// TODO: Implement {func}",
                            location_hint="end"
                        ))
        
        return edits
    
    def _guess_header_for_symbol(self, symbol: str) -> Optional[str]:
        """Guess which header file might contain a symbol"""
        # Common patterns
        patterns = {
            r'tt::tt_metal::': 'tt_metal/host_api.hpp',
            r'ttnn::': 'ttnn/tensor/tensor.hpp',
            r'TT_FATAL': 'tt_metal/common/assert.hpp',
            r'log_': 'tt_metal/common/logger.hpp',
        }
        
        for pattern, header in patterns.items():
            if re.search(pattern, symbol):
                return header
        
        return None
    
    # Add this method to the TreeSitterEditor class in tree_sitter_editor.py

    def _replace_code(self, tree_id: str, edit: 'CodeEdit') -> str:
        """
        Replace code for a specific target.
        
        Args:
            tree_id: The tree identifier
            edit: The edit operation containing target info and new content
            
        Returns:
            Updated tree_id
        """
        from tree_sitter_tool import query, replace_span
        
        # Find the target based on edit.target_type and edit.target_name
        if edit.target_type == "function":
            query_str = f"""
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @fn_name)
            ) @fn_def
            """
        elif edit.target_type == "class":
            query_str = f"""
            (class_specifier
                name: (type_identifier) @class_name
            ) @class_spec
            """
        elif edit.target_type == "include":
            query_str = """
            (preproc_include) @include
            """
        else:
            raise ValueError(f"Unsupported target type for replace: {edit.target_type}")
        
        # Run the query
        results = query(tree_id, query_str)
        
        # Find the specific target
        target_node = None
        for result in results:
            if edit.target_type == "function" and result['name'] == 'fn_name':
                if result['text'] == edit.target_name:
                    # Find the full function definition
                    for r in results:
                        if r['name'] == 'fn_def' and r['byte_range'][0] <= result['byte_range'][0] <= r['byte_range'][1]:
                            target_node = r
                            break
                    break
            elif edit.target_type == "class" and result['name'] == 'class_name':
                if result['text'] == edit.target_name:
                    # Find the full class definition
                    for r in results:
                        if r['name'] == 'class_spec' and r['byte_range'][0] <= result['byte_range'][0] <= r['byte_range'][1]:
                            target_node = r
                            break
                    break
            # Add more target types as needed
        
        if not target_node:
            raise ValueError(f"Target {edit.target_type} '{edit.target_name}' not found")
        
        # Replace the content
        start, end = target_node['byte_range']
        return replace_span(tree_id, start, end, edit.content)