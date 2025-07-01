import os
import re
from pathlib import Path
from typing import List, Dict, Set, Optional
from ttnn_op_generator.tools.tree_sitter_tool import parse_file, query

def extract_apis_from_header(header_path: str, base_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Extract API declarations from a header file without implementations.
    
    Args:
        header_path: Path to header file (can be relative to TT-Metal)
        base_path: Base path to search from (defaults to TT_METAL_PATH)
        
    Returns:
        Dictionary containing:
        - functions: List of function signatures
        - template_functions: List of template function signatures
        - classes: List of class names
        - structs: List of struct names
        - enums: List of enum names
        - typedefs: List of typedef names
        - namespaces: List of namespace names
        - usings: List of using declarations
    """
    if base_path is None:
        base_path = os.environ.get("TT_METAL_PATH", "/home/user/tt-metal")
    
    # Find the header file - handle the fact that includes use "ttnn/..." but 
    # actual paths are "ttnn/cpp/ttnn/..."
    full_path = None
    
    # For device-relative paths (like device/eltwise_multiply_custom_op.hpp),
    # these are relative to the operation's output directory
    if header_path.startswith("device/"):
        # This is a local file that will be generated, skip validation
        return {
            "error": f"Local file (will be generated): {header_path}",
            "local": True
        }
    
    # Common include path mappings in TT-Metal
    path_mappings = [
        # TTNN headers: include "ttnn/X" -> file at "ttnn/cpp/ttnn/X"
        ("ttnn/", "ttnn/cpp/ttnn/"),
        # TT-Metal headers: include "tt_metal/X" -> file at "tt_metal/X"
        ("tt_metal/", "tt_metal/"),
        # Common headers: try as-is first
        ("", ""),
        # Sometimes headers are in tt_metal/include/
        ("", "tt_metal/include/"),
        # Sometimes in tt_metal/api/
        ("", "tt_metal/api/"),
    ]
    
    base_path = Path(base_path)
    
    for include_prefix, file_prefix in path_mappings:
        if header_path.startswith(include_prefix) or include_prefix == "":
            # Remove the include prefix and add the file prefix
            relative_path = header_path
            if include_prefix:
                relative_path = header_path[len(include_prefix):]
            
            test_path = base_path / file_prefix / relative_path
            
            if test_path.exists():
                full_path = test_path
                break
    
    if full_path is None:
        # Last resort: search for the file
        header_name = Path(header_path).name
        print(f"[API Extractor] Searching for {header_name}...")
        
        # Search in common directories
        search_dirs = ["ttnn", "tt_metal", "tt_eager"]
        for search_dir in search_dirs:
            dir_path = base_path / search_dir
            if dir_path.exists():
                # Use rglob to search recursively
                matches = list(dir_path.rglob(header_name))
                if matches:
                    full_path = matches[0]  # Take the first match
                    print(f"[API Extractor] Found at: {full_path}")
                    break
    
    if full_path is None or not full_path.exists():
        return {"error": f"Header file not found: {header_path}"}
    
    print(f"[API Extractor] Parsing {full_path}")
    
    # Parse with tree-sitter
    tree_id = parse_file(str(full_path))
    
    apis = {
        "functions": [],
        "template_functions": [],
        "classes": [],
        "structs": [],
        "enums": [],
        "typedefs": [],
        "namespaces": [],
        "usings": []
    }
    
    try:
        # Extract regular functions (declarations AND definitions)
        func_query = """
        [
            (declaration
                type: (_)? @return_type
                declarator: (function_declarator
                    declarator: (identifier) @fn_name
                    parameters: (parameter_list) @params)
            ) @fn_decl
            
            (function_definition
                type: (_)? @return_type
                declarator: (function_declarator
                    declarator: (identifier) @fn_name
                    parameters: (parameter_list) @params)
            ) @fn_decl
        ]
        """
        
        func_results = query(tree_id, func_query)
        
        # Group by declaration
        func_groups = {}
        for result in func_results:
            if result['name'] == 'fn_decl':
                start = result['byte_range'][0]
                func_groups[start] = {'decl': result}
        
        for result in func_results:
            if result['name'] != 'fn_decl':
                for start in func_groups:
                    if result['byte_range'][0] >= start and result['byte_range'][1] <= func_groups[start]['decl']['byte_range'][1]:
                        func_groups[start][result['name']] = result
        
        # Build function signatures
        for group in func_groups.values():
            if 'fn_name' in group:
                signature = ""
                if 'return_type' in group:
                    # Strip ALWI prefix if present
                    return_type = group['return_type']['text']
                    if return_type.startswith("ALWI "):
                        return_type = return_type[5:]  # Remove "ALWI "
                    signature += return_type + " "
                signature += group['fn_name']['text']
                if 'params' in group:
                    signature += group['params']['text']
                apis['functions'].append(signature)
        
        # Extract template functions (declarations AND definitions)
        template_func_query = """
        [
            (template_declaration
                (declaration
                    type: (_)? @return_type
                    declarator: (function_declarator
                        declarator: (identifier) @fn_name
                        parameters: (parameter_list)? @params)
                )
            ) @template_fn_decl
            
            (template_declaration
                (function_definition
                    type: (_)? @return_type
                    declarator: (function_declarator
                        declarator: (identifier) @fn_name
                        parameters: (parameter_list)? @params)
                )
            ) @template_fn_decl
        ]
        """
        
        template_func_results = query(tree_id, template_func_query)
        
        # Group template function results
        template_func_groups = {}
        for result in template_func_results:
            if result['name'] == 'template_fn_decl':
                start = result['byte_range'][0]
                template_func_groups[start] = {'decl': result}
        
        for result in template_func_results:
            if result['name'] != 'template_fn_decl':
                for start in template_func_groups:
                    if result['byte_range'][0] >= start and result['byte_range'][1] <= template_func_groups[start]['decl']['byte_range'][1]:
                        template_func_groups[start][result['name']] = result
        
        # Build template function signatures
        for group in template_func_groups.values():
            if 'fn_name' in group:
                signature = ""
                if 'return_type' in group:
                    # Strip ALWI prefix if present
                    return_type = group['return_type']['text']
                    if return_type.startswith("ALWI "):
                        return_type = return_type[5:]  # Remove "ALWI "
                    signature += return_type + " "
                signature += group['fn_name']['text']
                if 'params' in group:
                    signature += group['params']['text']
                apis['template_functions'].append(signature)
        
        # Extract classes (only with bodies)
        class_query = """
        (class_specifier
            name: (type_identifier) @class_name
            body: (field_declaration_list)
        ) @class_decl
        """
        
        class_results = query(tree_id, class_query)
        for result in class_results:
            if result['name'] == 'class_name':
                apis['classes'].append(result['text'])
        
        # Extract structs (only with bodies)
        struct_query = """
        (struct_specifier
            name: (type_identifier) @struct_name
            body: (field_declaration_list)
        ) @struct_decl
        """
        
        struct_results = query(tree_id, struct_query)
        for result in struct_results:
            if result['name'] == 'struct_name':
                apis['structs'].append(result['text'])
        
        # Extract enums
        enum_query = """
        (enum_specifier
            name: (type_identifier) @enum_name
        ) @enum_decl
        """
        
        enum_results = query(tree_id, enum_query)
        for result in enum_results:
            if result['name'] == 'enum_name':
                apis['enums'].append(result['text'])
        
        # Extract typedefs
        typedef_query = """
        (type_definition
            declarator: (type_identifier) @typedef_name
        ) @typedef_decl
        """
        
        typedef_results = query(tree_id, typedef_query)
        for result in typedef_results:
            if result['name'] == 'typedef_name':
                apis['typedefs'].append(result['text'])
        
        # Extract using declarations
        using_query = """
        (using_declaration
            (qualified_identifier) @using_name
        ) @using_decl
        """
        
        using_results = query(tree_id, using_query)
        for result in using_results:
            if result['name'] == 'using_name':
                apis['usings'].append(result['text'])
        
        # Extract namespaces
        namespace_query = """
        (namespace_definition
            name: (namespace_identifier) @ns_name
        ) @namespace_decl
        """
        
        namespace_results = query(tree_id, namespace_query)
        for result in namespace_results:
            if result['name'] == 'ns_name':
                apis['namespaces'].append(result['text'])
        
        # For constants, use regex patterns
        # Read the file content for regex matching
        with open(full_path, 'r') as f:
            content = f.read()
            
        const_patterns = [
            r'constexpr\s+\w+\s+(\w+)\s*=',
            r'static\s+const\s+\w+\s+(\w+)\s*=',
            r'const\s+\w+\s+(\w+)\s*=\s*\w+'
        ]
        
        constants = []
        for pattern in const_patterns:
            matches = re.findall(pattern, content)
            constants.extend(matches)
        
        apis['constants'] = list(set(constants))
        
    except Exception as e:
        print(f"[API Extractor] Error during parsing: {str(e)}")
        # Continue with what we have
    
    # Remove duplicates
    for key in apis:
        if isinstance(apis[key], list):
            apis[key] = list(set(apis[key]))
    
    print(f"[API Extractor] Found: {len(apis['functions'])} functions, "
          f"{len(apis['template_functions'])} template functions, "
          f"{len(apis['classes'])} classes, "
          f"{len(apis['usings'])} using declarations")
    
    return apis

def analyze_header_dependencies(header_paths: List[str], base_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Analyze multiple headers and their available APIs.
    
    Returns a mapping of header path to available APIs.
    """
    results = {}
    
    for header in header_paths:
        results[header] = extract_apis_from_header(header, base_path)
    
    return results