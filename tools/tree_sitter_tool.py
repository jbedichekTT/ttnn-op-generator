"""
Tree-sitter utility wrapper for C++ - Correct Implementation
===========================================================

Based on the actual py-tree-sitter API where:
- parser.language = LANGUAGE (not set_language)
- query.captures() returns a dict of {capture_name: [nodes]}
- query.matches() returns [(pattern_index, {capture_name: [nodes]})]
"""

import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

try:
    import tree_sitter
    from tree_sitter import Language, Parser
except ImportError:
    raise ImportError("tree_sitter is required. Install with: pip install tree-sitter")

# ---------------------------------------------------------------------------
# Global state management
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _THIS_DIR / "_ts_build"
_LIB_PATH = _BUILD_DIR / "cpp.so"
_CPP_GRAMMAR_REPO = _THIS_DIR / "vendor" / "tree-sitter-cpp"

# Global state
_CPP_LANGUAGE: Optional[Language] = None
_PARSER: Optional[Parser] = None
_LOCK = threading.Lock()
_TREES: Dict[str, Tuple[Any, bytes]] = {}


def _build_cpp_grammar():
    """Build the C++ grammar shared library if it doesn't exist."""
    print(f"Building C++ grammar at {_LIB_PATH}...")
    _BUILD_DIR.mkdir(exist_ok=True)
    
    # Clone the grammar if needed
    if not _CPP_GRAMMAR_REPO.exists():
        print("Cloning tree-sitter-cpp grammar...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/tree-sitter/tree-sitter-cpp.git",
            str(_CPP_GRAMMAR_REPO)
        ], check=True)
    
    # Build the library
    Language.build_library(
        str(_LIB_PATH),
        [str(_CPP_GRAMMAR_REPO)]
    )
    print("C++ grammar built successfully!")


def _ensure_cpp_language() -> Language:
    """
    Return a tree_sitter.Language for C++.
    """
    # Try pre-compiled wheel first
    try:
        import tree_sitter_cpp as tscpp
        print("[DEBUG] Using pre-compiled tree_sitter_cpp wheel")
        return Language(tscpp.language())
    except ImportError:
        print("[DEBUG] tree_sitter_cpp not found, building from source")
    
    # Build from source
    if not _LIB_PATH.exists():
        _build_cpp_grammar()
    
    # Load the built library
    return Language(str(_LIB_PATH), "cpp")


def _initialize_parser():
    """Initialize parser on first use"""
    global _CPP_LANGUAGE, _PARSER
    with _LOCK:
        if _CPP_LANGUAGE is None:
            _CPP_LANGUAGE = _ensure_cpp_language()
            _PARSER = Parser()
            # Use property assignment, not set_language method!
            _PARSER.language = _CPP_LANGUAGE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(file_path: Union[str, Path]) -> str:
    """
    Parse a file and cache the syntax tree.
    
    Args:
        file_path: Path to the C++ file to parse
        
    Returns:
        tree_id: Unique identifier for the cached tree
    """
    _initialize_parser()
    
    # Resolve the file path
    p = Path(file_path)
    if not p.is_absolute():
        if p.exists():
            p = p.resolve()
        else:
            # Try environment paths
            for env_var in ["TT_METAL_HOME", "TTNN_OUTPUT_DIR"]:
                env_path = os.environ.get(env_var)
                if env_path:
                    alt_path = Path(env_path) / file_path
                    if alt_path.exists():
                        p = alt_path.resolve()
                        break
    
    # Read the file
    if p.exists() and p.is_file():
        source = p.read_bytes()
        print(f"[DEBUG] Successfully read file: {p} ({len(source)} bytes)")
    else:
        print(f"[WARNING] File not found: {p}")
        source = b""
    
    # Parse the file
    tree = _PARSER.parse(source)
    tree_id = uuid.uuid4().hex
    
    # Cache the tree and source
    with _LOCK:
        _TREES[tree_id] = (tree, source)
    
    return tree_id


def has_errors(tree_id: str) -> bool:
    """Check if the cached tree contains syntax errors."""
    with _LOCK:
        tree, _ = _TREES[tree_id]
    
    def check_errors(node):
        if node.type == "ERROR":
            return True
        for child in node.children:
            if check_errors(child):
                return True
        return False
    
    return check_errors(tree.root_node)


def query(tree_id: str, query_str: str) -> List[Dict[str, Any]]:
    """
    Run a Tree-sitter query against the cached tree.
    
    According to the documentation, query.captures() returns a dictionary
    where keys are capture names and values are lists of nodes.
    
    Args:
        tree_id: Identifier of the cached tree
        query_str: Tree-sitter query string
        
    Returns:
        List of dictionaries with keys: name, text, byte_range
    """
    if tree_id not in _TREES:
        raise ValueError(f"Tree '{tree_id}' not loaded")

    with _LOCK:
        tree, src = _TREES[tree_id]
    
    # Create the query using the language from the parser
    q = _PARSER.language.query(query_str)
    
    # Get captures - should return a dict
    captures_dict = q.captures(tree.root_node)
    
    results = []
    
    if isinstance(captures_dict, dict):
        # Modern API: dict of capture_name -> [nodes]
        for capture_name, nodes in captures_dict.items():
            for node in nodes:
                snippet = src[node.start_byte:node.end_byte].decode("utf-8", "ignore")
                results.append({
                    "name": capture_name,
                    "text": snippet,
                    "byte_range": (node.start_byte, node.end_byte),
                })
    else:
        # Fallback for older API or unexpected format
        print(f"[WARNING] Unexpected captures format: {type(captures_dict)}")
        
        # Try to handle as iterable
        try:
            for item in captures_dict:
                if isinstance(item, tuple) and len(item) == 2:
                    # Old API: (node, capture_name) tuples
                    node, capture_name = item
                    snippet = src[node.start_byte:node.end_byte].decode("utf-8", "ignore")
                    results.append({
                        "name": capture_name,
                        "text": snippet,
                        "byte_range": (node.start_byte, node.end_byte),
                    })
        except Exception as e:
            print(f"[ERROR] Failed to process captures: {e}")
    
    return results


def replace_span(tree_id: str, start_byte: int, end_byte: int, replacement: str) -> str:
    """
    Replace a byte span in the cached source and reparse.
    
    Args:
        tree_id: Identifier of the cached tree
        start_byte: Start position of the span to replace
        end_byte: End position of the span to replace
        replacement: New text to insert
        
    Returns:
        tree_id: Same tree_id (for compatibility)
    """
    _initialize_parser()
    
    with _LOCK:
        tree, src = _TREES[tree_id]
    
    # Perform the replacement
    new_src = src[:start_byte] + replacement.encode() + src[end_byte:]
    
    # Reparse the modified source
    new_tree = _PARSER.parse(new_src)
    
    # Update the cache
    with _LOCK:
        _TREES[tree_id] = (new_tree, new_src)
    
    return tree_id


def get_tree_content(tree_id: str) -> str:
    """Get the current source content for a given tree_id."""
    with _LOCK:
        tree, src = _TREES.get(tree_id, (None, None))
        if src is None:
            raise KeyError(f"Tree ID '{tree_id}' not found in cache")
        return src.decode('utf-8', 'ignore')


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_function_stub(file_path: Union[str, Path], func_name: str) -> Optional[str]:
    """
    Extract a function and replace its body with a TODO comment.
    """
    tree_id = parse_file(file_path)
    
    # Query for function definitions
    func_query = """
    (function_definition
        declarator: (function_declarator 
            declarator: (identifier) @fn_name)
        body: (compound_statement) @fn_body
    ) @fn_def
    """
    
    results = query(tree_id, func_query)
    
    # Group results by function definition
    fn_defs = {}
    for result in results:
        if result['name'] == 'fn_def':
            start, end = result['byte_range']
            fn_defs[start] = {'def': result, 'name': None, 'body': None}
    
    # Associate names and bodies with their definitions
    for result in results:
        if result['name'] in ['fn_name', 'fn_body']:
            # Find which function definition this belongs to
            start, end = result['byte_range']
            for fn_start in fn_defs:
                fn_end = fn_defs[fn_start]['def']['byte_range'][1]
                if start >= fn_start and end <= fn_end:
                    if result['name'] == 'fn_name':
                        fn_defs[fn_start]['name'] = result
                    else:
                        fn_defs[fn_start]['body'] = result
                    break
    
    # Find the requested function
    for fn_info in fn_defs.values():
        if fn_info['name'] and fn_info['name']['text'] == func_name:
            if fn_info['body']:
                body_span = fn_info['body']['byte_range']
                replace_span(tree_id, body_span[0], body_span[1], 
                           "{\n    // TODO: implement\n}")
                return get_tree_content(tree_id)
    
    return None


def quick_syntax_check(code: str) -> bool:
    """Check if a code string has syntax errors."""
    _initialize_parser()
    
    tree = _PARSER.parse(code.encode())
    
    def check_errors(node):
        if node.type == "ERROR":
            return True
        for child in node.children:
            if check_errors(child):
                return True
        return False
    
    return not check_errors(tree.root_node)


# ---------------------------------------------------------------------------
# CLI interface for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Tree-sitter C++ utility")
    parser.add_argument("file", help="C++ file to parse")
    parser.add_argument("--func", help="Extract stub for function")
    parser.add_argument("--query", help="Run a custom query")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    args = parser.parse_args()
    
    if args.func:
        stub = extract_function_stub(args.file, args.func)
        print(stub if stub else f"Function '{args.func}' not found.")
    
    elif args.query:
        tree_id = parse_file(args.file)
        results = query(tree_id, args.query)
        print(json.dumps(results, indent=2))
    
    else:
        tree_id = parse_file(args.file)
        print(f"Tree ID: {tree_id}")
        print(f"Has errors: {has_errors(tree_id)}")
        
        # Run a simple test query
        results = query(tree_id, "(function_definition) @func")
        print(f"Found {len(results)} function definitions")
        
        if args.debug and results:
            print("\nFirst 3 results:")
            for i, result in enumerate(results[:3]):
                print(f"\n{i+1}. {result['name']}:")
                print(f"   Text: {result['text'][:100]}...")
                print(f"   Range: {result['byte_range']}")