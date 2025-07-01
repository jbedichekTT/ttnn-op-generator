# tool_config.py

# Define all tools with their enabled/disabled status
# This is mainly for reference and individual tool toggling
ENABLED_TOOLS = {
    # Database/API tools
    "get_apis_from_header": False,
    "find_header_for_api": False,
    "search_apis": False,
    
    # File operations
    "find_files_in_repository": False,
    "extract_symbols_from_files": False,
    "read_ttnn_example_files": False,
    
    # Code analysis
    "find_api_usages": False,
    "parse_and_analyze_code": False,
    "apply_targeted_edits": False,
    
    # Validation tools
    "validate_includes_for_file": False,
    "extract_apis_from_header": False,
    "resolve_namespace_and_verify": False,
    
    # Documentation
    "search_tt_metal_docs": False,
    "check_common_namespace_issues": False,
}

# Define tool sets for different use cases
TOOL_SETS = {
    "planning": [
        "get_apis_from_header",
        "find_header_for_api", 
        "search_apis"
    ],
    "generation": [
        "find_api_usages"
    ],
    "full_analysis": [
        "find_files_in_repository",
        "extract_symbols_from_files",
        "read_ttnn_example_files",
        "find_api_usages",
        "parse_and_analyze_code"
    ],
    "debugging": [
        "resolve_namespace_and_verify",
        "check_common_namespace_issues",
        "validate_includes_for_file"
    ],
    "all": None  # None means all tools
}

# Choose active set - CHANGE THIS to switch between presets
ACTIVE_SET = "generation"  # Options: "planning", "generation", "full_analysis", "debugging", "all", or None

# Optional: Override specific tools regardless of set
# Useful for temporarily adding/removing tools without changing the set
TOOL_OVERRIDES = {
    # "enable": ["search_tt_metal_docs"],  # Always enable these
    # "disable": ["apply_targeted_edits"]  # Always disable these
}