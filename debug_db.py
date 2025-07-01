#!/usr/bin/env python3
"""Test the exact return format of find_header_for_api_tool"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.database_query_tool import find_header_for_api_tool

def test_return_format(db_path="include_api_database.json"):
    """Test what find_header_for_api_tool actually returns"""
    
    test_cases = [
        ("cb_pop_front", "functions"),
        ("acos_tile", "functions"),
        ("nonexistent_func", "functions")
    ]
    
    print("Testing find_header_for_api_tool return format:\n")
    
    for func_name, api_type in test_cases:
        print(f"Looking up: {func_name}")
        result = find_header_for_api_tool(func_name, api_type, db_path)
        
        print(f"Result type: {type(result)}")
        print(f"Result: {json.dumps(result, indent=2)}")
        
        # Show how to check if found
        if isinstance(result, dict):
            if "error" in result:
                print("❌ NOT FOUND - has 'error' key")
            elif "defined_in" in result:
                print(f"✅ FOUND - header: {result['defined_in']}")
            else:
                print("❓ UNEXPECTED FORMAT")
        else:
            print(f"❓ UNEXPECTED TYPE: {type(result)}")
        
        print("-" * 50)

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "include_api_database.json"
    test_return_format(db_path)