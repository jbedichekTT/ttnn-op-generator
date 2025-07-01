#!/usr/bin/env python3
"""
API Database Query Tools
========================

Tools for querying the API definition database to find:
1. Which APIs are defined in a given header
2. Which header defines a given API
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple


class APIDatabase:
    """Wrapper class for the API database."""
    
    def __init__(self, database_path: str = "include_api_database.json"):
        self.database_path = Path(database_path)
        self._db = None
        self._load_database()
    
    def _load_database(self):
        """Load the database from disk."""
        if not self.database_path.exists():
            raise FileNotFoundError(f"API database not found at: {self.database_path}")
        
        with open(self.database_path, 'r') as f:
            self._db = json.load(f)
            
        # Validate database structure
        required_keys = ["metadata", "headers", "api_to_header"]
        for key in required_keys:
            if key not in self._db:
                raise ValueError(f"Invalid database format: missing '{key}' section")
    
    def get_metadata(self) -> Dict:
        """Get database metadata."""
        return self._db["metadata"]
    
    def get_all_headers(self) -> List[str]:
        """Get list of all headers in the database."""
        return list(self._db["headers"].keys())
    
    def get_all_apis(self) -> List[str]:
        """Get list of all APIs in the database."""
        return list(self._db["api_to_header"].keys())
    
    def get_apis_from_header(self, header_path: str) -> Dict[str, List[str]]:
        """Get all APIs defined by a specific header."""
        # Normalize the header path - remove leading slashes
        header_path = header_path.lstrip('/')
        
        # Try exact match first
        if header_path in self._db["headers"]:
            return self._db["headers"][header_path]
        
        # Try with common variations
        variations = [
            header_path,
            f"ttnn/{header_path}",
            f"tt_metal/{header_path}",
            f"tt_metal/include/{header_path}",
        ]
        
        for variant in variations:
            if variant in self._db["headers"]:
                return self._db["headers"][variant]
        
        # If not found, try partial matching
        matches = []
        for stored_header in self._db["headers"]:
            if header_path in stored_header or stored_header.endswith(f"/{header_path}"):
                matches.append(stored_header)
        
        if len(matches) == 1:
            return self._db["headers"][matches[0]]
        elif len(matches) > 1:
            # Return info about multiple matches
            return {
                "error": "Multiple headers matched",
                "matches": matches,
                "hint": "Please use a more specific path"
            }
        
        return {
            "error": f"Header not found: {header_path}",
            "available_headers": self._find_similar_headers(header_path)[:5]
        }
    
    def find_header_for_api(self, api_name: str, api_type: Optional[str] = None) -> Union[str, Dict]:
        """Find which header defines a specific API."""
        
        # Helper function to extract function name from signature
        def extract_function_name(signature: str) -> str:
            """Extract function name from a full signature."""
            # Remove ALWI prefix if present
            sig = signature
            if sig.startswith("ALWI "):
                sig = sig[5:]
            
            # For functions, extract name before parentheses
            if '(' in sig:
                # Handle return type + name + params
                parts = sig.split('(')[0].strip().split()
                if parts:
                    return parts[-1]  # Last part before '(' is the function name
            
            # For other types, return as-is
            return sig.strip()
        
        if api_type:
            # Search with specific type
            # First try exact match
            key = f"{api_type}::{api_name}"
            if key in self._db["api_to_header"]:
                return self._db["api_to_header"][key]
            
            # Then try matching by extracting function names
            for stored_key, header in self._db["api_to_header"].items():
                if stored_key.startswith(f"{api_type}::"):
                    stored_api = stored_key.split("::", 1)[1]
                    if extract_function_name(stored_api) == api_name:
                        return header
        
        # Search all types
        matches = []
        api_types_to_search = ["functions", "template_functions", "classes", "structs", "enums", "typedefs", "usings"]
        
        for search_type in api_types_to_search:
            # First try exact match
            key = f"{search_type}::{api_name}"
            if key in self._db["api_to_header"]:
                matches.append((key, self._db["api_to_header"][key]))
                continue
            
            # Then search by extracting function names from signatures
            for stored_key, header in self._db["api_to_header"].items():
                if stored_key.startswith(f"{search_type}::"):
                    stored_api = stored_key.split("::", 1)[1]
                    if extract_function_name(stored_api) == api_name:
                        matches.append((stored_key, header))
                        break
        
        if not matches:
            # Try partial matching
            partial_matches = []
            for key, header in self._db["api_to_header"].items():
                api_part = key.split("::", 1)[1] if "::" in key else key
                extracted_name = extract_function_name(api_part)
                if api_name.lower() in extracted_name.lower():
                    partial_matches.append((key, header))
            
            if partial_matches:
                return {
                    "error": f"Exact match not found for '{api_name}'",
                    "partial_matches": partial_matches[:10],
                    "hint": "Use the full API name or specify the type"
                }
            else:
                return {
                    "error": f"API not found: {api_name}",
                    "hint": "Try searching for a different name or check available APIs"
                }
        
        if len(matches) == 1:
            return matches[0][1]
        
        # Multiple matches found
        return {
            "multiple_definitions": True,
            "matches": matches,
            "hint": f"Specify the type to disambiguate (e.g., api_type='template_functions' for template function '{api_name}')"
        }
    
    def _find_similar_headers(self, header_path: str) -> List[str]:
        """Find headers with similar names."""
        header_name = Path(header_path).name
        similar = []
        
        for stored_header in self._db["headers"]:
            if header_name in stored_header:
                similar.append(stored_header)
        
        return sorted(similar)


# Tool executor functions
def get_apis_from_header_tool(header_path: str, database_path: str = "include_api_database.json") -> Dict:
    """
    Tool to get all APIs defined in a header file.
    
    Args:
        header_path: Path to the header file (e.g., "ttnn/tensor/tensor.hpp")
        database_path: Path to the API database JSON file
        
    Returns:
        Dictionary containing APIs by type or error information
    """
    try:
        db = APIDatabase(database_path)
        result = db.get_apis_from_header(header_path)
        
        if "error" in result:
            return result
        
        # Format the output nicely
        output = {
            "header": header_path,
            "api_summary": {}
        }
        
        total_apis = 0
        for api_type, api_list in result.items():
            if isinstance(api_list, list) and api_list:
                output["api_summary"][api_type] = {
                    "count": len(api_list),
                    "items": api_list
                }
                total_apis += len(api_list)
        
        output["total_apis"] = total_apis
        
        return output
        
    except FileNotFoundError:
        return {
            "error": f"Database not found at: {database_path}",
            "hint": "Run build_api_database.py first to create the database"
        }
    except Exception as e:
        return {
            "error": f"Failed to query database: {str(e)}"
        }


def find_header_for_api_tool(api_name: str, api_type: Optional[str] = None, 
                           database_path: str = "include_api_database.json") -> Dict:
    """
    Tool to find which header defines a specific API.
    
    Args:
        api_name: Name of the API to search for (e.g., "Tensor", "multiply")
        api_type: Optional type hint ("functions", "classes", "structs", etc.)
        database_path: Path to the API database JSON file
        
    Returns:
        Dictionary containing the header path or error information
    """
    print(f"[Database tool] searching for {api_name}...")
    
    try:
        db = APIDatabase(database_path)
        result = db.find_header_for_api(api_name, api_type)
        
        if isinstance(result, str):
            # Found a single header
            print(f"[Database tool] Found {result} for {api_name}")
            return {
                "api": api_name,
                "type": api_type or "auto-detected",
                "defined_in": result,
                "include_path": result
            }
        else:
            # Error or multiple matches
            return result
            
    except FileNotFoundError:
        return {
            "error": f"Database not found at: {database_path}",
            "hint": "Run build_api_database.py first to create the database"
        }
    except Exception as e:
        return {
            "error": f"Failed to query database: {str(e)}"
        }


def search_apis_tool(search_term: str, database_path: str = "include_api_database.json") -> Dict:
    """
    Tool to search for APIs by partial name match.
    
    Args:
        search_term: Term to search for in API names
        database_path: Path to the API database JSON file
        
    Returns:
        Dictionary containing matching APIs and their headers
    """
    print(f"[Database tool] searching for {search_term}...")
    
    try:
        db = APIDatabase(database_path)
        
        matches = []
        for api_key, header in db._db["api_to_header"].items():
            if search_term.lower() in api_key.lower():
                api_type, api_name = api_key.split("::", 1) if "::" in api_key else ("unknown", api_key)
                matches.append({
                    "api": api_name,
                    "type": api_type,
                    "header": header
                })
        
        print(f"[Database tool] results: {matches[:3] if matches else []}")
        
        return {
            "search_term": search_term,
            "count": len(matches),
            "matches": matches[:50]  # Limit to first 50 matches
        }
        
    except Exception as e:
        return {
            "error": f"Search failed: {str(e)}"
        }


def get_database_stats_tool(database_path: str = "include_api_database.json") -> Dict:
    """
    Tool to get statistics about the API database.
    
    Args:
        database_path: Path to the API database JSON file
        
    Returns:
        Dictionary containing database statistics
    """
    try:
        db = APIDatabase(database_path)
        metadata = db.get_metadata()
        
        # Count APIs by type
        api_counts = {}
        for api_key in db._db["api_to_header"]:
            if "::" in api_key:
                api_type = api_key.split("::", 1)[0]
                api_counts[api_type] = api_counts.get(api_type, 0) + 1
        
        return {
            "database_info": metadata,
            "statistics": {
                "total_headers": len(db.get_all_headers()),
                "total_apis": len(db.get_all_apis()),
                "apis_by_type": api_counts
            }
        }
        
    except Exception as e:
        return {
            "error": f"Failed to get stats: {str(e)}"
        }


# Tool executor registry (for easy integration)
TOOL_EXECUTORS = {
    "get_apis_from_header": get_apis_from_header_tool,
    "find_header_for_api": find_header_for_api_tool,
    "search_apis": search_apis_tool,
    "get_database_stats": get_database_stats_tool
}


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query the API definition database")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subcommand: apis-in-header
    apis_parser = subparsers.add_parser("apis-in-header", help="Get APIs defined in a header")
    apis_parser.add_argument("header", help="Header path (e.g., ttnn/tensor/tensor.hpp)")
    
    # Subcommand: find-api
    find_parser = subparsers.add_parser("find-api", help="Find header that defines an API")
    find_parser.add_argument("api", help="API name to search for")
    find_parser.add_argument("--type", help="API type (functions, classes, etc.)")
    
    # Subcommand: search
    search_parser = subparsers.add_parser("search", help="Search for APIs by name")
    search_parser.add_argument("term", help="Search term")
    
    # Subcommand: stats
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    
    # Common arguments
    parser.add_argument("--db", default="include_api_database.json", help="Database path")
    
    args = parser.parse_args()
    
    if args.command == "apis-in-header":
        result = get_apis_from_header_tool(args.header, args.db)
        print(json.dumps(result, indent=2))
        
    elif args.command == "find-api":
        result = find_header_for_api_tool(args.api, args.type, args.db)
        print(json.dumps(result, indent=2))
        
    elif args.command == "search":
        result = search_apis_tool(args.term, args.db)
        print(json.dumps(result, indent=2))
        
    elif args.command == "stats":
        result = get_database_stats_tool(args.db)
        print(json.dumps(result, indent=2))
        
    else:
        parser.print_help()