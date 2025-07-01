#!/usr/bin/env python3
"""
Build a database mapping APIs to the header files that DEFINE them.

This script:
1. Scans header files in the TT-Metal repository
2. Extracts APIs defined in each header
3. Creates a mapping: API -> header that defines it
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

from tools.api_extractor import extract_apis_from_header


class IncludeAPIDatabase:
    """Database mapping APIs to their defining headers."""
    
    def __init__(self, tt_metal_path: str = None):
        self.tt_metal_path = Path(tt_metal_path or os.environ.get("TT_METAL_PATH", "/home/user/tt-metal"))
        self.database = {
            "metadata": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tt_metal_path": str(self.tt_metal_path),
                "version": "2.0"
            },
            "headers": {},  # header_path -> APIs it defines
            "api_to_header": {},  # API -> header that defines it
        }
        
    def build_database(self, scan_dirs: List[str] = None):
        """Build the database by scanning header files."""
        if scan_dirs is None:
            scan_dirs = [
                #"ttnn/cpp/ttnn",  # This captures all TTNN headers
                "ttnn/api/ttnn",
                "tt_metal/include",
                "tt_metal/hw/inc",
                "tt_metal/impl",
                "tt_eager",
                "tt_metal/api/tt-metalium"
            ]
        
        print(f"Building API Definition Database")
        print(f"Scanning directories: {scan_dirs}")
        print("=" * 80)
        
        # Step 1: Find all header files
        all_headers = self._find_all_headers(scan_dirs)
        
        # Step 2: Extract APIs from each header
        self._extract_apis_from_headers(all_headers)
        
        # Step 3: Build API -> header mapping
        self._build_api_mapping()
        
        print(f"\nDatabase built successfully!")
        print(f"Total headers analyzed: {len(self.database['headers'])}")
        print(f"Total unique APIs found: {len(self.database['api_to_header'])}")
        
    def _find_all_headers(self, scan_dirs: List[str]) -> List[Path]:
        """Find all header files in the given directories."""
        all_headers = []
        
        print(f"TT-Metal base path: {self.tt_metal_path}")
        print(f"Base path exists: {self.tt_metal_path.exists()}")
        
        for scan_dir in scan_dirs:
            dir_path = self.tt_metal_path / scan_dir
            print(f"\nChecking directory: {dir_path}")
            print(f"  Exists: {dir_path.exists()}")
            
            if not dir_path.exists():
                print(f"  Warning: Directory not found: {dir_path}")
                # Try to list what's actually in the parent directory
                parent = dir_path.parent
                if parent.exists():
                    print(f"  Contents of {parent}:")
                    try:
                        items = list(parent.iterdir())[:10]  # First 10 items
                        for item in items:
                            print(f"    - {item.name}")
                    except:
                        pass
                continue
                
            print(f"  Scanning {scan_dir}...")
            
            # Find all header files
            headers = list(dir_path.rglob("*.hpp")) + list(dir_path.rglob("*.h"))
            print(f"  Raw headers found: {len(headers)}")
            
            # Filter out certain types of headers
            filtered_headers = []
            skip_patterns = [
                "/kernels/", "/tests/", "/examples/", "_test.", 
                #"metal/", "hlk_", "llk_", "noc/", "risc/", "brisc/"
            ]
            
            for header in headers:
                header_str = str(header)
                if not any(pattern in header_str for pattern in skip_patterns):
                    filtered_headers.append(header)
                    
            all_headers.extend(filtered_headers)
            print(f"  Filtered headers: {len(filtered_headers)}")
            
            # Show a few examples
            if filtered_headers:
                print(f"  Example headers found:")
                for h in filtered_headers[:3]:
                    print(f"    - {h}")
                
        print(f"\nTotal headers found: {len(all_headers)}")
        return all_headers
        
    def _extract_apis_from_headers(self, headers: List[Path]):
        """Extract APIs defined in each header file."""
        print(f"\nExtracting APIs from {len(headers)} header files...")
        
        processed = 0
        errors = 0
        
        for header_path in headers:
            processed += 1
            
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(headers)} headers...")
            
            # Get the include path (how you'd include this file)
            include_path = self._get_include_path(header_path)
            if not include_path:
                continue
                
            # Extract APIs
            apis = extract_apis_from_header(str(header_path), str(self.tt_metal_path))
            
            if "error" in apis:
                errors += 1
                continue
            else:
                # Only store if APIs were found
                total_apis = sum(len(api_list) for api_list in apis.values() if isinstance(api_list, list))
                if total_apis > 0:
                    self.database["headers"][include_path] = apis
                
        print(f"\nAPI extraction complete. Errors: {errors}")
        
    def _get_include_path(self, header_path: Path) -> str:
        """Convert absolute path to include path."""
        header_str = str(header_path)
        
        # Try different base paths to create the include path
        base_paths = [
            (self.tt_metal_path / "ttnn" / "cpp", ""),  # Files under ttnn/cpp use includes like "ttnn/..."
            (self.tt_metal_path / "tt_metal" / "include", ""),
            (self.tt_metal_path / "tt_metal", "tt_metal/"),
            (self.tt_metal_path, ""),
        ]
        
        for base_path, prefix in base_paths:
            try:
                rel_path = header_path.relative_to(base_path)
                include_path = prefix + str(rel_path)
                # Normalize the path
                if include_path.startswith("ttnn/ttnn/"):
                    include_path = include_path[5:]  # Remove duplicate ttnn/
                return include_path
            except ValueError:
                continue
                
        return None
        
    def _build_api_mapping(self):
        """Build the mapping from each API to its defining header."""
        print("\nBuilding API -> header mapping...")
        
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
        
        api_count = 0
        duplicates = defaultdict(list)
        
        for header_path, apis in self.database["headers"].items():
            # Map each API type
            for api_type, api_list in apis.items():
                if not isinstance(api_list, list):
                    continue
                    
                for api_name in api_list:
                    # For functions and template_functions, extract just the function name
                    if api_type in ["functions", "template_functions"]:
                        simple_name = extract_function_name(api_name)
                        key = f"{api_type}::{simple_name}"
                    elif api_type == "usings":
                        # Extract the last part of qualified identifier
                        parts = api_name.split("::")
                        simple_name = parts[-1] if parts else api_name
                        key = f"usings::{simple_name}"
                    else:
                        # For other types, use the name as-is
                        key = f"{api_type}::{api_name}"
                    
                    # Check for duplicates
                    if key in self.database["api_to_header"]:
                        duplicates[key].append(header_path)
                    else:
                        self.database["api_to_header"][key] = header_path
                        api_count += 1
                        
        print(f"Mapped {api_count} unique APIs")
        
        if duplicates:
            print(f"\nFound {len(duplicates)} APIs defined in multiple headers:")
            for api, headers in list(duplicates.items())[:10]:
                all_headers = [self.database["api_to_header"][api]] + headers
                print(f"  {api}: {', '.join(all_headers)}")
                
    def save(self, output_file: str = "include_api_database.json"):
        """Save the database to a JSON file."""
        output_path = Path(output_file)
        
        # Update metadata
        self.database["metadata"]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_path, 'w') as f:
            json.dump(self.database, f, indent=2, sort_keys=True)
            
        print(f"\nDatabase saved to: {output_path}")
        
        # Also save a human-readable summary
        self._save_summary(output_path.with_suffix('.txt'))
        
    def _save_summary(self, summary_path: Path):
        """Save a human-readable summary of the database."""
        with open(summary_path, 'w') as f:
            f.write("API Definition Database Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Created: {self.database['metadata']['created']}\n")
            f.write(f"TT-Metal Path: {self.database['metadata']['tt_metal_path']}\n")
            f.write(f"Total Headers: {len(self.database['headers'])}\n")
            f.write(f"Total API Definitions: {len(self.database['api_to_header'])}\n\n")
            
            # Sample API mappings
            f.write("Sample API -> Header Mappings:\n")
            f.write("-" * 40 + "\n")
            
            # Look for common APIs
            common_apis = ["functions::Tensor", "classes::Device", "classes::Program", 
                          "classes::Buffer", "functions::multiply"]
            
            for api in common_apis:
                if api in self.database["api_to_header"]:
                    header = self.database["api_to_header"][api]
                    f.write(f"{api}:\n  -> {header}\n\n")
                    
            # Show more mappings
            f.write("\nAdditional API Mappings:\n")
            f.write("-" * 40 + "\n")
            
            sample_apis = list(self.database["api_to_header"].items())[:30]
            for api, header in sample_apis:
                f.write(f"{api}: {header}\n")
                    
        print(f"Summary saved to: {summary_path}")


def find_header_for_api(api_name: str, api_type: str = None, 
                       database_path: str = "include_api_database.json") -> str:
    """Find which header defines a specific API."""
    with open(database_path, 'r') as f:
        db = json.load(f)
        
    if api_type:
        key = f"{api_type}::{api_name}"
        return db["api_to_header"].get(key, None)
    else:
        # Search all types for EXACT matches
        for api_type in ["functions", "template_functions", "classes", "structs", "enums", "typedefs", "usings"]:
            key = f"{api_type}::{api_name}"
            if key in db["api_to_header"]:
                return db["api_to_header"][key]
        return None


def get_apis_from_header(header_path: str, 
                        database_path: str = "include_api_database.json") -> Dict:
    """Get all APIs defined by a specific header."""
    with open(database_path, 'r') as f:
        db = json.load(f)
        
    return db["headers"].get(header_path, {})


def main():
    """Build the API definition database."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build a database mapping APIs to their defining headers"
    )
    
    parser.add_argument(
        "--tt-metal-path",
        default=os.environ.get("TT_METAL_PATH", "/home/user/tt-metal"),
        help="Path to TT-Metal repository"
    )
    
    parser.add_argument(
        "--output",
        default="include_api_database.json",
        help="Output database file"
    )
    
    parser.add_argument(
        "--scan-dirs",
        nargs="+",
        help="Directories to scan (relative to tt-metal root)"
    )
    
    args = parser.parse_args()
    
    # Create database builder
    builder = IncludeAPIDatabase(args.tt_metal_path)
    
    # Build the database
    builder.build_database(args.scan_dirs)
    
    # Save it
    builder.save(args.output)
    
    # Example searches
    print("\n" + "=" * 80)
    print("Example API Searches:")
    print("=" * 80)
    
    # Search for some common APIs
    test_apis = ["Tensor", "Device", "Program", "Buffer", "multiply", "register_operation"]
    
    for api in test_apis:
        header = find_header_for_api(api, database_path=args.output)
        if header:
            print(f"\n'{api}' is defined in: {header}")
        else:
            print(f"\n'{api}' not found in database")


if __name__ == "__main__":
    main()