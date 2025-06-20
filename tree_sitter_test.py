#!/usr/bin/env python3
"""
Show the raw output from parse_and_analyze_code - exactly what gets fed to the LLM
"""

import json
from tools import parse_and_analyze_code

# Parse the file
file_path = "/home/user/tt-metal/ttnn/cpp/ttnn/operations/eltwise_add_custom/device/eltwise_add_custom_program_factory.cpp"

# Get the analysis - this is what the LLM receives
analysis = parse_and_analyze_code(file_path)

# Print it exactly as the LLM would see it
print("RAW OUTPUT THAT GETS FED TO THE LLM:")
print("=" * 80)
print(json.dumps(analysis, indent=2))