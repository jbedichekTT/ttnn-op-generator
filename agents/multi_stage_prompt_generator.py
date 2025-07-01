import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import time

from ttnn_op_generator.agents.multi_stage_generator import (
    APIReference, GenerationPlan, MultiStageGenerator
)
# Import the new database tools
from ttnn_op_generator.tools.database_query_tool import (
    get_apis_from_header_tool, 
    find_header_for_api_tool,
    search_apis_tool
)


class MultiStagePromptGenerator(MultiStageGenerator):
    """Enhanced multi-stage generation with API database queries and prompt export."""
    
    def __init__(self, agent, database_path: str = "include_api_database.json"):
        super().__init__(agent)
        self.context_dir = Path("context")
        self.context_dir.mkdir(exist_ok=True)
        self.database_path = database_path
        self.header_apis_cache = {}  # Cache of header -> APIs from database
        self.generated_files = {}  # Track generated files
        self.generated_snippets = {}  # Store key snippets from generated files
        
        # Known APIs that might not be in the database (e.g., macros)
        self.known_apis = {
            "register_operation": {
                "likely_headers": [
                    "ttnn/decorators.hpp",
                    "ttnn/cpp/ttnn/decorators.hpp"
                ],
                "api_type": "template_functions",
                "note": "Template function for registering operations"
            },
            "REGISTER_OPERATION": {
                "likely_headers": ["ttnn/decorators.hpp"],
                "note": "Macro version (if exists)"
            }
        }
        
    def generate_file_multi_stage(self, file_key: str, base_prompt: str, 
                                  dependencies: List[str] = None) -> str:
        """Enhanced multi-stage generation with prompt export."""
        print(f"\n[Enhanced Multi-Stage] Starting for {file_key}")
        
        # Stage 1: Planning with API identification
        plan = self._planning_stage(file_key, base_prompt, dependencies or [])
        
        # Export plan to file
        self._export_plan(file_key, plan)
        
        # Stage 2: Validation with database lookups
        validation_success = self._validation_stage(plan)
        
        # Stage 3: Refinement (if needed)
        #if not validation_success:
        #    plan = self._refinement_stage(plan, base_prompt)
            # Re-validate after refinement
        #    validation_success = self._validation_stage(plan)
        
        # Stage 4: Build final prompt and export
        final_prompt = self._build_execution_prompt(plan, base_prompt)
        self._export_prompt(file_key, final_prompt)
        
        # Stage 5: Execution
        code = self._execution_stage(plan, base_prompt)
        
        return code
        
    def _planning_stage(self, file_key: str, base_prompt: str, 
                        dependencies: List[str]) -> GenerationPlan:
        """Enhanced planning that identifies required APIs and headers."""
        print(f"[Enhanced Planning] Analyzing requirements for {file_key}")
        
        dep_context = self._build_dependency_context(dependencies)
        
        planning_prompt = f"""Analyze the requirements for generating {self.agent.files[file_key]['name']} for the {self.operation_name} operation.

        {base_prompt}

        {dep_context}

        Based on modern TT-Metal/TTNN architecture, create a DETAILED PLAN.

        Important considerations:
        - Use relative paths for includes (e.g., "ttnn/tensor/tensor.hpp", "ttnn/run_operation.hpp")
        - Look at the includes of other operations for examples
        - Common TTNN APIs are in the ttnn namespace
        - Common TT-Metal APIs are in tt::tt_metal namespace
        - Device operations typically use DeviceOperation<YourOpName> pattern
        - Registration might use template functions or macros (check api_type: "template_functions" for templates)

        Only search for APIs that are required by the operation.  Do NOT search for the same over and over, if you know you need it but can't find it, output it anyway and it will be fixed manually. 

        Output a JSON object with this EXACT structure:
        {{
            "required_apis": [
                {{
                    "name": "multiply",  
                    "namespace": "ttnn",
                    "api_type": "functions",  // or "template_functions", "classes", "structs", etc.
                    "purpose": "perform element-wise multiplication",
                    "expected_signature": "Tensor multiply(const Tensor& a, const Tensor& b, ...)"
                }}
            ],
            "required_includes": [
                {{
                    "path": "ttnn/tensor/tensor.hpp",
                    "reason": "Tensor type definition",
                    "expected_apis": ["Tensor", "is_tensor_on_device"]
                }}
            ],
            "namespace_imports": ["ttnn", "tt::tt_metal", "tt::tt_metal::operation"],
            "key_patterns": [
                "Use DeviceOperation<{self.operation_name}> for device op",
                "Use ProgramWithCallbacks for program creation"
            ],
            "common_mistakes_to_avoid": [
                "Don't use deprecated APIs",
                "Don't forget const& for input parameters",
                "Include proper error handling"
            ]
        }}"""

        messages = [{"role": "user", "content": planning_prompt}]
        response = self.agent.get_generation_with_tools(messages)
        
        # Check if response is valid
        if response is None:
            print("[Planning] Warning: Agent returned None response, using fallback plan")
            # Create a minimal fallback plan
            plan = GenerationPlan(
                file_key=file_key,
                file_name=self.agent.files[file_key]['name'],
                dependencies=dependencies
            )
            plan.required_apis = []
            plan.required_includes = []
            plan.namespace_imports = ["ttnn", "tt::tt_metal"]
            return plan
        
        plan = self._parse_enhanced_planning_response(response, file_key, dependencies)
    
        print(f"[Planning] Identified {len(plan.required_apis)} APIs, {len(plan.required_includes)} includes")
        return plan

    def _build_dependency_context(self, dependencies: List[str]) -> str:
        """Build context from dependent files."""
        if not dependencies:
            return "No dependencies for this file."
            
        context_parts = ["Dependencies from already generated files:"]
        
        for dep in dependencies:
            # Check if this dependency has been generated
            if hasattr(self, 'generated_files') and dep in self.generated_files:
                context_parts.append(f"\n--- {dep} ---")
                context_parts.append(f"File: {self.agent.files[dep]['name']}")
                # Add a snippet or summary of the generated file
                if hasattr(self, 'generated_snippets') and dep in self.generated_snippets:
                    context_parts.append(f"Key elements: {self.generated_snippets[dep]}")
            else:
                context_parts.append(f"\n--- {dep} (not yet generated) ---")
                    
        return "\n".join(context_parts)
    
    def _execution_stage(self, plan: GenerationPlan, base_prompt: str) -> str:
        """Execute the generation with the validated plan."""
        print(f"[Enhanced Execution] Generating code for {plan.file_name}")
        
        # Build the final execution prompt
        final_prompt = self._build_execution_prompt(plan, base_prompt)
        
        # Call the agent to generate code
        messages = [{"role": "user", "content": final_prompt}]
        response = self.agent.get_generation_with_tools(messages)
        
        # Check if response is valid
        if response is None:
            print("[Enhanced Execution] Warning: Agent returned None response")
            return f"// ERROR: Failed to generate code for {plan.file_name}\n// Agent returned no response"
        
        # Extract code from response (assuming it's wrapped in code blocks)
        code = response
        
        # Try to extract from code blocks if present
        import re
        code_match = re.search(r'```(?:cpp|c\+\+|cuda)?\n([\s\S]*?)\n```', response)
        if code_match:
            code = code_match.group(1)
        
        # Store the generated file info
        self.generated_files[plan.file_key] = code
        
        # Extract key elements for context (first few non-comment lines)
        lines = [line.strip() for line in code.split('\n') 
                if line.strip() and not line.strip().startswith('//')]
        self.generated_snippets[plan.file_key] = ' | '.join(lines[:3])
        
        return code
        
    def _validation_stage(self, plan: GenerationPlan) -> bool:
        """Enhanced validation using database tools."""
        print(f"\n[Enhanced Validation] Validating APIs and headers using database")
        
        all_valid = True
        api_to_header_map = {}  # Track where each API is found
        
        # Step 1: Validate all headers exist and get their APIs
        print(f"[Enhanced Validation] Checking {len(plan.required_includes)} headers...")
        
        valid_headers = []
        for include_info in plan.required_includes:
            if isinstance(include_info, dict):
                header_path = include_info.get('path', '')
                expected_apis = include_info.get('expected_apis', [])
            else:
                header_path = str(include_info)
                expected_apis = []
                
            # Query the database for this header
            result = get_apis_from_header_tool(header_path, self.database_path)
            
            if 'error' in result:
                print(f"  ✗ {header_path}: {result['error']}")
                if 'matches' in result:
                    print(f"    Possible matches: {', '.join(result['matches'][:3])}")
                all_valid = False
            else:
                total_apis = result.get('total_apis', 0)
                print(f"  ✓ {header_path}: {total_apis} APIs found")
                valid_headers.append(header_path)
                
                # Cache the APIs from this header
                self.header_apis_cache[header_path] = result.get('api_summary', {})
                
                # Check if expected APIs are present
                if expected_apis:
                    for expected_api in expected_apis:
                        found = self._check_api_in_header_result(expected_api, result)
                        if not found:
                            print(f"    ⚠ Expected API '{expected_api}' not found in {header_path}")
        
        # Step 2: Validate each required API can be found
        print(f"\n[Enhanced Validation] Locating {len(plan.required_apis)} required APIs...")
        
        for api in plan.required_apis:
            # Use the database to find where this API is defined
            api_type = getattr(api, 'api_type', None)
            result = find_header_for_api_tool(api.name, api_type, self.database_path)
            
            if 'error' in result:
                print(f"  ✗ {api.namespace}::{api.name} - {result['error']}")
                api.validated = False
                api.validation_error = result.get('error', 'Not found')
                
                # Try searching for partial matches
                if 'partial_matches' in result:
                    print(f"    Partial matches found:")
                    for match, header in result['partial_matches'][:3]:
                        print(f"      - {match} in {header}")
                        
                all_valid = False
            else:
                # API found!
                header = result['defined_in']
                print(f"  ✓ {api.namespace}::{api.name} found in {header}")
                
                api.validated = True
                api.include_path = header
                api_to_header_map[api.name] = header
                
                # Check if this header is already in our includes
                if header not in [self._get_header_path(inc) for inc in plan.required_includes]:
                    print(f"    ⚠ Header {header} not in required_includes - adding it")
                    plan.required_includes.append({
                        "path": header,
                        "reason": f"Required for {api.name}",
                        "expected_apis": [api.name]
                    })
        
        # Step 3: Search for any APIs that weren't found
        unvalidated_apis = [api for api in plan.required_apis if not api.validated]
        if unvalidated_apis:
            print(f"\n[Enhanced Validation] Searching for {len(unvalidated_apis)} missing APIs...")
            for api in unvalidated_apis:
                self._search_for_api_in_database(api)
                if api.validated:
                    # Re-check validation
                    if api.include_path and api.include_path not in [self._get_header_path(inc) for inc in plan.required_includes]:
                        plan.required_includes.append({
                            "path": api.include_path,
                            "reason": f"Required for {api.name} (found via search)",
                            "expected_apis": [api.name]
                        })
        
        plan.validated = all_valid
        return all_valid
        
    def _check_api_in_header_result(self, api_name: str, header_result: dict) -> bool:
        """Check if an API name exists in the header query result."""
        api_summary = header_result.get('api_summary', {})
        
        for api_type, api_info in api_summary.items():
            if isinstance(api_info, dict) and 'items' in api_info:
                for item in api_info['items']:
                    if api_name in item:
                        return True
        return False
        
    def _get_header_path(self, include_info) -> str:
        """Extract header path from include info."""
        if isinstance(include_info, dict):
            return include_info.get('path', '')
        return str(include_info)
        
    def _search_for_api_in_database(self, api: APIReference):
        """Search for an API using the database tools."""
        print(f"    Searching for '{api.name}'...")
        
        # Check if it's a known API that might not be in the database
        if api.name in self.known_apis:
            known_info = self.known_apis[api.name]
            print(f"    → Known API: {known_info['note']}")
            # Use the first likely header
            if known_info['likely_headers']:
                api.validated = True
                api.include_path = known_info['likely_headers'][0]
                api.validation_error = None
                # Set api_type if provided in known_apis
                if 'api_type' in known_info and not hasattr(api, 'api_type'):
                    api.api_type = known_info['api_type']
                print(f"    → Using known header: {api.include_path}")
                return
        
        # First try exact search with type hint
        api_type = getattr(api, 'api_type', None)
        result = find_header_for_api_tool(api.name, api_type, self.database_path)
        
        if 'error' not in result:
            # Found it!
            api.validated = True
            api.include_path = result['defined_in']
            api.namespace = api.namespace  # Keep original namespace for now
            print(f"    → Found: {api.name} in {api.include_path}")
            return
            
        # Try partial search
        search_result = search_apis_tool(api.name, self.database_path)
        
        if search_result.get('count', 0) > 0:
            matches = search_result.get('matches', [])
            
            # Look for best match based on namespace
            best_match = None
            for match in matches:
                if match['api'] == api.name:
                    # Exact name match
                    if api.namespace and api.namespace in match.get('type', ''):
                        # Namespace also matches
                        best_match = match
                        break
                    elif not best_match:
                        best_match = match
                        
            if best_match:
                api.validated = True
                api.include_path = best_match['header']
                print(f"    → Found via search: {best_match['api']} ({best_match['type']}) in {best_match['header']}")
                return
                
            # Show top matches for manual review
            print(f"    Possible matches:")
            for match in matches[:3]:
                print(f"      - {match['api']} ({match['type']}) in {match['header']}")
                
    def _build_dependency_context(self, dependencies: List[str]) -> str:
        """Build context from dependent files."""
        if not dependencies:
            return "No dependencies for this file."
            
        context_parts = ["Dependencies from already generated files:"]
        
        for dep in dependencies:
            if dep in self.generated_files:
                context_parts.append(f"\n--- {dep} ---")
                context_parts.append(f"File: {self.agent.files[dep]['name']}")
                # Add a snippet or summary of the generated file
                if hasattr(self, 'generated_snippets') and dep in self.generated_snippets:
                    context_parts.append(f"Key elements: {self.generated_snippets[dep]}")
                    
        return "\n".join(context_parts)
    
    def _refinement_stage(self, plan: GenerationPlan, base_prompt: str) -> GenerationPlan:
        """Refine the plan based on validation results."""
        print(f"\n[Enhanced Refinement] Refining plan based on validation")
        
        # Identify issues
        missing_apis = [api for api in plan.required_apis if not api.validated]
        missing_headers = []
        
        for include_info in plan.required_includes:
            header_path = self._get_header_path(include_info)
            result = get_apis_from_header_tool(header_path, self.database_path)
            if 'error' in result:
                missing_headers.append(header_path)
        
        refinement_prompt = f"""The validation found issues with the plan. Please refine it:

        MISSING APIs ({len(missing_apis)}):
        {chr(10).join(f"- {api.namespace}::{api.name} - {api.validation_error}" for api in missing_apis)}

        MISSING HEADERS ({len(missing_headers)}):
        {chr(10).join(f"- {h}" for h in missing_headers)}

        AVAILABLE ALTERNATIVES:
        {self._get_available_alternatives(missing_apis)}

        Please provide a refined plan with corrected API names and header paths.
        Use the same JSON structure as before, but with corrections based on what's actually available.

        Original context:
        {base_prompt}"""

        messages = [{"role": "user", "content": refinement_prompt}]
        response = self.agent.get_generation_with_tools(messages)
        
        # Check if response is valid
        if response is None:
            print("[Enhanced Refinement] Warning: Agent returned None response, keeping original plan")
            return plan
        
        # Parse refined plan
        refined_plan = self._parse_enhanced_planning_response(response, plan.file_key, plan.dependencies)
        
        # Merge validated APIs from original plan
        for api in plan.required_apis:
            if api.validated:
                # Keep validated APIs
                matching = [a for a in refined_plan.required_apis if a.name == api.name]
                if not matching:
                    refined_plan.required_apis.append(api)
                    
        return refined_plan
        
    def _get_available_alternatives(self, missing_apis: List[APIReference]) -> str:
        """Get available alternatives for missing APIs."""
        alternatives = []
        
        for api in missing_apis[:5]:  # Limit to first 5
            # Search for similar APIs
            search_result = search_apis_tool(api.name, self.database_path)
            if search_result.get('count', 0) > 0:
                alternatives.append(f"\nFor '{api.name}':")
                for match in search_result['matches'][:3]:
                    alternatives.append(f"  - {match['api']} ({match['type']}) in {match['header']}")
                    
        return '\n'.join(alternatives) if alternatives else "No alternatives found"
        
    def _build_execution_prompt(self, plan: GenerationPlan, base_prompt: str) -> str:
        """Build the final execution prompt with validated API information."""
        
        # Build validated API context
        api_context = self._build_validated_api_context(plan)
        
        # Build header context
        header_context = self._build_header_context(plan)
        
        # Build pattern context
        pattern_context = ""
        if hasattr(plan, 'key_patterns') and plan.key_patterns:
            pattern_context = "\nKey Implementation Patterns:\n" + \
                            '\n'.join(f"- {p}" for p in plan.key_patterns)
                            
        mistakes_context = ""
        if hasattr(plan, 'common_mistakes') and plan.common_mistakes:
            mistakes_context = "\nCommon Mistakes to Avoid:\n" + \
                             '\n'.join(f"- {m}" for m in plan.common_mistakes)
        
        final_prompt = f"""{base_prompt}

        VALIDATED API INFORMATION:
        {api_context}

        REQUIRED HEADERS:
        {header_context}

        NAMESPACE USAGE:
        {', '.join(plan.namespace_imports)}
        {pattern_context}
        {mistakes_context}

        Generate the complete implementation for {plan.file_name}.
        Ensure all includes are present and use the validated API signatures, and only those signatures and includes, don't add new ones."""

        return final_prompt
        
    def _build_validated_api_context(self, plan: GenerationPlan) -> str:
        """Build context from validated APIs."""
        lines = []
        
        for api in plan.required_apis:
            if api.validated:
                lines.append(f"- {api.namespace}::{api.name}")
                lines.append(f"  Include: {api.include_path}")
                if api.signature:
                    lines.append(f"  Signature: {api.signature}")
                if hasattr(api, 'purpose'):
                    lines.append(f"  Purpose: {api.purpose}")
                lines.append("")
                
        return '\n'.join(lines)
        
    def _build_header_context(self, plan: GenerationPlan) -> str:
        """Build context about headers and their APIs."""
        lines = []
        
        for include_info in plan.required_includes:
            if isinstance(include_info, dict):
                header_path = include_info.get('path', '')
                reason = include_info.get('reason', '')
            else:
                header_path = str(include_info)
                reason = ""
                
            lines.append(f'#include "{header_path}"')
            if reason:
                lines.append(f"  // {reason}")
                
            # Add info about what APIs this header provides
            if header_path in self.header_apis_cache:
                api_summary = self.header_apis_cache[header_path]
                api_counts = []
                for api_type, info in api_summary.items():
                    if isinstance(info, dict) and 'count' in info:
                        api_counts.append(f"{info['count']} {api_type}")
                if api_counts:
                    lines.append(f"  // Provides: {', '.join(api_counts)}")
                    
        return '\n'.join(lines)
        
    def _export_plan(self, file_key: str, plan: GenerationPlan):
        """Export the generation plan to a file."""
        plan_file = self.context_dir / f"{self.operation_name}_{file_key}_plan.json"
        
        plan_data = {
            "file_key": file_key,
            "file_name": plan.file_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "required_apis": [
                {
                    "name": api.name,
                    "namespace": api.namespace,
                    "api_type": getattr(api, 'api_type', 'unknown'),
                    "include_path": api.include_path,
                    "signature": api.signature,
                    "validated": api.validated,
                    "validation_error": api.validation_error,
                    "purpose": getattr(api, 'purpose', None)
                }
                for api in plan.required_apis
            ],
            "required_includes": plan.required_includes,
            "namespace_imports": plan.namespace_imports,
            "key_patterns": getattr(plan, 'key_patterns', []),
            "common_mistakes": getattr(plan, 'common_mistakes', []),
            "dependencies": plan.dependencies,
            "validated": plan.validated
        }
        
        with open(plan_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
            
        print(f"[Context Export] Plan saved to: {plan_file}")
        
    def _export_prompt(self, file_key: str, prompt: str):
        """Export the final execution prompt to a file."""
        prompt_file = self.context_dir / f"{self.operation_name}_{file_key}_prompt.txt"
        
        with open(prompt_file, 'w') as f:
            f.write(f"# Final Execution Prompt for {file_key}\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Operation: {self.operation_name}\n")
            f.write("#" * 80 + "\n\n")
            f.write(prompt)
            
        print(f"[Context Export] Prompt saved to: {prompt_file}")
        
    def _parse_enhanced_planning_response(self, response: str, file_key: str, 
                                        dependencies: List[str]) -> GenerationPlan:
        """Parse the enhanced planning response."""
        plan = GenerationPlan(
            file_key=file_key,
            file_name=self.agent.files[file_key]['name'],
            dependencies=dependencies
        )
        
        # Extract JSON from response
        json_match = re.search(r'```json\n([\s\S]*?)\n```', response)
        if json_match:
            try:
                plan_data = json.loads(json_match.group(1))
                
                # Parse APIs with additional metadata
                for api_data in plan_data.get("required_apis", []):
                    api = APIReference(
                        name=api_data.get("name", ""),
                        namespace=api_data.get("namespace", ""),
                    )
                    # Store additional metadata as attributes
                    if "api_type" in api_data:
                        api.api_type = api_data["api_type"]
                    if "purpose" in api_data:
                        api.purpose = api_data["purpose"]
                    if "expected_signature" in api_data:
                        api.expected_signature = api_data["expected_signature"]
                    plan.required_apis.append(api)
                    
                # Parse includes with metadata
                plan.required_includes = plan_data.get("required_includes", [])
                    
                # Parse namespaces
                plan.namespace_imports = plan_data.get("namespace_imports", [])
                
                # Store patterns and mistakes as attributes
                if "key_patterns" in plan_data:
                    plan.key_patterns = plan_data["key_patterns"]
                if "common_mistakes_to_avoid" in plan_data:
                    plan.common_mistakes = plan_data["common_mistakes_to_avoid"]
                
            except json.JSONDecodeError as e:
                print(f"[Warning] Could not parse enhanced planning JSON: {e}")
        
        return plan