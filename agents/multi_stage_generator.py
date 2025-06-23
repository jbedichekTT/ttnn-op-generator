import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import time

@dataclass
class APIReference:
    """Represents a validated API reference with its namespace and include"""
    name: str
    namespace: str = ""
    include_path: str = ""
    signature: str = ""
    validated: bool = False
    validation_error: Optional[str] = None
    usage_examples: List[str] = field(default_factory=list)

@dataclass
class GenerationPlan:
    """Represents a plan for generating a file"""
    file_key: str
    file_name: str
    required_apis: List[APIReference] = field(default_factory=list)
    required_includes: List[str] = field(default_factory=list)
    namespace_imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other file keys this depends on
    context: str = ""
    validated: bool = False
    
class MultiStageGenerator:
    """Multi-stage generation system for TTNN operations"""
    
    def __init__(self, agent):
        """Initialize with reference to the main TTNNOperationAgent"""
        self.agent = agent
        self.api_key = agent.api_key
        self.model = agent.model
        self.operation_name = agent.operation_name
        self.max_refinement_iterations = 3
        
    def generate_file_multi_stage(self, file_key: str, base_prompt: str, 
                                  dependencies: List[str] = None) -> str:
        """
        Generate a file using multi-stage approach
        
        Args:
            file_key: The key identifying which file to generate
            base_prompt: The base generation prompt
            dependencies: List of file keys this file depends on
            
        Returns:
            Generated code string
        """
        print(f"\n[Multi-Stage Generation] Starting for {file_key}")
        
        # Stage 1: Planning
        plan = self._planning_stage(file_key, base_prompt, dependencies or [])
        
        # Stage 2: Validation
        validation_success = self._validation_stage(plan)
        
        # Stage 3: Refinement (if needed)
        if not validation_success:
            plan = self._refinement_stage(plan, base_prompt)
        
        # Stage 4: Execution
        code = self._execution_stage(plan, base_prompt)
        
        return code
        
    def _planning_stage(self, file_key: str, base_prompt: str, 
                       dependencies: List[str]) -> GenerationPlan:
        """
        Stage 1: Plan what APIs and includes are needed
        """
        print(f"[Stage 1: Planning] Analyzing requirements for {file_key}")
        
        # Build context from dependencies
        dep_context = self._build_dependency_context(dependencies)
        
        planning_prompt = f"""Analyze the requirements for generating {self.agent.files[file_key]['name']} for the {self.operation_name} operation.

        {base_prompt}

        {dep_context}

        Your task is to create a DETAILED PLAN of all the APIs, includes, and namespaces needed.

        Output a JSON object with this structure:
        {{
            "required_apis": [
                {{
                    "name": "API_function_name",
                    "namespace": "expected_namespace",
                    "purpose": "why this API is needed"
                }}
            ],
            "required_includes": [
                {{
                    "path": "include/path.hpp",
                    "reason": "why this include is needed"
                }}
            ],
            "namespace_imports": ["namespace1", "namespace2"],
            "key_considerations": ["consideration1", "consideration2"]
        }}

        Be comprehensive - list ALL APIs you'll need to use, including:
        - Device management functions
        - Buffer/Tensor operations  
        - Kernel compilation functions
        - Program creation functions
        - Any TTNN-specific APIs

        Remember this is for the {self.operation_name} operation."""

        messages = [{"role": "user", "content": planning_prompt}]
        response = self.agent.get_generation_with_tools(messages)
        
        # Parse the plan from response
        plan = self._parse_planning_response(response, file_key, dependencies)
        
        print(f"[Planning Complete] Found {len(plan.required_apis)} APIs and {len(plan.required_includes)} includes")
        return plan
        
    def _validation_stage(self, plan: GenerationPlan) -> bool:
        """
        Stage 2: Validate all APIs and includes exist
        """
        print(f"[Stage 2: Validation] Validating {len(plan.required_apis)} APIs")
        
        all_valid = True
        
        # Validate APIs
        for api in plan.required_apis:
            print(f"  Validating API: {api.name}")
            
            # Prepare validation prompt with tool usage
            validation_prompt = f"""Validate that the API '{api.name}' exists in the TT-Metal/TTNN codebase.

            Expected namespace: {api.namespace}

            Use the available tools to:
            1. First use find_api_usages to find examples of how '{api.name}' is used
            2. Based on the results, identify the correct namespace and signature

            Provide the results in this JSON format:
            {{
                "found": true/false,
                "correct_namespace": "actual_namespace",
                "signature": "full_function_signature",
                "include_path": "required_include.hpp",
                "usage_example": "example usage"
            }}"""

            messages = [{"role": "user", "content": validation_prompt}]
            response = self.agent.get_generation_with_tools(messages)
            
            # Parse validation results
            validation_result = self._parse_validation_response(response, api)
            
            if not validation_result:
                all_valid = False
                api.validated = False
                api.validation_error = "Could not find API"
            else:
                api.validated = True
                if validation_result.get("correct_namespace"):
                    api.namespace = validation_result["correct_namespace"]
                if validation_result.get("include_path"):
                    api.include_path = validation_result["include_path"]
                if validation_result.get("signature"):
                    api.signature = validation_result["signature"]
                if validation_result.get("usage_example"):
                    api.usage_examples.append(validation_result["usage_example"])
                    
        # Validate includes
        include_validation_prompt = f"""Verify these include paths exist in TT-Metal:
            {json.dumps(plan.required_includes, indent=2)}

            For each include, confirm if it exists and suggest corrections if needed.
            Use the find_files_in_repository tool to verify the correctness of the include paths.

            Output JSON:
            {{
                "validated_includes": [
                    {{
                        "original": "original/path.hpp",
                        "valid": true/false,
                        "corrected": "corrected/path.hpp"  // if correction needed
                    }}
                ]
            }}"""

        messages = [{"role": "user", "content": include_validation_prompt}]
        response = self.agent.get_generation_with_tools(messages)
        
        # Update includes based on validation
        self._update_includes_from_validation(plan, response)
        
        plan.validated = all_valid
        print(f"[Validation Complete] Success: {all_valid}")
        return all_valid
        
    def _refinement_stage(self, plan: GenerationPlan, base_prompt: str) -> GenerationPlan:
        """
        Stage 3: Refine the plan based on validation results
        """
        print(f"[Stage 3: Refinement] Refining plan based on validation results")
        
        for iteration in range(self.max_refinement_iterations):
            print(f"  Refinement iteration {iteration + 1}")
            
            # Build refinement prompt with validation results
            failed_apis = [api for api in plan.required_apis if not api.validated]
            
            if not failed_apis:
                print("  All APIs validated successfully")
                break
                
            refinement_prompt = f"""The following APIs could not be validated:
            {self._format_failed_apis(failed_apis)}

            Based on the TT-Metal/TTNN API structure, please:
            1. Suggest alternative APIs that achieve the same purpose
            2. Provide the correct namespaces and includes

            Original requirement:
            {base_prompt}

            Output a JSON object with corrected API references:
            {{
                "corrected_apis": [
                    {{
                        "original_name": "failed_api_name",
                        "replacement_name": "correct_api_name",
                        "namespace": "correct_namespace",
                        "include_path": "correct/include.hpp",
                        "reason": "why this replacement works"
                    }}
                ],
                "additional_apis": [
                    // Any additional APIs discovered during refinement
                ]
            }}"""

            messages = [{"role": "user", "content": refinement_prompt}]
            response = self.agent.get_generation_with_tools(messages)
            
            # Apply refinements
            self._apply_refinements(plan, response)
            
            # Re-validate
            if self._validation_stage(plan):
                break
                
        return plan
        
    def _execution_stage(self, plan: GenerationPlan, base_prompt: str) -> str:
        """
        Stage 4: Generate the actual code using validated APIs
        """
        print(f"[Stage 4: Execution] Generating code with validated APIs")
        
        # Build enhanced prompt with validated information
        execution_prompt = self._build_execution_prompt(plan, base_prompt)
        
        messages = [{"role": "user", "content": execution_prompt}]
        response = self.agent.get_generation_with_tools(messages)
        
        # Extract code from response
        code = self.agent.parse_response(response)
        
        return code
        
    def _build_dependency_context(self, dependencies: List[str]) -> str:
        """Build context from dependent files"""
        if not dependencies:
            return ""
            
        context = "\nContext from dependencies:\n"
        for dep_key in dependencies:
            if dep_key in self.agent.files and self.agent.files[dep_key]["code"]:
                context += f"\n--- {self.agent.files[dep_key]['name']} ---\n"
                context += f"{self.agent.files[dep_key]['code']}\n"
                
        return context
        
    def _parse_planning_response(self, response: str, file_key: str, 
                                dependencies: List[str]) -> GenerationPlan:
        """Parse the planning response into a GenerationPlan object"""
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
                
                # Parse APIs
                for api_data in plan_data.get("required_apis", []):
                    api = APIReference(
                        name=api_data.get("name", ""),
                        namespace=api_data.get("namespace", ""),
                    )
                    plan.required_apis.append(api)
                    
                # Parse includes
                for inc_data in plan_data.get("required_includes", []):
                    if isinstance(inc_data, dict):
                        plan.required_includes.append(inc_data.get("path", ""))
                    else:
                        plan.required_includes.append(inc_data)
                        
                # Parse namespaces
                plan.namespace_imports = plan_data.get("namespace_imports", [])
                
            except json.JSONDecodeError:
                print("[Warning] Could not parse planning JSON")
                
        return plan
        
    def _parse_validation_response(self, response: str, api: APIReference) -> Optional[Dict]:
        """Parse validation response for an API"""
        json_match = re.search(r'```json\n([\s\S]*?)\n```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Try to extract information from text if JSON parsing fails
        result = {}
        
        # Look for namespace patterns
        ns_match = re.search(r'namespace[:\s]+(\w+(?:::\w+)*)', response, re.IGNORECASE)
        if ns_match:
            result["correct_namespace"] = ns_match.group(1)
            
        # Look for include patterns  
        inc_match = re.search(r'#include\s*[<"]([^>"]+)[>"]', response)
        if inc_match:
            result["include_path"] = inc_match.group(1)
            
        return result if result else None
        
    def _update_includes_from_validation(self, plan: GenerationPlan, response: str):
        """Update includes based on validation response"""
        json_match = re.search(r'```json\n([\s\S]*?)\n```', response)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                validated = data.get("validated_includes", [])
                
                new_includes = []
                for item in validated:
                    if item.get("valid"):
                        new_includes.append(item.get("original"))
                    elif item.get("corrected"):
                        new_includes.append(item.get("corrected"))
                        
                plan.required_includes = new_includes
            except json.JSONDecodeError:
                pass
                
    def _format_failed_apis(self, failed_apis: List[APIReference]) -> str:
        """Format failed APIs for refinement prompt"""
        result = []
        for api in failed_apis:
            result.append(f"- {api.name} (expected namespace: {api.namespace})")
            if api.validation_error:
                result.append(f"  Error: {api.validation_error}")
        return "\n".join(result)
        
    def _apply_refinements(self, plan: GenerationPlan, response: str):
        """Apply refinements from the refinement response"""
        json_match = re.search(r'```json\n([\s\S]*?)\n```', response)
        if not json_match:
            return
            
        try:
            refinements = json.loads(json_match.group(1))
            
            # Apply corrections
            for correction in refinements.get("corrected_apis", []):
                original_name = correction.get("original_name")
                
                # Find and update the API
                for api in plan.required_apis:
                    if api.name == original_name:
                        api.name = correction.get("replacement_name", api.name)
                        api.namespace = correction.get("namespace", api.namespace)
                        api.include_path = correction.get("include_path", api.include_path)
                        api.validated = False  # Reset validation status
                        
            # Add new APIs
            for new_api_data in refinements.get("additional_apis", []):
                new_api = APIReference(
                    name=new_api_data.get("name"),
                    namespace=new_api_data.get("namespace", ""),
                    include_path=new_api_data.get("include_path", "")
                )
                plan.required_apis.append(new_api)
                
        except json.JSONDecodeError:
            print("[Warning] Could not parse refinement JSON")
            
    def _build_execution_prompt(self, plan: GenerationPlan, base_prompt: str) -> str:
        """Build the final execution prompt with all validated information"""
        
        # Format validated APIs
        api_section = "VALIDATED APIs (use these exact namespaces and signatures):\n"
        for api in plan.required_apis:
            if api.validated:
                api_section += f"\n- {api.name}:\n"
                api_section += f"  Namespace: {api.namespace}\n"
                if api.signature:
                    api_section += f"  Signature: {api.signature}\n"
                if api.include_path:
                    api_section += f"  Include: #include \"{api.include_path}\"\n"
                if api.usage_examples:
                    api_section += f"  Example: {api.usage_examples[0]}\n"
                    
        # Format includes
        include_section = "REQUIRED INCLUDES:\n"
        for inc in plan.required_includes:
            include_section += f'#include "{inc}"\n'
            
        # Add includes from APIs
        api_includes = set()
        for api in plan.required_apis:
            if api.include_path and api.validated:
                api_includes.add(api.include_path)
        for inc in api_includes:
            include_section += f'#include "{inc}"\n'
            
        # Format namespaces
        namespace_section = ""
        if plan.namespace_imports:
            namespace_section = "\nUSE THESE NAMESPACE DECLARATIONS:\n"
            for ns in plan.namespace_imports:
                namespace_section += f"using namespace {ns};\n"
                
        # Build final prompt
        execution_prompt = f"""{base_prompt}

        IMPORTANT: Use the following validated APIs and includes:

        {api_section}

        {include_section}

        {namespace_section}

        Generate the complete code for {plan.file_name}.
        Ensure you use the EXACT namespaces and APIs as validated above.
        The code must compile with the TT-Metal framework.

        Provide the complete file enclosed in ```cpp``` tags."""

        return execution_prompt
        
    def generate_all_files_multi_stage(self) -> bool:
        """
        Generate all files using multi-stage approach
        """
        print(f"\n[Multi-Stage Workflow] Generating all files for {self.operation_name}")
        
        # Same order as original implementation
        file_gen_order = [
            ("hpp", [], "header file"),
            ("cpp", ["hpp"], "implementation file"),
            ("op-hpp", ["hpp"], "device operation header"),
            ("op", ["op-hpp", "hpp"], "device operation implementation"),
            ("program-factory-hpp", ["hpp"], "program factory header"),
            ("program-factory", ["program-factory-hpp", "op-hpp"], "program factory implementation"),
            ("reader", ["program-factory-hpp", "op-hpp"], "reader kernel"),
            ("writer", ["program-factory-hpp", "op-hpp"], "writer kernel"),
            ("compute", ["program-factory-hpp", "op-hpp"], "compute kernel"),
            ("pybind-hpp", ["hpp"], "Python binding header"),
            ("pybind-cpp", ["pybind-hpp", "hpp"], "Python binding implementation"),
            ("cmake", [], "CMake configuration"),
        ]
        
        success = True
        
        for file_key, dependencies, description in file_gen_order:
            print(f"\n{'='*80}")
            print(f"Generating {description}: {self.agent.files[file_key]['name']}")
            print(f"{'='*80}")
            
            # Get base prompt from agent's prompt system
            from prompts import (
                HPP_CONTEXT, CPP_CONTEXT, DEVICE_OP_CONTEXT,
                PROGRAM_FACTORY_CONTEXT, KERNEL_CONTEXT, 
                PYBIND_CONTEXT, CMAKE_CONTEXT
            )
            
            context_map = {
                "hpp": HPP_CONTEXT,
                "cpp": CPP_CONTEXT,
                "op-hpp": DEVICE_OP_CONTEXT,
                "op": DEVICE_OP_CONTEXT,
                "program-factory-hpp": PROGRAM_FACTORY_CONTEXT,
                "program-factory": PROGRAM_FACTORY_CONTEXT,
                "reader": KERNEL_CONTEXT,
                "writer": KERNEL_CONTEXT,
                "compute": KERNEL_CONTEXT,
                "pybind-hpp": PYBIND_CONTEXT,
                "pybind-cpp": PYBIND_CONTEXT,
                "cmake": CMAKE_CONTEXT,
            }
            
            context = context_map.get(file_key, "")
            base_prompt = (
                f"Generate the code for the file `{self.agent.files[file_key]['name']}` "
                f"for the `{self.operation_name}` operation.\n\n{context}"
            )
            
            # Use multi-stage generation
            try:
                code = self.generate_file_multi_stage(file_key, base_prompt, dependencies)
                self.agent.save_file(file_key, code)
            except Exception as e:
                print(f"[Error] Failed to generate {file_key}: {str(e)}")
                success = False
                # Continue with other files even if one fails
                
        return success