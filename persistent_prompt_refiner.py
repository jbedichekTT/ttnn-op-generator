# persistent_prompt_refiner.py
from typing import Dict, List, Tuple, Set, Any, Optional
import requests
import json
import re
from datetime import datetime
from pathlib import Path


class PersistentPromptRefiner:
    """
    Analyzes build errors and iteratively refines generation prompts to prevent recurring mistakes.
    Includes persistent storage of refinements across runs.
    """

    def __init__(
        self,
        operation_name: str,
        db_path: str = "ttnn_refinements_db.json",
        auto_save: bool = True,
        load_from_db: bool = True,
    ):
        """
        Initialize the prompt refiner with optional persistent storage.

        Args:
            operation_name: Name of the operation being generated
            db_path: Path to the JSON database file
            auto_save: Whether to automatically save refinements to database
            load_from_db: Whether to load existing refinements from database
        """
        self.operation_name = operation_name
        self.db_path = Path(db_path)
        self.auto_save = auto_save

        # Initialize database
        self.refinements_data = self._load_database() if load_from_db else self._create_empty_database()

        # Load existing refinements for this operation
        if load_from_db:
            db_data = self._get_refinements_for_operation(operation_name, include_global=True)
            self.refinements = db_data.get("refinements", self._empty_refinements())
            self.api_corrections = db_data.get("api_corrections", {})
            self.include_corrections = db_data.get("include_corrections", {})

            total_loaded = sum(len(v) for v in self.refinements.values())
            if total_loaded > 0:
                print(f"[PersistentPromptRefiner] Loaded {total_loaded} existing refinements from database")
            print(self.get_refinement_summary())  # Display active refinements
        else:
            self.refinements = self._empty_refinements()
            self.api_corrections = {}
            self.include_corrections = {}

        # Track what's new in this session
        self.session_refinements = self._empty_refinements()
        self.session_api_corrections = {}
        self.session_include_corrections = {}

        # For compatibility
        self.error_patterns = {}

    def _empty_refinements(self) -> Dict[str, List[str]]:
        """Return empty refinements structure."""
        return {
            "hpp": [],
            "cpp": [],
            "op-hpp": [],
            "op": [],
            "program-factory-hpp": [],
            "program-factory": [],
            "reader": [],
            "writer": [],
            "compute": [],
            "pybind-hpp": [],
            "pybind-cpp": [],
            "cmake": [],
        }

    def _load_database(self) -> Dict:
        """Load existing refinements from file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                print(f"[PersistentPromptRefiner] Loaded database with {len(data.get('operations', {}))} operations")
                return data
            except Exception as e:
                print(f"[PersistentPromptRefiner] Error loading database: {e}. Starting fresh.")
                return self._create_empty_database()
        else:
            return self._create_empty_database()

    def _create_empty_database(self) -> Dict:
        """Create empty database structure."""
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "global_refinements": self._empty_refinements(),
            "api_corrections": {},
            "include_corrections": {},
            "operations": {},
        }

    def analyze_errors_and_refine_prompts(
        self, error_output: str, file_contents: Dict[str, str], api_key: str, model: str
    ) -> Dict[str, List[str]]:
        """
        Analyze build errors and generate specific refinements for each file type's prompt.
        Returns a dictionary of file keys to lists of refinement instructions.
        Automatically saves to database if auto_save is enabled.
        """
        analysis_prompt = f"""You are analyzing build errors to improve code generation prompts.

        Build errors:
        {error_output}

        Current file contents:
        {self._format_file_contents(file_contents)}

        Your task is to analyze these errors and create SPECIFIC instructions that would prevent these errors in future code generation.

        Focus on:
        1. Incorrect #include statements - what should be included instead?
        2. Wrong API usage - what's the correct API signature or namespace?
        3. Missing dependencies - what needs to be defined or included first?
        4. Incorrect syntax patterns - what's the correct pattern for this framework?

        For each error pattern you identify, provide:
        - The file type it affects (hpp, cpp, op-hpp, etc.)
        - A specific instruction to add to the generation prompt
        - An example of the correct pattern

        Use the tools available to you to find the correct API usages.

        This is the output format you need to return (JSON), follow it exactly, and enclose it in ``` like this:
        ```json
        {{
            "refinements": {{
                "hpp": [
                    "Always include <ttnn/tensor/tensor.hpp> before using Tensor types",
                    "Use 'const Tensor&' for input parameters, not 'Tensor'"
                ],
                "cpp": [
                    "The correct namespace for operations is 'ttnn::operations::eltwise', not just 'ttnn'",
                    "Always validate tensor layouts using TT_FATAL with is_tensor_on_device() check"
                ],
                "op-hpp": [
                    "Device operations must inherit from 'tt::tt_metal::operation::DeviceOperation<YourOpName>'"
                ]
            }},
            "api_corrections": {{
                "create_program": "Should be 'operation::ProgramWithCallbacks create_program(...)'",
                "run": "Use 'operation::run(program, input_tensors, output_tensors)' not 'operation::launch()'"
            }},
            "include_corrections": {{
                "tensor.hpp": "ttnn/tensor/tensor.hpp",
                "operation.hpp": "ttnn/operations/eltwise/device/eltwise_op.hpp"
            }}
        }}
        ```"""

        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

        API_URL = "https://api.anthropic.com/v1/messages"

        payload = {
            "model": model,
            "max_tokens": 8192,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": [{"role": "user", "content": analysis_prompt}],
        }

        print("[PersistentPromptRefiner] Analyzing errors to improve generation prompts...")
        print(f"[DEBUG] Error output length: {len(error_output)}")
        print(f"[DEBUG] File contents keys: {list(file_contents.keys())}")

        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            print(f"[DEBUG] API Response status: {response.status_code}")
            response.raise_for_status()

            response_json = response.json()
            print(f"[DEBUG] Response keys: {response_json.keys()}")

            # Fixed content extraction - handle different response structures
            response_text = ""

            # Check if 'content' exists and is a list
            if "content" in response_json:
                content = response_json["content"]
                print(f"[DEBUG] Content type: {type(content)}")

                if isinstance(content, list):
                    # Extract text from content blocks
                    for block in content:
                        print(f"[DEBUG] Block type: {block.get('type', 'unknown')}")
                        if isinstance(block, dict):
                            if "text" in block:
                                response_text += block["text"]
                            elif "content" in block:
                                # Sometimes nested content
                                response_text += str(block["content"])
                    print(f"[DEBUG] Extracted text from list of blocks")
                elif isinstance(content, str):
                    # Direct string content
                    response_text = content
                    print(f"[DEBUG] Content is direct string")
                elif isinstance(content, dict) and "text" in content:
                    # Single content block
                    response_text = content["text"]
                    print(f"[DEBUG] Content is single block with text")

            # Alternative: check for 'completion' field (some API versions)
            elif "completion" in response_json:
                response_text = response_json["completion"]
                print(f"[DEBUG] Using 'completion' field")

            print(f"[DEBUG] Response text length: {len(response_text)}")
            print(f"[DEBUG] Response text preview: {response_text[:500]}...")

            if not response_text:
                print("[DEBUG] Empty response text, dumping full response:")
                print(json.dumps(response_json, indent=2)[:1000])

            # Extract JSON from response
            json_match = re.search(r"```json\n([\s\S]*?)\n```", response_text)
            if json_match:
                json_str = json_match.group(1)
                print(f"[DEBUG] Found JSON block, length: {len(json_str)}")

                analysis_result = json.loads(json_str)
                print(f"[DEBUG] Parsed JSON successfully")
                print(f"[DEBUG] Refinements keys: {list(analysis_result.get('refinements', {}).keys())}")

                # Track new refinements for this session
                new_refinements_count = 0
                for file_type, new_refinements in analysis_result.get("refinements", {}).items():
                    print(f"[DEBUG] Processing {file_type}: {len(new_refinements)} refinements")
                    if file_type in self.refinements:
                        existing = set(self.refinements[file_type])
                        for refinement in new_refinements:
                            if refinement not in existing:
                                self.refinements[file_type].append(refinement)
                                self.session_refinements[file_type].append(refinement)
                                existing.add(refinement)
                                new_refinements_count += 1
                                print(f"[DEBUG] Added refinement for {file_type}: {refinement[:50]}...")

                # Track new corrections
                new_api_corrections = analysis_result.get("api_corrections", {})
                new_include_corrections = analysis_result.get("include_corrections", {})

                print(f"[DEBUG] API corrections: {len(new_api_corrections)}")
                print(f"[DEBUG] Include corrections: {len(new_include_corrections)}")

                for k, v in new_api_corrections.items():
                    if k not in self.api_corrections:
                        self.session_api_corrections[k] = v
                self.api_corrections.update(new_api_corrections)

                for k, v in new_include_corrections.items():
                    if k not in self.include_corrections:
                        self.session_include_corrections[k] = v
                self.include_corrections.update(new_include_corrections)

                # Auto-save if enabled and there are new refinements
                if self.auto_save and (
                    self.session_refinements or self.session_api_corrections or self.session_include_corrections
                ):
                    print(f"[DEBUG] Auto-saving refinements...")
                    self._save_refinements(build_success=False)

                print(f"[PersistentPromptRefiner] Generated {new_refinements_count} new refinements")
                print(f"[DEBUG] Total refinements now: {sum(len(v) for v in self.refinements.values())}")
                return self.refinements
            else:
                print("[DEBUG] No JSON block found in response")
                print("[DEBUG] Attempting to find JSON without code block markers...")

                # Try to find JSON without the code block markers
                json_match = re.search(r'\{[\s\S]*"refinements"[\s\S]*\}', response_text)
                if json_match:
                    try:
                        analysis_result = json.loads(json_match.group(0))
                        print("[DEBUG] Found and parsed JSON without code blocks")
                        # Process the refinements (same code as above)
                        # ... (duplicate the processing code here)
                    except:
                        print("[DEBUG] Failed to parse extracted JSON")

                # If still no luck, use fallback
                print("[DEBUG] Using fallback refinement generation")
                return self._generate_fallback_refinements(error_output)

        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON decode error: {e}")
            print(f"[DEBUG] JSON string that failed: {json_str if 'json_str' in locals() else 'Not extracted'}")
            return self._generate_fallback_refinements(error_output)
        except Exception as e:
            print(f"[PersistentPromptRefiner] Error analyzing errors: {e}")
            print(f"[DEBUG] Exception type: {type(e).__name__}")
            import traceback

            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return self._generate_fallback_refinements(error_output)

    def _add_refinement(self, file_type: str, refinement: str):
        """Helper to add a refinement if it doesn't exist"""
        if file_type in self.refinements and refinement not in self.refinements[file_type]:
            self.refinements[file_type].append(refinement)
            self.session_refinements[file_type].append(refinement)

    def _add_api_correction(self, wrong: str, correct: str):
        """Helper to add API correction"""
        if wrong not in self.api_corrections:
            self.api_corrections[wrong] = correct
            self.session_api_corrections[wrong] = correct

    def _add_include_correction(self, wrong: str, correct: str):
        """Helper to add include correction"""
        if wrong not in self.include_corrections:
            self.include_corrections[wrong] = correct
            self.session_include_corrections[wrong] = correct

    def _format_file_contents(self, file_contents: Dict[str, str]) -> str:
        """Format file contents for the analysis prompt."""
        formatted = ""
        for file_key, content in file_contents.items():
            if content:
                # Show first 30 lines of each file
                lines = content.split("\n")[:30]
                formatted += f"\n--- {file_key} ---\n"
                formatted += "\n".join(lines)
                if len(content.split("\n")) > 30:
                    formatted += "\n... (truncated)"
        return formatted

    def apply_refinements_to_prompt(self, original_prompt: str, file_key: str) -> str:
        """
        Apply accumulated refinements to a generation prompt for a specific file.
        """
        if file_key not in self.refinements or not self.refinements[file_key]:
            return original_prompt

        refined_prompt = original_prompt + "\n\n**IMPORTANT LESSONS FROM PREVIOUS BUILD ATTEMPTS:**\n"
        refined_prompt += "Based on analysis of build errors, follow these specific requirements:\n"

        for i, refinement in enumerate(self.refinements[file_key], 1):
            refined_prompt += f"{i}. {refinement}\n"

        # Add API corrections if any
        if self.api_corrections:
            refined_prompt += "\n**CORRECT API USAGE:**\n"
            for wrong_api, correct_api in self.api_corrections.items():
                refined_prompt += f"- Instead of '{wrong_api}', use: {correct_api}\n"

        # Add include corrections if any
        if self.include_corrections and file_key in ["hpp", "cpp", "op-hpp", "op"]:
            refined_prompt += "\n**CORRECT INCLUDE PATHS:**\n"
            for wrong_inc, correct_inc in self.include_corrections.items():
                refined_prompt += f"- Replace '#include <{wrong_inc}>' with '#include <{correct_inc}>'\n"

        refined_prompt += "\nThese instructions are based on actual build errors and must be followed exactly."

        return refined_prompt

    def get_refinement_summary(self) -> str:
        """Get a summary of all refinements made so far."""
        summary = f"[Prompt Refinement Summary for {self.operation_name}]\n"
        total_refinements = sum(len(v) for v in self.refinements.values())
        summary += f"Total refinements: {total_refinements}\n"

        for file_key, refinements in self.refinements.items():
            if refinements:
                summary += f"\n{file_key}:\n"
                for r in refinements:
                    summary += f"  - {r}\n"

        return summary

    # Database methods
    def _save_refinements(self, build_success: bool = False):
        """Save refinements for the current operation."""
        timestamp = datetime.now().isoformat()

        # Initialize operation entry if it doesn't exist
        if self.operation_name not in self.refinements_data["operations"]:
            self.refinements_data["operations"][self.operation_name] = {
                "first_seen": timestamp,
                "last_updated": timestamp,
                "attempts": 0,
                "successes": 0,
                "refinements": {},
                "api_corrections": {},
                "include_corrections": {},
            }

        op_data = self.refinements_data["operations"][self.operation_name]
        op_data["last_updated"] = timestamp
        op_data["attempts"] += 1
        if build_success:
            op_data["successes"] += 1

        # Merge refinements (avoiding duplicates)
        for file_type, new_refinements in self.session_refinements.items():
            if new_refinements:  # Only process if there are new refinements
                if file_type not in op_data["refinements"]:
                    op_data["refinements"][file_type] = []

                existing = set(op_data["refinements"][file_type])
                for refinement in new_refinements:
                    if refinement not in existing:
                        op_data["refinements"][file_type].append(refinement)

        # Update API and include corrections
        op_data["api_corrections"].update(self.session_api_corrections)
        op_data["include_corrections"].update(self.session_include_corrections)

        # Update global refinements if they appear frequently
        self._update_global_refinements()

        # Save to file
        self._save_to_file()

        # Clear session tracking
        self.session_refinements = self._empty_refinements()
        self.session_api_corrections = {}
        self.session_include_corrections = {}

    def mark_build_success(self):
        """Mark that the current operation built successfully with these refinements."""
        if self.auto_save:
            self._save_refinements(build_success=True)

    def _update_global_refinements(self):
        """Promote frequently occurring refinements to global status."""
        # Count how many operations have each refinement
        refinement_counts = {}

        for op_name, op_data in self.refinements_data["operations"].items():
            for file_type, op_refinements in op_data.get("refinements", {}).items():
                for refinement in op_refinements:
                    key = f"{file_type}:{refinement}"
                    refinement_counts[key] = refinement_counts.get(key, 0) + 1

        # If a refinement appears in 3+ operations, make it global
        threshold = 3
        for key, count in refinement_counts.items():
            if count >= threshold:
                file_type, refinement = key.split(":", 1)
                if refinement not in self.refinements_data["global_refinements"].get(file_type, []):
                    self.refinements_data["global_refinements"][file_type].append(refinement)
                    print(f"[PersistentPromptRefiner] Promoted to global: {file_type} - {refinement}")

        # Update global API and include corrections
        self.refinements_data["api_corrections"].update(self.api_corrections)
        self.refinements_data["include_corrections"].update(self.include_corrections)

    def _get_refinements_for_operation(self, operation_name: str, include_global: bool = True) -> Dict:
        """Get all refinements relevant to an operation."""
        result = {"refinements": {}, "api_corrections": {}, "include_corrections": {}}

        # Start with global refinements if requested
        if include_global:
            result["refinements"] = {k: v.copy() for k, v in self.refinements_data["global_refinements"].items()}
            result["api_corrections"] = self.refinements_data["api_corrections"].copy()
            result["include_corrections"] = self.refinements_data["include_corrections"].copy()

        # Add operation-specific refinements
        if operation_name in self.refinements_data["operations"]:
            op_data = self.refinements_data["operations"][operation_name]

            # Merge refinements
            for file_type, refinements in op_data.get("refinements", {}).items():
                if file_type not in result["refinements"]:
                    result["refinements"][file_type] = []

                # Add unique refinements
                existing = set(result["refinements"][file_type])
                for r in refinements:
                    if r not in existing:
                        result["refinements"][file_type].append(r)

            # Update corrections
            result["api_corrections"].update(op_data.get("api_corrections", {}))
            result["include_corrections"].update(op_data.get("include_corrections", {}))

        return result

    def _save_to_file(self):
        """Save database to file."""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.refinements_data, f, indent=2)
            print(f"[PersistentPromptRefiner] Saved refinements to {self.db_path}")
        except Exception as e:
            print(f"[PersistentPromptRefiner] Error saving database: {e}")

    def export_human_readable(self, output_path: str = "refinements_summary.txt"):
        """Export refinements in human-readable format."""
        with open(output_path, "w") as f:
            f.write("TTNN Operation Refinements Database\n")
            f.write("=" * 50 + "\n\n")

            # Global refinements
            f.write("GLOBAL REFINEMENTS (Apply to all operations):\n")
            f.write("-" * 40 + "\n")
            for file_type, refinements in self.refinements_data["global_refinements"].items():
                if refinements:
                    f.write(f"\n{file_type}:\n")
                    for r in refinements:
                        f.write(f"  • {r}\n")

            # API Corrections
            if self.refinements_data["api_corrections"]:
                f.write("\n\nGLOBAL API CORRECTIONS:\n")
                f.write("-" * 40 + "\n")
                for wrong, correct in self.refinements_data["api_corrections"].items():
                    f.write(f"  ✗ {wrong}\n  ✓ {correct}\n\n")

            # Include Corrections
            if self.refinements_data["include_corrections"]:
                f.write("\nGLOBAL INCLUDE CORRECTIONS:\n")
                f.write("-" * 40 + "\n")
                for wrong, correct in self.refinements_data["include_corrections"].items():
                    f.write(f"  ✗ #include <{wrong}>\n  ✓ #include <{correct}>\n\n")

            # Operation-specific refinements
            f.write("\n\nOPERATION-SPECIFIC REFINEMENTS:\n")
            f.write("=" * 50 + "\n")

            for op_name, op_data in self.refinements_data["operations"].items():
                success_rate = (op_data["successes"] / op_data["attempts"] * 100) if op_data["attempts"] > 0 else 0
                f.write(f"\n{op_name}:\n")
                f.write(f"  Attempts: {op_data['attempts']}, Success Rate: {success_rate:.1f}%\n")
                f.write(f"  Last Updated: {op_data['last_updated']}\n")

                for file_type, refinements in op_data.get("refinements", {}).items():
                    if refinements:
                        f.write(f"\n  {file_type}:\n")
                        for r in refinements:
                            f.write(f"    • {r}\n")

        print(f"[PersistentPromptRefiner] Exported human-readable summary to {output_path}")
