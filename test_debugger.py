import signal
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import os
import subprocess
import re


class TestDebugger:
    """Handles test execution and debugging for TTNN operations"""

    def __init__(self, tt_metal_path: Path, operation_name: str):
        self.tt_metal_path = tt_metal_path
        self.operation_name = operation_name
        self.test_timeout = 60  # seconds
        self.debug_markers = []

    @contextmanager
    def timeout(self, seconds):
        """Context manager for running commands with timeout"""

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test execution timed out after {seconds} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def run_test_with_capture(self, test_path: str) -> Tuple[bool, str, str]:
        """Run test with timeout and capture all output"""
        print(f"[Test Execution] Running {test_path}")

        env = os.environ.copy()
        # env["PYTHONPATH"] = str(self.tt_metal_path)
        # env["TT_METAL_HOME"] = str(self.tt_metal_path)
        # Enable verbose logging for TT-Metal
        env["TT_METAL_LOGGER_LEVEL"] = "DEBUG"
        env["LOGGER_LEVEL"] = "DEBUG"

        try:
            # Run with timeout
            with self.timeout(self.test_timeout):
                result = subprocess.run(
                    ["python", test_path], capture_output=True, text=True, env=env, cwd=self.tt_metal_path
                )

            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr

            if not success:
                stderr += f"\n[Exit Code: {result.returncode}]"

        except TimeoutError as e:
            success = False
            stdout = "Test execution timed out - possible hang or infinite loop"
            stderr = str(e) + "\n\nPossible causes:\n"
            stderr += "- Kernel deadlock or infinite loop\n"
            stderr += "- Waiting for device operation that never completes\n"
            stderr += "- Synchronization issue between host and device\n"

        return success, stdout, stderr

    def extract_test_errors(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract and categorize test errors"""
        errors = {
            "error_type": "unknown",
            "stack_trace": [],
            "assertions": [],
            "device_errors": [],
            "tensor_errors": [],
            "kernel_errors": [],
            "last_successful_operation": None,
            "debug_output": [],
        }

        all_output = stdout + "\n" + stderr
        lines = all_output.split("\n")

        # Detect error type
        if "timed out" in all_output:
            errors["error_type"] = "timeout"
        elif "Segmentation fault" in all_output:
            errors["error_type"] = "segfault"
        elif "AssertionError" in all_output:
            errors["error_type"] = "assertion"
        elif "RuntimeError" in all_output:
            errors["error_type"] = "runtime"
        elif "CUDA error" in all_output or "Device error" in all_output:
            errors["error_type"] = "device"

        # Extract stack trace
        in_traceback = False
        for line in lines:
            if "Traceback" in line:
                in_traceback = True
            elif in_traceback:
                if line.strip() and not line.startswith(" "):
                    in_traceback = False
                else:
                    errors["stack_trace"].append(line)

        # Extract assertion failures
        for line in lines:
            if "assert" in line.lower() or "ASSERT" in line:
                errors["assertions"].append(line)

        # Extract tensor/shape errors
        tensor_patterns = [
            r"shape mismatch",
            r"incompatible shapes",
            r"tensor.*dimension",
            r"Expected.*shape.*but got",
        ]
        for line in lines:
            if any(re.search(p, line, re.I) for p in tensor_patterns):
                errors["tensor_errors"].append(line)

        # Extract kernel errors
        kernel_patterns = [r"kernel.*failed", r"kernel.*error", r"compute.*error", r"program.*error"]
        for line in lines:
            if any(re.search(p, line, re.I) for p in kernel_patterns):
                errors["kernel_errors"].append(line)

        # Find last successful operation (useful for hangs)
        operation_patterns = [
            r"Running operation: (\w+)",
            r"Executing: (\w+)",
            r"DEBUG.*operation.*(\w+)",
        ]
        for line in reversed(lines):
            for pattern in operation_patterns:
                match = re.search(pattern, line)
                if match:
                    errors["last_successful_operation"] = match.group(1)
                    break
            if errors["last_successful_operation"]:
                break

        # Collect debug output
        for line in lines:
            if any(marker in line for marker in ["DEBUG", "TRACE", "Custom op:"]):
                errors["debug_output"].append(line)

        return errors
