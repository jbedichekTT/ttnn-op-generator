# epic_workflow_runner.py
"""Runner that uses Epic front-end with the workflow backend."""

from pathlib import Path
from ttnn_op_generator.front_end.front_end_parser import parse_workflow_file
from ttnn_op_generator.core.epic_to_workflow_adapter import EpicToWorkflowAdapter
from ttnn_op_generator.agents.ttnn_agent import TTNNOperationAgent
from ttnn_op_generator.core.workflow_graph import GraphExecutor
from ttnn_op_generator.core.node_types import NodeContext

class EpicWorkflowRunner:
    """Runs Epic workflows using the TTNN backend."""
    
    def __init__(self, tt_metal_path: str = "/path/to/tt-metal"):
        self.tt_metal_path = tt_metal_path
        self.adapter = EpicToWorkflowAdapter()
        
    def run_workflow(self, workflow_file: str, operation_type: str = "add"):
        """Parse and run an Epic workflow file."""
        
        # 1. Parse the Epic workflow
        print(f"Parsing workflow from: {workflow_file}")
        epic_ir = parse_workflow_file(workflow_file)
        
        # 2. Convert to WorkflowGraph
        print("Converting to workflow graph...")
        workflow_graph = self.adapter.convert(epic_ir)
        
        # 3. Create agent with the operation
        print(f"Creating agent for operation: {operation_type}")
        agent = TTNNOperationAgent(
            operation_type=operation_type,
            tt_metal_path=self.tt_metal_path
        )
        
        # 4. Set the workflow
        agent.set_workflow_graph(workflow_graph)
        
        # 5. Execute
        print("Executing workflow...")
        success = agent.build_operation()
        
        return success


# Usage example
if __name__ == "__main__":
    # Create workflow file
    workflow_content = """
        /TEMPLATE ttnn/cpp/ttnn/operations/eltwise/binary/
        /PROMPT
        Generate a complete TTNN add operation that adds two tensors element-wise.
        Use the template structure and ensure all files compile correctly.
        /RO examples/reference_add.cpp
        /RUN
        /DEBUG_LOOP
        /EXIT
        """
    
    with open("add_operation_workflow.txt", "w") as f:
        f.write(workflow_content)
    
    # Run the workflow
    runner = EpicWorkflowRunner()
    success = runner.run_workflow("add_operation_workflow.txt", operation_type="add")
    
    print(f"\nWorkflow completed: {'SUCCESS' if success else 'FAILED'}")