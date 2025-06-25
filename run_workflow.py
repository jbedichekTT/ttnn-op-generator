import os
from pathlib import Path
from ttnn_op_generator.front_end.front_end_parser import parse_workflow_file
from ttnn_op_generator.core.epic_to_workflow_adapter import EpicToWorkflowAdapter
from ttnn_op_generator.agents.ttnn_agent import TTNNOperationAgent
from ttnn_op_generator.core.workflow_graph import GraphExecutor
from ttnn_op_generator.core.node_types import NodeContext

def run_generation():
    # Configuration
    TT_METAL_PATH = "/home/user/tt-metal"  # Update this path
    WORKFLOW_FILE = "/home/user/tt-metal/ttnn_op_generator/workflow_configs/eltwise_multiply_custom.txt"
    
    # Ensure the workflow file exists
    if not Path(WORKFLOW_FILE).exists():
        print(f"Error: Workflow file {WORKFLOW_FILE} not found!")
        return False
    
    # Parse the workflow
    print("=" * 60)
    print("Parsing workflow file...")
    epic_ir = parse_workflow_file(WORKFLOW_FILE)
    
    print("=== WORKFLOW FILE CONTENT ===")
    with open(WORKFLOW_FILE, 'r') as f:
        content = f.read()
        print(content) 
    print("=== END FILE CONTENT ===")
    
    # Convert to workflow graph
    print("Converting to workflow graph...")
    adapter = EpicToWorkflowAdapter()
    workflow_graph = adapter.convert(epic_ir)
    
    # Create agent for multiply operation
    print("Creating TTNN agent for custom operation...")
    agent = TTNNOperationAgent(
        operation_type="multiply",
        tt_metal_path=TT_METAL_PATH,
        custom_suffix="custom",
        build_retries=3,
        run_tests=False,  # Set to True if you want to run tests
        enable_logging=True,
        log_dir="workflow_logs/multiply_generation"
    )
    
    # Set the workflow
    agent.set_workflow_graph(workflow_graph)
    
    # Execute the workflow
    print("=" * 60)
    print("Starting workflow execution...")
    print("=" * 60)
    
    success = agent.build_operation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Operation: eltwise_multiply_custom")
    print(f"Success: {'YES' if success else 'NO'}")
    print(f"Output directory: {agent.output_dir}")
    
    if success:
        print("\nGenerated files:")
        for file_key, file_info in agent.files.items():
            if file_info.get("code"):
                print(f"  ✓ {file_info['name']}")
    
    return success

if __name__ == "__main__":
    # Set up environment
    os.environ["ANTHROPIC_API_KEY"] = ""
    
    # Run the generation
    success = run_generation()
    
    if success:
        print("\n✅ Multiply operation generated successfully!")
        print("\nNext steps:")
        print("1. Review the generated code")
        print("2. Run tests with: ./build_metal.sh && ./test_eltwise_multiply_custom")
        print("3. Integrate into your TTNN application")
    else:
        print("\n❌ Generation failed. Check the logs for details.")