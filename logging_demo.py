#!/usr/bin/env python3
"""
Complete demonstration of the logging and visualization system.

This script shows:
1. Running a workflow with automatic logging
2. Creating an interactive visualization
3. Analyzing the execution
4. Opening the results in a browser
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.ttnn_agent import TTNNOperationAgent
from utils.interactive_visualizer import InteractiveVisualizer
from utils.graph_visualizer import visualize_graph, visualize_execution_path


def run_demo():
    """Run a complete logging demonstration."""
    print("TTNN Operation Generator - Logging Demo")
    print("=" * 50)
    
    # Step 1: Create and run an operation
    print("\n1. Creating agent and running workflow...")
    
    agent = TTNNOperationAgent(
        operation_type="demo_multiply",
        tt_metal_path="/home/user/tt-metal",  # Update this path
        custom_suffix="logged_demo"
    )
    
    # Use default workflow
    agent.use_default_workflow()
    
    # Visualize the workflow structure first
    print("\n2. Creating workflow structure visualization...")
    workflow_viz_path = visualize_graph(
        agent.workflow_graph, 
        "demo_workflow_structure",
        format="png"
    )
    print(f"   Workflow structure saved to: {workflow_viz_path}")
    
    # Run the workflow
    print("\n3. Executing workflow (this will be logged automatically)...")
    success = agent.build_operation()
    
    print(f"\n   Execution {'SUCCEEDED' if success else 'FAILED'}")
    
    # Step 2: Get the execution log
    print("\n4. Loading execution log...")
    
    executions = agent.logger.list_executions()
    if not executions:
        print("   ERROR: No executions found!")
        return
        
    last_exec = executions[-1]
    exec_log = agent.logger.load_execution_log(last_exec['execution_id'])
    
    if not exec_log:
        print("   ERROR: Could not load execution log!")
        return
        
    # Step 3: Display execution statistics
    print("\n5. Execution Statistics:")
    print(f"   - Execution ID: {exec_log.execution_id}")
    print(f"   - Workflow: {exec_log.workflow_name}")
    print(f"   - Duration: {exec_log.duration:.3f} seconds")
    print(f"   - Nodes executed: {len(exec_log.node_logs)}")
    print(f"   - Success: {'Yes' if exec_log.success else 'No'}")
    
    # Show node breakdown
    status_counts = {}
    for node_log in exec_log.node_logs:
        status_counts[node_log.status] = status_counts.get(node_log.status, 0) + 1
    
    print("\n   Node Status Breakdown:")
    for status, count in status_counts.items():
        print(f"   - {status}: {count} nodes")
    
    # Show execution path
    print("\n   Execution Path:")
    for i, node_name in enumerate(exec_log.execution_path[:10]):
        node_log = exec_log.get_node_log(node_name)
        status_symbol = {
            'success': '✓',
            'failure': '✗',
            'skip': '○'
        }.get(node_log.status.lower(), '?')
        
        print(f"   {i+1:2d}. {status_symbol} {node_name} ({node_log.duration:.3f}s)")
    
    if len(exec_log.execution_path) > 10:
        print(f"   ... and {len(exec_log.execution_path) - 10} more nodes")
    
    # Step 4: Create visualizations
    print("\n6. Creating visualizations...")
    
    # Interactive HTML view
    visualizer = InteractiveVisualizer()
    html_path = visualizer.create_interactive_view(
        agent.workflow_graph,
        exec_log,
        f"demo_execution_{exec_log.execution_id}"
    )
    print(f"   Interactive view: {html_path}")
    
    # Static execution path
    path_viz = visualize_execution_path(
        agent.workflow_graph,
        exec_log.execution_path,
        f"demo_execution_path_{exec_log.execution_id}",
        format="png"
    )
    print(f"   Execution path: {path_viz}")
    
    # Step 5: Show sample node details
    print("\n7. Sample Node Details:")
    
    # Find an interesting node (e.g., one that failed or took long)
    interesting_node = None
    
    # Look for a failed node
    for node_log in exec_log.node_logs:
        if node_log.status == 'failure':
            interesting_node = node_log
            break
    
    # If no failures, find the slowest node
    if not interesting_node:
        interesting_node = max(exec_log.node_logs, key=lambda x: x.duration)
    
    if interesting_node:
        print(f"\n   Details for node: {interesting_node.node_name}")
        print(f"   - Type: {interesting_node.node_type}")
        print(f"   - Status: {interesting_node.status}")
        print(f"   - Duration: {interesting_node.duration:.3f}s")
        
        if interesting_node.message:
            print(f"   - Message: {interesting_node.message}")
            
        if interesting_node.error:
            print(f"   - Error: {interesting_node.error[:100]}...")
            
        if interesting_node.outputs:
            print(f"   - Output keys: {list(interesting_node.outputs.keys())}")
    
    # Step 6: Open in browser
    print("\n8. Opening interactive visualization...")
    
    # Convert to absolute path for browser
    abs_html_path = os.path.abspath(html_path)
    
    # Try to open in browser
    try:
        webbrowser.open(f"file://{abs_html_path}")
        print(f"   Opened in default browser")
        print(f"\n   If browser didn't open, manually open:")
        print(f"   {abs_html_path}")
    except Exception as e:
        print(f"   Could not open browser: {e}")
        print(f"   Please manually open: {abs_html_path}")
    
    # Final instructions
    print("\n" + "="*50)
    print("Demo Complete!")
    print("\nIn the interactive view you can:")
    print("- Click on any node to see its inputs/outputs")
    print("- Hover over nodes to see tooltips")
    print("- Drag to pan, scroll to zoom")
    print("- View the complete execution timeline")
    print("\nLog files are saved in:")
    print(f"  {agent.logger.log_dir}/")
    

if __name__ == "__main__":
    run_demo()