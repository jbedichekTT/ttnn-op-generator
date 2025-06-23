#!/usr/bin/env python3
"""
Example usage of configuration-driven TTNN operation generation.
"""

import argparse
from pathlib import Path
from ttnn_op_generator.agents.ttnn_agent_config import TTNNOperationAgentConfig


def main():
    parser = argparse.ArgumentParser(description="Generate TTNN operations from config files")
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--tt-metal-path",
        default="/home/user/tt-metal",
        help="Path to TT-Metal repository"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the workflow graph"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without executing"
    )
    
    args = parser.parse_args()
    
    # Create agent from config
    print(f"Loading configuration from: {args.config}")
    agent = TTNNOperationAgentConfig(
        config_file=args.config,
        tt_metal_path=args.tt_metal_path
    )
    
    # Show operation info
    print(f"\nOperation: {agent.operation_name}")
    print(f"Type: {agent.operation_config.type}")
    print(f"Description: {agent.operation_config.description}")
    
    # Show files to be generated
    print(f"\nFiles to generate ({len(agent.files)}):")
    for file_key in agent.get_file_generation_order():
        file_info = agent.files[file_key]
        print(f"  - {file_info['name']} ({file_key})")
        
    # Visualize workflow if requested
    if args.visualize:
        from ttnn_op_generator.utils.graph_visualizer import visualize_graph
        output_path = visualize_graph(
            agent.workflow_graph,
            f"{agent.operation_name}_workflow"
        )
        print(f"\nWorkflow visualization saved to: {output_path}")
        
    # Execute if not dry run
    if not args.dry_run:
        print("\nStarting operation generation...")
        success = agent.build_operation()
        
        if success:
            print("\n✅ Operation generated successfully!")
        else:
            print("\n❌ Operation generation failed")
            return 1
    else:
        print("\n[Dry run - no files generated]")
        
    return 0


if __name__ == "__main__":
    exit(main())