"""Utilities for visualizing workflow graphs."""

from typing import Optional, Dict, Any
from ttnn_op_generator.core.workflow_graph import WorkflowGraph
from ttnn_op_generator.core.graph_nodes import (
    GenerateFileNode, BuildVerificationNode, DebugAnalysisNode,
    DebugFixNode, LoopControlNode, SetupNode
)


def visualize_graph(
    graph: WorkflowGraph, 
    output_file: str = "workflow", 
    format: str = "png",
    show_conditions: bool = True
) -> str:
    """
    Visualize the workflow graph using graphviz.
    
    Args:
        graph: The workflow graph to visualize
        output_file: Base name for output file (without extension)
        format: Output format (png, pdf, svg, etc.)
        show_conditions: Whether to show edge conditions
        
    Returns:
        Path to the generated file
    """
    try:
        import graphviz
    except ImportError:
        print("graphviz not installed. Run: pip install graphviz")
        return ""
    
    dot = graphviz.Digraph(
        name=graph.name,
        comment=f'Workflow: {graph.name}',
        graph_attr={
            'rankdir': 'TB',
            'splines': 'polyline',
            'nodesep': '0.8',
            'ranksep': '1.2'
        }
    )
    
    # Define node styles based on type
    node_styles = {
        GenerateFileNode: {
            'shape': 'box',
            'style': 'filled,rounded',
            'fillcolor': 'lightblue',
            'fontname': 'Arial'
        },
        BuildVerificationNode: {
            'shape': 'box',
            'style': 'filled',
            'fillcolor': 'lightgreen',
            'fontname': 'Arial'
        },
        DebugAnalysisNode: {
            'shape': 'ellipse',
            'style': 'filled',
            'fillcolor': 'lightyellow',
            'fontname': 'Arial'
        },
        DebugFixNode: {
            'shape': 'box',
            'style': 'filled,rounded',
            'fillcolor': 'lightcoral',
            'fontname': 'Arial'
        },
        LoopControlNode: {
            'shape': 'diamond',
            'style': 'filled',
            'fillcolor': 'lightgray',
            'fontname': 'Arial'
        },
        SetupNode: {
            'shape': 'ellipse',
            'style': 'filled',
            'fillcolor': 'lightsteelblue',
            'fontname': 'Arial'
        }
    }
    
    # Add nodes
    for name, node in graph.nodes.items():
        # Get style based on node type
        style = {}
        for node_type, node_style in node_styles.items():
            if isinstance(node, node_type):
                style = node_style
                break
        
        # Default style if no match
        if not style:
            style = {'shape': 'ellipse', 'fontname': 'Arial'}
        
        # Add description if available
        label = name
        if hasattr(node, 'description') and node.description:
            label = f"{name}\\n({node.description})"
        elif hasattr(node, 'config'):
            # Add file key for generation nodes
            if 'file_key' in node.config:
                label = f"{name}\\n[{node.config['file_key']}]"
        
        dot.node(name, label, **style)
    
    # Mark start and end nodes specially
    if graph.start_node:
        dot.node(
            '_start', 
            'START', 
            shape='circle', 
            style='filled', 
            fillcolor='green',
            fontsize='10'
        )
        dot.edge('_start', graph.start_node, style='dashed')
    
    for end_node in graph.end_nodes:
        dot.node(
            f'_end_{end_node}', 
            'END', 
            shape='doublecircle', 
            style='filled', 
            fillcolor='red',
            fontsize='10'
        )
        dot.edge(end_node, f'_end_{end_node}', style='dashed')
    
    # Add edges
    for edge in graph.edges:
        edge_attr = {'fontname': 'Arial', 'fontsize': '10'}
        
        if edge.label:
            edge_attr['label'] = edge.label
        elif show_conditions and edge.condition:
            # Try to extract condition info
            if hasattr(edge.condition, '__name__'):
                edge_attr['label'] = edge.condition.__name__
            elif 'SUCCESS' in str(edge.condition):
                edge_attr['label'] = 'success'
                edge_attr['color'] = 'green'
            elif 'FAILURE' in str(edge.condition):
                edge_attr['label'] = 'failure'
                edge_attr['color'] = 'red'
        
        dot.edge(edge.source, edge.target, **edge_attr)
    
    # Render the graph
    output_path = dot.render(output_file, format=format, cleanup=True)
    print(f"Graph visualization saved to: {output_path}")
    
    return output_path


def visualize_execution_path(
    graph: WorkflowGraph,
    execution_path: list,
    output_file: str = "execution_path",
    format: str = "png"
) -> str:
    """
    Visualize the workflow graph with the execution path highlighted.
    
    Args:
        graph: The workflow graph
        execution_path: List of node names in execution order
        output_file: Base name for output file
        format: Output format
        
    Returns:
        Path to the generated file
    """
    try:
        import graphviz
    except ImportError:
        print("graphviz not installed. Run: pip install graphviz")
        return ""
    
    dot = graphviz.Digraph(
        name=f"{graph.name}_execution",
        comment=f'Execution Path for: {graph.name}',
        graph_attr={'rankdir': 'TB', 'splines': 'polyline'}
    )
    
    # Track which edges were traversed
    traversed_edges = set()
    for i in range(len(execution_path) - 1):
        traversed_edges.add((execution_path[i], execution_path[i + 1]))
    
    # Add nodes
    for name, node in graph.nodes.items():
        style = {'shape': 'box', 'fontname': 'Arial'}
        
        # Highlight executed nodes
        if name in execution_path:
            style['style'] = 'filled'
            style['fillcolor'] = 'lightgreen'
            # Add execution order
            exec_index = execution_path.index(name) + 1
            label = f"{name}\\n[{exec_index}]"
        else:
            style['style'] = 'filled'
            style['fillcolor'] = 'lightgray'
            label = name
        
        dot.node(name, label, **style)
    
    # Add edges
    for edge in graph.edges:
        edge_attr = {'fontname': 'Arial', 'fontsize': '10'}
        
        # Highlight traversed edges
        if (edge.source, edge.target) in traversed_edges:
            edge_attr['color'] = 'green'
            edge_attr['penwidth'] = '2'
        else:
            edge_attr['color'] = 'gray'
            edge_attr['style'] = 'dashed'
        
        if edge.label:
            edge_attr['label'] = edge.label
            
        dot.edge(edge.source, edge.target, **edge_attr)
    
    # Render
    output_path = dot.render(output_file, format=format, cleanup=True)
    print(f"Execution path visualization saved to: {output_path}")
    
    return output_path


def create_workflow_documentation(graph: WorkflowGraph) -> str:
    """
    Generate markdown documentation for a workflow graph.
    
    Args:
        graph: The workflow graph to document
        
    Returns:
        Markdown string documenting the workflow
    """
    doc = [f"# Workflow: {graph.name}", ""]
    
    # Overview
    doc.append("## Overview")
    doc.append(f"- **Total Nodes**: {len(graph.nodes)}")
    doc.append(f"- **Total Edges**: {len(graph.edges)}")
    doc.append(f"- **Start Node**: {graph.start_node or 'Not defined'}")
    doc.append(f"- **End Nodes**: {', '.join(graph.end_nodes) if graph.end_nodes else 'Not defined'}")
    doc.append("")
    
    # Node documentation
    doc.append("## Nodes")
    doc.append("")
    
    for name, node in graph.nodes.items():
        doc.append(f"### {name}")
        doc.append(f"- **Type**: `{node.__class__.__name__}`")
        
        if hasattr(node, 'description') and node.description:
            doc.append(f"- **Description**: {node.description}")
            
        if hasattr(node, 'config') and node.config:
            doc.append("- **Configuration**:")
            for key, value in node.config.items():
                if not callable(value):  # Skip function configs
                    doc.append(f"  - `{key}`: {value}")
                    
        # Document incoming and outgoing edges
        incoming = graph.get_incoming_edges(name)
        outgoing = graph.get_outgoing_edges(name)
        
        if incoming:
            doc.append("- **Incoming from**: " + 
                      ", ".join(f"`{e.source}`" for e in incoming))
        if outgoing:
            doc.append("- **Outgoing to**: " + 
                      ", ".join(f"`{e.target}`" for e in outgoing))
            
        doc.append("")
    
    # Edge documentation
    doc.append("## Edges")
    doc.append("")
    doc.append("| Source | Target | Condition/Label |")
    doc.append("|--------|--------|-----------------|")
    
    for edge in graph.edges:
        condition = edge.label or "Always"
        doc.append(f"| `{edge.source}` | `{edge.target}` | {condition} |")
    
    doc.append("")
    
    # Validation results
    issues = graph.validate()
    if issues:
        doc.append("## Validation Issues")
        doc.append("")
        for issue in issues:
            doc.append(f"- ⚠️ {issue}")
        doc.append("")
    
    return "\n".join(doc)