"""Interactive visualization of workflow execution logs."""

import json
import html
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ttnn_op_generator.core.workflow_graph import WorkflowGraph
from ttnn_op_generator.core.execution_logger import ExecutionLog, NodeExecutionLog


class InteractiveVisualizer:
    """Create interactive HTML visualizations of workflow executions."""
    
    def __init__(self, output_dir: str = "workflow_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_interactive_view(
        self, 
        graph: WorkflowGraph, 
        execution_log: ExecutionLog,
        output_file: str = None
    ) -> str:
        """
        Create an interactive HTML visualization of a workflow execution.
        
        Args:
            graph: The workflow graph
            execution_log: The execution log
            output_file: Output filename (without extension)
            
        Returns:
            Path to the generated HTML file
        """
        if output_file is None:
            output_file = f"{graph.name}_{execution_log.execution_id}"
            
        html_content = self._generate_html(graph, execution_log)
        
        output_path = self.output_dir / f"{output_file}.html"
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        print(f"[InteractiveVisualizer] Created interactive view: {output_path}")
        return str(output_path)
        
    def _generate_html(self, graph: WorkflowGraph, execution_log: ExecutionLog) -> str:
        """Generate the HTML content with embedded visualization."""
        
        # Convert execution log to JavaScript-friendly format
        nodes_data = self._prepare_nodes_data(graph, execution_log)
        edges_data = self._prepare_edges_data(graph, execution_log)
        
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Workflow Execution: {workflow_name} - {execution_id}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
        }}
        #network {{
            flex: 1;
            border-right: 1px solid #ddd;
        }}
        #details {{
            width: 400px;
            overflow-y: auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        #details h2 {{
            margin-top: 0;
            color: #333;
        }}
        #details .section {{
            background: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        #details .section h3 {{
            margin-top: 0;
            color: #666;
            font-size: 16px;
        }}
        #details pre {{
            background: #f0f0f0;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
            font-size: 12px;
        }}
        #details .status {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        #details .status.success {{
            background: #4CAF50;
            color: white;
        }}
        #details .status.failure {{
            background: #f44336;
            color: white;
        }}
        #details .status.skip {{
            background: #9E9E9E;
            color: white;
        }}
        #summary {{
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }}
        .metric-label {{
            color: #666;
        }}
        .metric-value {{
            font-weight: bold;
        }}
        #legend {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 3px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div id="network"></div>
    <div id="details">
        <h2>Workflow Execution Details</h2>
        
        <div id="summary">
            <h3>Execution Summary</h3>
            <div class="metric">
                <span class="metric-label">Workflow:</span>
                <span class="metric-value">{workflow_name}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Execution ID:</span>
                <span class="metric-value">{execution_id}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Status:</span>
                <span class="status {overall_status}">{overall_status_text}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Start Time:</span>
                <span class="metric-value">{start_time}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Duration:</span>
                <span class="metric-value">{duration:.2f}s</span>
            </div>
            <div class="metric">
                <span class="metric-label">Nodes Executed:</span>
                <span class="metric-value">{nodes_executed}</span>
            </div>
        </div>
        
        <div id="node-details">
            <p style="color: #666; text-align: center;">Click on a node to see details</p>
        </div>
    </div>
    
    <div id="legend">
        <strong>Node Status</strong>
        <div class="legend-item">
            <div class="legend-color" style="background: #4CAF50;"></div>
            <span>Success</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f44336;"></div>
            <span>Failed</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #9E9E9E;"></div>
            <span>Skipped</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #E0E0E0;"></div>
            <span>Not Executed</span>
        </div>
    </div>
    
    <script>
        // Node and edge data
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        
        // Execution log data
        const executionLog = {execution_log_json};
        
        // Create network
        const container = document.getElementById('network');
        const data = {{
            nodes: new vis.DataSet(nodesData),
            edges: new vis.DataSet(edgesData)
        }};
        
        const options = {{
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 150,
                    nodeSpacing: 200
                }}
            }},
            physics: {{
                enabled: false
            }},
            nodes: {{
                shape: 'box',
                font: {{
                    size: 14
                }},
                borderWidth: 2
            }},
            edges: {{
                arrows: 'to',
                smooth: {{
                    type: 'cubicBezier',
                    roundness: 0.4
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 0
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        
        // Handle node clicks
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                showNodeDetails(nodeId);
            }}
        }});
        
        // Show node details
        function showNodeDetails(nodeId) {{
            const nodeLog = executionLog.node_logs.find(log => log.node_name === nodeId);
            const detailsDiv = document.getElementById('node-details');
            
            if (!nodeLog) {{
                detailsDiv.innerHTML = `
                    <div class="section">
                        <h3>Node: ${{nodeId}}</h3>
                        <p style="color: #666;">This node was not executed</p>
                    </div>
                `;
                return;
            }}
            
            let html = `
                <div class="section">
                    <h3>Node: ${{nodeId}}</h3>
                    <div class="metric">
                        <span class="metric-label">Type:</span>
                        <span class="metric-value">${{nodeLog.node_type}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="status ${{nodeLog.status.toLowerCase()}}">${{nodeLog.status}}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Duration:</span>
                        <span class="metric-value">${{nodeLog.duration.toFixed(3)}}s</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Start Time:</span>
                        <span class="metric-value">${{new Date(nodeLog.start_time * 1000).toLocaleTimeString()}}</span>
                    </div>
                </div>
            `;
            
            // Add message if present
            if (nodeLog.message) {{
                html += `
                    <div class="section">
                        <h3>Message</h3>
                        <p>${{escapeHtml(nodeLog.message)}}</p>
                    </div>
                `;
            }}
            
            // Add error if present
            if (nodeLog.error) {{
                html += `
                    <div class="section" style="border-left: 3px solid #f44336;">
                        <h3>Error</h3>
                        <pre>${{escapeHtml(nodeLog.error)}}</pre>
                    </div>
                `;
            }}
            
            // Add inputs
            if (nodeLog.inputs && Object.keys(nodeLog.inputs).length > 0) {{
                html += `
                    <div class="section">
                        <h3>Inputs</h3>
                        <pre>${{escapeHtml(JSON.stringify(nodeLog.inputs, null, 2))}}</pre>
                    </div>
                `;
            }}
            
            // Add outputs
            if (nodeLog.outputs && Object.keys(nodeLog.outputs).length > 0) {{
                html += `
                    <div class="section">
                        <h3>Outputs</h3>
                        <pre>${{escapeHtml(JSON.stringify(nodeLog.outputs, null, 2))}}</pre>
                    </div>
                `;
            }}
            
            // Add context state
            if (nodeLog.context_state && Object.keys(nodeLog.context_state).length > 0) {{
                html += `
                    <div class="section">
                        <h3>Context State</h3>
                        <pre>${{escapeHtml(JSON.stringify(nodeLog.context_state, null, 2))}}</pre>
                    </div>
                `;
            }}
            
            detailsDiv.innerHTML = html;
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        // Auto-select first executed node
        if (executionLog.execution_path.length > 0) {{
            setTimeout(() => {{
                showNodeDetails(executionLog.execution_path[0]);
            }}, 100);
        }}
    </script>
</body>
</html>"""
        
        # Format the template
        return html_template.format(
            workflow_name=graph.name,
            execution_id=execution_log.execution_id,
            overall_status='success' if execution_log.success else 'failure',
            overall_status_text='Success' if execution_log.success else 'Failed',
            start_time=datetime.fromtimestamp(execution_log.start_time).strftime('%Y-%m-%d %H:%M:%S'),
            duration=execution_log.duration or 0,
            nodes_executed=len(execution_log.node_logs),
            nodes_json=json.dumps(nodes_data),
            edges_json=json.dumps(edges_data),
            execution_log_json=json.dumps(execution_log.to_dict())
        )
        
    def _prepare_nodes_data(self, graph: WorkflowGraph, execution_log: ExecutionLog) -> List[Dict]:
        """Prepare node data for visualization."""
        nodes_data = []
        
        # Map of node execution status
        node_status = {}
        for log in execution_log.node_logs:
            node_status[log.node_name] = log.status.lower()
            
        # Color scheme
        status_colors = {
            'success': '#4CAF50',
            'failure': '#f44336',
            'skip': '#9E9E9E',
            'pending': '#FFC107'
        }
        
        for node_name, node in graph.nodes.items():
            status = node_status.get(node_name, 'not_executed')
            
            # Get execution details
            node_log = execution_log.get_node_log(node_name)
            
            # Build title
            title_parts = [f"Node: {node_name}"]
            if node_log:
                title_parts.append(f"Status: {node_log.status}")
                title_parts.append(f"Duration: {node_log.duration:.3f}s")
                if node_log.message:
                    title_parts.append(f"Message: {node_log.message}")
                    
            # Determine color
            if status == 'not_executed':
                color = '#E0E0E0'
                border_color = '#BDBDBD'
            else:
                color = status_colors.get(status, '#E0E0E0')
                border_color = color
                
            # Add execution order if executed
            label = node_name
            if node_name in execution_log.execution_path:
                order = execution_log.execution_path.index(node_name) + 1
                label = f"{node_name}\n[{order}]"
                
            nodes_data.append({
                'id': node_name,
                'label': label,
                'title': '\n'.join(title_parts),
                'color': {
                    'background': color,
                    'border': border_color
                },
                'font': {
                    'color': 'white' if status != 'not_executed' else '#666'
                }
            })
            
        return nodes_data
        
    def _prepare_edges_data(self, graph: WorkflowGraph, execution_log: ExecutionLog) -> List[Dict]:
        """Prepare edge data for visualization."""
        edges_data = []
        
        # Track which edges were traversed
        traversed = set()
        path = execution_log.execution_path
        for i in range(len(path) - 1):
            traversed.add((path[i], path[i + 1]))
            
        for edge in graph.edges:
            edge_tuple = (edge.source, edge.target)
            
            # Style based on whether edge was traversed
            if edge_tuple in traversed:
                color = '#4CAF50'
                width = 3
                dashes = False
            else:
                color = '#CCCCCC'
                width = 1
                dashes = True
                
            edge_data = {
                'from': edge.source,
                'to': edge.target,
                'color': color,
                'width': width,
                'dashes': dashes
            }
            
            # Add label if present
            if edge.label:
                edge_data['label'] = edge.label
                edge_data['font'] = {'size': 12}
                
            edges_data.append(edge_data)
            
        return edges_data