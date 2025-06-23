"""Workflow graph definition and execution engine."""

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from collections import deque
from ttnn_op_generator.core.node_types import Node, NodeResult, NodeStatus, NodeContext, Edge

if TYPE_CHECKING:
    from .execution_logger import ExecutionLogger


class WorkflowGraph:
    """Represents the workflow as a directed graph."""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.start_node: Optional[str] = None
        self.end_nodes: Set[str] = set()
        
    def add_node(self, node: Node) -> 'WorkflowGraph':
        """Add a node to the graph."""
        self.nodes[node.name] = node
        if not self.start_node:
            self.start_node = node.name
        return self
        
    def add_edge(self, edge: Edge) -> 'WorkflowGraph':
        """Add an edge to the graph."""
        self.edges.append(edge)
        # Validate that nodes exist
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not found in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not found in graph")
        return self
        
    def add_simple_edge(self, source: str, target: str) -> 'WorkflowGraph':
        """Add a simple unconditional edge."""
        return self.add_edge(Edge(source, target))
        
    def set_start_node(self, node_name: str) -> 'WorkflowGraph':
        """Set the starting node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        self.start_node = node_name
        return self
        
    def add_end_node(self, node_name: str) -> 'WorkflowGraph':
        """Mark a node as an end node."""
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        self.end_nodes.add(node_name)
        return self
        
    def get_next_nodes(self, current_node: str, result: NodeResult) -> List[str]:
        """Get the next nodes based on current node and result."""
        next_nodes = []
        for edge in self.edges:
            if edge.source == current_node and edge.should_traverse(result):
                next_nodes.append(edge.target)
        return next_nodes
    
    def get_incoming_edges(self, node_name: str) -> List[Edge]:
        """Get all edges leading to a node."""
        return [edge for edge in self.edges if edge.target == node_name]
    
    def get_outgoing_edges(self, node_name: str) -> List[Edge]:
        """Get all edges leaving from a node."""
        return [edge for edge in self.edges if edge.source == node_name]
    
    def validate(self) -> List[str]:
        """Validate the graph structure and return any issues."""
        issues = []
        
        if not self.start_node:
            issues.append("No start node defined")
            
        # Check for unreachable nodes
        reachable = self._get_reachable_nodes()
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            issues.append(f"Unreachable nodes: {unreachable}")
            
        # Check for nodes with no outgoing edges (potential end nodes)
        for node_name in self.nodes:
            if not self.get_outgoing_edges(node_name) and node_name not in self.end_nodes:
                # This might be intentional, so just warn
                issues.append(f"Node '{node_name}' has no outgoing edges and is not marked as end node")
                
        return issues
    
    def _get_reachable_nodes(self) -> Set[str]:
        """Get all nodes reachable from the start node."""
        if not self.start_node:
            return set()
            
        reachable = set()
        queue = deque([self.start_node])
        
        while queue:
            node = queue.popleft()
            if node in reachable:
                continue
            reachable.add(node)
            
            # Add all possible next nodes (regardless of condition)
            for edge in self.get_outgoing_edges(node):
                queue.append(edge.target)
                
        return reachable


class GraphExecutor:
    """Executes a workflow graph."""
    
    def __init__(self, graph: WorkflowGraph, max_visits_per_node: int = 10, 
                 logger: Optional['ExecutionLogger'] = None):
        self.graph = graph
        self.max_visits_per_node = max_visits_per_node
        self.logger = logger
        
    def execute(self, context: NodeContext) -> bool:
        """Execute the graph starting from the start node."""
        if not self.graph.start_node:
            print("[GraphExecutor] No start node defined")
            return False
            
        # Start execution logging
        if self.logger:
            execution_metadata = {
                'agent_operation': getattr(context.agent, 'operation_name', 'unknown'),
                'node_count': len(self.graph.nodes),
                'edge_count': len(self.graph.edges)
            }
            self.logger.start_execution(self.graph.name, execution_metadata)
            
        # Validate graph
        issues = self.graph.validate()
        if issues:
            print(f"[GraphExecutor] Graph validation warnings:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Track execution
        visit_counts = {}
        execution_path = []
        queue = deque([self.graph.start_node])
        
        while queue:
            current_node_name = queue.popleft()
            
            # Check visit count to prevent infinite loops
            visit_count = visit_counts.get(current_node_name, 0)
            if visit_count >= self.max_visits_per_node:
                print(f"[GraphExecutor] Node '{current_node_name}' visited {visit_count} times, skipping")
                continue
                
            visit_counts[current_node_name] = visit_count + 1
            execution_path.append(current_node_name)
            
            # Get the node
            node = self.graph.nodes.get(current_node_name)
            if not node:
                print(f"[GraphExecutor] Node '{current_node_name}' not found")
                continue
                
            print(f"\n{'='*60}")
            print(f"[GraphExecutor] Executing: {current_node_name}")
            print(f"{'='*60}")
            
            # Execute the node
            try:
                # Log node start
                if self.logger:
                    node_type = type(node).__name__
                    self.logger.log_node_start(current_node_name, node_type, context)
                    
                result = node.execute(context)
                context.node_outputs[current_node_name] = result
                
                # Log node end
                if self.logger:
                    self.logger.log_node_end(current_node_name, node_type, result, context)
                
                print(f"[GraphExecutor] Result: {result.status.value}")
                if result.message:
                    print(f"[GraphExecutor] Message: {result.message}")
                    
            except Exception as e:
                print(f"[GraphExecutor] Error executing node: {str(e)}")
                result = NodeResult(
                    NodeStatus.FAILURE,
                    {"error": str(e)},
                    f"Execution error: {str(e)}"
                )
                context.node_outputs[current_node_name] = result
                
                # Log error
                if self.logger:
                    node_type = type(node).__name__
                    self.logger.log_node_end(current_node_name, node_type, result, context)
            
            # Handle skip status
            if result.status == NodeStatus.SKIP:
                print(f"[GraphExecutor] Node skipped")
                continue
            
            # Get next nodes based on result
            next_nodes = self.graph.get_next_nodes(current_node_name, result)
            
            # Handle loop control nodes specially
            from .graph_nodes import LoopControlNode
            if isinstance(node, LoopControlNode):
                should_continue = result.data.get('should_continue', False)
                if not should_continue:
                    print(f"[GraphExecutor] Loop exit condition met")
                    # Find nodes that bypass the loop
                    # This is a simplified approach - in practice you might want
                    # to mark loop exit edges explicitly
                    continue
            
            # Add next nodes to queue
            for next_node in next_nodes:
                print(f"[GraphExecutor] Queueing next: {next_node}")
                queue.append(next_node)
                
        # Print execution summary
        print(f"\n[GraphExecutor] Execution complete")
        print(f"[GraphExecutor] Path: {' -> '.join(execution_path)}")
        
        # Determine success based on end nodes or specific success criteria
        success = self._determine_success(context)
        print(f"[GraphExecutor] Overall success: {success}")
        
        # Finalize logging
        if self.logger:
            execution_id = self.logger.finalize_execution(success)
            print(f"[GraphExecutor] Execution logged with ID: {execution_id}")
        
        return success
    
    def _determine_success(self, context: NodeContext) -> bool:
        """Determine if the workflow execution was successful."""
        # If end nodes are defined, check their status
        if self.graph.end_nodes:
            for end_node in self.graph.end_nodes:
                result = context.node_outputs.get(end_node)
                if result and result.status == NodeStatus.SUCCESS:
                    return True
            return False
            
        # Otherwise, look for any build success
        for node_name, result in context.node_outputs.items():
            if 'build' in node_name.lower() and result.status == NodeStatus.SUCCESS:
                return True
                
        return False


class GraphBuilder:
    """Fluent interface for building workflow graphs."""
    
    def __init__(self, name: str = "custom"):
        self.graph = WorkflowGraph(name)
        
    def add_node(self, node: Node) -> 'GraphBuilder':
        """Add a node to the graph."""
        self.graph.add_node(node)
        return self
        
    def add_edge(self, source: str, target: str, 
                 condition: Optional[callable] = None, 
                 label: Optional[str] = None) -> 'GraphBuilder':
        """Add an edge to the graph."""
        edge = Edge(source, target, condition, label)
        self.graph.add_edge(edge)
        return self
        
    def add_sequence(self, *node_names: str) -> 'GraphBuilder':
        """Add a sequence of nodes connected by simple edges."""
        for i in range(len(node_names) - 1):
            self.add_edge(node_names[i], node_names[i + 1])
        return self
        
    def set_start(self, node_name: str) -> 'GraphBuilder':
        """Set the start node."""
        self.graph.set_start_node(node_name)
        return self
        
    def add_end(self, node_name: str) -> 'GraphBuilder':
        """Add an end node."""
        self.graph.add_end_node(node_name)
        return self
        
    def build(self) -> WorkflowGraph:
        """Build and return the graph."""
        return self.graph