"""Base types and enums for the workflow graph system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum


class NodeStatus(Enum):
    """Status of a node execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    SKIP = "skip"
    RETRY = "retry"
    PENDING = "pending"


@dataclass
class NodeResult:
    """Result of a node execution."""
    status: NodeStatus
    data: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"NodeResult(status={self.status.value}, message={self.message})"


@dataclass
class NodeContext:
    """Shared context passed between nodes."""
    agent: Any  # Will be TTNNOperationAgent
    global_state: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, NodeResult] = field(default_factory=dict)
    iteration_counts: Dict[str, int] = field(default_factory=dict)
    
    def get_node_output(self, node_name: str) -> Optional[NodeResult]:
        """Safely get output from a previous node."""
        return self.node_outputs.get(node_name)
    
    def set_global(self, key: str, value: Any):
        """Set a global state value."""
        self.global_state[key] = value
        
    def get_global(self, key: str, default: Any = None) -> Any:
        """Get a global state value."""
        return self.global_state.get(key, default)


class Node(ABC):
    """Base class for all graph nodes."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        self.description = kwargs.get('description', '')
        
    @abstractmethod
    def execute(self, context: NodeContext) -> NodeResult:
        """Execute the node's logic."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class Edge:
    """Edge connecting two nodes with optional condition."""
    
    def __init__(
        self, 
        source: str, 
        target: str, 
        condition: Optional[Callable[[NodeResult], bool]] = None,
        label: Optional[str] = None
    ):
        self.source = source
        self.target = target
        self.condition = condition or (lambda r: True)
        self.label = label or ""
        
    def should_traverse(self, result: NodeResult) -> bool:
        """Check if this edge should be traversed given the result."""
        return self.condition(result)
    
    def __repr__(self) -> str:
        return f"Edge({self.source} -> {self.target}, label='{self.label}')"