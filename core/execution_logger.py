"""Execution logging system for workflow graphs."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict

from ttnn_op_generator.core.node_types import NodeResult, NodeStatus, NodeContext, Node


@dataclass
class NodeExecutionLog:
    """Log entry for a single node execution."""
    node_name: str
    node_type: str
    start_time: float
    end_time: float
    duration: float
    status: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    message: Optional[str] = None
    context_state: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert any non-serializable objects
        data['start_time_str'] = datetime.fromtimestamp(self.start_time).isoformat()
        data['end_time_str'] = datetime.fromtimestamp(self.end_time).isoformat()
        return data


@dataclass
class ExecutionLog:
    """Complete execution log for a workflow run."""
    execution_id: str
    workflow_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: Optional[bool] = None
    node_logs: List[NodeExecutionLog] = field(default_factory=list)
    execution_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node_log(self, log: NodeExecutionLog):
        """Add a node execution log."""
        self.node_logs.append(log)
        self.execution_path.append(log.node_name)
        
    def finalize(self, success: bool):
        """Mark the execution as complete."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'execution_id': self.execution_id,
            'workflow_name': self.workflow_name,
            'start_time': self.start_time,
            'start_time_str': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': self.end_time,
            'end_time_str': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            'duration': self.duration,
            'success': self.success,
            'node_logs': [log.to_dict() for log in self.node_logs],
            'execution_path': self.execution_path,
            'metadata': self.metadata
        }
        
    def get_node_log(self, node_name: str) -> Optional[NodeExecutionLog]:
        """Get the log for a specific node."""
        for log in self.node_logs:
            if log.node_name == node_name:
                return log
        return None


class ExecutionLogger:
    """Logger for workflow executions."""
    
    def __init__(self, log_dir: str = "workflow_logs", enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.enabled = enabled
        self.current_log: Optional[ExecutionLog] = None
        
    def start_execution(self, workflow_name: str, metadata: Dict[str, Any] = None) -> str:
        """Start logging a new execution."""
        if not self.enabled:
            return ""
            
        execution_id = str(uuid.uuid4())[:8]
        self.current_log = ExecutionLog(
            execution_id=execution_id,
            workflow_name=workflow_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        print(f"[ExecutionLogger] Started logging execution: {execution_id}")
        return execution_id
        
    def log_node_start(self, node_name: str, node_type: str, context: NodeContext) -> float:
        """Log the start of a node execution."""
        if not self.enabled or not self.current_log:
            return time.time()
            
        start_time = time.time()
        
        # Capture inputs
        inputs = {
            'config': getattr(context.agent.workflow_graph.nodes.get(node_name), 'config', {}) if hasattr(context.agent, 'workflow_graph') else {},
            'global_state': self._serialize_state(context.global_state),
            'iteration_counts': dict(context.iteration_counts),
            'previous_outputs': self._get_relevant_outputs(node_name, context)
        }
        
        # Store in context for later
        context.set_global(f'_log_start_{node_name}', {
            'start_time': start_time,
            'inputs': inputs
        })
        
        return start_time
        
    def log_node_end(self, node_name: str, node_type: str, 
                     result: NodeResult, context: NodeContext):
        """Log the end of a node execution."""
        if not self.enabled or not self.current_log:
            return
            
        end_time = time.time()
        
        # Get start info
        start_info = context.get_global(f'_log_start_{node_name}', {})
        start_time = start_info.get('start_time', end_time)
        inputs = start_info.get('inputs', {})
        
        # Create log entry
        log = NodeExecutionLog(
            node_name=node_name,
            node_type=node_type,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            status=result.status.value,
            inputs=inputs,
            outputs=self._serialize_state(result.data),
            error=result.data.get('error') if result.status == NodeStatus.FAILURE else None,
            message=result.message,
            context_state={
                'global_state': self._serialize_state(context.global_state),
                'iteration_counts': dict(context.iteration_counts)
            }
        )
        
        self.current_log.add_node_log(log)
        
        # Clean up
        context.global_state.pop(f'_log_start_{node_name}', None)
        
    def finalize_execution(self, success: bool):
        """Finalize and save the execution log."""
        if not self.enabled or not self.current_log:
            return
            
        self.current_log.finalize(success)
        
        # Save to file
        log_file = self.log_dir / f"execution_{self.current_log.execution_id}_{self.current_log.workflow_name}.json"
        
        with open(log_file, 'w') as f:
            json.dump(self.current_log.to_dict(), f, indent=2, default=str)
            
        print(f"[ExecutionLogger] Saved execution log to: {log_file}")
        
        # Also save a summary
        self._save_summary()
        
        # Clear current log
        execution_id = self.current_log.execution_id
        self.current_log = None
        
        return execution_id
        
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state data for logging."""
        serialized = {}
        
        for key, value in state.items():
            # Skip internal logging keys
            if key.startswith('_log_'):
                continue
                
            try:
                # Try to JSON serialize to check
                json.dumps(value)
                serialized[key] = value
            except:
                # Convert non-serializable objects
                if hasattr(value, '__dict__'):
                    serialized[key] = {
                        '_type': type(value).__name__,
                        '_repr': repr(value)[:200]
                    }
                else:
                    serialized[key] = {
                        '_type': type(value).__name__,
                        '_value': str(value)[:200]
                    }
                    
        return serialized
        
    def _get_relevant_outputs(self, node_name: str, context: NodeContext) -> Dict[str, Any]:
        """Get outputs from nodes that might be relevant to this node."""
        relevant = {}
        
        # Get outputs from recently executed nodes
        for name, result in list(context.node_outputs.items())[-5:]:
            if name != node_name:
                relevant[name] = {
                    'status': result.status.value,
                    'data': self._serialize_state(result.data)
                }
                
        return relevant
        
    def _save_summary(self):
        """Save a summary of all executions."""
        summary_file = self.log_dir / "execution_summary.json"
        
        # Load existing summary
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {'executions': []}
            
        # Add current execution
        summary['executions'].append({
            'execution_id': self.current_log.execution_id,
            'workflow_name': self.current_log.workflow_name,
            'timestamp': self.current_log.start_time,
            'timestamp_str': datetime.fromtimestamp(self.current_log.start_time).isoformat(),
            'duration': self.current_log.duration,
            'success': self.current_log.success,
            'node_count': len(self.current_log.node_logs),
            'execution_path': self.current_log.execution_path[:10]  # First 10 nodes
        })
        
        # Keep only last 100 executions
        summary['executions'] = summary['executions'][-100:]
        
        # Save
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def load_execution_log(self, execution_id: str) -> Optional[ExecutionLog]:
        """Load a previously saved execution log."""
        # Find the log file
        for log_file in self.log_dir.glob(f"execution_{execution_id}_*.json"):
            with open(log_file, 'r') as f:
                data = json.load(f)
                
            # Reconstruct ExecutionLog
            log = ExecutionLog(
                execution_id=data['execution_id'],
                workflow_name=data['workflow_name'],
                start_time=data['start_time'],
                end_time=data.get('end_time'),
                duration=data.get('duration'),
                success=data.get('success'),
                execution_path=data['execution_path'],
                metadata=data.get('metadata', {})
            )
            
            # Reconstruct NodeExecutionLogs
            for node_data in data['node_logs']:
                node_log = NodeExecutionLog(
                    node_name=node_data['node_name'],
                    node_type=node_data['node_type'],
                    start_time=node_data['start_time'],
                    end_time=node_data['end_time'],
                    duration=node_data['duration'],
                    status=node_data['status'],
                    inputs=node_data.get('inputs', {}),
                    outputs=node_data.get('outputs', {}),
                    error=node_data.get('error'),
                    message=node_data.get('message'),
                    context_state=node_data.get('context_state', {})
                )
                log.node_logs.append(node_log)
                
            return log
            
        return None
        
    def list_executions(self) -> List[Dict[str, Any]]:
        """List all logged executions."""
        summary_file = self.log_dir / "execution_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                return summary.get('executions', [])
                
        return []