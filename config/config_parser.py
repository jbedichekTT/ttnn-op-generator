"""
YAML Configuration Parser for TTNN Operations
============================================

Parses YAML configuration files and provides structured access to
operation definitions, file specifications, and workflow definitions.
"""

import yaml
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field


@dataclass
class FileConfig:
    """Configuration for a single file to be generated."""
    key: str
    path: str
    description: str
    prompt: str
    dependencies: List[str] = field(default_factory=list)
    context: str = ""
    skip_if_exists: bool = False
    
    def resolve_path(self, variables: Dict[str, str]) -> str:
        """Resolve variables in the file path."""
        path = self.path
        for var, value in variables.items():
            path = path.replace(f"{{{var}}}", str(value))
        return path


@dataclass
class NodeConfig:
    """Configuration for a workflow node."""
    name: str
    type: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    next: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    

@dataclass
class WorkflowConfig:
    """Configuration for the workflow."""
    start: str
    nodes: Dict[str, NodeConfig]
    description: str = ""
    

@dataclass
class OperationConfig:
    """Complete configuration for a TTNN operation."""
    # Required fields (no defaults)
    name: str
    type: str
    class_name: str
    python_name: str
    files: Dict[str, FileConfig]
    workflow: WorkflowConfig
    
    # Optional fields (with defaults)
    description: str = ""
    templates: Dict[str, str] = field(default_factory=dict)
    variables: Dict[str, str] = field(default_factory=dict)
    

class ConfigParser:
    """Parser for TTNN operation configuration files."""
    
    def __init__(self, config_path: str):
        """Initialize parser with configuration file path."""
        self.config_path = Path(config_path)
        self.raw_config = None
        self.operation_config = None
        
    def load(self) -> OperationConfig:
        """Load and parse the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)
            
        self.operation_config = self._parse_config(self.raw_config)
        self._validate_config(self.operation_config)
        
        return self.operation_config
        
    def _parse_config(self, raw: Dict[str, Any]) -> OperationConfig:
        """Parse raw YAML into structured configuration."""
        # Parse operation metadata
        op_data = raw.get('operation', {})
        
        name = op_data.get('name', '')
        op_type = op_data.get('type', '')
        
        # Build variables for substitution
        variables = {
            'name': name,
            'type': op_type,
            'class_name': op_data.get('metadata', {}).get('class_name', ''),
            'python_name': op_data.get('metadata', {}).get('python_name', ''),
        }
        
        # Add custom variables
        variables.update(op_data.get('variables', {}))
        
        # Parse files
        files = {}
        for file_key, file_data in raw.get('files', {}).items():
            files[file_key] = FileConfig(
                key=file_key,
                path=file_data.get('path', ''),
                description=file_data.get('description', ''),
                prompt=self._process_template(file_data.get('prompt', ''), raw.get('templates', {})),
                dependencies=file_data.get('dependencies', []),
                context=file_data.get('context', ''),
                skip_if_exists=file_data.get('skip_if_exists', False)
            )
            
        # Parse workflow
        workflow_data = raw.get('workflow', {})
        nodes = {}
        
        for node_name, node_data in workflow_data.get('nodes', {}).items():
            nodes[node_name] = NodeConfig(
                name=node_name,
                type=node_data.get('type', ''),
                description=node_data.get('description', ''),
                config=self._parse_node_config(node_data, files),
                next=node_data.get('next'),
                on_success=node_data.get('on_success'),
                on_failure=node_data.get('on_failure')
            )
            
        workflow = WorkflowConfig(
            start=workflow_data.get('start', 'setup'),
            nodes=nodes,
            description=workflow_data.get('description', '')
        )
        
        return OperationConfig(
            name=name,
            type=op_type,
            class_name=variables['class_name'],
            python_name=variables['python_name'],
            description=op_data.get('description', ''),
            files=files,
            workflow=workflow,
            templates=raw.get('templates', {}),
            variables=variables
        )
        
    def _parse_node_config(self, node_data: Dict[str, Any], files: Dict[str, FileConfig]) -> Dict[str, Any]:
        """Parse node-specific configuration."""
        node_type = node_data.get('type', '')
        config = {}
        
        if node_type in ['generate', 'generate_group']:
            # File generation nodes
            generates = node_data.get('generates', [])
            if isinstance(generates, str):
                generates = [generates]
            config['files'] = generates
            config['parallel'] = node_data.get('parallel', False)
            
        elif node_type == 'debug_loop':
            config['max_attempts'] = node_data.get('max_attempts', 3)
            config['fix_strategy'] = node_data.get('fix_strategy', 'regenerate')
            
        elif node_type == 'build_verification':
            config['timeout'] = node_data.get('timeout', 1200)
            config['continue_on_warning'] = node_data.get('continue_on_warning', True)
            
        # Add any custom config
        config.update(node_data.get('config', {}))
        
        return config
        
    def _process_template(self, text: str, templates: Dict[str, str]) -> str:
        """Process template references in text."""
        # Replace template references like {{template_name}}
        def replace_template(match):
            template_name = match.group(1)
            return templates.get(template_name, match.group(0))
            
        return re.sub(r'\{\{(\w+)\}\}', replace_template, text)
        
    def _validate_config(self, config: OperationConfig):
        """Validate the configuration for consistency."""
        errors = []
        
        # Check required fields
        if not config.name:
            errors.append("Operation name is required")
        if not config.type:
            errors.append("Operation type is required")
            
        # Check workflow has start node
        if config.workflow.start not in config.workflow.nodes:
            errors.append(f"Workflow start node '{config.workflow.start}' not found")
            
        # Check all file dependencies exist
        for file_key, file_config in config.files.items():
            for dep in file_config.dependencies:
                if dep not in config.files:
                    errors.append(f"File '{file_key}' depends on unknown file '{dep}'")
                    
        # Check all node references exist
        for node_name, node in config.workflow.nodes.items():
            for next_node in [node.next, node.on_success, node.on_failure]:
                if next_node and next_node not in config.workflow.nodes and next_node != 'done':
                    errors.append(f"Node '{node_name}' references unknown node '{next_node}'")
                    
        # Check generate nodes reference valid files
        for node_name, node in config.workflow.nodes.items():
            if node.type in ['generate', 'generate_group']:
                for file_key in node.config.get('files', []):
                    if file_key not in config.files:
                        errors.append(f"Node '{node_name}' references unknown file '{file_key}'")
                        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
            
    def get_file_order(self) -> List[str]:
        """Get the correct file generation order based on dependencies."""
        files = self.operation_config.files
        
        # Topological sort
        visited = set()
        order = []
        
        def visit(file_key: str, stack: Set[str]):
            if file_key in stack:
                raise ValueError(f"Circular dependency detected: {' -> '.join(stack)} -> {file_key}")
                
            if file_key in visited:
                return
                
            stack.add(file_key)
            
            # Visit dependencies first
            for dep in files[file_key].dependencies:
                visit(dep, stack.copy())
                
            visited.add(file_key)
            order.append(file_key)
            
        # Visit all files
        for file_key in files:
            if file_key not in visited:
                visit(file_key, set())
                
        return order