"""
Configuration-driven TTNN Operation Agent
========================================

Extension of TTNNOperationAgent that supports YAML configuration files.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from ttnn_op_generator.agents.ttnn_agent import TTNNOperationAgent
from ttnn_op_generator.config.config_parser import ConfigParser
from ttnn_op_generator.config.config_workflow_builder import ConfigWorkflowBuilder


class TTNNOperationAgentConfig(TTNNOperationAgent):
    """TTNN Operation Agent with configuration file support."""
    
    def __init__(self, config_file: str, **kwargs):
        """
        Initialize agent from configuration file.
        
        Args:
            config_file: Path to YAML configuration file
            **kwargs: Additional arguments to override config
        """
        # Load configuration
        self.config_parser = ConfigParser(config_file)
        self.operation_config = self.config_parser.load()
        
        # Extract operation parameters from config
        operation_type = kwargs.get('operation_type', self.operation_config.type)
        custom_suffix = kwargs.get('custom_suffix', 'custom')
        
        # Initialize parent with config values
        super().__init__(
            operation_type=operation_type,
            custom_suffix=custom_suffix,
            **kwargs
        )
        
        # Override with config values
        self.operation_name = self.operation_config.name
        self.operation_class_name = self.operation_config.class_name
        self.python_function_name = self.operation_config.python_name
        
        if self.operation_config.settings:
            settings = self.operation_config.settings
        
        # Enable multi-stage if configured
        if settings.use_multi_stage:
            self.enable_multi_stage_generation()
            print(f"[Config] Multi-stage generation enabled")
        
        # Apply build timeout
        if hasattr(self, 'build_timeout'):
            self.build_timeout = settings.build_timeout
            
        # Store settings for use by nodes
        self.config_settings = settings
        # Build files structure from config
        self._build_files_structure()
        
        # Build and set workflow
        self._build_workflow()
        
    def _build_files_structure(self):
        """Build the files dictionary from configuration."""
        self.files = {}
        
        for file_key, file_config in self.operation_config.files.items():
            resolved_path = file_config.resolve_path(self.operation_config.variables)
            self.files[file_key] = {
                "name": resolved_path,
                "code": "",
                "config": file_config  # Store config for reference
            }
            
    def _build_workflow(self):
        """Build workflow graph from configuration."""
        builder = ConfigWorkflowBuilder(self.operation_config)
        self.workflow_graph = builder.build_graph()
        print(f"[Config Agent] Built workflow from config: {self.workflow_graph.name}")
        
    def get_file_generation_order(self):
        """Get the correct file generation order based on dependencies."""
        return self.config_parser.get_file_order()
        
    def get_file_prompt(self, file_key: str) -> str:
        """Get the prompt for a specific file from configuration."""
        file_config = self.operation_config.files.get(file_key)
        if not file_config:
            raise ValueError(f"No configuration found for file: {file_key}")
            
        # Resolve variables in prompt
        prompt = file_config.prompt
        for var, value in self.operation_config.variables.items():
            prompt = prompt.replace(f"{{{var}}}", value)
            
        return prompt
        
    def reload_config(self):
        """Reload configuration from file."""
        self.operation_config = self.config_parser.load()
        self._build_files_structure()
        self._build_workflow()
        print("[Config Agent] Configuration reloaded")