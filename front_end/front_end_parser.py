import networkx as nx
from enum import Enum
from typing import List, Tuple, Any
import os
from pathlib import Path


# 1. Opcode Enum and Markers
class Opcode(Enum):
    """
    Defines the operation codes for nodes in the workflow graph.
    """
    TEMPLATE = "TEMPLATE"
    PROMPT = "PROMPT"
    READ_ONLY = "READ_ONLY"
    RUN = "RUN"
    DEBUG_LOOP = "DEBUG_LOOP"
    CONDITIONAL = "CONDITIONAL"
    EXIT = "EXIT"
    COMMAND = "RUN_COMMAND"

# Front End parser: parse User input prompt.txt file markers
FE_MARKERS = [
    "/TEMPLATE",
    "/PROMPT",
    "/RUN",
    "/DEBUG_LOOP",
    "/EXIT",
    "/MULTI_STAGE"
]

# Markers within a node after the IR_MARKER parsing, that trigger second order effects
INTRA_NODE_MARKERS = ["/RO"]

# 2. EpicIR Class
class EpicIR():
    """
    Represents the Intermediate Representation (IR) of the workflow as a directed graph.
    """
    first_node = None
    graph = None
    node_counter = 0

    def __init__(self):
        """Initializes an empty EpicIR graph."""
        self.graph = nx.DiGraph()
        self.node_counter = 0 # Reset counter for each new EpicIR instance

    def set_first_node(self, first_node: str):
        """Sets the first node of the graph."""
        self.first_node = first_node

    def get_next_node_counter(self) -> str:
        """Increments and returns the next node counter as a string."""
        self.node_counter += 1
        return str(self.node_counter)

    def add_node(self, opcode: Opcode, contents: dict = {}, name: str = None) -> str:
        """
        Adds a new node to the graph.

        Args:
            opcode (Opcode): The opcode of the node.
            contents (dict): A dictionary containing the node's contents.
            name (str, optional): A specific name for the node. If None, a default name
                                  based on opcode and counter is generated.

        Returns:
            str: The name of the newly added node.
        """
        if name is None:
            node_name = opcode.value.lower() + "_" + self.get_next_node_counter()
        else:
            node_name = name + "_" + self.get_next_node_counter()

        self.graph.add_node(node_name, opcode=opcode, contents=contents)
        if self.first_node is None:
            self.first_node = node_name
        return node_name

# 3. Utility Functions for Parsing
def get_string_from_file(file_path: str) -> str:
    """
    Reads the content of a file and returns it as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def parse_ir_markers(input_file_content: str) -> List[str]:
    """
    Parses a text string containing markers and extracts sections.
    Each section starts with a marker and goes until the next marker or end of content.
    Markers are included in the section.
    """
    if not any(marker in input_file_content for marker in FE_MARKERS):
        return []

    marker_positions = []
    for marker in FE_MARKERS:
        pos = 0
        while True:
            pos = input_file_content.find(marker, pos)
            if pos == -1:
                break
            marker_positions.append((pos, marker))
            pos += len(marker)

    marker_positions.sort()

    sections = []
    for i in range(len(marker_positions)):
        start_pos = marker_positions[i][0]
        end_pos = marker_positions[i + 1][0] if i + 1 < len(marker_positions) else len(input_file_content)
        section = input_file_content[start_pos:end_pos].strip()
        if section:
            sections.append(section)
    return sections

def extract_next_word_after_marker(input_string: str, marker: str) -> List[str]:
    """
    Extracts all words that come immediately after each instance of a specified marker.
    """
    if not input_string or marker not in input_string:
        return []

    found_words = []
    start_index = 0

    while True:
        marker_index = input_string.find(marker, start_index)
        if marker_index == -1:
            break

        remaining_text = input_string[marker_index + len(marker):].strip()
        if remaining_text:
            words = remaining_text.split()
            if words:
                found_words.append(words[0])

        start_index = marker_index + len(marker)
    return found_words

# 4. Graph Building Logic
def build_default_run_node(epic: EpicIR) -> str:
    """
    Builds a default RUN node for the EpicIR graph.
    """
    new_node = epic.add_node(opcode=Opcode.RUN, contents={})
    return new_node

def add_simple_edge(epic: EpicIR, previous_node: str, new_node: str) -> str:
    """
    Adds a simple edge between two nodes and updates the previous node reference.
    """
    if previous_node is not None:
        epic.graph.add_edge(previous_node, new_node)
    previous_node = new_node
    return previous_node

def parse_workflow_file(input_file_path: str) -> EpicIR:
    """
    Parses an input text file containing workflow markers and constructs an EpicIR graph.

    Args:
        input_file_path (str): The path to the input text file.

    Returns:
        EpicIR: An EpicIR object representing the parsed workflow graph.
    """
    epic = EpicIR()
    input_content = get_string_from_file(input_file_path)
    ir_marker_list = parse_ir_markers(input_content)

    previous_node = None
    for ir_marker in ir_marker_list:
        ir_marker_first_word = ir_marker.split()[0]
        ir_marker_without_first_word = " ".join(ir_marker.split()[1:])

        if ir_marker_first_word == "/TEMPLATE":
            new_node = epic.add_node(opcode=Opcode.TEMPLATE, contents={"path": ir_marker_without_first_word})
            previous_node = add_simple_edge(epic, previous_node, new_node)
        elif ir_marker_first_word == "/PROMPT":
            prompt_content = ir_marker_without_first_word
            
            # Check if it's a file reference
            if prompt_content.startswith("@file:"):
                prompt_file = prompt_content[6:].strip()
                base_dir = Path(input_file_path).parent
                prompt_content = load_prompt_from_file(prompt_file, base_dir)
            
            # Handle /RO markers within the prompt content
            ro_list = extract_next_word_after_marker(prompt_content, "/RO")


        elif ir_marker_first_word == "/RUN":
            new_node = build_default_run_node(epic)
            previous_node = add_simple_edge(epic, previous_node, new_node)
        elif ir_marker_first_word == "/DEBUG_LOOP":
            new_node = epic.add_node(opcode=Opcode.DEBUG_LOOP, contents={})
            previous_node = add_simple_edge(epic, previous_node, new_node)
        elif ir_marker_first_word == "/EXIT":
            new_node = epic.add_node(opcode=Opcode.EXIT, contents={})
            previous_node = add_simple_edge(epic, previous_node, new_node)
        elif ir_marker_first_word == "/MULTI_STAGE":
            # Add a node that signals multi-stage should be enabled
            new_node = epic.add_node(opcode=Opcode.COMMAND, contents={"command": "enable_multi_stage"})
            previous_node = add_simple_edge(epic, previous_node, new_node)

    return epic

def load_prompt_from_file(prompt_path: str, base_dir: Path = None) -> str:
    """Load prompt content from file."""
    if base_dir and not Path(prompt_path).is_absolute():
        prompt_path = base_dir / prompt_path
    
    with open(prompt_path, 'r') as f:
        return f.read().strip()

# Example Usage (for demonstration and testing)
if __name__ == "__main__":
    # Create a dummy test workflow file
    test_file_path = "test_workflow.txt"
    test_file_content = """
        /TEMPLATE input_const/templates/build_scripts/
        /PROMPT
        Don't install new requirements.
        Write a python program that adds two numbers.
        /RO src/data.txt
        /PROMPT
        Make a dir called tests and write some tests in there. Don't install new requirements.
        /RO another_file.csv
        /RUN
        /DEBUG_LOOP
        /EXIT
        """
    with open(test_file_path, "w") as f:
        f.write(test_file_content)

    print(f"Parsing workflow from: {test_file_path}")
    parsed_epic_ir = parse_workflow_file(test_file_path)

    print("\n--- Parsed Workflow Graph (Nodes) ---")
    for node_name, data in parsed_epic_ir.graph.nodes(data=True):
        print(f"Node: {node_name}, Opcode: {data['opcode'].value}, Contents: {data['contents']}")

    print("\n--- Parsed Workflow Graph (Edges) ---")
    for u, v in parsed_epic_ir.graph.edges():
        print(f"{u} -> {v}")

    # Clean up the dummy file
    os.remove(test_file_path)
    print(f"\nCleaned up {test_file_path}")