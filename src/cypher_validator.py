# Copied from https://github.com/bSharpCyclist/cypher-direction-competition/tree/main

import re

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

class PatternStatus(Enum):
    UNVALIDATED = 0
    EXISTS = 1
    REVERSE = 2
    SOURCE_EQUALS_TARGET = 3
    DOES_NOT_EXIST = 4
    DO_NOT_CORRECT = 5

@dataclass
class ParsedPattern:
    source_node_variable: str
    source_node_label: str
    source_node_properties: str
    relationship_variable: str
    relationship_label: str
    relationship_properties: str
    target_node_variable: str
    target_node_label: str
    target_node_properties: str
    direction: str
    raw: str  # parsed pattern as it appears in the query
    query: str
    status: PatternStatus

class CypherValidator:
    """
    A class for validating Cypher queries.

    Attributes:
    _schema_pattern (str): A regex pattern for matching schema.
    _node_pattern (str): A regex pattern for matching a node.
    _relationship_pattern (str): A regex pattern for matching a relationship.
    _forward_relationship_pattern (str): A regex pattern for matching forward relationship.
    _backward_relationship_pattern (str): A regex pattern for matching backward relationship.
    _undirected_relationship_pattern (str): A regex pattern for matching undirected relationship.
    _forward_pattern (str): A regex pattern for matching forward pattern.
    _backward_pattern (str): A regex pattern for matching backward pattern.
    _undirected_pattern (str): A regex pattern for matching undirected pattern.
    _patterns (List[Tuple[str, str]]): A list of regex patterns and their corresponding direction.
    _parsed_patterns (List[ParsedPattern]): A list of parsed patterns.
    _parsed_schemas (List[Tuple[str, str, str]]): A list of parsed schemas.
    _variable_label_mapping (List[Tuple[str, str]]): A list of variable label mappings.
    """
    
    _schema_pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    _node_pattern = r'\(([\w]+)?:?([\w`:]+)?\s?(\{.*?\})?\)'
    _relationship_pattern = r'\[?([\w]+)?:?([\w`*|!.]+)?\s*?(\{.*?\})?\]?'

    _forward_relationship_pattern = r'-' + _relationship_pattern + r'->'
    _backward_relationship_pattern = r'<-' + _relationship_pattern + r'-'
    _undirected_relationship_pattern = r'-' + _relationship_pattern + r'-'

    _forward_pattern = _node_pattern + _forward_relationship_pattern + _node_pattern
    _backward_pattern = _node_pattern + _backward_relationship_pattern + _node_pattern
    _undirected_pattern = _node_pattern + _undirected_relationship_pattern + _node_pattern

    _patterns = [(_forward_pattern, '->'), (_backward_pattern, '<-'), (_undirected_pattern, '--')]

    def __init__(self):
        """
        Initializes CypherValidator with empty lists for parsed patterns, parsed schemas, and variable label mappings.
        """
        self._parsed_patterns = []
        self._parsed_schemas = []
        self._variable_label_mapping = []
    
    def _parse_schema(self, schema: str) -> None:
        """
        Parses the input schema string to identify individual schema elements and store them as tuples.
        
        These tuples are appended to the internal `_parsed_schemas` list.
        
        Parameters:
        -----------
        schema : str
            The input schema string containing elements to be parsed.
        
        Returns:
        --------
        None
        
        Side Effects:
        -------------
        Modifies the internal `_parsed_schemas` list, appending tuples that represent the parsed elements
        of the schema.
        """
        self._parsed_schemas = re.findall(self._schema_pattern, schema)
        print(f"Parsed schema: {self._parsed_schemas}")
    
    def _parse_pattern(self, query: str) -> None:
        """
        Parses the input query to identify specific patterns and directions.
        
        This method iterates through the internal list of `_patterns` and attempts to find matches
        in the input `query`. For each match, a new `ParsedPattern` instance is created and populated
        with relevant details such as the matched groups and direction.
        
        The created `ParsedPattern` instances are then appended to the internal `_parsed_patterns` list.
        
        Parameters:
        -----------
        query : str
            The input query string containing patterns and directions to be parsed.
        
        Returns:
        --------
        None
        
        Side Effects:
        -------------
        Modifies the internal `_parsed_patterns` list, appending instances of `ParsedPattern`
        that represent the parsed elements of the query.
        """
        for pattern, direction in self._patterns:
            tmp_query = query
            match = re.search(pattern, tmp_query)

            while match:
                # Unpack the matched groups and create a new ParsedQuery instance
                parsed_pattern = ParsedPattern(*match.groups(), direction, match.group(), query, PatternStatus.UNVALIDATED)

                # If there are backticks in the node and relationship labels, remove them
                for label_attr in ['source_node_label', 'relationship_label', 'target_node_label']:
                    label = getattr(parsed_pattern, label_attr, None)
                    if label is not None:
                        setattr(parsed_pattern, label_attr, label.replace('`', ''))

                # Append the new ParsedQuery instance to the results
                self._parsed_patterns.append(parsed_pattern)

                # Adjust the start position for the next search to be the beginning of the last matched group
                start_pos = match.start(2)
                tmp_query = tmp_query[start_pos-1:] # -1 to include the parenthesis

                match = re.search(pattern, tmp_query)

    def _map_variables(self, query: str) -> None:
        """
        Maps variables to their corresponding labels based on the query string.
        
        This method scans the input query to find pairs of variables and their labels, then updates the internal
        `_variable_label_mapping` list with these mappings. The variables and labels are found based on a specified
        regex pattern. 
        
        Parameters:
        -----------
        query : str
            The input query string containing variables and their labels that need to be mapped.
        
        Returns:
        --------
        None
        
        Side Effects:
        -------------
        Modifies the internal `_variable_label_mapping` list, appending tuples in the format (variable, label).
        """
        pattern = r'\((\w+):([\w:]+)\)'
        match = re.findall(pattern, query)

        for var, label in match:
            if ':' in label:
                labels = label.split(':')
                for lbl in labels:
                    self._variable_label_mapping.append((var, lbl))
            else:
                self._variable_label_mapping.append((var, label))

    def _validate_pattern(self) -> None:
        """
        Validates the parsed query patterns against the parsed schema.
        
        This method iterates through each parsed pattern, checks its validity and sets its status accordingly.
        The statuses can be one of the following: EXISTS, REVERSE, SOURCE_EQUALS_TARGET, DO_NOT_CORRECT, and DOES_NOT_EXIST,
        as defined in the PatternStatus enum.
        
        The method relies on helper functions `_validate_node_labels` and `_validate_relationship_labels` to validate the
        individual components of the pattern.
        
        - If the source and target node labels are the same, the status is set to SOURCE_EQUALS_TARGET.
        - If the relationship is undirected (`--`), the status is set to DO_NOT_CORRECT.
        - If the pattern exists or can be reversed based on the schema, the status is set to EXISTS or REVERSE respectively.
        - If none of the above conditions are met, the status is set to DOES_NOT_EXIST.
        
        This method modifies the internal `_parsed_patterns` list in-place.
        
        Returns:
        --------
        None
        """
        for parsed_pattern in self._parsed_patterns:

            # If the relationship is between two nodes of the same labels, there is nothing to validate or correct
            if parsed_pattern.source_node_label and parsed_pattern.target_node_label and parsed_pattern.source_node_label == parsed_pattern.target_node_label:
                parsed_pattern.status = PatternStatus.SOURCE_EQUALS_TARGET
                continue

            # If the input query has an undirected relationship in the pattern, we do not correct it.
            if parsed_pattern.direction == '--':
                parsed_pattern.status = PatternStatus.DO_NOT_CORRECT
                continue

            for parsed_schema in self._parsed_schemas:
                if not self._validate_relationship_labels(parsed_pattern, parsed_schema[1]):
                    continue

                direction_mapping = {
                    '->': [(parsed_schema[0], parsed_schema[2], PatternStatus.EXISTS),
                        (parsed_schema[2], parsed_schema[0], PatternStatus.REVERSE)],
                    '<-': [(parsed_schema[2], parsed_schema[0], PatternStatus.EXISTS),
                        (parsed_schema[0], parsed_schema[2], PatternStatus.REVERSE)]
                }

                for source_label, target_label, status in direction_mapping[parsed_pattern.direction]:
                    if self._validate_node_labels(parsed_pattern, source_label, target_label):
                        parsed_pattern.status = status
                        break

                if parsed_pattern.status != PatternStatus.UNVALIDATED:
                    break

            if parsed_pattern.status == PatternStatus.UNVALIDATED:
                parsed_pattern.status = PatternStatus.DOES_NOT_EXIST

    def _validate_relationship_labels(self, parsed_pattern: ParsedPattern, rel_schema: str) -> bool:
        """
        Validates relationship labels in a parsed pattern against a given relationship schema.
        
        This function compares the relationship labels found in the `ParsedPattern` instance 
        with a specified relationship schema. It returns True if the parsed pattern's 
        relationship label matches with the given schema, or False otherwise. It can also handle 
        negation using '!', and multiple relationship labels separated by a '|'.
        
        Parameters:
        -----------
        parsed_pattern : ParsedPattern
            An instance of ParsedPattern containing the relationship label to be validated.
            
        rel_schema : str
            The relationship schema against which the label will be validated.
            
        Returns:
        --------
        bool
            - True if the relationship label in `parsed_pattern` matches `rel_schema`.
            - True if the relationship label in `parsed_pattern` negates (`!`) `rel_schema`.
            - False otherwise.
        """
        if parsed_pattern.relationship_label is None:
            return True
        
        # We may have multiple relationship labels in the pattern, so we need to split them
        parsed_pattern_relationship_labels = parsed_pattern.relationship_label.split('|') if parsed_pattern.relationship_label is not None else [None]

        for label in parsed_pattern_relationship_labels:
            if label == rel_schema:
                return True
            elif label.startswith('!') and label[1:] != rel_schema:
                return True
            
        return False
    
    def _validate_node_labels(self, parsed_pattern: ParsedPattern, source_schema: str, target_schema: str) -> bool:
        """
        Validates the labels of source and target nodes in a given pattern based on a schema.
        
        Parameters:
        -----------
        parsed_pattern : ParsedPattern
            An object containing the parsed pattern to validate.
            
        source_schema : str
            The label of the source node in the schema.
            
        target_schema : str
            The label of the target node in the schema.
            
        Returns:
        --------
        bool:
            True if the node labels in the pattern match the source and target labels in the schema, False otherwise.
            
        Internal Functions:
        -------------------
        get_labels(node_label, node_variable) -> list:
            A helper function that returns a list of labels for a node. If `node_label` is None, it checks for any mapping
            between `node_variable` and labels stored in `_variable_label_mapping`.
            
        Notes:
        ------
        - The method uses a helper function `get_labels` to obtain the labels for source and target nodes.
        - Validation checks are done based on the obtained labels and the schema labels passed as arguments.
        """
        def get_labels(node_label, node_variable):
            labels = []
            if node_label is None:
                for variable, label in self._variable_label_mapping:
                    if node_variable == variable:
                        labels.append(label)
            else:
                labels = node_label.split(':')
            return labels

        source_labels = get_labels(parsed_pattern.source_node_label, parsed_pattern.source_node_variable)
        target_labels = get_labels(parsed_pattern.target_node_label, parsed_pattern.target_node_variable)

        if not source_labels and not target_labels:
            return True

        if not source_labels:
            return target_schema in target_labels

        if not target_labels:
            return source_schema in source_labels

        return source_schema in source_labels and target_schema in target_labels

    def _update_query(self, query: str) -> str:
        """
        Updates a query based on the statuses of parsed patterns.
        
        This function iterates over each parsed pattern and modifies the query accordingly based on the pattern's status. 
        If the pattern's status is DOES_NOT_EXIST, the updated query is set to an empty string. If the pattern's status is 
        REVERSE, the direction of the relationship in the pattern is reversed in the updated query.

        Parameters:
        -----------
        query : str
            The initial query string that needs to be updated.
            
        Returns:
        --------
        str:
            The updated query string based on the statuses of parsed patterns.
            
        Notes:
        ------
        - This function uses Python's re library to search and replace substrings within the query.
        - The pattern's status must be one of the PatternStatus Enum values: UNVALIDATED, EXISTS, REVERSE, SOURCE_EQUALS_TARGET,
        DOES_NOT_EXIST, DO_NOT_CORRECT.
        """
        
        updated_query = query

        for parsed_pattern in self._parsed_patterns:
            if parsed_pattern.status == PatternStatus.DOES_NOT_EXIST:
                return ''

            if parsed_pattern.status in [PatternStatus.SOURCE_EQUALS_TARGET, PatternStatus.DO_NOT_CORRECT]:
                continue

            if parsed_pattern.status == PatternStatus.REVERSE:
                if parsed_pattern.direction == '<-':
                    if '<--' in parsed_pattern.raw:
                        substring = parsed_pattern.raw.replace('<--', '-->')
                    else:
                        match = re.search(r'\)(<-)\[(?:[^]]*)\](-)\(', parsed_pattern.raw)
                        substring = parsed_pattern.raw[:match.start(1)] + '-' + parsed_pattern.raw[match.end(1):match.start(2)] + '->' + parsed_pattern.raw[match.end(2):]
                elif parsed_pattern.direction == '->':
                    if '-->' in parsed_pattern.raw:
                        substring = parsed_pattern.raw.replace('-->', '<--')
                    else:
                        match = re.search(r'\)(-)\[(?:[^]]*)\](->)\(', parsed_pattern.raw)
                        substring = parsed_pattern.raw[:match.start(1)] + '<-' + parsed_pattern.raw[match.end(1):match.start(2)] + '-' + parsed_pattern.raw[match.end(2):]
                updated_query = updated_query.replace(parsed_pattern.raw, substring)

        return updated_query

    def validate_query(self, schema: str, query: str) -> Tuple[str, List[ParsedPattern]]:
        """
        Validates and optionally updates a query based on a provided schema.
        
        This is the main public method of the class that orchestrates the validation process. It first parses the schema and 
        the query pattern to understand their structure. Then, it maps variables within the query, validates the parsed
        patterns, and finally updates the query as necessary based on its validation status.
        
        Parameters:
        -----------
        schema : str
            The schema string that defines the structure and constraints of the data model.
        
        query : str
            The query string that needs to be validated and optionally updated.
            
        Returns:
        --------
        Tuple[str, List[ParsedPattern]]:
            A tuple containing two elements:
                1. The updated (or original) query string.
                2. A list of parsed patterns with their statuses.
        """
        self._parse_schema(schema)
        self._parse_pattern(query)
        self._map_variables(query)
        self._validate_pattern()
        fixed_query = self._update_query(query)

        return fixed_query, self._parsed_patterns