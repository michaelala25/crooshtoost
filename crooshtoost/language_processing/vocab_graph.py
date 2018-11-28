"""
The VocabGraph maintains a graph of known vocabulary, and how it's related to other terms in the
VocabGraph. Nodes of the graph are terms in the vocabulary, and edges connect terms that are related
by some "relation". The possible relations are given by the "WordRelations" enum.
"""

from collections import defaultdict
from enum import Enum

class WordRelations(Enum):
    UNKNOWN = 0

    IS_ATTR_OF = 1
    HAS_ATTR = 2

    IS_METHOD_OF = 3
    HAS_METHOD = 4

    IS_ARG_OF = 5
    HAS_ARG = 6

    IS_KWARG_OF = 7
    HAS_KWARG = 8

    IS_VARG_OF = 9
    HAS_VARG = 10

    IS_VKWARG_OF = 11
    HAS_VKWARG = 12

    IS_ELEM_OF_LIST = 13
    HAS_LIST_ELEM = 14

    IS_ELEM_OF_DICT = 15
    HAS_DICT_ELEM = 16

    IS_VALUE_OF = 17
    HAS_VALUE = 18

    IS_KERAS_LAYER_OF = 19
    HAS_KERAS_LAYER = 20

    IS_KERAS_CALLBACK_OF = 21
    HAS_KERAS_CALLBACK = 22

class WordNode:
    @property
    def connections(self):
        return self._connections

    def __init__(self, value):
        self.value = value
        self._connections = set()

    def add_connection(self, node, relation):
        self._connections.add(WordEdge(self, node, relation))

    def remove_connection(self, node):
        to_remove = set() # Possible for multiple connections to the same node with different relation types
        for connection in self._connections:
            if connection.second_node == node:
                to_remove.add(connection)
        self._connections -= to_remove

    def remove_connection_by_name(self, out_name, relation=None):
        to_remove = set()
        for connection in self._connections:
            if (relation is None or connection.relation == relation) and \
                connection.second_node.value == out_name: 
                to_remove.add(connection)
        self._connections -= to_remove

class WordEdge:

    def __init__(self, first_node, second_node, relation):
        self.first_node = first_node
        self.second_node = second_node
        self.relation = relation

class VocabGraph:

    def __init__(self, default_words):
        self._graph_by_values = defaultdict(set)
        for word in default_words:
            self._graph_by_values[word].append(WordNode(word))

    def add_node(self, value, context, relation):
        """
        Add a node to the vocabulary graph, connected to the `context` node with the given `relation`,
        and return the newly created node.
        """
        node = WordNode(value)
        if (context is not None):
            node.add_connection(context, relation)

        self._graph_by_values[value].add(node)

        return node