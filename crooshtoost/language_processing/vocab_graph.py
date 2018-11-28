"""
The VocabGraph maintains a graph of known vocabulary, and how it's related to other terms in the
VocabGraph. Nodes of the graph are terms in the vocabulary, and edges connect terms that are related
by some "relation". The possible relations are given by the "WordRelations" enum.
"""

from collections import defaultdict

class _WordRelation:
    def __init__(self, name, opposite=None):
        self.name = name
        self.opposite = opposite

    def __repr__(self):
        return self.name
    __str__ = __repr__

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

def opposite_relations(a, b):
    r1 = _WordRelation(a)
    r2 = _WordRelation(b)

    r1.opposite = r2
    r2.opposite = r1
    return r1, r2

class WordRelations:
    """
    An 'Enum' (not really) containing WordRelation types.
    """
    UNKNOWN, _ = opposite_relations("UNKNOWN", "UNKNOWN")

    IS_ATTR_OF, HAS_ATTR     = opposite_relations("IS_ATTR_OF", "HAS_ATTR")
    IS_METHOD_OF, HAS_METHOD = opposite_relations("IS_METHOD_OF", "HAS_METHOD")
    IS_ARG_OF, HAS_ARG       = opposite_relations("IS_ARG_OF", "HAS_ARG")
    IS_KWARG_OF, HAS_KWARG   = opposite_relations("IS_KWARG_OF", "HAS_KWARG")
    IS_VARG_OF, HAS_VARG     = opposite_relations("IS_VARG_OF", "HAS_VARG")
    IS_VKWARG_OF, HAS_VKWARG = opposite_relations("IS_VKWARG_OF", "HAS_VKWARG")
    IS_ELEM_OF_L, HAS_L_ELEM = opposite_relations("IS_ELEM_OF_L", "HAS_L_ELEM")
    IS_ELEM_OF_D, HAS_D_ELEM = opposite_relations("IS_ELEM_OF_D", "HAS_D_ELEM")
    IS_VALUE_OF, HAS_VALUE   = opposite_relations("IS_VALUE_OF", "HAS_VALUE")

    IS_KERAS_LAYER_OF, HAS_KERAS_LAYER       = opposite_relations("IS_KERAS_LAYER_OF", "HAS_KERAS_LAYER")
    IS_KERAS_CALLBACK_OF, HAS_KERAS_CALLBACK = opposite_relations("IS_KERAS_CALLBACK_OF", "HAS_KERAS_CALLBACK")

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

    def __repr__(self):
        if (self.connections):
            self_str = "Node(%s) -> [" % self.value
            self_str += ", ".join("%s : Node(%s)" % (c.relation, c.second_node.value) for c in self._connections)
            self_str += "]"
            return self_str
        else:
            return "Node(%s)" % self.value

    __str__ = __repr__

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