"""
The VocabGraph maintains a graph of known vocabulary, and how it's related to other terms in the
VocabGraph. Nodes of the graph are terms in the vocabulary, and edges connect terms that are related
by some "relation". The possible relations are given by the "WordRelations" enum.
"""

from collections import defaultdict
import types

_function_type = types.FunctionType
_lambda_type = types.LambdaType
_method_type = types.MethodType

class _WordRelation:
    def __init__(self, name, opposite=None):
        self.name = name
        self.opposite = opposite

    def __repr__(self):
        return self.name
    __str__ = __repr__

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

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

    IS_ATTR_OF, HAS_ATTR               = opposite_relations("IS_ATTR_OF",   "HAS_ATTR")
    IS_METHOD_OF, HAS_METHOD           = opposite_relations("IS_METHOD_OF", "HAS_METHOD")
    IS_FUNC_ATTR_OF, HAS_FUNC_ATTR     = opposite_relations("IS_FUNC_ATTR_OF", "HAS_FUNC_ATTR")
    IS_LAMBDA_ATTR_OF, HAS_LAMBDA_ATTR = opposite_relations("IS_LAMBDA_ATTR_OF", "HAS_LAMBDA_ATTR")
    IS_ARG_OF, HAS_ARG                 = opposite_relations("IS_ARG_OF",    "HAS_ARG")
    IS_KWARG_OF, HAS_KWARG             = opposite_relations("IS_KWARG_OF",  "HAS_KWARG")
    IS_VARG_OF, HAS_VARG               = opposite_relations("IS_VARG_OF",   "HAS_VARG")
    IS_VKWARG_OF, HAS_VKWARG           = opposite_relations("IS_VKWARG_OF", "HAS_VKWARG")
    IS_ELEM_OF_L, HAS_L_ELEM           = opposite_relations("IS_ELEM_OF_L", "HAS_L_ELEM")
    IS_ELEM_OF_D, HAS_D_ELEM           = opposite_relations("IS_ELEM_OF_D", "HAS_D_ELEM")
    IS_VALUE_OF, HAS_VALUE             = opposite_relations("IS_VALUE_OF",  "HAS_VALUE")

    IS_KERAS_LAYER_OF, HAS_KERAS_LAYER       = opposite_relations("IS_KERAS_LAYER_OF", "HAS_KERAS_LAYER")
    IS_KERAS_CALLBACK_OF, HAS_KERAS_CALLBACK = opposite_relations("IS_KERAS_CALLBACK_OF", "HAS_KERAS_CALLBACK")

    @classmethod
    def _get_attr_rel(cls, type):
        if issubclass(type, _function_type):
            return cls.IS_FUNC_ATTR_OF
        elif issubclass(type, _method_type):
            return cls.IS_METHOD_OF
        elif issubclass(type, _lambda_type):
            return cls.IS_LAMBDA_ATTR_OF
        return cls.IS_ARG_OF

class WordNode:
    @property
    def connections(self):
        return self._connections

    def __init__(self, value):
        self.value = value
        # The connections are directed. If a <-> b, then a has a connection to b 
        # and b has a connection to a, rather than both objects having both connnections.
        self._connections = set()

    def add_connection(self, node, relation):
        self._connections.add(WordEdge(self, node, relation))
        node._connections.add(WordEdge(node, self, relation.opposite))

    def _reset_connections(self):
        self._connections = set()

    def get_connections_to_value(self, value, relation=None):
        connections = []
        for connection in self._connections: 
            if (relation is None or connection.relation == relation) and \
                connection.second_node.value == value:
                connections.append(connection)
        return connections

    def _remove_connections(self, node):
        # Possible for multiple connections to the same node with different relation types
        connections_to_remove = set() 
        for connection in self._connections:
            if connection.second == node:
                connections_to_remove.add(connection)
        self._connections -= connections_to_remove

    def remove_connections(self, connections):
        self._connections -= set(connections)

    def remove_connections_to_node(self, node):
        """
        Remove any connections between this node and the given node.
        This will also remove any connections from the given node to
        the first node.
        """
        self._remove_connections(node)
        node._remove_connections(self)

    def remove_connections_by_value(self, node_name, relation=None):
        connections_to_remove = set()
        for connection in self._connections:
            if (relation is None or connection.relation == relation) and \
                connection.second.value == node_name:
                connections_to_remove.add(connection)
                connection.second._remove_connections(self)
        self._connections -= connections_to_remove

    def __repr__(self):
        if (self.connections):
            self_str = "Node(%s <%s>) -> [" % (self.value, hash(self))
            self_str += ", ".join("%s : Node(%s)" % (c.relation, c.second.value) for c in self._connections)
            self_str += "]"
            return self_str
        else:
            return "Node(%s)" % self.value

    __str__ = __repr__

# TODO: Just a namedtuple?
class WordEdge:
    """
    WordEdge represents a directed connection of type `relation` connecting `first` to `second`.
    As an example, "model.scheduler" would be represented by the nodes "model" and "scheduler", connected
    via a WordEdge of type "IS_ATTR_OF", with `first` == "scheduler" and second "model" (i.e., 
    `first` is `relation` of `second`).
    """

    def __init__(self, first_node, second_node, relation):
        self.first = first_node
        self.second = second_node
        self.relation = relation

    def __eq__(self, other):
        return isinstance(other, WordEdge) and \
               (self.first == other.first)   and \
               (self.second == other.second) and \
               (self.relation == other.relation)

    def __hash__(self):
        return hash((self.first, self.second, self.relation))

class VocabGraph:

    def __init__(self, default_words):
        self._graph_by_values = defaultdict(set)
        for word in default_words:
            self._graph_by_values[word].append(WordNode(word))

    def add_node(self, value, context, relation):
        """
        Add a node with the given `value` to the vocabulary graph, connected to the `context`
        node with the given `relation`, and return the newly created node.
        """
        node = WordNode(value)
        if (context is not None):
            # This will automatically add the opposite connection to the context node.
            node.add_connection(context, relation)

        self._graph_by_values[value].add(node)

        return node

    def remove_node_by_value(self, value, context, recursive=True, initial_relation=None):
        """
        Remove a node with the given `value` attached to the given `context` node
        from the vocab graph.

        If `recursive=True`, recursively remove all subsequent out-nodes connected to the
        first removed node.

        This will fail silently if no nodes with the given `value` (and connected with the
        given `initial_relation` if not None) are found connected to `context`.
        """
        connections = context.get_connections_to_value(value, relation=initial_relation)
        if not connections:
            return
        
        # Remove connections to the initial node...
        context.remove_connections(connections)

        if not recursive:
            return

        # ...then continue by removing the deeper nodes from "memory".
        # We remove the nodes themselves from self._graph_by_values, and
        # remove their connections.

        to_remove = {c.second_node for c in connections}
        while to_remove:
            next_node = to_remove.pop()

            # Try-except block is faster than an if statement when it is _unlikely_ for
            # an exception to be raised in the try block, which in this case it is (why
            # would there be a duplicate in the vocab graph, unless I'm not understanding
            # something about this program.)
            try:
                self._graph_by_values[next_node.value].remove(next_node)
            except ValueError:
                # The only way we can encounter a node in this process that _isn't_ in the
                # graph is if there's somehow a cycle in the graph, which _should_ mean
                # we've already processed the node and removed all it's connections as well.
                # In this case, if we were to continue removing it's connected nodes we would
                # cause an infinite loop.
                continue

            for connection in next_node.connections:
                to_remove.add(connection.second_node)
            
            next_node._reset_connections()

            
            