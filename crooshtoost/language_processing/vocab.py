"""
Global vocab managing utilities.

Essential question: what does CrooshToost know about for this session? The utilities in this module are for
answering the more specific question: what vocabulary is CrooshToost familiar with in this session. This includes
model names, model layer names, model callbacks, model parameters, etc.

"Vocabulary" in CrooshToost refers to anything that can be operated on/with corresponding to the current model.
This can include parameter names, attribute names, method names, method argument names, etc.
"""

# Idea
# I think internally the vocabulary should be stored as a sort of graph of relations between
# names and objects. For example, "model.layers" should be stored as an IS_ATTR_OF relation
# connecting "layers" to "model" (directed). "node_index" should be stored as an IS_ARG_OF
# relation connecting "node_index" to "get_input_mask_at", which is itself stored as an
# IS_METHOD_OF relation connecting "get_input_mask_at" to "model".
#
# The types of relationships the graph maintains can then be used to search for likely objects
# or commands matching noun-phrases and intent extracted from the user input. They will also
# help to structure the execution of the command itself.

from .vocab_graph import VocabGraph, WordRelations

from platform import python_version

import inspect
import numpy as np
import tensorflow as tf
import types, typing

if python_version() < '3.6':
    # Easiest examples I could find of each
    typing_MethodDescriptorType = None # Still not sure
    typing_MethodWrapperType = type("".__repr__)
    typing_WrapperDescriptorType = type(str.__repr__)
else:
    typing_MethodDescriptorType = typing.MethodDescriptorType
    typing_MethodWrapperType = typing.MethodWrapperType
    typing_WrapperDescriptorType = typing.WrapperDescriptorType

_DEFAULT_VOCAB = [
    # Default vocabulary to include.
]

# These are types to ignore exploring further. They don't have "unusual" attributes
# that we wish to explore and add to our vocabulary.
#
# There are tons of other classes we could be checking here, but for now lets just assume
# that anything beyond this list is considered "interesting".
_BASE_TYPES = {
    bool, int, float, complex, str,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, np.double,
    np.long, np.longcomplex, np.longdouble, np.longlong,
    bytes, bytearray, memoryview, type, property,
    list, tuple, range, dict, set, frozenset,
    type(...), type(NotImplemented), # type(None), Not too sure about type(None) yet...
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    types.CodeType,
    types.CoroutineType,
    types.FrameType,
    types.FunctionType,
    types.GeneratorType,
    types.LambdaType,
    types.MappingProxyType,
    types.MethodType,
    types.ModuleType, # probably should explore modules?
    types.SimpleNamespace,
    types.TracebackType,
    typing_MethodDescriptorType,
    typing_MethodWrapperType,
    typing_WrapperDescriptorType,
    tf.dtypes.DType
}

_function_type = types.FunctionType
_lambda_type = types.LambdaType
_method_type = types.MethodType

_FUNCTION_TYPES = {
    _function_type,
    _lambda_type,
    _method_type,
    type # we check classes as well because their __init__ methods might have important vocabulary
}

# Maximum recursion depth of objects to explore when looking for vocabulary
_MAX_RECURSION_DEPTH = 2

def _is_base_type(t):
    if t in _BASE_TYPES: # Quick Check
        return True
    return any(issubclass(t, x) for x in _BASE_TYPES)

def _is_function_type(t):
    return any(issubclass(t, x) for x in _FUNCTION_TYPES)

def _is_magic_name(name):
    return name.startswith("__")

def _is_private_name(name):
    """
    Check if `name` starts with a single underscore. Note that this will return _False_
    for magic names like "__init__".
    """
    return name.startswith("_") and not _is_magic_name(name)

def _extract_vocab_recursive(obj, vg, context, depth=0):
    # Algorithm: For now, iterate recursively through every object's "dir".
    #
    # If a name is private (starts with a single underscore), we ignore it only if there's
    # an equivalent non-private name in the object dir. An example of this is tensorflow's
    # Operation._node_def, which has corresponding public attribute Operation.node_def.
    #
    # If an object is of type list, dict, or tuple, then the recursive extraction also applies
    # to each of the objects in the iterable (we limit the discovery to these types because these
    # are the safest classes to iterate over).
    #
    # If an object is of type string, and the string is non-empty, then the value of the 
    # string is added to the vocab as well.
    #
    # If an object is a function type, inspect its arguments. If the object has **kwargs,
    # we can try to guess some of the parameters by looking at the object's __doc__.
    
    if depth == _MAX_RECURSION_DEPTH:
        return

    obj_dir = dir(obj)
    for name in obj_dir:
        if _is_magic_name(name):
            # Ignore magic stuff.
            continue

        if _is_private_name(name) and name[1:] in obj_dir:
            # Check here for private names with public counterparts
            continue

        attr = None
        try:
            attr = getattr(obj, name)
        except AttributeError:
            # Sometimes keras will throw an AttributeError for values that aren't initialized
            # yet in the current tensorflow session.
            continue
        finally:
            # TODO: We should really also store a list of "ignore" types, for which we don't add the name.
            relation_type = WordRelations.IS_ATTR_OF # Default
            if attr is not None and isinstance(attr, _method_type):
                # If attr is a method, change the default relation type.
                relation_type = WordRelations.IS_METHOD_OF
            next_context = vg.add_node(name, context, relation_type)

        attr_type = type(attr)
        if _is_base_type(attr_type) and attr is not None:
            # Note: when entering an iterable to search for new items, we actually increment
            # the depth by 2, since we're entering both the iterable attribute itself _and_
            # it's items. 
            if attr_type == dict:
                for key, value in attr.items():
                    if _is_private_name(name) or _is_magic_name(name):
                        continue
                    # We build a node for the keyword itself built off the current context
                    kw_context = vg.add_node(key, next_context, WordRelations.IS_ELEM_OF_DICT)
                    _extract_vocab_recursive(value, vg, kw_context, depth+2)

            elif attr_type == list or attr_type == tuple:
                for i, item in enumerate(attr):
                    # This isn't quite right, how should this be handled?
                    elem_context = vg.add_node(i, next_context, WordRelations.IS_ELEM_OF_LIST)
                    _extract_vocab_recursive(item, vg, elem_context, depth+2)
            
            elif attr_type == str and attr:
                vg.add_node(attr, next_context, WordRelations.IS_VALUE_OF)

            elif _is_function_type(attr_type):
                try:
                    parameters = inspect.signature(attr).parameters
                except ValueError: # Sometimes no signature is found. Happens occasionally for weird tf objects
                    continue

                learn_from_doc = False
                for param_name, param in parameters.items():
                    if (param_name == "args" and param.kind == inspect.Parameter.VAR_POSITIONAL) or \
                       (param_name == "kwargs" and param.kind == inspect.Parameter.VAR_KEYWORD):
                        learn_from_doc = True
                        continue

                    relation_type = WordRelations.IS_ARG_OF
                    if (param.kind == inspect.Parameter.VAR_POSITIONAL):
                        relation_type = WordRelations.IS_VARG_OF
                    elif (param.kind == inspect.Parameter.KEYWORD_ONLY):
                        relation_type = WordRelations.IS_KWARG_OF
                    elif (param.kind == inspect.Parameter.VAR_KEYWORD):
                        relation_type = WordRelations.IS_VKWARG_OF

                    vg.add_node(param_name, next_context, relation_type)

                if learn_from_doc:
                    _learn_kwargs_from_doc(attr.__doc__, vg)
        else:
            _extract_vocab_recursive(attr, vg, next_context, depth+1)

def _learn_kwargs_from_doc(docstr, vocab):
    # TIME FOR SOME NLP :O
    if not docstr:
        return

class VocabExtractor:
    """
    A VocabExtractor maintains a Vocabulary for a CrooshToost session, with a Vocabulary consisting
    of a set of words CrooshToost is familiar with related to the current session's model.
    """

    def __init__(self):
        self.vocab = VocabGraph(_DEFAULT_VOCAB[::])

    def extract_initial_vocab(self, model):
        model_node = self.vocab.add_node(model.name, None, None)

        # Model Layers
        for layer in model.layers:
            self.vocab.add_node(
                layer.name, model_node, WordRelations.IS_KERAS_LAYER_OF)

        # List of the callbacks
        if (hasattr(model, "callbacks")):
            for callback in model.callbacks: # `model.callbacks` doesn't exist in keras!
                self.vocab.add_node(
                    type(callback).__name__, model_node, WordRelations.IS_KERAS_CALLBACK_OF)

        _extract_vocab_recursive(model, self.vocab, model_node)
    
    def follow(self, dict, contexts={}):
        """
        At certain CrooshToost managed pieces of code (model fitting, preprocessing code run in CT, etc.),
        VocabExtractor.follow(dict, contexts) is used to add potentially new vocabulary to the currently
        maintained vocabulary along with the context(s) in which that vocabulary is found.

        The idea of this method is to "decorate" important function calls. So for example, instead of calling
        model.fit(learning_rate=)
        """
        return dict
        
if __name__ == "__main__":
    # quick testing utilities

    from pprint import pprint
    from keras.models import Model
    from keras.layers import Input, Dense, Softmax
    from keras.callbacks import LearningRateScheduler, TensorBoard, LambdaCallback
    from keras.optimizers import SGD

    x = inputs = Input(shape=(10,))
    x = Dense(40)(x)
    x = Dense(10)(x)
    x = Softmax()(x)

    model = Model(inputs=inputs, outputs=x)

    model.callbacks = [
        LearningRateScheduler(schedule = lambda epoch: 1/(1 + epoch)**.55),
        TensorBoard(),
        LambdaCallback(on_epoch_begin=lambda epoch, logs={}: None)
    ]

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)

    vocab_extractor = VocabExtractor()
    vocab_extractor.extract_initial_vocab(model)

    pprint(vocab_extractor.vocab)