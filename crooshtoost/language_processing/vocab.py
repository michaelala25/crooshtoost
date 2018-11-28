"""
Global vocab managing utilities.

Essential question: what does CrooshToost know about for this session? The utilities in this module are for
answering the more specific question: what vocabulary is CrooshToost familiar with in this session. This includes
model names, model layer names, model callbacks, model parameters, etc.

"Vocabulary" in CrooshToost refers to anything that can be operated on/with corresponding to the current model.
This can include parameter names, attribute names, method names, method argument names, etc.
"""

import numpy as np
import tensorflow as tf
import types, typing
from platform import python_version

if python_version() < '3.6':
    # Easiest examples I could find of each
    typing_MethodDescriptorType = None
    typing_MethodWrapperType = type("".__repr__)
    typing_WrapperDescriptorType = type(str.__repr__)
else:
    typing_MethodDescriptorType = typing.MethodDescriptorType
    typing_MethodWrapperType = typing.MethodWrapperType
    typing_WrapperDescriptorType = typing.WrapperDescriptorType

__DEFAULT_VOCAB = [
    # Default vocabulary to check through.
]

# These are types to ignore exploring further. They don't have "unusual" attributes
# that we wish to explore and add to our vocabulary.
#
# There are tons of other classes we could be checking here, but for now lets just assume
# that anything beyond this list is considered "interesting".
__BASE_TYPES = {
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

# Maximum recursion depth of objects to explore when looking for vocabulary
__MAX_RECURSION_DEPTH = 4

def _is_base_type(t):
    if t in __BASE_TYPES: # Quick Check
        return True
    return any(issubclass(t, x) for x in __BASE_TYPES)

def _is_magic_name(name):
    return name.startswith("__")

def _is_private_name(name):
    """
    Check if `name` starts with a single underscore. Note that this will return _False_
    for magic names like "__init__".
    """
    return name.startswith("_") and not _is_magic_name(name)

def _extract_vocab_recursive(obj, vocab, depth=0):
    # Algorithm: For now, iterate recursively through every object's "dir", ignoring (for now)
    # methods and magic names (but including names with a single underscore)
    #
    # If a name is private (starts with a single underscore), we ignore it only if there's
    # an equivalent non-private name in the object dir. An example of this is tensorflow's
    # Operation._node_def, which has corresponding public attribute Operation.node_def.
    
    if depth == __MAX_RECURSION_DEPTH:
        return vocab

    obj_dir = dir(obj)
    for name in obj_dir:
        if _is_magic_name(name):
            # Ignore magic stuff.
            continue

        if _is_private_name(name) and name[1:] in obj_dir:
            # Check here for private names with public counterparts
            continue

        try:
            attr = getattr(obj, name)
        except AttributeError:
            # Sometimes keras will throw an AttributeError for values that aren't initialized
            # yet in the current tensorflow session.
            continue
        finally:
            # TODO: We should really also store a list of "ignore" types, for which we don't add the name.
            vocab.add(name)

        attr_type = type(attr)
        if _is_base_type(attr_type):
            # TODO: If t is None, this might indicate an important parameter to pay attention to
            # which simply hasn't been initialized yet.
            continue

        _extract_vocab_recursive(attr, vocab, depth=depth+1)

class VocabExtractor:
    """
    A VocabExtractor maintains a Vocabulary for a CrooshToost session, with a Vocabulary consisting
    of a set of words CrooshToost is familiar with related to the current session's model.
    """

    def __init__(self):
        self.vocab = set(__DEFAULT_VOCAB[::])

    def extract_initial_vocab(self, model):
        self.vocab.add(model.name)

        # Model Layers
        for layer in model.layers:
            self.vocab.add(layer.name)

        # List of the callbacks
        if (hasattr(model, "callbacks")):
            for callback in model.callbacks: # `model.callbacks` doesn't exist in keras!
                self.vocab.add(type(callback).__name__)

        _extract_vocab_recursive(model, self.vocab)
    
    def follow(self, dict, contexts={}):
        """
        At certain CrooshToost managed pieces of code (model fitting, preprocessing code run in CT, etc.),
        VocabExtractor.follow(dict, contexts) is used to add potentially new vocabulary to the currently
        maintained vocabulary along with the context(s) in which that vocabulary is found.
        """
        pass
        
if __name__ == "__main__":
    # quick testing utilities

    from pprint import pprint
    from keras.models import Model
    from keras.layers import Input, Dense, Softmax

    x = inputs = Input(shape=(10,))
    x = Dense(40)(x)
    x = Dense(10)(x)
    x = Softmax()(x)

    model = Model(inputs=inputs, outputs=x)
    vocab_extractor = VocabExtractor()
    vocab_extractor.extract_initial_vocab(model)

    pprint(vocab_extractor.vocab)