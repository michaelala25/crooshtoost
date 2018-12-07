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

# TODO: Figure out how "possible" it is to automatically monitor mutable objects for changes in
#       their __dict__.
#       Interesting Fact: most objects' __dict__'s are *not* readonly ;)

from .vocab_graph import VocabGraph, WordRelations
from ..globals import GLOBALS

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
    types.GeneratorType,
    types.MappingProxyType,
    types.ModuleType, # TODO: probably should explore modules?
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

# Maximum length of an iterable to explore
_MAX_ITERABLE_LEN = 50

# The name format for items found in literals
_ITERABLE_ITEM_NAME_FORMAT = "%s_Item"

def _is_base_type(t):
    return t in _BASE_TYPES or any(issubclass(t, x) for x in _BASE_TYPES)

def _is_function_type(t):
    return any(issubclass(t, x) for x in _FUNCTION_TYPES)

def _is_magic_name(name):
    return name.startswith("__")

def _is_private_name(name):
    """
    Check if `name` starts with a single underscore. Note that this will return *False*
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
    
    if depth >= _MAX_RECURSION_DEPTH or _is_base_type(type(obj)):
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
            relation_type = WordRelations._get_attr_rel(attr_type)
            next_context = vg.add_node(name, context, relation_type)

        attr_type = type(attr)
        if (_is_base_type(attr_type) or _is_function_type(attr_type)) and attr is not None:

            # TODO: What happens if this list gets added to or removed from?

            # Note: when entering an iterable to search for new items, we actually increment
            # the depth by 2, since we're entering both the iterable attribute itself _and_
            # it's items. 
            if issubclass(attr_type, dict) and len(attr) <= _MAX_ITERABLE_LEN:
                for key, value in attr.items():
                    if _is_private_name(name) or _is_magic_name(name):
                        continue

                    if _is_base_type(type(value)):
                        continue
                    # We build a node for the keyword itself built off the current context
                    kw_context = vg.add_node(key, next_context, WordRelations.IS_ELEM_OF_D)

                    _extract_vocab_recursive(value, vg, kw_context, depth+2)

            elif issubclass(attr_type, (list, tuple)) and len(attr) <= _MAX_ITERABLE_LEN:
                # TODO: This is REALLY fucky. The name given to items in a list is "list_name[index]". What if
                # the index of the item changes? I think a few things have to happen. First, lists and tuples
                # should be handled differently, since tuples are immutable (but can still be appended to
                # apparently with +). Second, the node name given to items found in the iterable should _not_
                # be dependent on the item's position in the iterable (unless, _maybe_ for tuples).
                for i, item in enumerate(attr):
                    if _is_base_type(type(item)):
                        continue

                    node_name = _ITERABLE_ITEM_NAME_FORMAT % name
                    elem_context = vg.add_node(node_name, next_context, WordRelations.IS_ELEM_OF_L)
                    _extract_vocab_recursive(item, vg, elem_context, depth + 2)
            
            elif attr_type == str and attr:
                # The reason we do this is so that we can pick up on things like object names or keywords.
                # For example, if we asked "What is the learning rate of model CNN1?", it wouldn't be enough
                # to know that 'model' has a parameter 'name', we would also need to know that that specific
                # model's 'name' is equal to "CNN1" (or close enough).
                vg.add_node(attr, next_context, WordRelations.IS_VALUE_OF)

            elif _is_function_type(attr_type):
                try:
                    parameters = inspect.signature(attr).parameters
                except ValueError: # Sometimes no signature is found. Happens occasionally for weird tf objects
                    continue

                learn_from_doc = False
                for param_name, param in parameters.items():
                    # If the name is "args" or "kwargs", then typically the list of valid keyword argument
                    # names can be found somewhere in the function's docstring (assuming it's well documented).
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
            # If this attribute isn't one of the base type cases above, 
            # continue searching through it's __dict__ for vocab.
            _extract_vocab_recursive(attr, vg, next_context, depth+1)

    # Now the fun stuff. We inject our own __getattribute__, __setattr__,
    # and __delattr__ into each object to dynamically update the vocab
    # graph.
    #
    # This does introduce significant overhead, and so can be turned off
    # with the global attribute ENABLE_DYNAMIC_VOCAB_UPDATES.
    if not GLOBALS.ENABLE_DYNAMIC_VOCAB_UPDATES:
        return

    _inject_dynamic_vocab_updates(obj, vg, context)

def _learn_kwargs_from_doc(docstr, vg):
    # TIME FOR SOME NLP :O
    if not docstr:
        return

class _DynamicVocabWatcher:
    """
    Empty base class for checking if we've already injected 
    our dynamic vocab updating code into an object.
    """
    pass

def _inject_dynamic_vocab_updates(obj, vg, context):
    """Inject code into obj to dynamically watch changes to the object's __dict__."""
    # WARNING: This code is ugly and magical.

    if isinstance(obj, _DynamicVocabWatcher):
        # Already injected dynamic vocab watching into the object.
        return
    
    # __slots__ indicates which attributes an object has access to. If __slots__ is set and
    # __slots__ doesn't contain __dict__, then the object's list of accessible attributes
    # won't change, meaning we don't need to worry about dynamically updating the vocab graph.
    #
    # Even if the object has __slots__ without __dict__, setting an attribute to a new 
    # value still warrants updating the vocab graph, meaning we still have to override
    # __setattr__.
    has_immutable_slots = hasattr(obj, "__slots__") and "__dict__" not in obj.__slots__

    # Notes
    # =====
    # IMPORTANT NOTE: Inside the new _getattr, _setattr_, _delattr, we *can't* use
    # hasattr, getattr, setattr, or delattr, as this will create an infinite loop.

    # If the object has __getattr__, then this will be called ONLY when the attribute
    # being retrieved is _not_ already an attribute of the object. Hence, if we're
    # inside __getattr__, then hasattr(self, name) is implicitly False.

    # It may not seem like we have to worry about __getattr__ or __getattribute__ (since when
    # does the _retrieval_ of an attribute change the object's dict?), however there is a chance
    # the class has a __getattr__ override that does something magical like automatically returning
    # None regardless of whether or not the attribute is defined.

    object_getattr = obj.__getattribute__ # This is the most basic "getattr" that doesn't involve any magic

    NoAttribute, NoItem = object(), object() # Unique identifiers

    _getattr_old = obj.__getattr__ if hasattr(obj, "__getattr__") else None
    _setattr_old = obj.__setattr__
    _delattr_old = obj.__delattr__
    
    def _getattr(self, name):
        # TODO: Do stuff in here.
        return _getattr_old(name)

    if has_immutable_slots:
        # If the object has immutable slots we can speed this up a ton by avoiding the
        # unnecessary try-except block.
        def _setattr(self, name, value):
            retrieved = object_getattr(self, name)
            if type(retrieved) != type(value):
                vg.remove_node_by_value(
                    name, context, 
                    recursive=True,
                    initial_relation=WordRelations._get_attr_rel(type(retrieved)))
                if not _is_base_type(type(value)):
                    vg.add_node(name, context, WordRelations._get_attr_rel(type(value)))
            _setattr_old(name, value)
    else:
        def _setattr(self, name, value):
            try:
                retrieved = object_getattr(self, name)
            except:
                retrieved = NoAttribute
            val_type = type(value)
            if retrieved == NoAttribute and not _is_base_type(val_type):
                vg.add_node(name, context, WordRelations._get_attr_rel(val_type))
            elif type(retrieved) != val_type:
                vg.remove_node_by_value(
                    name, context, 
                    recursive=True, 
                    initial_relation=WordRelations._get_attr_rel(type(retrieved)))
                if not _is_base_type(val_type):
                    vg.add_node(name, context, WordRelations._get_attr_rel(val_type))
            _setattr_old(name, value)
            
    def _delattr(self, name):
        try:
            retrieved = object_getattr(self, name)
        except AttributeError:
            # if we can't retrieve the object then we can't delete it either
            # so just call delattr and throw the error regardless.
            pass
        _delattr_old(name)
        vg.remove_node_by_value(
            name, context, 
            recursive=True, 
            initial_relation=WordRelations._get_attr_rel(type(retrieved)))

    _setitem = _delitem = None
    if isinstance(obj, (dict, list, tuple)):
        if isinstance(obj, (list, tuple)):
            node_name = _ITERABLE_ITEM_NAME_FORMAT % context.value
            new_node_format = lambda item: node_name
            word_relation = WordRelations.IS_ELEM_OF_L
        else:
            new_node_format = lambda item: item
            word_relation = WordRelations.IS_ELEM_OF_D
        _getitem_old = obj.__getitem__
        _setitem_old = obj.__setitem__
        _delitem_old = obj.__delitem__
        def _setitem(self, item, value):
            try:
                retrieved = _getitem_old(item)
            except:
                retrieved = NoItem
            val_type = type(value)
            if retrieved == NoItem and not _is_base_type(val_type):
                vg.add_node(new_node_format(item), context, word_relation)
            else:
                if type(retrieved) != val_type:
                    vg.remove_node_by_value(
                        new_node_format(item), context, 
                        recursive=True, 
                        initial_relation=word_relation)
                if not _is_base_type(val_type):
                    vg.add_node(new_node_format(item), context, word_relation)
            _setitem_old(item, value)

        def _delitem(self, item):
            _delitem_old(item)
            vg.remove_node_by_value(
                new_node_format(item), context, 
                recursive=True, 
                initial_relation=word_relation)

    new__dict__ = {
            "__getattr__"  : _getattr,
            "__setattr__"  : _setattr,
            "__delattr__"  : _delattr
        }
    if _setitem:
        new__dict__["__setitem__"] = _setitem
    if _delitem:
        new__dict__["__delitem__"] = _delitem

    try:
        # Magic
        obj.__class__ = type(
            "_%s" % obj.__class__.__name__,
            (obj.__class__, _DynamicVocabWatcher),
            new__dict__
        )
    except:
        # Sometimes an object's metaclass prevents it from being subclassed, preventing the
        # above trick from working. If this is the case, then maybe there's something else
        # we can do?
        
        # TODO (low priority): Figure out if there's some other way to monitor updates dynamically.
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
            # TODO: Ensure models managed by CT have a "callbacks" attribute.
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

        model.fit(x=inputs, y=outputs, batch_size=4, epochs=50, callbacks=callbacks)

        you would call
        model.fit(
            vocab_extractor.follow(
                {"x" : inputs, "y" : outputs, "batch_size" : 4, "epochs" : 50, "callbacks" : callbacks},
                contexts=contexts))
        """

        # TODO: Implement this properly
        # TODO: Figure out what "contexts" should be, or if it's even necessary.
        raise NotImplementedError
        # return dict # Eventually
        
if __name__ == "__main__":
    # quick testing utilities

    def test_keras():
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
    
    def test_vg():
        class A:
            def __init__(self, attr1, attr2):
                self.attr1 = attr1
                self.attr2 = attr2

            def method_1(self):
                pass

            def method_2(self, arg1, arg2):
                pass

        class B:
            def __init__(self):
                self.property = "Hey!"
            
            def method_1(self, thing):
                pass

        class C:
            def __init__(self):
                self.dict = {
                    "Entry 1" : 1,
                    "Entry 2" : 2
                }
                self.list = [D(), E()]

                self.function = lambda hey, you, *args, **kwargs: None

        class D:
            cls_thing = ":O"
    
        class E:
            def __init__(self):
                self.a = A(1, 2)

        a = A(B(), C())

        from pprint import pprint
        vg = VocabGraph([])
        root_node = vg.add_node("a", None, None)
        _extract_vocab_recursive(a, vg, root_node)
        pprint(vg._graph_by_values)

    test_vg()