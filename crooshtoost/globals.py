import logging

class Globals:
    __INSTANCE__ = None
    
    def __new__(cls):
        if not cls.__INSTANCE__:
            cls.__INSTANCE__ = object.__new__(cls)
        return cls.__INSTANCE__

GLOBALS = Globals()

# The default global logging level throughout the program.
GLOBALS.LOGGING_LEVEL = logging.DEBUG

# This global variable determines whether we choose to dynamically update
# the vocab graph throughout the runtime of the program.
#
# The reason you may want to turn this off is because the dynamic updating
# code introduces significant overhead. Also, it's a reasonable assumption
# that the working vocab graph doesn't update very often.
GLOBALS.ENABLE_DYNAMIC_VOCAB_UPDATES = True