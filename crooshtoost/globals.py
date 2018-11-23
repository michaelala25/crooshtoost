import logging

class Globals:
    __INSTANCE__ = None
    
    def __new__(cls):
        if cls.__INSTANCE__:
            return cls.__INSTANCE__
        return object.__new__(cls)

GLOBALS = Globals()

GLOBALS.LOGGING_LEVEL = logging.DEBUG