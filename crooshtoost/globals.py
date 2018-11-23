import logging

class Globals:
    __INSTANCE__ = None
    
    def __new__(cls):
        if not cls.__INSTANCE__:
            cls.__INSTANCE__ = object.__new__(cls)
        return cls.__INSTANCE__

GLOBALS = Globals()

GLOBALS.LOGGING_LEVEL = logging.DEBUG