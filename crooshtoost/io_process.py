"""
The IOProcess handles user input and output (both via SMS text).

The NLP Intent and Entity extraction does not happen here, but the IOProcess does facilitate
interactions between the I/O and the various NLP algorithms.
"""

import multiprocessing as mp

class IOProcess(mp.Process):
    
    def __init__(self):
        super().__init__()
        
    def run(self):
        pass