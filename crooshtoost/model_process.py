"""
The ModelProcess handles model related tasks such as training, validating, evaluating,
testing, visualizing, etc.
"""

import multiprocessing as mp

class ModelProcess(mp.Process):
    
    def __init__(self):
        super().__init__()

    def run(self):
        pass