"""
The ModelProcess handles model related tasks such as training, validating, evaluating,
testing, visualizing, etc.
"""

import multiprocessing as mp

class ModelProcess(mp.Process):
    
    def __init__(self, message_queue):
        super().__init__()
        self.message_queue = message_queue

    def run(self):
        # This object's logic is actually pretty complicated.
        # 
        # While the model is being trained, the message_queue has to be read by the
        # CTCallback object.
        #
        # After training, we have to enter an environment where we can either analyze
        # the current model, modify it, load new data, load a new model, or retrain it.
        # In this environment, the message_queue is read by the ModelProcess itself.
        pass