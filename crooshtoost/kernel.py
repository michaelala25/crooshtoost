"""
The CTKernel is the central object which manages the two main CT Threads, the IOThread and
the ModelManagerThread.

The CTKernel facilitates communication between these two threads.
"""

from .io_process import IOProcess
from .model_process import ModelProcess

class CTKernel:

    def __init__(self):
        pass

    def set_model(self, model):
        pass

    def _setup_processes(self):
        self.io_process = IOProcess()
        self.model_process = ModelProcess()

    def run(self):
        self._setup_processes()

        self.io_process.start()
        self.model_process.start()

        self.io_process.join()
        self.model_process.join()