"""
core.py exposes a public interface through the CrooshToost object.
"""

from .flask_process import FlaskProcess
from .model_process import ModelProcess

from multiprocessing import Queue

class CrooshToost:

    """
    initial_settings=None,
    data_downloader=None,
    data_processor=None,
    train_setup=None
    """

    def __init__(self):
        pass

    def set_model(self, model):
        pass

    def _setup_processes(self):
        self.message_queue = Queue()

        self.io_process = FlaskProcess(self.message_queue)
        self.model_process = ModelProcess(self.message_queue)

    def run(self):
        self._setup_processes()

        self.io_process.start()
        self.model_process.start()

        self.io_process.join()
        self.model_process.join()