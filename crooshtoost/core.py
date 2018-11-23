"""
core.py exposes a public interface through the CrooshToost object.
"""

from .flask_process import FlaskProcess
from .model_process import ModelProcess
from .tunneling import ServeoTunnel

from multiprocessing import Queue

class CrooshToost:

    """
    initial_settings=None,
    data_downloader=None,
    data_processor=None,
    train_setup=None
    """

    @property
    def account_sid(self):
        return self.twilio_account_sid
    @account_sid.setter
    def account_sid(self, x):
        self.twilio_account_sid = x

    @property
    def auth_token(self):
        return self.twilio_auth_token
    @auth_token.setter
    def auth_token(self, x):
        self.twilio_auth_token = x

    @property
    def phone_number(self):
        return self.twilio_phone_number
    @phone_number.setter
    def phone_number(self, x):
        self.twilio_phone_number = x

    def __init__(self,
                account_sid=None,
                auth_token=None,
                phone_number=None,
                localhost_port=None,
                localhost_tunneling_method=ServeoTunnel(),
                name="CrooshToost"):
        self.twilio_account_sid = account_sid
        self.twilio_auth_token = auth_token
        self.twilio_phone_number = phone_number

        self.localhost_port = localhost_port
        self.localhost_tunneling_method = localhost_tunneling_method

        self.name = name

        self.flask_process = None
        self.model_process = None

    def set_model(self, model):
        pass

    def _setup_processes(self):
        self.message_queue = Queue()

        self.flask_process = FlaskProcess(
            self.message_queue,
            self.twilio_account_sid,
            self.twilio_auth_token,
            self.twilio_phone_number,
            self.localhost_port, 
            self.localhost_tunneling_method,
            self.name)

        self.model_process = ModelProcess(self.message_queue)

    def run(self):
        self._setup_processes()

        self.flask_process.start()
        self.model_process.start()

        self.flask_process.join()
        self.model_process.join()