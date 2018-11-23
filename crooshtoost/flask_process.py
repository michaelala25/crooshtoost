"""
The FlaskProcess handles user input and output (both via SMS text).

The NLP Intent and Entity extraction does not happen here, but the FlaskProcess does facilitate
interactions between the I/O and the various NLP algorithms.
"""

from flask import Flask
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from multiprocessing import Process, Queue

class FlaskProcess(Process):
    
    def __init__(self, message_queue):
        super().__init__()
        self.message_queue = message_queue

    def expose_localhost(self):
        pass
        
    def run(self):
        pass