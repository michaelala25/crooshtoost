"""
The FlaskProcess handles user input and output (both via SMS text).

The NLP Intent and Entity extraction does not happen here, but the FlaskProcess does facilitate
interactions between the I/O and the various NLP algorithms.
"""

from .globals import GLOBALS

from flask import Flask, request
from twilio.rest import Client
from twilio.base.exceptions import TwilioException
from twilio.twiml.messaging_response import MessagingResponse

from multiprocessing import Process, Queue

import logging

class FlaskProcess(Process):

    def __init__(self, message_queue, account_sid, auth_token, 
                 phone_number, port, tunneling_method, name):
        super().__init__()
        self.message_queue = message_queue

        self.account_sid = account_sid
        self.auth_token = auth_token
        self.phone_number = phone_number

        self.port = port
        self.tunneling_method = tunneling_method

        self.name = name

        # Public url is resolved once a tunnel is created with the tunneling_method.
        self.public_url = None

        self.twilio_client = None
        self.flask_app = None

    def _expose_localhost(self):
        self.public_url = self.tunneling_method.expose(self.port)
        
    def _initialize(self):
        self._expose_localhost()

        self.twilio_client = Client(self.account_sid, self.auth_token)
        phone_numbers = self.twilio_client.incoming_phone_numbers.list(phone_number=self.phone_number)
        if not len(phone_numbers):
            raise TwilioException(
                "No phone number %s found for the given account SID (%s)." % (self.phone_number, self.account_sid))

        if len(phone_numbers) > 1 and GLOBALS.LOGGING_LEVEL <= logging.WARNING:
            logging.log(
                logging.WARNING, 
                "Multiple incoming phone numbers with given number " +
                "%s found, taking the first one." % self.phone_number
                )
        # Just take the first matching number, it's probably good enough
        phone_number = phone_numbers[0]

        # Update the phone number on Twilio to take our publicly exposed 
        # localhost server as the sms_url.
        phone_number.update(sms_url=self.public_url + "/sms")

        self._make_routes()

    def _make_routes(self):
        self.flask_app = Flask(self.name)
        self.flask_app.route("/sms", methods=["GET", "POST"])(self._sms_reply)

    def _sms_reply(self):
        # This is where all the good stuff goes :D
        message = request.form["Body"]

        response = MessagingResponse()
        response.message("test message")
        return str(response)

    def run(self):
        self._initialize()

        self.flask_app.run(debug=True)