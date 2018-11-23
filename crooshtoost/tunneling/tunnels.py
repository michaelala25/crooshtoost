from abc import ABC, abstractmethod

import json
import subprocess
import urllib

class TunnelingException(Exception):
    pass

class AbstractTunnel(ABC):
    @abstractmethod
    def expose(self, localhost_port):
        """
        Expose the localhost server, and return the public URL to connect to the localhost.
        """
        pass

class NgrokTunnel(AbstractTunnel):
    def expose(self, localhost_port):
        subprocess.Popen(["ngrok", "http", localhost_port])

        # Thanks to github user "peakwinter" for help figuring out this code.
        # Source: https://github.com/peakwinter/python-ngrok
        local_url = "http://127.0.0.1:4040/api/tunnels"
        request = urllib.request.Request(local_url)
        request.add_header("Content-Type", "application/json")
        response = urllib.request.urlopen(request)
        tunnels = json.loads(response.read())["tunnels"]

        if not len(tunnels):
            raise TunnelingException("Couldn't create a tunnel with ngrok.")

        # Return the https tunnel by default, or else just the first tunnel we find.
        for tunnel in tunnels:
            if tunnel["public_url"].startswith("https"):
                return tunnel["public_url"]
        else:
            return tunnels[0]["public_url"]

class ServeoTunnel(AbstractTunnel):
    def __init__(self, subdomain="crooshtoost.servero.net"):
        self.subdomain = subdomain

    def expose(self, localhost_port):
        subprocess.Popen([
            "ssh", "-R", 
            "%s:80:localhost:%s" % (self.subdomain, localhost_port), 
            "serveo.net"])
        return "https:%s" % self.subdomain

class CustomTunnel(AbstractTunnel):
    def __init__(self, expose_method):
        self.expose_method = expose_method

    def expose(self, localhost_port):
        return self.expose_method(localhost_port)