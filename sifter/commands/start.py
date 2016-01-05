import logging
import os
from .base import ICommand
from zope.interface import implementer

__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'
logger = logging.getLogger(__name__)

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_TRAINING_FILE = 'data/RC_2010-02-20k.json'


@implementer(ICommand)
class StartCommand:
    command = "start"
    usage = "start"
    short = "Start the twister event loop."
    docs = """See the sifter.yaml configuration for details."""

    def __init__(self):
        self.vocabulary_size = _VOCABULARY_SIZE
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"

    def setenv(self, env):
        self.env = env

    def validate(self, args):
        return args

    def shutdown(self):
        logger.info("shutting down")

    def execute(self, args=None):
        ok = True
        self.load_model()
        self.run_twisted()
        return ok

    def load_model(self):
        pass

    def run_twisted(self):
        from twisted.internet import protocol, reactor

        class SifterProtocol(protocol.Protocol):
            pass

        class SifterFactory(protocol.ServerFactory):
            protocol = SifterProtocol

        reactor.listenTCP(1079, SifterFactory())

        logger.info("starting twisted event loop")
        reactor.addSystemEventTrigger('before', 'shutdown', self.shutdown)
        reactor.run()
