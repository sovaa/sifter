import logging
from zope.interface import Interface, Attribute

__author__ = 'Oscar Eriksson <oscar.eriks@gmail.com>'
logger = logging.getLogger(__name__)


class ICommand(Interface):
    command = Attribute("""Command string""")
    usage = Attribute("""Usage string""")
    short = Attribute("""Short command description""")
    docs = Attribute("""Long documentation string""")

    def setenv(self, env):
        """
        Set the environment in which the command is running.
        """

    def validate(self, args):
        """
        Validate the arguments.
        @throws RuntimeException if arguments are invalid in some way.
        """

    def execute(self, *args):
        """
        Execute the command with the specified set of arguments.
        """
