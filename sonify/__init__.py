from importlib.metadata import version

__version__ = version('sonify')
del version

from .sonify import sonify
