from importlib.metadata import version

__version__ = version('sonify')
del version

__all__ = ['sonify']

from .sonify import sonify
