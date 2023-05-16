try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .operator import WaveletDec2, stackedWaveletDec

__all__ = (
    "WaveletDec2",
    "stackedWaveletDec",
)
