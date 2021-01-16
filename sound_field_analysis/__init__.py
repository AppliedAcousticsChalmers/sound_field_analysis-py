"""
Analyze, visualize and process sound field data recorded by spherical microphone
arrays.
"""

from ._version import __version__

__all__ = ["io", "gen", "process", "sph", "utils", "lebedev"]

try:
    import plotly

    __all__.append("plot")  # provide `plot` component if `plotly` is available
    del plotly
except ModuleNotFoundError:
    pass

from . import *  # import all components in `__all__`
