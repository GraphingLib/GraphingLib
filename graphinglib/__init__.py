"""
GraphingLib
===========

An object oriented wrapper combining the functionalities of Matplotlib and Scipy.

Provides one-line commands for:
    1. creating 1D and 2D plottable objects like curve plots, scatter plots, etc.
    2. creating curve fits using any function (builtin or not)
    3. getting the derivative, integral or tangent (and many other) of a curve

Also provides the ability to create multiple visual styles for your plots and save
them to be used anytime you want.

Notes
-----

Graphinglib uses the same named colors, line styles, marker styles and other conventions
as Matplotlib. For further reference on those conventions, see the
[Matplotlib documentation](https://matplotlib.org/stable/index.html).
"""

from .data_plotting_1d import Curve, Histogram, Scatter
from .data_plotting_2d import Contour, Heatmap, Stream, VectorField
from .figure import Figure
from .file_manager import get_color, get_colors
from .fits import (
    FitFromExponential,
    FitFromFunction,
    FitFromGaussian,
    FitFromLog,
    FitFromPolynomial,
    FitFromSine,
    FitFromSquareRoot,
)
from .graph_elements import GraphingException, Hlines, Point, Table, Text, Vlines
from .multifigure import MultiFigure
from .shapes import Circle, Rectangle, Arrow, Line
from ._version import __version__
