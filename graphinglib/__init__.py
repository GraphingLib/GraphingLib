"""
# GraphingLib

A simpler way to visualise data.
"""

from .data_plotting_1d import Curve, Histogram, Scatter
from .data_plotting_2d import Heatmap
from .figure import Figure
from .fits import (
    FitFromExponential,
    FitFromGaussian,
    FitFromLog,
    FitFromPolynomial,
    FitFromSine,
    FitFromSquareRoot,
    FitFromFunction,
)
from .graph_elements import GraphingException, Hlines, Point, Text, Vlines
