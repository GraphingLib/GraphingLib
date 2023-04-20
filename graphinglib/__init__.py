"""
# GraphingLib

A simpler way to visualise data.
"""

from .data_plotting_1d import Curve, Dashed, Histogram, Scatter
from .figure import Figure
from .fits import (
    FitFromExponential,
    FitFromGaussian,
    FitFromPolynomial,
    FitFromSine,
    FitFromSquareRoot,
)
from .graph_elements import GraphingException, Hlines, Point, Text, Vlines
