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

from ._version import __version__
from .data_plotting_1d import Curve, Histogram, Plottable1D, Scatter
from .data_plotting_2d import Contour, Heatmap, Plottable2D, Stream, VectorField
from .figure import Figure
from .file_manager import (
    get_color,
    get_colors,
    get_default_style,
    get_styles,
    set_default_style,
)
from .fits import (
    FitFromExponential,
    FitFromFOTF,
    FitFromFunction,
    FitFromGaussian,
    FitFromLog,
    FitFromPolynomial,
    FitFromSine,
    FitFromSquareRoot,
)
from .exceptions import (
    GraphingException,
    GraphingLibError,
    IncompatibleArgumentsError,
    InvalidOperationError,
    InvalidParameterError,
    InvalidParameterTypeError,
    LayoutError,
    MissingOptionalDependencyError,
    PlottingError,
    StyleFileError,
    StyleNotFoundError,
    UnsupportedFeatureError,
)
from .graph_elements import (
    Hlines,
    Plottable,
    PlottableAxMethod,
    Point,
    Table,
    Text,
    Vlines,
)
from .legend_artists import LegendElement, LegendLine, LegendMarker, LegendPatch
from .inherit import INHERIT, Inherit, Styled, is_inherit

# MultiFigure is deprecated but intentionally re-exported for backward compatibility.
from .multifigure import MultiFigure  # ty: ignore[deprecated]
from .shapes import Arrow, Circle, Ellipse, Line, Polygon, Rectangle
from .smart_figure import SmartFigure, SmartFigureWCS, SmartTwinAxis
from .tools import MathematicalObject

__all__ = [
    "__version__",
    "Curve",
    "Histogram",
    "Plottable1D",
    "Scatter",
    "Contour",
    "Heatmap",
    "Plottable2D",
    "Stream",
    "VectorField",
    "Figure",
    "get_color",
    "get_colors",
    "get_default_style",
    "get_styles",
    "set_default_style",
    "FitFromExponential",
    "FitFromFOTF",
    "FitFromFunction",
    "FitFromGaussian",
    "FitFromLog",
    "FitFromPolynomial",
    "FitFromSine",
    "FitFromSquareRoot",
    "GraphingException",
    "GraphingLibError",
    "InvalidParameterError",
    "InvalidParameterTypeError",
    "IncompatibleArgumentsError",
    "InvalidOperationError",
    "LayoutError",
    "StyleNotFoundError",
    "StyleFileError",
    "MissingOptionalDependencyError",
    "UnsupportedFeatureError",
    "PlottingError",
    "Hlines",
    "Plottable",
    "PlottableAxMethod",
    "Point",
    "Table",
    "Text",
    "Vlines",
    "LegendElement",
    "LegendLine",
    "LegendMarker",
    "LegendPatch",
    "INHERIT",
    "Inherit",
    "Styled",
    "is_inherit",
    "MultiFigure",
    "Arrow",
    "Circle",
    "Ellipse",
    "Line",
    "Polygon",
    "Rectangle",
    "SmartFigure",
    "SmartFigureWCS",
    "SmartTwinAxis",
    "MathematicalObject",
]
