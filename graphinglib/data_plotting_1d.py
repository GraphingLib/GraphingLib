from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from types import NoneType
from typing import Any, Callable, Optional, Protocol, cast, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.colors import Colormap, Normalize, is_color_like, to_rgba, to_rgba_array
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import nonsingular
from numpy.typing import ArrayLike
from pyperclip import copy as copy_to_clipboard
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import interp1d

from ._animation import (
    _DRAW_SESSION,
    _ERROR_STYLE_PROPERTIES,
    _AnimationSession,
    _ArtistEdit,
    _attach_artists,
    _bound_session,
    _combine_edits,
    _configure_colorbar,
    _CurveBands,
    _draw_live_errorbars,
    _edit,
    _error_array,
    _error_bounds,
    _error_style_edits,
    _fill_edit,
    _reserve_errorbar_directions,
    _stage_live_value,
)
from .exceptions import (
    IncompatibleArgumentsError,
    InvalidParameterError,
    InvalidParameterTypeError,
)
from .graph_elements import Plottable, Point
from .inherit import INHERIT, Inherit, Styled, is_inherit, resolve_or, strip_inherit
from .tools import MathematicalObject, get_contrasting_shade

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

_SNAPPING_INTERPOLATION_METHODS = frozenset({"nearest", "previous", "next"})


class _KeepError:
    pass


_KEEP_ERROR = _KeepError()


def _set_point_data(
    element: Curve | Scatter,
    x_data: ArrayLike,
    y_data: ArrayLike,
    x_error: ArrayLike | None | _KeepError,
    y_error: ArrayLike | None | _KeepError,
) -> None:
    x, y = np.array(x_data, copy=True), np.array(y_data, copy=True)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(
            "x_data and y_data must be one-dimensional and have the same length."
        )
    old_dx, old_dy = (
        getattr(element, "_x_error", None),
        getattr(element, "_y_error", None),
    )
    dx = _error_array(
        old_dx if isinstance(x_error, _KeepError) else x_error, x.size, "x_error"
    )
    dy = _error_array(
        old_dy if isinstance(y_error, _KeepError) else y_error, x.size, "y_error"
    )
    session = _bound_session(element)
    if session is not None:
        previous = element._x_data, element._y_data, old_dx, old_dy

        def restore() -> None:
            element._x_data, element._y_data, element._x_error, element._y_error = (
                previous
            )

        session.queue_points(
            element,
            x,
            y,
            dx,
            dy,
            show_errorbars=element._show_errorbars,
            restore=restore,
        )
    element._x_data, element._y_data, element._x_error, element._y_error = x, y, dx, dy


def _add_live_errorbars(
    element: Curve | Scatter,
    x_error: ArrayLike | None,
    y_error: ArrayLike | None,
    **styles: Any,
) -> None:
    element.set_data(
        element.x_data,
        element.y_data,
        x_error=_KEEP_ERROR if x_error is None else x_error,
        y_error=_KEEP_ERROR if y_error is None else y_error,
    )
    for name, value in styles.items():
        if not is_inherit(value):
            setattr(element, name, value)
    element.show_errorbars = True


@runtime_checkable
class Fit(Protocol):
    """
    Dummy class to allow type hinting of Fit objects.
    """

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified axes.
        """
        pass

    def show_residual_curves(
        self,
        sigma_multiplier: float,
        color: str,
        line_width: float,
        line_style: str,
    ) -> None:
        pass

    def get_residuals(self) -> np.ndarray:
        pass


@runtime_checkable
class Plottable1D(Plottable, Protocol):
    """
    Dummy class to allow type hinting of Plottable1D objects.
    """

    def to_desmos(self, decimal_precision: int = 2, to_clipboard: bool = False) -> str:
        pass


def _check_same_length(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    """
    Raises an evocative error if two data arrays don't have matching shapes.

    Validating here, at the boundary, surfaces the mistake at the user's call site instead
    of much later as a cryptic matplotlib error during plotting.
    """
    if a.shape != b.shape:
        raise IncompatibleArgumentsError(
            f"{name_a} and {name_b} must have the same shape, but got "
            f"{a.shape} and {b.shape}."
        )


def _check_error_shape(
    name_err: str, err: np.ndarray, name_data: str, data: np.ndarray
) -> None:
    """
    Raises an evocative error if an error array's shape can't line up with its data.

    Matplotlib accepts a scalar, a length-n 1D array, or a (2, n) array (asymmetric); any
    other shape is caught here rather than deep inside ``errorbar`` at plotting time.
    """
    n = data.shape[0] if data.ndim >= 1 else 1
    valid = (
        err.ndim == 0
        or (err.ndim == 1 and err.shape[0] == n)
        or (err.ndim == 2 and err.shape == (2, n))
    )
    if not valid:
        raise IncompatibleArgumentsError(
            f"{name_err} must be a scalar, a length-{n} 1D array, or a (2, {n}) array to "
            f"match {name_data}, but got shape {err.shape}."
        )


def _format_desmos_points(
    x_data: ArrayLike, y_data: ArrayLike, decimal_precision: int = 2
) -> str:
    """
    Gives the data points in a Desmos-readable format.

    .. note::
        NaN values are ignored.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    sorted_indices = np.argsort(x_data)
    sorted_x_data = x_data[sorted_indices]
    sorted_y_data = y_data[sorted_indices]

    # Change exponential formatting to be interpretable by Desmos.
    def format_tex(num: str, exponent: str):
        num = num.rstrip("0")
        if exponent == "+00":
            return str(num)
        else:
            return rf"{num}\cdot10^" + "{" + str(int(exponent)) + "}"

    formatted_points = "["
    for x, y in zip(sorted_x_data, sorted_y_data):
        if np.isnan(x) or np.isnan(y):
            continue
        x_num, x_exponent = f"{x:.{decimal_precision:d}e}".split("e")
        y_num, y_exponent = f"{y:.{decimal_precision:d}e}".split("e")
        formatted_points += (
            f"({format_tex(x_num, x_exponent)},{format_tex(y_num, y_exponent)}),"
        )
    formatted_points = formatted_points[:-1] + "]"
    return formatted_points


@dataclass
class Curve(Plottable1D, MathematicalObject):
    """
    This class implements a general continuous curve.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        Arrays of x and y values to be plotted.
    label : str, optional
        Label to be displayed in the legend.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : float
        Width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Style of the curve.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the curve.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> None:
        self.handle = None
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        _check_same_length("x_data", self._x_data, "y_data", self._y_data)
        self._label = label
        self._color = color
        self._line_width = line_width
        self._line_style = line_style
        self._alpha = alpha

        self._x_error = None
        self._y_error = None

        self._show_errorbars: bool = False
        self._errorbars_color: Styled[str | None] = None
        self._errorbars_line_width: Styled[float | None] = None
        self._cap_thickness: Styled[float | None] = None
        self._cap_width: Styled[float | None] = None

        self._show_error_curves: bool = False
        self._error_curves_fill_between: Styled[bool] = False
        self._error_curves_color: Styled[str | None] = None
        self._error_curves_line_style: Styled[str | None] = None
        self._error_curves_line_width: Styled[float | None] = None

        self._fill_between_bounds: Optional[tuple[float, float]] = None
        self._fill_between_other_curve: Curve | None = None
        self._fill_between_color: Styled[str | None] = None

    @classmethod
    def from_function(
        cls,
        func: Callable[[np.ndarray], ArrayLike],
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        number_of_points: int = 500,
    ) -> Self:
        """
        Creates a :class:`~graphinglib.data_plotting_1d.Curve` from a function and a range of x values.

        Parameters
        ----------
        func : Callable[[ArrayLike], ArrayLike]
            Function to be plotted. Works with regular functions and lambda functions.
        x_min, x_max : float
            The :class:`~graphinglib.data_plotting_1d.Curve` will be plotted between these two values.
        label : str, optional
            Label to be displayed in the legend.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        number_of_points : int
            Number of points to be used to plot the curve (resolution).
            Defaults to 500.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object created from the given function and x range.
        """
        x_data = np.linspace(x_min, x_max, number_of_points)
        y_data = func(x_data)
        return cls(x_data, y_data, label, color, line_width, line_style, alpha)

    @property
    def x_data(self) -> np.ndarray:
        return self._x_data

    @x_data.setter
    def x_data(self, x_data: ArrayLike) -> None:
        if _bound_session(self) is None:
            self._x_data = np.asarray(x_data)
        else:
            self.set_data(x_data, self._y_data)

    @property
    def y_data(self) -> np.ndarray:
        return self._y_data

    @y_data.setter
    def y_data(self, y_data: ArrayLike) -> None:
        if _bound_session(self) is None:
            self._y_data = np.asarray(y_data)
        else:
            self.set_data(self._x_data, y_data)

    @property
    def x_error(self) -> np.ndarray | None:
        return self._x_error

    @x_error.setter
    def x_error(self, x_error: ArrayLike) -> None:
        if _bound_session(self) is None:
            self._x_error = np.asarray(x_error)
        else:
            self.set_data(self._x_data, self._y_data, x_error=x_error)

    @property
    def y_error(self) -> np.ndarray | None:
        return self._y_error

    @y_error.setter
    def y_error(self, y_error: ArrayLike) -> None:
        if _bound_session(self) is None:
            self._y_error = np.asarray(y_error)
        else:
            self.set_data(self._x_data, self._y_data, y_error=y_error)

    def set_data(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        *,
        x_error: ArrayLike | None | _KeepError = _KEEP_ERROR,
        y_error: ArrayLike | None | _KeepError = _KEEP_ERROR,
    ) -> Self:
        """Replace coordinates and uncertainties as one validated update.

        Parameters
        ----------
        x_data, y_data : ArrayLike
            One-dimensional coordinate arrays of the same length.
        x_error, y_error : ArrayLike or None, optional
            Scalar, per-point, or lower/upper (2 by N) nonnegative errors.
            Omit an error to retain it or pass None to clear it. On Cartesian
            axes, this can add or remove a direction during animation.

        Returns
        -------
        Curve
            The updated curve. During an animation the line, bars, and caps
            are refreshed together before drawing the frame.
        """
        _set_point_data(self, x_data, y_data, x_error, y_error)
        return self

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = _stage_live_value(self, "_label", label)

    @property
    def color(self) -> Styled[str]:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = _stage_live_value(self, "_color", color)

    @property
    def line_width(self) -> float | Inherit:
        return self._line_width

    @line_width.setter
    def line_width(self, line_width: float | Inherit) -> None:
        self._line_width = _stage_live_value(self, "_line_width", line_width)

    @property
    def line_style(self) -> Styled[str]:
        return self._line_style

    @line_style.setter
    def line_style(self, line_style: str) -> None:
        self._line_style = _stage_live_value(self, "_line_style", line_style)

    @property
    def alpha(self) -> float | Inherit:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Inherit) -> None:
        self._alpha = _stage_live_value(self, "_alpha", alpha)

    @property
    def show_errorbars(self) -> bool:
        return self._show_errorbars

    @show_errorbars.setter
    def show_errorbars(self, show_errorbars: bool) -> None:
        self._show_errorbars = _stage_live_value(self, "_show_errorbars", show_errorbars)

    @property
    def errorbars_color(self) -> Styled[str | None]:
        return self._errorbars_color

    @errorbars_color.setter
    def errorbars_color(self, errorbars_color: str) -> None:
        self._errorbars_color = _stage_live_value(self, "_errorbars_color", errorbars_color)

    @property
    def errorbars_line_width(self) -> Styled[float | None]:
        return self._errorbars_line_width

    @errorbars_line_width.setter
    def errorbars_line_width(self, errorbars_line_width: float | Inherit) -> None:
        self._errorbars_line_width = _stage_live_value(self, "_errorbars_line_width", errorbars_line_width)

    @property
    def cap_thickness(self) -> Styled[float | None]:
        return self._cap_thickness

    @cap_thickness.setter
    def cap_thickness(self, cap_thickness: float | Inherit) -> None:
        self._cap_thickness = _stage_live_value(self, "_cap_thickness", cap_thickness)

    @property
    def cap_width(self) -> Styled[float | None]:
        return self._cap_width

    @cap_width.setter
    def cap_width(self, cap_width: float | Inherit) -> None:
        self._cap_width = _stage_live_value(self, "_cap_width", cap_width)

    @property
    def show_error_curves(self) -> bool:
        return self._show_error_curves

    @show_error_curves.setter
    def show_error_curves(self, show_error_curves: bool) -> None:
        self._show_error_curves = _stage_live_value(self, "_show_error_curves", show_error_curves)

    @property
    def error_curves_fill_between(self) -> Styled[bool]:
        return self._error_curves_fill_between

    @error_curves_fill_between.setter
    def error_curves_fill_between(self, error_curves_fill_between: bool) -> None:
        self._error_curves_fill_between = _stage_live_value(
            self, "_error_curves_fill_between", error_curves_fill_between
        )

    @property
    def error_curves_color(self) -> Styled[str | None]:
        return self._error_curves_color

    @error_curves_color.setter
    def error_curves_color(self, error_curves_color: str) -> None:
        self._error_curves_color = _stage_live_value(self, "_error_curves_color", error_curves_color)

    @property
    def error_curves_line_style(self) -> Styled[str | None]:
        return self._error_curves_line_style

    @error_curves_line_style.setter
    def error_curves_line_style(self, error_curves_line_style: str) -> None:
        self._error_curves_line_style = _stage_live_value(self, "_error_curves_line_style", error_curves_line_style)

    @property
    def error_curves_line_width(self) -> Styled[float | None]:
        return self._error_curves_line_width

    @error_curves_line_width.setter
    def error_curves_line_width(self, error_curves_line_width: float | Inherit) -> None:
        self._error_curves_line_width = _stage_live_value(self, "_error_curves_line_width", error_curves_line_width)

    @property
    def fill_between_bounds(self) -> tuple[float, float] | None:
        return self._fill_between_bounds

    @fill_between_bounds.setter
    def fill_between_bounds(
        self, fill_between_bounds: tuple[float, float] | None
    ) -> None:
        session = _bound_session(self)
        if fill_between_bounds is not None:
            if session is None:
                self._area_fill_configured = True
            else:
                fill_between_bounds = tuple(fill_between_bounds)
        self._fill_between_bounds = _stage_live_value(
            self, "_fill_between_bounds", fill_between_bounds
        )

    @property
    def fill_between_other_curve(self) -> Curve | None:
        return self._fill_between_other_curve

    @fill_between_other_curve.setter
    def fill_between_other_curve(self, fill_between_other_curve: Curve | None) -> None:
        session = _bound_session(self)
        name = "_fill_between_other_curve"
        previous = self._fill_between_other_curve
        if (
            session is not None
            and fill_between_other_curve is not None
            and not isinstance(fill_between_other_curve, Curve)
        ):
            raise InvalidParameterTypeError(
                "The fill boundary must be a Curve or None."
            )
        self._fill_between_other_curve = _stage_live_value(
            self, name, fill_between_other_curve
        )
        if session is not None:

            def restore_link() -> None:
                session.link_point_source(
                    self, name, self._fill_between_other_curve, previous
                )

            # Restore the link before the separate rollback of the property value.
            session.remember(self, "fill_boundary_link", restore_link)
            session.link_point_source(self, name, previous, fill_between_other_curve)

    @property
    def fill_between_color(self) -> Styled[str | None]:
        return self._fill_between_color

    @fill_between_color.setter
    def fill_between_color(self, fill_between_color: Styled[str | None]) -> None:
        self._fill_between_color = _stage_live_value(self, "_fill_between_color", fill_between_color)

    def __eq__(self, other: object) -> bool:
        """
        Defines the equality between two curves.
        """
        if not isinstance(other, Curve):
            return False
        return bool(
            np.equal(self.x_data, other.x_data).all()
            and np.equal(self.y_data, other.y_data).all()
        )

    def __add__(self, other: Curve | float) -> Curve:
        """
        Defines the addition of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    return Curve(x_data, y_data + other._y_data)
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    return Curve(x_data, y_data + self._y_data)

            new_y_data = self._y_data + other._y_data
            return Curve(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data + other
            return Curve(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only add a curve to another curve or a number."
            )

    def __sub__(self, other: Curve | float) -> Curve:
        """
        Defines the subtraction of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    return Curve(x_data, y_data - other._y_data)
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    return Curve(x_data, self._y_data - y_data)
            new_y_data = self._y_data - other._y_data
            return Curve(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data - other
            return Curve(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only subtract a curve from another curve or a number."
            )

    def __mul__(self, other: Curve | float) -> Curve:
        """
        Defines the multiplication of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    return Curve(x_data, y_data * other._y_data)
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    return Curve(x_data, y_data * self._y_data)
            new_y_data = self._y_data * other._y_data
            return Curve(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data * other
            return Curve(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only multiply a curve by another curve or a number."
            )

    def __truediv__(self, other: Curve | float) -> Curve:
        """
        Defines the division of two curves or a curve and a number.
        """
        if isinstance(other, Curve):
            if not np.array_equal(self._x_data, other._x_data):
                if len(self._x_data) > len(other._x_data):
                    x_data = other._x_data
                    y_data = interp1d(self._x_data, self._y_data)(x_data)
                    return Curve(x_data, y_data / other._y_data)
                else:
                    x_data = self._x_data
                    y_data = interp1d(other._x_data, other._y_data)(x_data)
                    return Curve(x_data, self._y_data / y_data)
            new_y_data = self._y_data / other._y_data
            return Curve(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data / other
            return Curve(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only divide a curve by another curve or a number."
            )

    def __pow__(self, other: float) -> Curve:
        """
        Defines the power of a curve to a number.
        """
        if isinstance(other, (int, float)):
            new_y_data = self._y_data**other
            return Curve(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only raise a curve to another curve or a number."
            )

    # Explicitly retain the protocol stub so type checkers treat Curve as concrete.
    __rpow__ = MathematicalObject.__rpow__

    def __iter__(self):
        """
        Defines the iteration of a curve. Returns the y values.
        """
        return iter(self._y_data)

    def __abs__(self) -> Curve:
        """
        Returns the absolute value of the curve.
        """
        return Curve(self._x_data, np.abs(self._y_data))

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_1d.Curve`.
        """
        return deepcopy(self)

    def create_slice_x(
        self,
        x1: float,
        x2: float,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        copy_first: bool = False,
    ) -> Curve:
        """
        Creates a slice of the curve between two x values.

        Parameters
        ----------
        x1, x2 : float
            The x values between which the slice is to be created.
        label : str, optional
            Label of the slice to be displayed in the legend.
        color : str
            Color of the slice.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the slice.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the slice.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the slice.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the slicing applied. Any
            other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve
            will be created with the slicing applied and the parameters passed to this method.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the slice of the original curve between the two x
        values.
        """
        mask = (self._x_data >= x1) & (self._x_data <= x2)
        x_data = self._x_data[mask]
        y_data = self._y_data[mask]
        if copy_first:
            copy = self.copy()
            copy._x_data = x_data
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != INHERIT:
                copy._color = color
            if line_width != INHERIT:
                copy._line_width = line_width
            if line_style != INHERIT:
                copy._line_style = line_style
            if alpha != INHERIT:
                copy._alpha = alpha
            return copy
        else:
            return Curve(x_data, y_data, label, color, line_width, line_style, alpha)

    def create_slice_y(
        self,
        y1: float,
        y2: float,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        copy_first: bool = False,
    ) -> Curve:
        """
        Creates a slice of the curve between two y values.

        Parameters
        ----------
        y1, y2 : float
            The y values between which the slice is to be created.
        label : str, optional
            Label of the slice to be displayed in the legend.
        color : str
            Color of the slice.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the slice.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the slice.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the slice.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the slicing applied. Any
            other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve
            will be created with the slicing applied and the parameters passed to this method.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the slice of the original curve between the two y
        values.
        """
        mask = (self._y_data >= y1) & (self._y_data <= y2)
        x_data = self._x_data[mask]
        y_data = self._y_data[mask]
        if copy_first:
            copy = self.copy()
            copy._x_data = x_data
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != INHERIT:
                copy._color = color
            if line_width != INHERIT:
                copy._line_width = line_width
            if line_style != INHERIT:
                copy._line_style = line_style
            if alpha != INHERIT:
                copy._alpha = alpha
            return copy
        else:
            return Curve(x_data, y_data, label, color, line_width, line_style, alpha)

    def add_errorbars(
        self,
        x_error: Optional[ArrayLike] = None,
        y_error: Optional[ArrayLike] = None,
        cap_width: float | Inherit = INHERIT,
        errorbars_color: str | Inherit = INHERIT,
        errorbars_line_width: float | Inherit = INHERIT,
        cap_thickness: float | Inherit = INHERIT,
    ) -> None:
        """
        Adds errorbars to the :class:`~graphinglib.data_plotting_1d.Curve`.

        Parameters
        ----------
        x_error, y_error : ArrayLike, optional
            Arrays of x and y errors. Use one or both.
            Values must be non-negative.
        cap_width : float
            Width of the errorbar caps.
            Typical range is ``0`` to ``10`` points.
            Default depends on the ``figure_style`` configuration.
        errorbars_color : str
            Color of the errorbars.
            ``"same as curve"`` uses the curve color.
            Default depends on the ``figure_style`` configuration.
        errorbars_line_width : float
            Width of the errorbars.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        cap_thickness : float
            Thickness of the errorbar caps.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        if _bound_session(self) is not None:
            _add_live_errorbars(
                self,
                x_error,
                y_error,
                cap_width=cap_width,
                errorbars_color=errorbars_color,
                errorbars_line_width=errorbars_line_width,
                cap_thickness=cap_thickness,
            )
            return
        self._show_errorbars = True
        self._errorbars_configured = True

        if x_error is not None:
            self._x_error = np.array(x_error)
            _check_error_shape("x_error", self._x_error, "x_data", self._x_data)

        if y_error is not None:
            self._y_error = np.array(y_error)
            _check_error_shape("y_error", self._y_error, "y_data", self._y_data)

        self._errorbars_color = errorbars_color
        self._errorbars_line_width = errorbars_line_width
        self._cap_thickness = cap_thickness
        self._cap_width = cap_width

    def add_error_curves(
        self,
        y_error: Optional[ArrayLike] = None,
        error_curves_color: str | Inherit = INHERIT,
        error_curves_line_style: str | Inherit = INHERIT,
        error_curves_line_width: float | Inherit = INHERIT,
        error_curves_fill_between: bool | Inherit = INHERIT,
    ) -> None:
        """
        Adds error curves to the :class:`~graphinglib.data_plotting_1d.Curve`.

        Parameters
        ----------
        y_error : ArrayLike, optional
            Array of y errors.
        error_curves_color : str
            Color of the error curves.
            ``"same as curve"`` uses the curve color.
            Default depends on the ``figure_style`` configuration.
        error_curves_line_style : str
            Line style of the error curves.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        error_curves_line_width : float
            Line width of the error curves.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        error_curves_fill_between : bool
            Whether or not to fill the area between the two error curves.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        if _bound_session(self) is not None:
            if y_error is not None:
                self.set_data(self._x_data, self._y_data, y_error=y_error)
            for name, value in (
                ("error_curves_color", error_curves_color),
                ("error_curves_line_style", error_curves_line_style),
                ("error_curves_line_width", error_curves_line_width),
                ("error_curves_fill_between", error_curves_fill_between),
            ):
                if not is_inherit(value):
                    setattr(self, name, value)
            self.show_error_curves = True
            return
        self._show_error_curves = True
        self._error_curves_configured = True

        if y_error is not None:
            self._y_error = np.array(y_error)

        self._error_curves_color = error_curves_color
        self._error_curves_line_style = error_curves_line_style
        self._error_curves_line_width = error_curves_line_width
        self._error_curves_fill_between = error_curves_fill_between

    def get_coordinates_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
    ) -> tuple[float, float]:
        """
        Gets the coordinates of the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value of the desired coordinates.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            For ``"nearest"``, ``"previous"``, and ``"next"``, the returned coordinates snap to an
            actual data point instead of interpolating; for every other kind, the returned x is the
            same as the queried ``x``.

            Defaults to "linear".

        Returns
        -------
        tuple[float, float]
            The coordinates of the curve at the given x value.
        """
        if interpolation_method in _SNAPPING_INTERPOLATION_METHODS:
            idx_interp = interp1d(
                self._x_data, np.arange(len(self._x_data)), kind=interpolation_method
            )
            idx = int(round(float(idx_interp(x))))
            return (float(self._x_data[idx]), float(self._y_data[idx]))
        return (
            x,
            float(interp1d(self._x_data, self._y_data, kind=interpolation_method)(x)),
        )

    def create_point_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
        label: Optional[str] = None,
        face_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> Point:
        """
        Creates a point on the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value of the point.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            For ``"nearest"``, ``"previous"``, and ``"next"``, the returned point snaps to an
            actual data point instead of interpolating; for every other kind, the point's x is the
            same as the queried ``x``.

            Defaults to "linear".
        label : str, optional
            Point's label to be displayed in the legend.
        face_color : str
            Face color of the point.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the point.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the point.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the point.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the point edge.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the point.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point`
            The point on the curve at the given x value.
        """
        x_val, y_val = self.get_coordinates_at_x(x, interpolation_method)
        point = Point(
            x_val,
            y_val,
            label=label,
            face_color=face_color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=line_width,
            alpha=alpha,
        )
        return point

    def get_coordinates_at_y(
        self,
        y: float,
        interpolation_method: str = "linear",
    ) -> list[tuple[float, float]]:
        """
        Gets the coordinates of the curve at a given y value. Can return multiple coordinate pairs if the curve crosses
        the y value multiple times.

        Parameters
        ----------
        y : float
            The y value of the desired coordinates.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            For ``"nearest"``, ``"previous"``, and ``"next"``, the returned coordinates snap to an
            actual data point instead of interpolating; for every other kind, the returned y is the
            same as the queried ``y``.

            Defaults to "linear".

        Returns
        -------
        list[tuple[float, float]]
            The coordinates of the points on the curve at the given y value.
        """
        xs = self._x_data
        ys = self._y_data
        crossings = np.where(np.diff(np.sign(ys - y)))[0]
        points: list[tuple[float, float]] = []
        for cross in crossings:
            x1, x2 = xs[cross], xs[cross + 1]
            y1, y2 = ys[cross], ys[cross + 1]
            if interpolation_method in _SNAPPING_INTERPOLATION_METHODS:
                idx_interp = interp1d([y1, y2], [0, 1], kind=interpolation_method)
                idx = int(round(float(idx_interp(y))))
                points.append((float((x1, x2)[idx]), float((y1, y2)[idx])))
            else:
                f = interp1d([y1, y2], [x1, x2], kind=interpolation_method)
                points.append((float(f(y)), y))
        return points

    def create_points_at_y(
        self,
        y: float,
        interpolation_method: str = "linear",
        label: str | None = None,
        face_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> list[Point]:
        """
        Gets the points on the curve at a given y value. Can return multiple Point objects if the curve crosses the y
        value multiple times.

        Parameters
        ----------
        y : float
            The y value of the desired points.
        interpolation_method : str
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            For ``"nearest"``, ``"previous"``, and ``"next"``, the returned points snap to actual
            data points instead of interpolating; for every other kind, the points' y is the same
            as the queried ``y``.

            Defaults to "linear".
        label : str, optional
            Point label to be displayed in the legend.
        face_color : str
            Face color of the point.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the point.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the point.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the point.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the point edge.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the point.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`]
            The Point objects on the curve at the given y value.
        """
        pairs = self.get_coordinates_at_y(y, interpolation_method)
        points = [
            Point(
                pair[0],
                pair[1],
                label=label,
                face_color=face_color,
                edge_color=edge_color,
                marker_size=marker_size,
                marker_style=marker_style,
                edge_width=line_width,
                alpha=alpha,
            )
            for pair in pairs
        ]
        return points

    def create_derivative_curve(
        self,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        copy_first: bool = False,
    ) -> Curve:
        """
        Creates a new curve which is the derivative of the original curve.

        Parameters
        ----------
        label : str, optional
            Label of the new curve to be displayed in the legend.
        color : str
            Color of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the new curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the new curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the new curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the derivative applied. Any
            other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve
            will be created with the derivative applied and the parameters passed to this method.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the derivative of the original curve.
        """
        x_data = self._x_data
        y_data = np.gradient(self._y_data, x_data)
        if copy_first:
            copy = self.copy()
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != INHERIT:
                copy._color = color
            if line_width != INHERIT:
                copy._line_width = line_width
            if line_style != INHERIT:
                copy._line_style = line_style
            if alpha != INHERIT:
                copy._alpha = alpha
            return copy
        else:
            return Curve(x_data, y_data, label, color, line_width, line_style, alpha)

    def create_integral_curve(
        self,
        initial_value: float = 0,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        copy_first: bool = False,
    ) -> Curve:
        """
        Creates a new curve which is the integral of the original curve.

        Parameters
        ----------
        initial_value : float, optional
            The value of the integral at the first x value (initial condition).
            Defaults to 0.
        label : str, optional
            Label of the new curve to be displayed in the legend.
        color : str
            Color of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the new curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the new curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the new curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the integral applied. Any
            other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve
            will be created with the integral applied and the parameters passed to this method.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Curve` object which is the integral of the original curve.
        """
        # calculate the integral curve using cumulative trapezoidal integration
        y_data = (
            cumulative_trapezoid(self._y_data, self._x_data, initial=0) + initial_value
        )
        if copy_first:
            copy = self.copy()
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != INHERIT:
                copy._color = color
            if line_width != INHERIT:
                copy._line_width = line_width
            if line_style != INHERIT:
                copy._line_style = line_style
            if alpha != INHERIT:
                copy._alpha = alpha
            return copy
        else:
            return Curve(
                self._x_data, y_data, label, color, line_width, line_style, alpha
            )

    def create_tangent_curve(
        self,
        x: float,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        copy_first: bool = False,
    ) -> Curve:
        """
        Creates a new curve which is the tangent to the original curve at a given x value.

        Parameters
        ----------
        x : float
            The x value at which the tangent is to be calculated.
        label : str, optional
            Label of the new curve to be displayed in the legend.
        color : str
            Color of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the new curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the new curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the new curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the tangent applied. Any
            other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve
            will be created with the tangent applied and the parameters passed to this method.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Curve`
            A :class:`~graphinglib.data_plotting_1d.Curve` object which is the tangent to the original curve at a given
            x value.
        """
        point = self.get_coordinates_at_x(x)
        gradient = self.create_derivative_curve().get_coordinates_at_x(x)[1]
        y_data = gradient * (self._x_data - x) + point[1]
        if copy_first:
            copy = self.copy()
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != INHERIT:
                copy._color = color
            if line_width != INHERIT:
                copy._line_width = line_width
            if line_style != INHERIT:
                copy._line_style = line_style
            if alpha != INHERIT:
                copy._alpha = alpha
            return copy
        else:
            tangent_curve = Curve(
                self._x_data, y_data, label, color, line_width, line_style, alpha
            )
            return tangent_curve

    def create_normal_curve(
        self,
        x: float,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        copy_first: bool = False,
    ) -> Curve:
        """
        Creates a new curve which is the normal to the original curve at a given x value.

        Parameters
        ----------
        x : float
            The x value at which the normal is to be calculated.
        label : str, optional
            Label of the new curve to be displayed in the legend.
        color : str
            Color of the new curve.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the new curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Style of the new curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the new curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the curve (with all its parameters) will be returned with the normal applied. Any
            other parameters passed to this method will also be applied to the copied curve. If ``False``, a new curve
            will be created with the normal applied and the parameters passed to this method.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Curve`
            A :class:`~graphinglib.data_plotting_1d.Curve` object which is the normal to the original curve at a given x
            value.
        """
        point = self.get_coordinates_at_x(x)
        gradient = self.create_derivative_curve().get_coordinates_at_x(x)[1]
        y_data = -1 / gradient * (self._x_data - x) + point[1]
        if copy_first:
            copy = self.copy()
            copy._y_data = y_data
            if label is not None:
                copy._label = label
            if color != INHERIT:
                copy._color = color
            if line_width != INHERIT:
                copy._line_width = line_width
            if line_style != INHERIT:
                copy._line_style = line_style
            if alpha != INHERIT:
                copy._alpha = alpha
            return copy
        else:
            normal_curve = Curve(
                self._x_data, y_data, label, color, line_width, line_style, alpha
            )
            return normal_curve

    def get_slope_at(self, x: float) -> float:
        """
        Calculates the slope of the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value at which the slope is to be calculated.

        Returns
        -------
        The slope of the curve (float) at the given x value.
        """
        return self.create_derivative_curve().get_coordinates_at_x(x)[1]

    def get_arc_length_between(self, x1: float, x2: float) -> float:
        """
        Calculates the arc length of the curve between two x values.

        Parameters
        ----------
        x1, x2 : float
            The x values between which the arc length is to be calculated.

        Returns
        -------
        The arc length of the curve (float) between the two given x values.
        """
        y_data = self._y_data
        x_data = self._x_data
        # f = interp1d(x_data, y_data)
        # x = np.linspace(x1, x2, 1000)
        # y = f(x)
        x = x_data[(x_data >= x1) & (x_data <= x2)]
        y = y_data[(x_data >= x1) & (x_data <= x2)]
        return trapezoid(np.sqrt(1 + np.gradient(y, x) ** 2), x)

    def get_area_between(
        self,
        x1: float,
        x2: float,
        fill_between: bool = False,
        fill_color: str | Inherit = INHERIT,
        other_curve: Optional[Self] = None,
    ) -> float:
        """
        Calculates the area between the curve and the x axis between two x values.
        This is the definite integral of the curve between the two x values.

        Parameters
        ----------
        x1, x2 : float
            The x values between which the area is to be calculated.
        fill_between : bool
            Whether to fill the specified area between the curve and the x axis when displaying.
            Defaults to ``False``.
        fill_color : str
            Color of the area between the curve and the x axis when ``fill_between`` is set to ``True``.
            ``"same as curve"`` uses the curve color.
            Default depends on the ``figure_style`` configuration.
        other_curve : :class:`~graphinglib.data_plotting_1d.Curve`, optional
            If specified, the area between the two curves will be calculated instead of the area between the curve and
            the x axis.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        The area (float) between the curve and the x axis (or between the two curves) between the two given x values.
        """
        if fill_between and _bound_session(self) is not None:
            area = self.get_area_between(x1, x2, other_curve=other_curve)
            self.fill_between_other_curve = other_curve
            self.fill_between_bounds = (x1, x2)
            if not is_inherit(fill_color):
                self.fill_between_color = fill_color
            return area
        if other_curve is None:
            if fill_between:
                self.fill_between_bounds = (x1, x2)
                self._fill_between_color = fill_color
            y_data = self._y_data
            x_data = self._x_data
            mask = (x_data >= x1) & (x_data <= x2)
            y = y_data[mask]
            x = x_data[mask]
            return trapezoid(y, x)
        else:
            if fill_between:
                self.fill_between_bounds = (x1, x2)
                self._fill_between_color = fill_color
                self._fill_between_other_curve = other_curve
            if np.array_equal(self._x_data, other_curve._x_data):
                # No need to interpolate
                mask = (self._x_data >= x1) & (self._x_data <= x2)
                common_x = self._x_data[mask]
                y1 = self._y_data[mask]
                y2 = other_curve._y_data[mask]
            else:
                # Interpolate to get common x values
                density_x1 = len(self._x_data) / (self._x_data[-1] - self._x_data[0])
                density_x2 = len(other_curve._x_data) / (
                    other_curve._x_data[-1] - other_curve._x_data[0]
                )
                higher_density = max(density_x1, density_x2)
                num_of_values = int(np.ceil(higher_density * (x2 - x1)))
                common_x = np.linspace(x1, x2, num_of_values)
                y1 = np.interp(common_x, self._x_data, self._y_data)
                y2 = np.interp(common_x, other_curve._x_data, other_curve._y_data)

            difference = y1 - y2
            area = trapezoid(difference, common_x)
            return area

    def get_intersection_coordinates(
        self,
        other: Self,
    ) -> list[tuple[float, float]]:
        """
        Calculates the coordinates of the intersection points between two curves.

        Parameters
        ----------
        other : :class:`~graphinglib.data_plotting_1d.Curve`
            The other curve to calculate the intersections with.

        Returns
        -------
        list[tuple[float, float]]
            A list of tuples of coordinates which are the intersection points between the two curves.
        """
        y = self._y_data - other._y_data
        s = np.abs(np.diff(np.sign(y))).astype(bool)
        intersections_x = self._x_data[:-1][s] + np.diff(self._x_data)[s] / (
            np.abs(y[1:][s] / y[:-1][s]) + 1
        )
        intersections_y = np.interp(intersections_x, self._x_data, self._y_data)
        points = []
        for i in range(len(intersections_x)):
            x_val = intersections_x[i]
            y_val = intersections_y[i]
            points.append((x_val, y_val))
        return points

    def to_desmos(self, decimal_precision: int = 2, to_clipboard: bool = False) -> str:
        """
        Gives the data points in a Desmos-readable format. The outputted string can then be pasted into a single Desmos
        cell and the object's data will be displayed.

        Parameters
        ----------
        decimal_precision : int, optional
            Specifies the number of decimals of the formatted points.
            Defaults to 2.
        to_clipboard : bool, optional
            Specifies whether the points should be directly copied to the user's clipboard in addition to being
            returned as a string.
            Defaults to False.

        Returns
        -------
        formatted points : str
            A list of tuples representing every data point.
        """
        formatted_points = _format_desmos_points(
            self._x_data, self._y_data, decimal_precision
        )
        if to_clipboard:
            copy_to_clipboard(formatted_points)
        return formatted_points

    def create_intersection_points(
        self,
        other: Self,
        labels: Optional[list[str] | str] = None,
        face_colors: list[str] | str | Inherit = INHERIT,
        edge_colors: list[str] | str | Inherit = INHERIT,
        marker_sizes: list[float] | float | Inherit = INHERIT,
        marker_styles: list[str] | str | Inherit = INHERIT,
        edge_widths: list[float] | float | Inherit = INHERIT,
        alphas: list[float] | float | Inherit = INHERIT,
    ) -> list[Point]:
        """
        Creates the intersection Points between two curves.

        Parameters
        ----------
        other : :class:`~graphinglib.data_plotting_1d.Curve`
            The other curve to calculate the intersections with.
        as_point_objects : bool
            Whether to return a list of :class:`~graphinglib.graph_elements.Point` objects (True) or a list of tuples of
            coordinates (False).
            Defaults to False.
        labels : list[str] or str, optional
            Labels of the intersection points to be displayed in the legend.
            If a single string is passed, all intersection points will have the same label.
        face_colors : list[str] or str
            Face colors of the intersection points.
            If a single string is passed, all intersection points will have the same color.
            Default depends on the ``figure_style`` configuration.
        edge_colors : list[str] or str
            Edge colors of the intersection points.
            If a single string is passed, all intersection points will have the same color.
            Default depends on the ``figure_style`` configuration.
        marker_sizes : list[float] or float
            Sizes of the intersection points.
            If a single float is passed, all intersection points will have the same size.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_styles : list[str] or str
            Styles of the intersection points.
            If a single string is passed, all intersection points will have the same style.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        edge_widths : list[float] or float
            Widths of the intersection points.
            If a single float is passed, all intersection points will have the same width.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        alphas : list[float] or float
            Opacities of the intersection points.
            If a single float is passed, all intersection points will have the same opacity.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`] or list[tuple[float, float]]
            A list of :class:`~graphinglib.graph_elements.Point` objects which are the intersection points between the
            two curves.
        """
        y = self._y_data - other._y_data
        s = np.abs(np.diff(np.sign(y))).astype(bool)
        intersections_x = self._x_data[:-1][s] + np.diff(self._x_data)[s] / (
            np.abs(y[1:][s] / y[:-1][s]) + 1
        )
        point_coords = self.get_intersection_coordinates(other)
        point_objects = []
        for i in range(len(intersections_x)):
            label: str | None
            if isinstance(labels, list):
                try:
                    label = labels[i]
                except IndexError:
                    label = None
            else:
                label = labels

            face_color: str | Inherit
            if isinstance(face_colors, list):
                face_color_values = cast(list[str], face_colors)
                try:
                    face_color = face_color_values[i]
                except IndexError:
                    face_color = INHERIT
            else:
                face_color = face_colors

            edge_color: str | Inherit
            if isinstance(edge_colors, list):
                edge_color_values = cast(list[str], edge_colors)
                try:
                    edge_color = edge_color_values[i]
                except IndexError:
                    edge_color = INHERIT
            else:
                edge_color = edge_colors

            marker_size: float | Inherit
            if isinstance(marker_sizes, list):
                marker_size_values = cast(list[float], marker_sizes)
                try:
                    marker_size = marker_size_values[i]
                except IndexError:
                    marker_size = INHERIT
            else:
                marker_size = marker_sizes

            marker_style: str | Inherit
            if isinstance(marker_styles, list):
                marker_style_values = cast(list[str], marker_styles)
                try:
                    marker_style = marker_style_values[i]
                except IndexError:
                    marker_style = INHERIT
            else:
                marker_style = marker_styles

            edge_width: float | Inherit
            if isinstance(edge_widths, list):
                edge_width_values = cast(list[float], edge_widths)
                try:
                    edge_width = edge_width_values[i]
                except IndexError:
                    edge_width = INHERIT
            else:
                edge_width = edge_widths

            alpha: float | Inherit
            if isinstance(alphas, list):
                alpha_values = cast(list[float], alphas)
                try:
                    alpha = alpha_values[i]
                except IndexError:
                    alpha = INHERIT
            else:
                alpha = alphas
            point = point_coords[i]
            point_objects.append(
                Point(
                    point[0],
                    point[1],
                    label=label,
                    face_color=face_color,
                    edge_color=edge_color,
                    marker_size=marker_size,
                    marker_style=marker_style,
                    edge_width=edge_width,
                    alpha=alpha,
                )
            )
        return point_objects

    def _shared_fill_data(
        self, other: Curve, bounds: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(*bounds, max(len(self._x_data), len(other._x_data)))
        return (
            x,
            interp1d(self._x_data, self._y_data)(x),
            interp1d(other._x_data, other._y_data)(x),
        )

    def _bind_live_area(
        self, session: _AnimationSession, axes: plt.Axes, fill: PolyCollection
    ) -> None:
        def prepare() -> _ArtistEdit:
            bounds = self._fill_between_bounds
            if bounds is None:
                return _edit(fill.set_visible, fill.get_visible(), False)
            limits = np.asarray(bounds)
            if (
                limits.shape != (2,)
                or limits.dtype.kind not in "iuf"
                or not np.isfinite(limits).all()
                or limits[0] > limits[1]
            ):
                raise InvalidParameterError(
                    "Animated Curve fill_between_bounds must be two finite, ordered endpoints."
                )
            left, right = limits
            if left == right:
                return _edit(fill.set_visible, fill.get_visible(), False)
            other = self._fill_between_other_curve
            if other is None:
                x, y = self._x_data, self._y_data
                within = (x >= left) & (x <= right)
                geometry = _fill_edit(fill, x, y, np.zeros_like(y), within)
            else:
                try:
                    x, y, boundary = self._shared_fill_data(other, (left, right))
                except ValueError as exc:
                    raise InvalidParameterError(
                        "Both curves must cover the animated fill interval."
                    ) from exc
                geometry = _fill_edit(fill, x, y, boundary)
            edits = []
            if fill.axes is None:
                edits.append(_attach_artists(axes, (fill,)))
            edits.extend((geometry, _edit(fill.set_visible, fill.get_visible(), True)))
            return _combine_edits(edits)

        boundary = "_fill_between_other_curve"
        session.bind_derived(
            self, ("point_data", "_fill_between_bounds", boundary), prepare
        )
        session.link_point_source(self, boundary, None, self._fill_between_other_curve)

    def _bind_live_styles(
        self,
        session: _AnimationSession,
        axes: plt.Axes,
        handle: ErrorbarContainer,
        lower: Line2D | None,
        upper: Line2D | None,
        error_fill: PolyCollection | None,
        area_fill: PolyCollection | None,
        reserved_errors: ErrorbarContainer | None,
    ) -> None:
        line = handle.lines[0]
        if line is None:
            return
        for name, mpl_name in (
            ("_color", "color"),
            ("_line_style", "linestyle"),
            ("_line_width", "linewidth"),
            ("_alpha", "alpha"),
        ):
            session.bind_property(self, name, line, mpl_name)

        initial_color = line.get_color()
        for errorbars in (handle, reserved_errors):
            if errorbars is None or not errorbars.lines[2]:
                continue
            _, error_caps, _ = errorbars.lines
            reserved_axes = (
                line.axes
                if errorbars is reserved_errors and isinstance(line.axes, plt.Axes)
                else None
            )
            initial_error_color = self._errorbars_color
            initial_error_width = self._errorbars_line_width
            initial_cap_thickness = self._cap_thickness
            cap_width_live = bool(error_caps) and (
                reserved_axes is not None
                or all(cap.axes is not None for cap in error_caps)
            )
            initial_cap_width = (
                error_caps[0].get_markersize() / 2 if cap_width_live else None
            )

            def prepare_errorbars(
                errorbars=errorbars,
                reserved_axes=reserved_axes,
                cap_width_live=cap_width_live,
                initial_cap_width=initial_cap_width,
            ) -> _ArtistEdit:
                color = resolve_or(self._errorbars_color, initial_error_color)
                if color is None or color == "same as curve":
                    color = resolve_or(self._color, initial_color)
                width = resolve_or(self._errorbars_line_width, initial_error_width)
                thickness = resolve_or(self._cap_thickness, initial_cap_thickness)
                curve_width = resolve_or(self._line_width, line.get_linewidth())
                return _combine_edits(
                    _error_style_edits(
                        errorbars,
                        color=None if is_inherit(color) else color,
                        line_width=curve_width
                        if width == "same as curve"
                        or width is None
                        and reserved_axes is not None
                        else resolve_or(width, None),
                        cap_width=resolve_or(self._cap_width, initial_cap_width)
                        if cap_width_live
                        else None,
                        cap_thickness=curve_width
                        if thickness == "same as curve"
                        else resolve_or(thickness, None),
                        alpha=resolve_or(self._alpha, initial_alpha),
                        show=self._show_errorbars,
                        directions=(
                            self._x_error is not None,
                            self._y_error is not None,
                        ),
                        reserved_axes=reserved_axes,
                        reserved_data=(
                            self._x_data,
                            self._y_data,
                            self._x_error,
                            self._y_error,
                        )
                        if reserved_axes is not None
                        else None,
                    )
                )

            initial_alpha = line.get_alpha()
            error_properties = (
                "point_data",
                "_errorbars_color",
                "_errorbars_line_width",
                "_color",
                "_line_width",
                "_alpha",
            )
            if not error_caps or cap_width_live:
                error_properties += ("_show_errorbars",)
            if cap_width_live:
                error_properties += ("_cap_width", "_cap_thickness")
            session.bind_derived(self, error_properties, prepare_errorbars)

        if lower is not None and upper is not None:
            detached = lower.axes is None
            initial_fill = resolve_or(self._error_curves_fill_between, True)

            def prepare_visibility() -> _ArtistEdit:
                shown = self._show_error_curves
                filled = resolve_or(self._error_curves_fill_between, initial_fill)
                if not all(
                    isinstance(value, (bool, np.bool_)) for value in (shown, filled)
                ):
                    raise InvalidParameterError(
                        "show_error_curves and error_curves_fill_between must be booleans."
                    )
                edits = []
                if detached and shown:
                    if lower.axes is None:
                        additions = (lower, upper)
                        if error_fill is not None:
                            additions = (*additions, error_fill)
                        edits.append(_attach_artists(axes, additions))
                    edits.extend(
                        _CurveBands(lower, upper, error_fill).edits(
                            self._x_data, self._y_data, self._y_error
                        )
                    )
                edits.extend(
                    _edit(artist.set_visible, artist.get_visible(), shown)
                    for artist in (lower, upper)
                )
                if error_fill is not None:
                    edits.append(
                        _edit(
                            error_fill.set_visible,
                            error_fill.get_visible(),
                            shown and filled,
                        )
                    )
                return _combine_edits(edits)

            properties = ("_show_error_curves", "_error_curves_fill_between")
            if detached:
                properties += ("point_data",)
            session.bind_derived(self, properties, prepare_visibility)
            for name, field, source in (
                ("_error_curves_color", "color", "_color"),
                ("_error_curves_line_style", "linestyle", "_line_style"),
                ("_error_curves_line_width", "linewidth", "_line_width"),
            ):
                self._bind_curve_decoration_style(
                    session, line, (lower, upper), name, field, source
                )
        if error_fill is not None:
            session.bind_property(self, "_color", error_fill, "facecolor")
        if area_fill is not None:
            self._bind_curve_decoration_style(
                session,
                line,
                (area_fill,),
                "_fill_between_color",
                "facecolor",
                "_color",
            )

    def _bind_curve_decoration_style(
        self,
        session: _AnimationSession,
        line: Line2D,
        artists: tuple[Line2D | PolyCollection, ...],
        name: str,
        field: str,
        source: str,
    ) -> None:
        initial = getattr(self, name)
        getter, setter = f"get_{field}", f"set_{field}"
        initial_source = getattr(
            line, "get_color" if field == "facecolor" else getter
        )()
        baselines = [getattr(artist, getter)() for artist in artists]

        def prepare() -> _ArtistEdit:
            requested = resolve_or(getattr(self, name), initial)
            if requested == "same as curve":
                requested = resolve_or(getattr(self, source), initial_source)
            return _combine_edits(
                [
                    _edit(
                        getattr(artist, setter),
                        getattr(artist, getter)(),
                        baseline
                        if requested is None or is_inherit(requested)
                        else requested,
                    )
                    for artist, baseline in zip(artists, baselines, strict=True)
                ]
            )

        session.bind_derived(self, (name, source), prepare)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified axes.
        """
        session = _DRAW_SESSION.get()
        params = {
            "color": self._color,
            "linewidth": self._line_width,
            "linestyle": self._line_style,
            "alpha": self._alpha,
        }
        if self._show_errorbars or (
            session is not None and getattr(self, "_errorbars_configured", False)
        ):
            params.update(
                {
                    "elinewidth": (
                        self._errorbars_line_width
                        if self._errorbars_line_width != "same as curve"
                        else self._line_width
                    ),
                    "capsize": self._cap_width,
                    "capthick": (
                        self._cap_thickness
                        if self._cap_thickness != "same as curve"
                        else self._line_width
                    ),
                    "ecolor": (
                        self._errorbars_color
                        if self._errorbars_color != "same as curve"
                        else self._color
                    ),
                }
            )
            params = strip_inherit(params)
            draw_errorbars = (
                axes.errorbar
                if session is None
                else partial(_draw_live_errorbars, axes, show=self._show_errorbars)
            )
            self.handle = draw_errorbars(
                self._x_data,
                self._y_data,
                xerr=self._x_error,
                yerr=self._y_error,
                label=self._label,
                zorder=z_order,
                **params,
            )
        else:
            params = strip_inherit(params)
            self.handle = axes.errorbar(
                self._x_data,
                self._y_data,
                label=self._label,
                zorder=z_order,
                **params,
            )
        lower = upper = error_fill = area_fill = None
        if self._show_error_curves or (
            session is not None and getattr(self, "_error_curves_configured", False)
        ):
            min_y, max_y = _error_bounds(self._y_data, self._y_error)

            params = {
                "color": (
                    self._error_curves_color
                    if self._error_curves_color != "same as curve"
                    else self.handle[0].get_color()
                ),
                "linestyle": (
                    self._error_curves_line_style
                    if self._error_curves_line_style != "same as curve"
                    else self._line_style
                ),
                "linewidth": (
                    self._error_curves_line_width
                    if self._error_curves_line_width != "same as curve"
                    else self._line_width
                ),
            }

            params = strip_inherit(params)

            if self._show_error_curves:
                (lower,) = axes.plot(self._x_data, min_y, **params)
                (upper,) = axes.plot(self._x_data, max_y, **params)
            else:
                # Dormant artists should affect neither limits nor the color cycle.
                lower = Line2D(self._x_data, min_y, visible=False, **params)
                upper = Line2D(self._x_data, max_y, visible=False, **params)
                axes.add_artist(lower)
                axes.add_artist(upper)
            if self._show_error_curves and resolve_or(
                self._error_curves_fill_between, True
            ):
                error_fill = axes.fill_between(
                    self._x_data,
                    max_y,
                    min_y,
                    facecolor=self.handle[0].get_color(),
                    alpha=0.2,
                )
            elif session is not None:
                error_fill = PolyCollection(
                    [], facecolor=self.handle[0].get_color(), alpha=0.2, visible=False
                )
                _fill_edit(error_fill, self._x_data, max_y, min_y).apply()
                axes.add_collection(error_fill, autolim=False)
        area_bounds = self._fill_between_bounds
        if area_bounds or (
            session is not None and getattr(self, "_area_fill_configured", False)
        ):
            params = strip_inherit(
                {
                    "alpha": 0.2,
                    "facecolor": (
                        self._fill_between_color
                        if self._fill_between_color != "same as curve"
                        else self.handle[0].get_color()
                    ),
                }
            )
            if area_bounds is None:
                area_fill = PolyCollection(
                    [], zorder=z_order - 2, visible=False, **params
                )
                axes.add_collection(area_fill, autolim=False)
            else:
                if self._fill_between_other_curve is not None:
                    x, y1, y2 = self._shared_fill_data(
                        self._fill_between_other_curve, area_bounds
                    )
                else:
                    x, y1, y2 = self._x_data, self._y_data, 0
                area_fill = axes.fill_between(
                    x,
                    y1,
                    y2,
                    where=((x >= area_bounds[0]) & (x <= area_bounds[1])).tolist(),
                    zorder=z_order - 2,
                    **params,
                )

        if session is not None:
            line = self.handle[0]
            existing_bands = _CurveBands(lower, upper, error_fill)
            if lower is None:
                color = resolve_or(self._error_curves_color, None)
                width = resolve_or(self._error_curves_line_width, None)
                style = resolve_or(self._error_curves_line_style, None)
                band_style: dict[str, Any] = {
                    "color": line.get_color()
                    if color in (None, "same as curve")
                    else color,
                    "linewidth": line.get_linewidth()
                    if width in (None, "same as curve")
                    else width,
                    "linestyle": line.get_linestyle()
                    if style == "same as curve"
                    else style or "--",
                }
                lower, upper = (
                    Line2D([], [], visible=False, **band_style) for _ in range(2)
                )
                error_fill = PolyCollection(
                    [], facecolor=line.get_color(), alpha=0.2, visible=False
                )
            if area_fill is None:
                color = resolve_or(self._fill_between_color, None)
                area_fill = PolyCollection(
                    [],
                    facecolor=line.get_color()
                    if color in (None, "same as curve")
                    else color,
                    alpha=0.2,
                    zorder=z_order - 2,
                    visible=False,
                )
            reserved_errors = _reserve_errorbar_directions(
                axes,
                self.handle,
                color=self.handle[0].get_color(),
                zorder=z_order - 0.1,
                cap_width=resolve_or(self._cap_width, None),
            )
            session.bind_legend_source(self, self.handle, lambda: self.label)
            session.bind_curve(
                self,
                self.handle,
                bands=existing_bands,
                reserved_errors=reserved_errors,
            )
            self._bind_live_area(session, axes, area_fill)
            self._bind_live_styles(
                session,
                axes,
                self.handle,
                lower,
                upper,
                error_fill,
                area_fill,
                reserved_errors,
            )


@dataclass
class Scatter(Plottable1D, MathematicalObject):
    """
    This class implements a general scatter plot.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        Arrays of x and y values to be plotted.
    label : str, optional
        Label to be displayed in the legend.
    face_color : str or ArrayLike or None
        Face color of the points. If an array of intensities is provided, the values are mapped to the specified color
        map. If None, marker faces are transparent.
        Default depends on the ``figure_style`` configuration.
    edge_color : str or ArrayLike or None
        Edge color of the points. If an array of intensities is provided, the values are mapped to the specified color
        map. If None, marker edges are transparent.
        Default depends on the ``figure_style`` configuration.
    color_map : str or Colormap
        Color map used when ``face_color`` or ``edge_color`` is an array of intensity values.
        Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
        Default depends on the ``figure_style`` configuration.
    color_map_range: tuple[float, float], optional
        The data range covered by the color map, given as ``(minimum, maximum)``.
    show_color_bar : bool
        Whether or not to display the color bar next to the plot.
        Default depends on the ``figure_style`` configuration.
    marker_size : float
        Size of the points.
        Typical range is ``10`` to ``100``.
        Default depends on the ``figure_style`` configuration.
    marker_edge_width: float
        Line width of the marker edges.
        Typical range is ``0`` to ``3`` points.
        Default depends on the ``figure_style`` configuration.
    marker_style : str
        Style of the points.
        Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
        ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
        ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the scatter plot.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``). They may also be arrays of
    intensity values, which are mapped through ``color_map``.
    """

    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        label: Optional[str] = None,
        face_color: str | ArrayLike | NoneType | Inherit = INHERIT,
        edge_color: str | ArrayLike | NoneType | Inherit = INHERIT,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_edge_width: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> None:
        """
        This class implements a general scatter plot.

        Parameters
        ----------
        x_data, y_data : ArrayLike
            Arrays of x and y values to be plotted.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str or ArrayLike or None
            Face color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker faces are transparent.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or ArrayLike or None
            Edge color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker edges are transparent.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map used when ``face_color`` or ``edge_color`` is an array of intensity values.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the points.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_edge_width: float
            Line width of the marker edges.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the points.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the scatter plot.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``). They may also be arrays of
        intensity values, which are mapped through ``color_map``.
        """
        self.handle = None
        self.errorbars_handle = None
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        _check_same_length("x_data", self._x_data, "y_data", self._y_data)
        self._label = label
        self._face_color = face_color
        self._edge_color = edge_color
        self._color_map = color_map
        self._color_map_range = color_map_range
        self._show_color_bar = show_color_bar
        self._marker_size = marker_size
        self._marker_edge_width = marker_edge_width
        self._marker_style = marker_style
        self._alpha = alpha

        self._x_error = None
        self._y_error = None

        self._show_errorbars: bool = False
        self._errorbars_line_width: Styled[float] = 1.0
        self._cap_width: Styled[float] = 3.0
        self._cap_thickness: Styled[float] = 1.0
        self._errorbars_color: Styled[str | None] = "same as scatter"
        self._color_bar_params: dict = {}
        self._reserve_color_bar = False

    @classmethod
    def from_function(
        cls,
        func: Callable[[np.ndarray], ArrayLike],
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        face_color: str | ArrayLike | NoneType | Inherit = INHERIT,
        edge_color: str | ArrayLike | NoneType | Inherit = INHERIT,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Inherit = INHERIT,
        marker_size: int | Inherit = INHERIT,
        marker_edge_width: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        number_of_points: int = 30,
    ) -> Self:
        """
        Creates a scatter plot from a function and a range of x values.

        Parameters
        ----------
        func : Callable[[ArrayLike], ArrayLike]
            The function to be plotted.
        x_min, x_max : float
            The scatter plot will be created for x values between x_min and x_max.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str or ArrayLike or None
            Face color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker faces are transparent.
        edge_color : str or ArrayLike or None
            Edge color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker edges are transparent.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map used when ``face_color`` or ``edge_color`` is an array of intensity values.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        marker_size : int
            Size of the points.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the points.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        marker_edge_width: float
            Line width of the marker edges.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the scatter plot.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        number_of_points : int
            Number of points to be plotted.
            Defaults to 30.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``). They may also be arrays of
        intensity values, which are mapped through ``color_map``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Scatter` object created from a function and a range of x values.
        """
        x_data = np.linspace(x_min, x_max, number_of_points)
        y_data = func(x_data)
        return cls(
            x_data,
            y_data,
            label,
            face_color,
            edge_color,
            color_map,
            color_map_range,
            show_color_bar,
            marker_size,
            marker_edge_width,
            marker_style,
            alpha,
        )

    @property
    def x_data(self) -> np.ndarray:
        return self._x_data

    @x_data.setter
    def x_data(self, x_data: ArrayLike) -> None:
        if _bound_session(self) is None:
            self._x_data = np.asarray(x_data)
        else:
            self.set_data(x_data, self._y_data)

    @property
    def y_data(self) -> np.ndarray:
        return self._y_data

    @y_data.setter
    def y_data(self, y_data: ArrayLike) -> None:
        if _bound_session(self) is None:
            self._y_data = np.asarray(y_data)
        else:
            self.set_data(self._x_data, y_data)

    @property
    def x_error(self) -> np.ndarray | None:
        return self._x_error

    @x_error.setter
    def x_error(self, x_error: ArrayLike) -> None:
        if _bound_session(self) is None:
            self._x_error = np.asarray(x_error)
        else:
            self.set_data(self._x_data, self._y_data, x_error=x_error)

    @property
    def y_error(self) -> np.ndarray | None:
        return self._y_error

    @y_error.setter
    def y_error(self, y_error: ArrayLike) -> None:
        if _bound_session(self) is None:
            self._y_error = np.asarray(y_error)
        else:
            self.set_data(self._x_data, self._y_data, y_error=y_error)

    def set_data(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        *,
        x_error: ArrayLike | None | _KeepError = _KEEP_ERROR,
        y_error: ArrayLike | None | _KeepError = _KEEP_ERROR,
    ) -> Self:
        """Replace point coordinates and optional errors in one update.

        The coordinate arrays must be one-dimensional and match in length.
        Errors can be scalars, per-point arrays, or lower/upper arrays.
        Omitted errors keep their current values. Pass None to clear an error
        direction. On Cartesian axes, errorbar directions can change during
        animation. When the count changes, assign any per-point colors
        for the new count in the same callback.
        """
        _set_point_data(self, x_data, y_data, x_error, y_error)
        return self

    @property
    def label(self) -> str | None:
        return self._label

    @label.setter
    def label(self, label: str | None) -> None:
        self._label = _stage_live_value(self, "_label", label)

    @property
    def face_color(self) -> Styled[str | ArrayLike | None]:
        return self._face_color

    @face_color.setter
    def face_color(self, face_color: Styled[str | ArrayLike | None]) -> None:
        self._face_color = _stage_live_value(self, "_face_color", face_color)

    @property
    def edge_color(self) -> Styled[str | ArrayLike | None]:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: Styled[str | ArrayLike | None]) -> None:
        self._edge_color = _stage_live_value(self, "_edge_color", edge_color)

    @property
    def color_map(self) -> Styled[str | Colormap]:
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: str | Colormap) -> None:
        self._color_map = _stage_live_value(self, "_color_map", color_map)

    @property
    def color_map_range(self) -> tuple[float, float] | None:
        return self._color_map_range

    @color_map_range.setter
    def color_map_range(self, color_map_range: tuple[float, float] | None) -> None:
        self._color_map_range = _stage_live_value(self, "_color_map_range", color_map_range)

    @property
    def show_color_bar(self) -> Styled[bool]:
        return self._show_color_bar

    @show_color_bar.setter
    def show_color_bar(self, show_color_bar: bool) -> None:
        self._show_color_bar = _stage_live_value(self, "_show_color_bar", show_color_bar)

    @property
    def marker_size(self) -> float | Inherit:
        return self._marker_size

    @marker_size.setter
    def marker_size(self, marker_size: float | Inherit) -> None:
        self._marker_size = _stage_live_value(self, "_marker_size", marker_size)

    @property
    def marker_edge_width(self) -> Styled[float]:
        return self._marker_edge_width

    @marker_edge_width.setter
    def marker_edge_width(self, value: float):
        self._marker_edge_width = _stage_live_value(self, "_marker_edge_width", value)

    @property
    def marker_style(self) -> Styled[str]:
        return self._marker_style

    @marker_style.setter
    def marker_style(self, marker_style: str) -> None:
        self._marker_style = _stage_live_value(self, "_marker_style", marker_style)

    @property
    def alpha(self) -> float | Inherit:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Inherit) -> None:
        self._alpha = _stage_live_value(self, "_alpha", alpha)

    @property
    def show_errorbars(self) -> bool:
        return self._show_errorbars

    @show_errorbars.setter
    def show_errorbars(self, show_errorbars: bool) -> None:
        self._show_errorbars = _stage_live_value(self, "_show_errorbars", show_errorbars)

    @property
    def errorbars_line_width(self) -> Styled[float]:
        return self._errorbars_line_width

    @errorbars_line_width.setter
    def errorbars_line_width(self, errorbars_line_width: Styled[float]) -> None:
        self._errorbars_line_width = _stage_live_value(self, "_errorbars_line_width", errorbars_line_width)

    @property
    def cap_width(self) -> Styled[float]:
        return self._cap_width

    @cap_width.setter
    def cap_width(self, cap_width: Styled[float]) -> None:
        self._cap_width = _stage_live_value(self, "_cap_width", cap_width)

    @property
    def cap_thickness(self) -> Styled[float]:
        return self._cap_thickness

    @cap_thickness.setter
    def cap_thickness(self, cap_thickness: Styled[float]) -> None:
        self._cap_thickness = _stage_live_value(self, "_cap_thickness", cap_thickness)

    @property
    def errorbars_color(self) -> Styled[str | None]:
        return self._errorbars_color

    @errorbars_color.setter
    def errorbars_color(self, errorbars_color: Styled[str | None]) -> None:
        self._errorbars_color = _stage_live_value(self, "_errorbars_color", errorbars_color)

    @property
    def color_bar_params(self) -> dict:
        return self._color_bar_params

    def __eq__(self, other: object) -> bool:
        """
        Defines the equality between two scatters.
        """
        if not isinstance(other, Scatter):
            return False
        return bool(
            np.equal(self.x_data, other.x_data).all()
            and np.equal(self.y_data, other.y_data).all()
        )

    def __add__(self, other: Scatter | float) -> Scatter:
        """
        Defines the addition of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise IncompatibleArgumentsError(
                    "Cannot add two scatter plots with different x values."
                )
            new_y_data = self._y_data + other._y_data
            return Scatter(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data + other
            return Scatter(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only add a scatter plot to another scatter plot or a number."
            )

    def __sub__(self, other: Scatter | float) -> Scatter:
        """
        Defines the subtraction of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise IncompatibleArgumentsError(
                    "Cannot subtract two scatter plots with different x values."
                )
            new_y_data = self._y_data - other._y_data
            return Scatter(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data - other
            return Scatter(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only subtract a scatter plot from another scatter plot or a number."
            )

    def __mul__(self, other: Scatter | float) -> Scatter:
        """
        Defines the multiplication of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise IncompatibleArgumentsError(
                    "Cannot multiply two scatter plots with different x values."
                )
            new_y_data = self._y_data * other._y_data
            return Scatter(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data * other
            return Scatter(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only multiply a scatter plot by another scatter plot or a number."
            )

    def __truediv__(self, other: Scatter | float) -> Scatter:
        """
        Defines the division of two scatter plots or a scatter plot and a number.
        """
        if isinstance(other, Scatter):
            try:
                assert np.array_equal(self._x_data, other._x_data)
            except AssertionError:
                raise IncompatibleArgumentsError(
                    "Cannot divide two scatter plots with different x values."
                )
            new_y_data = self._y_data / other._y_data
            return Scatter(self._x_data, new_y_data)
        elif isinstance(other, (int, float)):
            new_y_data = self._y_data / other
            return Scatter(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only divide a scatter plot by another scatter plot or a number."
            )

    def __pow__(self, other: float) -> Scatter:
        """
        Defines the power of a scatter plot to a number.
        """
        if isinstance(other, (int, float)):
            new_y_data = self._y_data**other
            return Scatter(self._x_data, new_y_data)
        else:
            raise InvalidParameterTypeError(
                "Can only raise a scatter plot to another scatter plot or a number."
            )

    # Explicitly retain the protocol stub so type checkers treat Scatter as concrete.
    __rpow__ = MathematicalObject.__rpow__

    def __iter__(self):
        """
        Defines the iteration of a scatter plot. Returns the y values.
        """
        return iter(self._y_data)

    def __abs__(self) -> Scatter:
        """
        Defines the absolute value of a scatter plot.
        """
        new_y_data = np.abs(self._y_data)
        return Scatter(self._x_data, new_y_data)

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_1d.Scatter` object.
        """
        return deepcopy(self)

    def create_slice_x(
        self,
        x_min: float,
        x_max: float,
        label: Optional[str] = None,
        face_color: str | ArrayLike | NoneType | Inherit = INHERIT,
        edge_color: str | ArrayLike | NoneType | Inherit = INHERIT,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_edge_width: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        copy_first: bool = False,
    ) -> Scatter:
        """
        Creates a slice of the scatter plot between two x values.

        Parameters
        ----------
        x_min, x_max : float
            The slice will be created between x_min and x_max.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str or ArrayLike or None
            Face color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker faces are transparent.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or ArrayLike or None
            Edge color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker edges are transparent.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map used when ``face_color`` or ``edge_color`` is an array of intensity values.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the points.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_edge_width: float
            Line width of the marker edges.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the points.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacities of the points.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the scatter plot (with all its parameters) will be returned with the slice applied.
            Any other parameters passed to this method will also be applied to the copied scatter plot. If ``False``, a
            new scatter plot will be created with the slice applied and the parameters passed to this method.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``). They may also be arrays of
        intensity values, which are mapped through ``color_map``.

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Scatter`
            A new :class:`~graphinglib.data_plotting_1d.Scatter` object which is a slice of the original scatter plot.
        """
        mask = (self._x_data >= x_min) & (self._x_data <= x_max)
        if copy_first:
            copy = self.copy()
            copy._x_data = self._x_data[mask]
            copy._y_data = self._y_data[mask]
            if label is not None:
                copy._label = label
            if face_color != INHERIT:
                copy._face_color = face_color
            if edge_color != INHERIT:
                copy._edge_color = edge_color
            if color_map != INHERIT:
                copy._color_map = color_map
            if color_map_range:
                copy._color_map_range = color_map_range
            if show_color_bar != INHERIT:
                copy._show_color_bar = show_color_bar
            if marker_size != INHERIT:
                copy._marker_size = marker_size
            if marker_edge_width != INHERIT:
                copy._marker_edge_width = marker_edge_width
            if marker_style != INHERIT:
                copy._marker_style = marker_style
            if alpha != INHERIT:
                copy._alpha = alpha
            return copy
        else:
            return Scatter(
                self._x_data[mask],
                self._y_data[mask],
                label,
                face_color,
                edge_color,
                color_map,
                color_map_range,
                show_color_bar,
                marker_size,
                marker_edge_width,
                marker_style,
                alpha,
            )

    def create_slice_y(
        self,
        y_min: float,
        y_max: float,
        label: Optional[str] = None,
        face_color: str | ArrayLike | NoneType | Inherit = INHERIT,
        edge_color: str | ArrayLike | NoneType | Inherit = INHERIT,
        color_map: str | Colormap | Inherit = INHERIT,
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_edge_width: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        copy_first: bool = False,
    ) -> Scatter:
        """
        Creates a slice of the scatter plot between two y values.

        Parameters
        ----------
        y_min, y_max : float
            The slice will be created between y_min and y_max.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str or ArrayLike or None
            Face color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker faces are transparent.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or ArrayLike or None
            Edge color of the points. If an array of intensities is provided, the values are mapped to the specified
            color map. If None, marker edges are transparent.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map used when ``face_color`` or ``edge_color`` is an array of intensity values.
            Examples include ``"viridis"``, ``"plasma"``, and ``"coolwarm"``.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range covered by the color map, given as ``(minimum, maximum)``.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the points.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_edge_width: float
            Line width of the marker edges.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the points.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacities of the points.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        copy_first : bool
            If ``True``, a copy of the scatter plot (with all its parameters) will be returned with the slice applied.
            Any other parameters passed to this method will also be applied to the copied scatter plot. If ``False``, a
            new scatter plot will be created with the slice applied and the parameters passed to this method.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``). They may also be arrays of
        intensity values, which are mapped through ``color_map``.

        Returns
        -------
        :class:`~graphinglib.data_plotting_1d.Scatter`
            A new :class:`~graphinglib.data_plotting_1d.Scatter` object which is a slice of the original scatter plot.
        """
        mask = (self._y_data >= y_min) & (self._y_data <= y_max)
        if copy_first:
            copy = self.copy()
            copy._x_data = self._x_data[mask]
            copy._y_data = self._y_data[mask]
            if label is not None:
                copy._label = label
            if face_color != INHERIT:
                copy._face_color = face_color
            if edge_color != INHERIT:
                copy._edge_color = edge_color
            if color_map != INHERIT:
                copy._color_map = color_map
            if color_map_range:
                copy._color_map_range = color_map_range
            if show_color_bar != INHERIT:
                copy._show_color_bar = show_color_bar
            if marker_size != INHERIT:
                copy._marker_size = marker_size
            if marker_edge_width != INHERIT:
                copy._marker_edge_width = marker_edge_width
            if marker_style != INHERIT:
                copy._marker_style = marker_style
            if alpha != INHERIT:
                copy._alpha = alpha
            return copy
        else:
            return Scatter(
                self._x_data[mask],
                self._y_data[mask],
                label,
                face_color,
                edge_color,
                color_map,
                color_map_range,
                show_color_bar,
                marker_size,
                marker_edge_width,
                marker_style,
                alpha,
            )

    def add_errorbars(
        self,
        x_error: Optional[ArrayLike] = None,
        y_error: Optional[ArrayLike] = None,
        cap_width: float | Inherit = INHERIT,
        errorbars_color: str | Inherit = INHERIT,
        errorbars_line_width: float | Inherit = INHERIT,
        cap_thickness: float | Inherit = INHERIT,
    ) -> None:
        """
        Adds errorbars to the scatter plot.

        Parameters
        ----------
        x_error, y_error : ArrayLike, optional
            Arrays of x and y errors. Use one or both.
            Values must be non-negative.
        cap_width : float
            Width of the errorbar caps.
            Typical range is ``0`` to ``10`` points.
            Default depends on the ``figure_style`` configuration.
        errorbars_color : str
            Color of the errorbars.
            ``"same as scatter"`` uses the scatter marker color.
            Default depends on the ``figure_style`` configuration.
        errorbars_line_width : float
            Width of the errorbars.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        cap_thickness : float
            Thickness of the errorbar caps.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        if _bound_session(self) is not None:
            _add_live_errorbars(
                self,
                x_error,
                y_error,
                cap_width=cap_width,
                errorbars_color=errorbars_color,
                errorbars_line_width=errorbars_line_width,
                cap_thickness=cap_thickness,
            )
            return
        self._show_errorbars = True
        self._errorbars_configured = True

        if x_error is not None:
            self._x_error = np.array(x_error)
            _check_error_shape("x_error", self._x_error, "x_data", self._x_data)

        if y_error is not None:
            self._y_error = np.array(y_error)
            _check_error_shape("y_error", self._y_error, "y_data", self._y_data)

        self._errorbars_color = errorbars_color
        self._errorbars_line_width = errorbars_line_width
        self._cap_thickness = cap_thickness
        self._cap_width = cap_width

    def set_color_bar_params(
        self,
        label: Optional[str] = None,
        position: Optional[str] = None,
        *,
        reserve: bool = False,
        **color_bar_params,
    ) -> None:
        """
        Sets the color bar parameters.

        Parameters
        ----------
        label : str, optional
            Label of the color bar.
        position : str, optional
            Position of the color bar relative to the ``Figure``. It can be "left",
            "right", "top" or "bottom". This also determines the orientation of the
            color bar (vertical if the color bar is plotted on the "left" or "right",
            horizontal otherwise). If None, the color bar is plotted on the right
            side of the ``Figure``.
            Values are ``"left"``, ``"right"``, ``"top"``, and ``"bottom"``.
        reserve : bool, optional
            Allocate space for a color bar at animation setup, even when
            ``show_color_bar=False``. Set ``show_color_bar=True`` in a callback
            to reveal it. Applies to data with a color scale. Defaults to ``False``.
        **color_bar_params:
            Additional keyword arguments are passed to ``plt.colorbar``.

        Notes
        -----
        During animation, update the label or major tick positions on an
        existing colorbar. Use ``label=""`` to clear its label and
        ``ticks=None`` to restore its tick rule from setup.
        """
        _configure_colorbar(self, label, position, reserve, color_bar_params)

    def get_coordinates_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
    ) -> tuple[float, float]:
        """
        Gets the coordinates of the point on the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value of the point.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            For ``"nearest"``, ``"previous"``, and ``"next"``, the returned coordinates snap to an
            actual data point instead of interpolating; for every other kind, the returned x is the
            same as the queried ``x``.

            Defaults to "linear".

        Returns
        -------
        tuple[float, float]
            The coordinates of the point on the curve at the given x value.
        """
        if interpolation_method in _SNAPPING_INTERPOLATION_METHODS:
            idx_interp = interp1d(
                self._x_data, np.arange(len(self._x_data)), kind=interpolation_method
            )
            idx = int(round(float(idx_interp(x))))
            return (float(self._x_data[idx]), float(self._y_data[idx]))
        return (
            x,
            float(interp1d(self._x_data, self._y_data, kind=interpolation_method)(x)),
        )

    def create_point_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
        label: Optional[str] = None,
        face_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_edge_width: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> Point:
        """
        Creates a Point on the curve at a given x value.

        Parameters
        ----------
        x : float
            The x value of the point.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            For ``"nearest"``, ``"previous"``, and ``"next"``, the returned point snaps to an
            actual data point instead of interpolating; for every other kind, the point's x is the
            same as the queried ``x``.

            Defaults to "linear".
        label : str, optional
            Label to be displayed in the legend.
        face_color : str
            Face color of the point.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the point.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the point.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_edge_width : float
            Width of the point edge.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the point.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the point.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        :class:`~graphinglib.graph_elements.Point`
            The Point on the curve at the given x value.
        """
        x_val, y_val = self.get_coordinates_at_x(x, interpolation_method)
        point = Point(
            x_val,
            y_val,
            label=label,
            face_color=face_color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=marker_edge_width,
            alpha=alpha,
        )
        return point

    def get_coordinates_at_y(
        self,
        y: float,
        interpolation_method: str = "linear",
    ) -> list[tuple[float, float]]:
        """
        Gets the coordinates the curve at a given y value. Can return multiple coordinate pairs if the curve crosses the
        y value multiple times.

        Parameters
        ----------
        y : float
            The y value of the point.
        interpolation_method : str,
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            For ``"nearest"``, ``"previous"``, and ``"next"``, the returned coordinates snap to an
            actual data point instead of interpolating; for every other kind, the returned y is the
            same as the queried ``y``.

            Defaults to "linear".

        Returns
        -------
        list[tuple[float, float]]
            The coordinates of the points on the curve at the given y value.
        """
        xs = self._x_data
        ys = self._y_data
        assert isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray)
        crossings = np.where(np.diff(np.sign(ys - y)))[0]
        points: list[tuple[float, float]] = []
        for cross in crossings:
            x1, x2 = xs[cross], xs[cross + 1]
            y1, y2 = ys[cross], ys[cross + 1]
            if interpolation_method in _SNAPPING_INTERPOLATION_METHODS:
                idx_interp = interp1d([y1, y2], [0, 1], kind=interpolation_method)
                idx = int(round(float(idx_interp(y))))
                points.append((float((x1, x2)[idx]), float((y1, y2)[idx])))
            else:
                f = interp1d([y1, y2], [x1, x2], kind=interpolation_method)
                points.append((float(f(y)), y))
        return points

    def create_points_at_y(
        self,
        y: float,
        interpolation_method: str = "linear",
        label: Optional[str] = None,
        face_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_edge_width: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> list[Point]:
        """
        Creates the Points on the curve at a given y value. Can return multiple Points if the curve crosses the y value
        multiple times.

        Parameters
        ----------
        y : float
            The y value of the desired points.
        interpolation_method : str
            The type of interpolation to be used, as defined in ``scipy.interpolate.interp1d``.

            .. seealso:: `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_

            For ``"nearest"``, ``"previous"``, and ``"next"``, the returned points snap to actual
            data points instead of interpolating; for every other kind, the points' y is the same
            as the queried ``y``.

            Defaults to "linear".
        label : str, optional
            Label to be displayed in the legend.
        face_color : str
            Face color of the point.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the point.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the point.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_edge_width : float
            Width of the point edge.
            Typical range is ``0`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the point.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the point.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        list[:class:`~graphinglib.graph_elements.Point`]
            The Point objects on the curve at the given y value.
        """
        coords = self.get_coordinates_at_y(y, interpolation_method)
        points = [
            Point(
                coord[0],
                coord[1],
                label=label,
                face_color=face_color,
                edge_color=edge_color,
                marker_size=marker_size,
                marker_style=marker_style,
                edge_width=marker_edge_width,
                alpha=alpha,
            )
            for coord in coords
        ]
        return points

    def to_desmos(self, decimal_precision: int = 2, to_clipboard: bool = False) -> str:
        """
        Gives the data points in a Desmos-readable format. The outputted string can then be pasted into a single Desmos
        cell and the object's data will be displayed.

        Parameters
        ----------
        decimal_precision : int, optional
            Specifies the number of decimals of the formatted points.
            Defaults to 2.
        to_clipboard : bool, optional
            Specifies whether the points should be directly copied to the user's clipboard in addition to being
            returned as a string.
            Defaults to False.

        Returns
        -------
        formatted points : str
            A list of tuples representing every data point.
        """
        formatted_points = _format_desmos_points(
            self._x_data, self._y_data, decimal_precision
        )
        if to_clipboard:
            copy_to_clipboard(formatted_points)
        return formatted_points

    @staticmethod
    def _color_intensities(value: Any) -> np.ndarray | None:
        if not isinstance(value, (list, tuple, np.ndarray)):
            return None
        # Numerical tuples in Matplotlib's RGB(A) range are single colors.
        if isinstance(value, tuple) and is_color_like(value):
            return None
        try:
            data = np.asarray(value)
        except ValueError:
            # Named colors and RGB tuples may be mixed in a literal color sequence.
            return None
        if data.ndim == 1 and data.size and data.dtype.kind in "biuf":
            return data
        return None

    @staticmethod
    def _prepare_scatter_color(
        value: Any,
        count: int,
        cmap: Colormap,
        limits: tuple[float, float],
    ) -> np.ndarray:
        intensities = Scatter._color_intensities(value)
        try:
            if intensities is not None:
                rgba = cmap(Normalize(*limits)(intensities))
            else:
                rgba = to_rgba_array("none" if value is None else value)
        except (ValueError, TypeError) as exc:
            raise InvalidParameterError(
                f"Invalid Scatter color specification ({exc})."
            ) from exc
        if len(rgba) not in (0, 1, count):
            raise InvalidParameterError(
                f"Scatter per-point colors must match the final point count ({count}); got {len(rgba)}."
            )
        return rgba

    def _bind_live_colors(
        self,
        session: _AnimationSession,
        collection: PathCollection,
        marker: MarkerStyle,
        colorbars: dict[str, ScalarMappable],
        cycle_color: str,
        errorbars: ErrorbarContainer | None,
        reserved_errors: ErrorbarContainer | None,
    ) -> None:
        initial_face, initial_edge = self._face_color, self._edge_color
        face_cycle = cycle_color
        if (
            is_inherit(initial_face)
            or isinstance(initial_face, str)
            and initial_face == "color cycle"
        ):
            drawn_face = collection.get_facecolor()
            if len(drawn_face) == 1:
                face_cycle = drawn_face
        initial_marker_style = resolve_or(
            self._marker_style, plt.rcParams["scatter.marker"]
        )
        active_marker_style = initial_marker_style
        initial_color_range = (
            tuple(self._color_map_range) if self._color_map_range is not None else None
        )
        initial_error_color = self._errorbars_color
        initial_error_width = self._errorbars_line_width
        initial_cap_thickness = self._cap_thickness
        cap_width_live = False
        visibility_live = False
        initial_cap_width = None
        drawn_error_color = None
        if errorbars is not None and errorbars.lines[2]:
            _, caps, segments = errorbars.lines
            cap_width_live = bool(caps) and all(cap.axes is not None for cap in caps)
            visibility_live = not caps or cap_width_live
            initial_cap_width = caps[0].get_markersize() / 2 if cap_width_live else None
            drawn_error_color = segments[0].get_color()
        contrast_color = get_contrasting_shade(plt.rcParams["axes.facecolor"])
        initial_cmap = plt.get_cmap(
            resolve_or(self._color_map, plt.rcParams["image.cmap"])
        )

        def initial_limits(name: str, value: Any) -> tuple[float, float]:
            if name in colorbars:
                lo, hi = colorbars[name].get_clim()
                if lo is not None and hi is not None:
                    return lo, hi
            if initial_color_range is not None:
                return min(initial_color_range), max(initial_color_range)
            values = self._color_intensities(value)
            if values is not None:
                finite = values[np.isfinite(values)]
                if finite.size:
                    return float(finite.min()), float(finite.max())
            return 0.0, 1.0

        face_limits = initial_limits("face", initial_face)
        edge_limits = initial_limits("edge", initial_edge)

        def prepare() -> _ArtistEdit:
            face = resolve_or(self._face_color, initial_face)
            edge = resolve_or(self._edge_color, initial_edge)
            if face is None and edge is None:
                raise IncompatibleArgumentsError(
                    "Set a face or edge color for the animated Scatter."
                )
            if is_inherit(face) or isinstance(face, str) and face == "color cycle":
                face = face_cycle
            if is_inherit(edge) or isinstance(edge, str) and edge == "color cycle":
                edge = cycle_color

            cmap = plt.get_cmap(resolve_or(self._color_map, initial_cmap))
            explicit = self._color_map_range
            if explicit is None or tuple(explicit) == initial_color_range:
                face_range, edge_range = face_limits, edge_limits
            else:
                if (
                    len(explicit) != 2
                    or not np.isfinite(explicit).all()
                    or explicit[0] == explicit[1]
                ):
                    raise InvalidParameterError(
                        "Scatter color_map_range must contain two distinct finite numbers."
                    )
                face_range = edge_range = min(explicit), max(explicit)
            count = self._x_data.size
            face_rgba = self._prepare_scatter_color(face, count, cmap, face_range)
            edge_rgba = (
                face_rgba
                if isinstance(edge, str) and edge == "face"
                else self._prepare_scatter_color(edge, count, cmap, edge_range)
            )
            for name, value, rgba in (
                ("face", face, face_rgba),
                ("edge", edge, edge_rgba),
            ):
                if (
                    name in colorbars
                    and self._color_intensities(value) is None
                    and (count or len(rgba))
                ):
                    raise InvalidParameterError(
                        f"The existing Scatter {name} colorbar requires numeric color values for that channel."
                    )
            marker_style = resolve_or(self._marker_style, initial_marker_style)
            current_marker = (
                marker
                if marker_style == initial_marker_style
                else MarkerStyle(marker_style)
            )
            if not current_marker.is_filled():
                edge_rgba = face_rgba
                if current_marker.get_fillstyle() == "none":
                    face_rgba = np.empty((0, 4))

            edits = [
                _edit(collection.set_facecolor, collection.get_facecolor(), face_rgba),
                _edit(collection.set_edgecolor, collection.get_edgecolor(), edge_rgba),
            ]
            if marker_style != active_marker_style:

                def set_marker(state) -> None:
                    nonlocal active_marker_style
                    style, paths = state
                    collection.set_paths(paths)
                    active_marker_style = style

                path = current_marker.get_path().transformed(
                    current_marker.get_transform()
                )
                edits.append(
                    _edit(
                        set_marker,
                        (active_marker_style, collection.get_paths()),
                        (marker_style, [path]),
                    )
                )
            for current in (errorbars, reserved_errors):
                if current is None or not current.lines[2]:
                    continue
                reserved_axes = (
                    collection.axes
                    if current is reserved_errors
                    and isinstance(collection.axes, plt.Axes)
                    else None
                )
                change_caps = reserved_axes is not None or cap_width_live
                default_cap_width = (
                    current.lines[1][0].get_markersize() / 2
                    if reserved_axes is not None
                    else initial_cap_width
                )
                color = resolve_or(self._errorbars_color, initial_error_color)
                if is_inherit(color):
                    color = (
                        drawn_error_color[0]
                        if drawn_error_color is not None
                        else current.lines[2][0].get_color()[0]
                    )
                elif color == "same as scatter":
                    color = self._error_color(edge_rgba, face_rgba, contrast_color)
                elif not isinstance(color, str):
                    raise InvalidParameterError(
                        "Scatter errorbars_color must be a string."
                    )
                edits.extend(
                    _error_style_edits(
                        current,
                        color=color,
                        line_width=resolve_or(
                            self._errorbars_line_width,
                            resolve_or(initial_error_width, None),
                        ),
                        cap_width=resolve_or(self._cap_width, default_cap_width)
                        if change_caps
                        else None,
                        cap_thickness=resolve_or(
                            self._cap_thickness, resolve_or(initial_cap_thickness, None)
                        ),
                        alpha=None,
                        show=self._show_errorbars,
                        directions=(
                            self._x_error is not None,
                            self._y_error is not None,
                        ),
                        reserved_axes=reserved_axes,
                        reserved_data=(
                            self._x_data,
                            self._y_data,
                            self._x_error,
                            self._y_error,
                        )
                        if reserved_axes is not None
                        else None,
                    )
                )

            for name, mappable in colorbars.items():
                limits_for_bar = face_range if name == "face" else edge_range
                previous = mappable.get_cmap(), mappable.get_clim()
                if previous == (cmap, limits_for_bar):
                    continue

                def set_scale(state, bar=mappable) -> None:
                    palette, color_limits = state
                    with (
                        bar.callbacks.blocked(signal="changed"),
                        bar.norm.callbacks.blocked(signal="changed"),
                    ):
                        bar.set_cmap(palette)
                        bar.set_clim(color_limits)
                    bar.changed()

                edits.append(
                    _edit(
                        set_scale,
                        previous,
                        (cmap, limits_for_bar),
                    )
                )

            return _combine_edits(edits)

        properties = (
            "point_data",
            "_face_color",
            "_edge_color",
            "_color_map",
            "_color_map_range",
            "_marker_style",
        )
        if drawn_error_color is not None or reserved_errors is not None:
            properties += (
                _ERROR_STYLE_PROPERTIES
                if cap_width_live or reserved_errors is not None
                else _ERROR_STYLE_PROPERTIES[:2]
            )
        if visibility_live or reserved_errors is not None:
            properties += ("_show_errorbars",)
        session.bind_derived(self, properties, prepare)

    @staticmethod
    def _error_color(edge: Any, face: Any, fallback: str) -> Any:
        if is_color_like(edge):
            return edge
        if is_color_like(face):
            return face
        return fallback

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified axes.
        """
        # Check that either face color or edge color is not None
        if self._face_color is None and self._edge_color is None:
            raise IncompatibleArgumentsError(
                "Both face color and edge color cannot be None. Please set at least one of them to a valid color."
            )

        session = _DRAW_SESSION.get()
        colorbars: dict[str, ScalarMappable] = {}
        show_colorbars = resolve_or(self._show_color_bar, False)
        prepare_colorbars = show_colorbars or (
            session is not None and self._reserve_color_bar
        )

        def marker_color(name: str, value: Any, cycle: str | None) -> Any:
            values = self._color_intensities(value)
            if values is not None:
                cmap = plt.get_cmap(
                    resolve_or(self._color_map, plt.rcParams["image.cmap"])
                )
                if self._color_map_range is not None:
                    low, high = min(self._color_map_range), max(self._color_map_range)
                else:
                    finite = values[np.isfinite(values)]
                    low, high = (finite.min(), finite.max()) if finite.size else (0, 1)
                # Colorbar expands a constant range. Map the starting markers
                # with that same range so unrelated frame edits cannot shift colors.
                if prepare_colorbars:
                    low, high = nonsingular(low, high, expander=0.1)
                norm = Normalize(low, high)
                if prepare_colorbars:
                    colorbars[name] = ScalarMappable(cmap=cmap, norm=norm)
                return cmap(norm(values))
            if value is None:
                return "none"
            if is_inherit(value) or isinstance(value, str) and value == "color cycle":
                return cycle
            return value

        mpl_face_color = marker_color("face", self._face_color, None)
        mpl_edge_color = marker_color(
            "edge", self._edge_color, kwargs.get("cycle_color")
        )
        params = {
            "s": self._marker_size,
            "marker": self._marker_style,
            "linewidth": self._marker_edge_width,
            "alpha": self._alpha,
        }
        params = strip_inherit(params)
        params["facecolors"] = mpl_face_color
        params["edgecolors"] = mpl_edge_color
        self.handle = axes.scatter(
            self._x_data,
            self._y_data,
            label=self._label,
            zorder=z_order,
            **params,
        )
        errorbars = None
        if self._show_errorbars or (
            session is not None and getattr(self, "_errorbars_configured", False)
        ):
            # Convert errorbars color to matplotlib notation
            if self._errorbars_color is None:
                raise InvalidParameterError(
                    "Errorbars color cannot be None. Please set the errorbars color to a valid color."
                )
            elif is_inherit(self._errorbars_color):
                # Use color cycle
                mpl_errorbars_color = None
            elif (
                isinstance(self._errorbars_color, str)
                and self._errorbars_color == "same as scatter"
            ):
                mpl_errorbars_color = self._error_color(
                    self.handle.get_edgecolor(),
                    self.handle.get_facecolor(),
                    get_contrasting_shade(plt.rcParams["axes.facecolor"]),
                )
            elif isinstance(self._errorbars_color, str):
                # Use specified color
                mpl_errorbars_color = self._errorbars_color
            else:
                raise InvalidParameterError("Errorbars color must be a string.")

            errorbar_params = {
                "markerfacecolor": None,
                "markeredgecolor": None,
                "elinewidth": self._errorbars_line_width,
                "capsize": self._cap_width,
                "capthick": self._cap_thickness,
                "linestyle": "none",
            }
            errorbar_params = strip_inherit(errorbar_params)
            errorbar_params["ecolor"] = mpl_errorbars_color
            draw_errorbars = (
                axes.errorbar
                if session is None
                else partial(_draw_live_errorbars, axes, show=self._show_errorbars)
            )
            errorbars = self.errorbars_handle = draw_errorbars(
                self._x_data,
                self._y_data,
                xerr=self._x_error,
                yerr=self._y_error,
                zorder=z_order - 1,
                **errorbar_params,
            )

        for mappable in colorbars.values():
            bar = plt.colorbar(mappable, ax=axes, **self._color_bar_params)
            if session is not None:
                session.bind_colorbar(
                    self, bar, initially_visible=show_colorbars
                )

        if session is not None:
            session.bind_legend_source(self, self.handle, lambda: self.label)
            reserved_errors = _reserve_errorbar_directions(
                axes,
                errorbars,
                color=kwargs.get("cycle_color", "C0"),
                zorder=z_order - 1,
                cap_width=resolve_or(self._cap_width, None),
            )
            session.bind_scatter(self, self.handle, errorbars, reserved_errors)
            marker = MarkerStyle(
                resolve_or(self._marker_style, plt.rcParams["scatter.marker"])
            )
            self._bind_live_colors(
                session,
                self.handle,
                marker,
                colorbars,
                kwargs.get("cycle_color", "C0"),
                errorbars,
                reserved_errors,
            )
            session.bind_property(
                self, "_marker_size", self.handle, "sizes", lambda value: [value]
            )
            session.bind_property(self, "_marker_edge_width", self.handle, "linewidth")
            session.bind_property(self, "_alpha", self.handle, "alpha")


@dataclass
class Histogram(Plottable1D):
    """
    This class implements a general histogram.

    Parameters
    ----------
    data : ArrayLike
        Array of values to be plotted.
    bins : int | ArrayLike
        If `bins` is an integer, it defines the number of equal_width bins to be used in the histogram.
        If `bins` is an array, it defines the bin edges to be used in the histogram, including the left edge of the
        first bin and the right edge of the last bin.
    label : str, optional
        Label to be displayed in the legend.
    face_color : str
        Face color of the histogram.
        Default depends on the ``figure_style`` configuration.
    edge_color : str
        Edge color of the histogram.
        Default depends on the ``figure_style`` configuration.
    hist_type : str
        Type of the histogram.
        Values are ``"bar"``, ``"barstacked"``, ``"step"``, and ``"stepfilled"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the histogram.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    line_width : float
        Width of the histogram edge.
        Typical range is ``0.5`` to ``3`` points.
        Default depends on the ``figure_style`` configuration.
    normalize : bool
        Whether or not to normalize the histogram.
        Default depends on the ``figure_style`` configuration.
    orientation: str
        Whether to plot the histogram on x-axis or on y-axis.
        Values are ``"vertical"`` and ``"horizontal"``.
        Default depends on the ``figure_style`` configuration.
    show_params : bool
        Whether or not to show the mean and standard deviation of the data.
        Default depends on the ``figure_style`` configuration.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        data: ArrayLike,
        bins: int,
        label: Optional[str] = None,
        face_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        hist_type: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        normalize: bool | Inherit = INHERIT,
        orientation: str | Inherit = INHERIT,
        show_params: bool | Inherit = INHERIT,
    ) -> None:
        """
        This class implements a general histogram.

        Parameters
        ----------
        data : ArrayLike
            Array of values to be plotted.
        bins : int | ArrayLike
            If `bins` is an integer, it defines the number of equal_width bins to be used in the histogram.
            If `bins` is an array, it defines the bin edges to be used in the histogram, including the left edge of the
            first bin and the right edge of the last bin.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str
            Face color of the histogram.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the histogram.
            Default depends on the ``figure_style`` configuration.
        hist_type : str
            Type of the histogram.
            Values are ``"bar"``, ``"barstacked"``, ``"step"``, and ``"stepfilled"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the histogram.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the histogram edge.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        normalize : bool
            Whether or not to normalize the histogram.
            Default depends on the ``figure_style`` configuration.
        orientation: str
            Whether to plot the histogram on x-axis or on y-axis.
            Values are ``"vertical"`` and ``"horizontal"``.
            Default depends on the ``figure_style`` configuration.
        show_params : bool
            Whether or not to show the mean and standard deviation of the data.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        self._bins = bins
        self._label = label
        self._face_color = face_color
        self._edge_color = edge_color
        self._hist_type = hist_type
        self._alpha = alpha
        self._line_width = line_width
        self._normalize = normalize
        self._orientation = orientation
        self._show_params = show_params
        self._histogram_cache: Optional[tuple[bool, tuple[np.ndarray, np.ndarray]]] = (
            None
        )
        self.data = np.asarray(data)

        self._show_pdf = False
        self._pdf_type = None
        self._pdf_show_mean = INHERIT
        self._pdf_show_std = INHERIT
        self._pdf_curve_color = INHERIT
        self._pdf_mean_color = INHERIT
        self._pdf_std_color = INHERIT

    @classmethod
    def from_fit_residuals(
        cls,
        fit: Fit,
        bins: int,
        label: Optional[str] = None,
        face_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        hist_type: str | Inherit = INHERIT,
        alpha: int | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        normalize: bool | Inherit = INHERIT,
        orientation: str | Inherit = INHERIT,
        show_params: bool | Inherit = INHERIT,
    ) -> Self:
        """
        Calculates the residuals of a fit and plots them as a histogram.

        Parameters
        ----------
        fit : Fit
            The fit from which the residuals are to be calculated.
        bins : int | ArrayLike
            If `bins` is an integer, it defines the number of equal_width bins to be used in the histogram.
            If `bins` is an array, it defines the bin edges to be used in the histogram, including the left edge of the
            first bin and the right edge of the last bin.
        label : str, optional
            Label to be displayed in the legend.
        face_color : str
            Face color of the histogram.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
            Edge color of the histogram.
            Default depends on the ``figure_style`` configuration.
        hist_type : str
            Type of the histogram.
            Values are ``"bar"``, ``"barstacked"``, ``"step"``, and ``"stepfilled"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the histogram.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the histogram edge.
            Typical range is ``0.5`` to ``3`` points.
            Default depends on the ``figure_style`` configuration.
        normalize : bool
            Whether or not to normalize the histogram.
            Default depends on the ``figure_style`` configuration.
        orientation: str
            Whether to plot the histogram on x-axis or on y-axis.
            Values are ``"vertical"`` and ``"horizontal"``.
            Default depends on the ``figure_style`` configuration.
        show_params : bool
            Whether or not to show the mean and standard deviation of the data.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Returns
        -------
        A :class:`~graphinglib.data_plotting_1d.Histogram` object created from the residuals of a fit.
        """
        residuals = fit.get_residuals()
        return cls(
            residuals,
            bins,
            label,
            face_color,
            edge_color,
            hist_type,
            alpha,
            line_width,
            normalize,
            orientation,
            show_params,
        )

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: ArrayLike) -> None:
        values = np.array(data)
        session = _bound_session(self)
        if session is not None and (
            values.ndim != 1
            or values.dtype.kind not in "biuf"
            or not np.isfinite(values).all()
        ):
            raise InvalidParameterError(
                "Animated Histogram data must be a one-dimensional array of finite numbers."
            )
        mean = np.mean(values) if values.size else float("nan")
        deviation = np.std(values) if values.size else float("nan")
        if session is not None:
            previous = self._data, self._mean, self._standard_deviation

            def restore() -> None:
                self._data, self._mean, self._standard_deviation = previous

            session.queue_property(self, "_data", values, restore)
        self._invalidate_histogram_cache()
        self._data, self._mean, self._standard_deviation = values, mean, deviation

    def _invalidate_histogram_cache(self) -> None:
        session = _bound_session(self)
        if session is not None:
            before = self._histogram_cache
            session.remember(
                self,
                "_histogram_cache",
                lambda: setattr(self, "_histogram_cache", before),
            )
        self._histogram_cache = None

    @property
    def bins(self) -> int | ArrayLike:
        return self._bins

    @bins.setter
    def bins(self, bins: int | ArrayLike) -> None:
        if _bound_session(self) is not None and not isinstance(bins, (Integral, str)):
            bins = np.array(bins, copy=True)
        self._bins = _stage_live_value(self, "_bins", bins)
        self._invalidate_histogram_cache()

    @property
    def label(self) -> str | None:
        return self._get_label()

    @label.setter
    def label(self, label: str | None) -> None:
        self._label = _stage_live_value(self, "_label", label)

    @property
    def face_color(self) -> Styled[str]:
        return self._face_color

    @face_color.setter
    def face_color(self, face_color: str) -> None:
        self._face_color = _stage_live_value(self, "_face_color", face_color)

    @property
    def edge_color(self) -> Styled[str]:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = _stage_live_value(self, "_edge_color", edge_color)

    @property
    def hist_type(self) -> Styled[str]:
        return self._hist_type

    @hist_type.setter
    def hist_type(self, hist_type: str) -> None:
        self._hist_type = _stage_live_value(self, "_hist_type", hist_type)

    @property
    def alpha(self) -> Styled[float]:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = _stage_live_value(self, "_alpha", alpha)

    @property
    def line_width(self) -> Styled[float]:
        return self._line_width

    @line_width.setter
    def line_width(self, line_width: float) -> None:
        self._line_width = _stage_live_value(self, "_line_width", line_width)

    @property
    def normalize(self) -> Styled[bool]:
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self._normalize = _stage_live_value(self, "_normalize", normalize)
        self._invalidate_histogram_cache()

    @property
    def orientation(self) -> Styled[str]:
        return self._orientation

    @orientation.setter
    def orientation(self, orientation: str) -> None:
        self._orientation = _stage_live_value(self, "_orientation", orientation)

    @property
    def show_params(self) -> Styled[bool]:
        return self._show_params

    @show_params.setter
    def show_params(self, show_params: bool) -> None:
        self._show_params = _stage_live_value(self, "_show_params", show_params)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def standard_deviation(self) -> float:
        return self._standard_deviation

    @property
    def parameters(self) -> tuple[float, float]:
        return self._mean, self._standard_deviation

    @property
    def _resolved_normalize(self) -> bool:
        return False if is_inherit(self._normalize) else bool(self._normalize)

    def _compute_histogram(self) -> tuple[np.ndarray, np.ndarray]:
        # The cache remembers which `density` value it was computed with, and is
        # recomputed if that value has changed since. This is needed because some
        # code (e.g. figure style resolution) changes `_normalize` without going
        # through the `normalize` setter, so the setter alone can't clear the cache.
        density = self._resolved_normalize
        if self._histogram_cache is None or self._histogram_cache[0] != density:
            self._histogram_cache = (
                density,
                np.histogram(self._data, bins=self._bins, density=density),
            )
        return self._histogram_cache[1]

    @property
    def bin_heights(self) -> np.ndarray:
        return self._compute_histogram()[0]

    @property
    def bin_edges(self) -> np.ndarray:
        return self._compute_histogram()[1]

    @property
    def bin_centers(self) -> np.ndarray:
        edges = self.bin_edges
        return (edges[:-1] + edges[1:]) / 2

    def __eq__(self, other: object) -> bool:
        """
        Defines the equality between two histograms.
        """
        if not isinstance(other, Histogram):
            return False
        return bool(
            np.equal(self.bin_heights, other.bin_heights).all()
            and np.equal(self.bin_centers, other.bin_centers).all()
        )

    def _get_label(self, default_show_params: bool = True) -> str | None:
        """
        Gives the label of the histogram (with or without parameters).
        """
        lab = self._label
        show_params = resolve_or(self._show_params, default_show_params)
        if lab and show_params:
            lab += (
                " :\n"
                + rf"$\mu$ = {0 if abs(self._mean) < 1e-3 else self._mean:.3f}, $\sigma$ = {self._standard_deviation:.3f}"
            )
        elif show_params:
            lab = rf"$\mu$ = {0 if abs(self._mean) < 1e-3 else self._mean:.3f}, $\sigma$ = {self._standard_deviation:.3f}"
        return lab

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_1d.Histogram` object.
        """
        return deepcopy(self)

    def _normal_normalized(self, x: ArrayLike) -> ArrayLike:
        """
        Calculates the normalized gaussian curve from the mean and standard deviation of the data.

        Parameters
        ----------
        x : ArrayLike
            The x values at which the gaussian curve is to be calculated.

        Returns
        -------
        The corresponding array of y values of the gaussian curve.
        """
        x = np.array(x)
        return (1 / (self._standard_deviation * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (((x - self._mean) / self._standard_deviation) ** 2)
        )

    def _normal_not_normalized(self, x: ArrayLike) -> ArrayLike:
        """
        Calculates the (not normalized) gaussian curve from the mean and standard deviation of the data.

        Parameters
        ----------
        x : ArrayLike
            The x values at which the gaussian curve is to be calculated.

        Returns
        -------
        The corresponding array of y values of the gaussian curve.
        """
        x = np.array(x)
        bin_heights, bin_edges = self._compute_histogram()
        total_area = np.sum(bin_heights * np.diff(bin_edges))
        return total_area * self._normal_normalized(x)

    def add_pdf(
        self,
        type: str = "normal",
        show_mean: bool | Inherit = INHERIT,
        show_std: bool | Inherit = INHERIT,
        curve_color: str | Inherit = INHERIT,
        mean_color: str | Inherit = INHERIT,
        std_color: str | Inherit = INHERIT,
    ) -> None:
        """
        Shows the probability density function of the histogram.

        Parameters
        ----------
        type : str
            The type of probability density function to be shown.
            Currently only ``"normal"`` is supported.
            Defaults to "normal".
        show_mean : bool
            Whether or not to show the mean of the data.
            Default depends on the ``figure_style`` configuration.
        show_std : bool
            Whether or not to show the standard deviation of the data.
            Default depends on the ``figure_style`` configuration.
        curve_color : str
            Color of the probability density function curve.
            Default depends on the ``figure_style`` configuration.
        mean_color : str
            Color of the mean line.
            Default depends on the ``figure_style`` configuration.
        std_color : str
            Color of the standard deviation lines.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        if type != "normal":
            raise InvalidParameterError(
                "Currently, only 'normal' distribution is supported."
            )
        self._stage_pdf_configuration()
        self._show_pdf = True
        self._pdf_type = type
        self._pdf_show_mean = show_mean
        self._pdf_show_std = show_std
        self._pdf_curve_color = curve_color
        self._pdf_mean_color = mean_color
        self._pdf_std_color = std_color

    def remove_pdf(self) -> None:
        """Hide the density curve and its guides."""
        self._stage_pdf_configuration()
        self._show_pdf = False

    def _stage_pdf_configuration(self) -> None:
        session = _bound_session(self)
        if session is not None:
            fields = (
                "_show_pdf",
                "_pdf_type",
                "_pdf_show_mean",
                "_pdf_show_std",
                "_pdf_curve_color",
                "_pdf_mean_color",
                "_pdf_std_color",
            )
            previous = tuple(getattr(self, field) for field in fields)

            def restore() -> None:
                for field, value in zip(fields, previous, strict=True):
                    setattr(self, field, value)

            session.queue_property(self, "_pdf_configuration", None, restore)

    def to_desmos(self, decimal_precision: int = 2, to_clipboard: bool = False) -> str:
        """
        Gives every bin's upper center in a Desmos-readable format. The outputted string can then be pasted into a
        single Desmos cell and the object's data will be displayed.

        Parameters
        ----------
        decimal_precision : int, optional
            Specifies the number of decimals of the formatted points.
            Defaults to 2.
        to_clipboard : bool, optional
            Specifies whether the points should be directly copied to the user's clipboard in addition to being
            returned as a string.
            Defaults to False.

        Returns
        -------
        formatted points : str
            A list of tuples representing every data point.
        """
        formatted_points = _format_desmos_points(
            self.bin_centers, self.bin_heights, decimal_precision
        )
        if to_clipboard:
            copy_to_clipboard(formatted_points)
        return formatted_points

    def _live_histogram(self, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._data.ndim != 1
            or self._data.dtype.kind not in "biuf"
            or not np.isfinite(self._data).all()
        ):
            raise InvalidParameterError(
                "Animated Histogram data must be a one-dimensional array of finite numbers."
            )
        try:
            values = (
                self._data.astype(np.uint8)
                if self._data.dtype.kind == "b"
                else self._data
            )
            counts, edges = np.histogram(values, bins=self._bins)
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError(
                f"Invalid animated Histogram bins ({exc})."
            ) from exc
        widths = np.diff(edges)
        if not np.isfinite(edges).all() or not (widths > 0).all():
            raise InvalidParameterError(
                "Animated Histogram bin edges must be finite and strictly increasing."
            )
        if normalize:
            total = counts.sum()
            counts = (
                counts / widths / total
                if total
                else np.zeros(counts.shape, dtype=float)
            )
        return counts, edges

    def _bind_live_histogram(
        self,
        session: _AnimationSession,
        axes: plt.Axes,
        artists,
        z_order: int,
        get_geometry,
    ) -> None:
        handle = self.handle
        assert isinstance(handle, Polygon)
        bars = list(artists.patches) if isinstance(artists, BarContainer) else []
        initial_type = resolve_or(self._hist_type, "bar")
        if isinstance(artists, list) and artists:
            step = artists[0]
        else:
            step = Polygon(
                np.empty((0, 2)), closed=False, visible=False, zorder=z_order - 1
            )
            axes.add_artist(step)
        initial_face = self._face_color
        initial_edge = self._edge_color
        initial_width = resolve_or(self._line_width, handle.get_linewidth())
        initial_alpha = resolve_or(self._alpha, 1.0)
        original_face = handle.get_facecolor()
        original_edge = handle.get_edgecolor()

        def prepare() -> _ArtistEdit:
            heights, edges, horizontal, _ = get_geometry()
            face = to_rgba(
                resolve_or(self._face_color, resolve_or(initial_face, original_face)),
                resolve_or(self._alpha, initial_alpha),
            )
            configured_edge = resolve_or(self._edge_color, initial_edge)
            edge = (
                original_edge
                if is_inherit(configured_edge)
                else to_rgba(configured_edge, 1)
            )
            width = resolve_or(self._line_width, initial_width)
            mode = resolve_or(self._hist_type, initial_type)
            if mode not in ("bar", "barstacked", "step", "stepfilled"):
                raise InvalidParameterError(
                    "Histogram hist_type must be 'bar', 'barstacked', 'step', or 'stepfilled'."
                )
            use_step = mode in ("step", "stepfilled")
            filled = mode == "stepfilled"
            x = np.repeat(edges, 2)
            y = np.r_[0, np.repeat(heights, 2), 0]
            xy = np.column_stack((y, x) if horizontal else (x, y))

            def set_step(state) -> None:
                closed, fill, vertices, visible = state
                step.set_closed(closed)
                step.set_fill(fill)
                step.set_xy(vertices)
                step.set_visible(visible)

            edits = [
                _edit(
                    set_step,
                    (
                        step.get_closed(),
                        step.get_fill(),
                        step.get_xy(),
                        step.get_visible(),
                    ),
                    (filled, filled, xy, use_step),
                )
            ]
            size = len(bars)
            widths = np.diff(edges)
            bounds = []
            if not use_step:
                bounds = (
                    list(zip(np.zeros_like(heights), edges[:-1], heights, widths))
                    if horizontal
                    else list(zip(edges[:-1], np.zeros_like(heights), widths, heights))
                )
            for index, bar in enumerate(bars):
                visible = index < len(bounds)
                if visible:
                    edits.append(
                        _edit(
                            lambda value, bar=bar: bar.set_bounds(*value),
                            bar.get_bbox().bounds,
                            bounds[index],
                        )
                    )
                edits.append(_edit(bar.set_visible, bar.get_visible(), visible))

            def grow() -> None:
                for x, y, w, h in bounds[size:]:
                    bar = Rectangle(
                        (x, y),
                        w,
                        h,
                        facecolor=face,
                        edgecolor=edge,
                        linewidth=width,
                        zorder=z_order - 1,
                    )
                    bars.append(bar)
                    axes.add_artist(bar)

            def shrink() -> None:
                for bar in reversed(bars[size:]):
                    if bar.axes is not None:
                        bar.remove()
                del bars[size:]

            edits.append(_ArtistEdit(grow, shrink))

            for target in (handle, step, *bars):
                edits.extend(
                    (
                        _edit(target.set_facecolor, target.get_facecolor(), face),
                        _edit(target.set_edgecolor, target.get_edgecolor(), edge),
                        _edit(target.set_linewidth, target.get_linewidth(), width),
                    )
                )
            return _combine_edits(edits)

        session.bind_derived(
            self,
            (
                "_data",
                "_bins",
                "_normalize",
                "_orientation",
                "_hist_type",
                "_face_color",
                "_edge_color",
                "_alpha",
                "_line_width",
            ),
            prepare,
        )

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified axes.
        """
        normalize_resolved = self._resolved_normalize
        params = {
            "facecolor": (
                to_rgba(self._face_color, resolve_or(self._alpha, 1.0))
                if not is_inherit(self._face_color)
                else INHERIT
            ),
            "edgecolor": (
                to_rgba(self._edge_color, 1)
                if not is_inherit(self._edge_color)
                else self._edge_color
            ),
            "linewidth": self._line_width,
        }
        params = strip_inherit(params)
        self.handle = Polygon(
            np.array([[0, 2, 2, 3, 3, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 0]]).T,
            **params,
        )
        params = {
            "facecolor": (
                to_rgba(self._face_color, resolve_or(self._alpha, 1.0))
                if not is_inherit(self._face_color)
                else INHERIT
            ),
            "edgecolor": (
                to_rgba(self._edge_color, 1)
                if not is_inherit(self._edge_color)
                else self._edge_color
            ),
            "histtype": self._hist_type,
            "linewidth": self._line_width,
            "density": normalize_resolved,
            "orientation": self._orientation,
        }
        params = strip_inherit(params)
        heights, edges, artists = axes.hist(
            self._data,
            bins=cast(Any, self._bins),
            label=self.label,  # uses the get_label() method
            zorder=z_order - 1,
            **params,
        )
        orientation = resolve_or(self._orientation, "vertical")
        horizontal = orientation == "horizontal"
        session = _DRAW_SESSION.get()

        cached_data = cached_bins = cached_density = None
        cached_geometry = None

        def geometry():
            nonlocal cached_data, cached_bins, cached_density, cached_geometry
            direction = resolve_or(self._orientation, orientation)
            if direction not in ("vertical", "horizontal"):
                raise InvalidParameterError(
                    "Histogram orientation must be 'vertical' or 'horizontal'."
                )
            density = resolve_or(self._normalize, normalize_resolved)
            if not isinstance(density, (bool, np.bool_)):
                raise InvalidParameterError("Histogram normalize must be a boolean.")
            if (
                cached_geometry is None
                or cached_data is not self._data
                or cached_bins is not self._bins
                or cached_density != density
            ):
                cached_geometry = self._live_histogram(density)
                cached_data, cached_bins, cached_density = (
                    self._data,
                    self._bins,
                    density,
                )
            return *cached_geometry, direction == "horizontal", density

        if session is not None:
            self._bind_live_histogram(session, axes, artists, z_order, geometry)
            show_params = resolve_or(self._show_params, True)
            session.bind_legend_source(
                self,
                self.handle,
                lambda: self._get_label(show_params),
                ("_label", "_data", "_show_params"),
            )
        if self._show_pdf or session is not None:
            self._plot_pdf(
                axes,
                z_order,
                np.asarray(heights),
                edges,
                horizontal,
                normalize_resolved,
                session,
                geometry,
            )

    def _plot_pdf(
        self,
        axes: plt.Axes,
        z_order: int,
        heights: np.ndarray,
        edges: np.ndarray,
        horizontal: bool,
        normalize: bool,
        session: _AnimationSession | None,
        get_geometry,
    ) -> None:
        def coordinates(
            counts: np.ndarray,
            boundaries: np.ndarray,
            horizontal: bool,
            normalize: bool,
        ):
            mean, std = self.parameters
            if not np.isfinite(std) or std <= 0:
                empty = np.empty(0)
                return ((empty, empty), (empty, empty), (empty, empty))

            area = 1 if normalize else np.sum(counts * np.diff(boundaries))
            x_data = np.linspace(boundaries[0], boundaries[-1], 500)
            y_data = area * np.asarray(self._normal_normalized(x_data))
            mean_position = np.array([mean])
            std_positions = np.array([mean - std, mean + std])
            line = (y_data, x_data) if horizontal else (x_data, y_data)
            return (
                line,
                (
                    mean_position,
                    area * np.asarray(self._normal_normalized(mean_position)),
                ),
                (
                    std_positions,
                    area * np.asarray(self._normal_normalized(std_positions)),
                ),
            )

        show_pdf = self._show_pdf
        empty = np.empty(0)
        empty_geometry = ((empty, empty),) * 3
        initial_geometry = (
            coordinates(heights, edges, horizontal, normalize)
            if show_pdf
            else empty_geometry
        )
        params = strip_inherit({"color": self._pdf_curve_color})
        if show_pdf:
            (pdf,) = axes.plot(*initial_geometry[0], zorder=z_order, **params)
        else:
            pdf = Line2D([], [], visible=False, zorder=z_order, **params)
            axes.add_artist(pdf)

        draw_guides = axes.hlines if horizontal else axes.vlines
        guide_artists = []
        for index, enabled, color in (
            (2, self._pdf_show_std, self._pdf_std_color),
            (1, self._pdf_show_mean, self._pdf_mean_color),
        ):
            enabled = resolve_or(enabled, True)
            params = strip_inherit({"colors": color})
            if show_pdf and enabled:
                positions, densities = initial_geometry[index]
                artist = draw_guides(
                    positions,
                    0,
                    densities,
                    linestyles="dashed",
                    zorder=z_order - 1,
                    **params,
                )
            elif session is not None:
                artist = LineCollection(
                    [],
                    linestyles="dashed",
                    zorder=z_order - 1,
                    visible=False,
                    **params,
                )
                axes.add_collection(artist, autolim=False)
            else:
                continue
            guide_artists.append(
                (artist, index, enabled, np.array(artist.get_colors(), copy=True))
            )

        if session is not None:
            initial_curve_color = pdf.get_color()

            def prepare() -> _ArtistEdit:
                counts, boundaries, horizontal, normalize = get_geometry()
                visible = self._show_pdf
                geometry = (
                    coordinates(counts, boundaries, horizontal, normalize)
                    if visible
                    else empty_geometry
                )
                edits = [
                    _edit(pdf.set_data, pdf.get_data(), geometry[0]),
                    _edit(pdf.set_visible, pdf.get_visible(), visible),
                    _edit(
                        pdf.set_color,
                        pdf.get_color(),
                        resolve_or(self._pdf_curve_color, initial_curve_color),
                    ),
                ]
                for artist, index, initial_enabled, initial_color in guide_artists:
                    if index == 1:
                        enabled, color = self._pdf_show_mean, self._pdf_mean_color
                    else:
                        enabled, color = self._pdf_show_std, self._pdf_std_color
                    positions, densities = geometry[index]
                    starts = np.column_stack((positions, np.zeros_like(positions)))
                    ends = np.column_stack((positions, densities))
                    segments = np.stack((starts, ends), axis=1)
                    if horizontal:
                        segments = segments[..., ::-1]
                    edits.extend(
                        (
                            _edit(artist.set_segments, artist.get_segments(), segments),
                            _edit(
                                artist.set_color,
                                artist.get_colors(),
                                resolve_or(color, initial_color),
                            ),
                            _edit(
                                artist.set_visible,
                                artist.get_visible(),
                                visible and resolve_or(enabled, initial_enabled),
                            ),
                        )
                    )
                return _combine_edits(edits)

            session.bind_derived(
                self,
                ("_data", "_bins", "_orientation", "_normalize", "_pdf_configuration"),
                prepare,
            )
