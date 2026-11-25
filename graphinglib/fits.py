from __future__ import annotations

from copy import deepcopy
from functools import partial
from inspect import signature
from math import isfinite
from numbers import Real
from typing import Any, Callable, Generic, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from typing_extensions import TypeVar

from ._animation import (
    _DRAW_SESSION,
    _AnimationSession,
    _ArtistEdit,
    _attach_artists,
    _bound_session,
    _combine_edits,
    _CurveBands,
    _edit,
    _reject_live_update,
    _stage_live_value,
)
from .data_plotting_1d import _KEEP_ERROR, Curve, Scatter, _KeepError
from .exceptions import InvalidParameterError, InvalidParameterTypeError, PlottingError
from .graph_elements import Point
from .inherit import INHERIT, Inherit, Styled, is_inherit, strip_inherit

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

_StandardDeviation = TypeVar("_StandardDeviation", default=np.ndarray)


def _run_curve_fit(func, x_data, y_data, **kwargs):
    """
    Runs ``scipy.optimize.curve_fit`` and wraps any failure with GraphingLib context.

    A non-converging fit otherwise surfaces as a bare scipy ``RuntimeError`` with no
    indication that it came from a GraphingLib fit.
    """
    try:
        return curve_fit(func, x_data, y_data, **kwargs)
    except Exception as exc:
        raise PlottingError(
            f"The curve fit did not succeed ({exc}). Try adjusting the initial guesses "
            "or increasing the maximum number of iterations."
        ) from exc


def _repair_positive_guess(
    guesses: ArrayLike, index: int, number_of_parameters: int
) -> np.ndarray:
    """
    Copies the initial guesses and forces the guess of a bounded parameter to be strictly positive.
    """
    guesses = np.array(guesses, dtype=float)
    if guesses.shape != (number_of_parameters,):
        raise InvalidParameterError(
            f"Expected {number_of_parameters} initial guesses, "
            f"but got an array of shape {guesses.shape}."
        )
    guesses[index] = max(abs(guesses[index]), 1e-10)
    return guesses


class GeneralFit(Curve, Generic[_StandardDeviation]):
    """
    Dummy class for curve fits. Defines the interface for all curve fits.

    .. attention:: Not to be used directly.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
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
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> None:
        """
        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
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
        self._curve_to_be_fit = curve_to_be_fit
        self._color = color
        self._line_width = line_width
        self._set_generated_label(label, prefix="$f(x) = $")
        self._line_style = line_style
        self._alpha = alpha

        self._function: Callable[[float | np.ndarray], float | np.ndarray]
        self._parameters: np.ndarray
        self._cov_matrix: np.ndarray
        self._standard_deviation: _StandardDeviation

        self._setup_attributes()

    def _setup_attributes(self) -> None:
        self._res_curves_to_be_plotted = False
        self._res_sigma_multiplier = None
        self._res_color = None
        self._res_line_width = None
        self._res_line_style = None

        self._show_errorbars: bool = False
        self._errorbars_color: Styled[str | None] = None
        self._errorbars_line_width: Styled[float | None] = None
        self._cap_thickness: Styled[float | None] = None
        self._cap_width: Styled[float | None] = None

        self._show_error_curves: bool = False
        self._error_curves_fill_between: bool = False
        self._error_curves_color = None
        self._error_curves_line_style = None
        self._error_curves_line_width = None

        self._fill_between_bounds: Optional[tuple[float, float]] = None
        self._fill_between_other_curve = None
        self._fill_between_color: Optional[str] = None

    def _set_generated_label(self, label: str | None, prefix: str = "") -> None:
        self._fit_label_base = label
        self._fit_label_prefix = prefix
        self._label = self._generated_label()

    def _generated_label(self) -> str:
        lead = f"{self._fit_label_base} : " if self._fit_label_base else ""
        return lead + self._fit_label_prefix + str(self)

    def _calculate_parameters(self) -> None:
        raise NotImplementedError

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        raise NotImplementedError

    def _solve(self) -> None:
        self._calculate_parameters()
        self._function = self._make_fitted_function()
        x = self._curve_to_be_fit.x_data
        self._x_data = np.linspace(np.min(x), np.max(x), max(len(x), 500))
        self._y_data = np.broadcast_to(
            self._function(self._x_data), self._x_data.shape
        ).copy()

    def refit(self, source: Curve | Scatter | None = None) -> Self:
        """Fit a new source or refresh the fit from its current source.

        The original model and solver options are reused. In an animation callback,
        assign the new samples to the source first, then call ``fit.refit()``.
        A refit updates the sampled line, residual curves, and fills on existing
        artists. A failed solve preserves the previous fit. The fit's equation
        label is refreshed unless it has been overridden.
        """
        source = self._curve_to_be_fit if source is None else source
        if not isinstance(source, (Curve, Scatter)):
            raise InvalidParameterTypeError("A fit source must be a Curve or Scatter.")
        if source is self:
            raise InvalidParameterError("A fit cannot use itself as the source.")
        x, y = np.asarray(source.x_data), np.asarray(source.y_data)
        if (
            x.ndim != 1
            or y.ndim != 1
            or not x.size
            or x.size != y.size
            or x.dtype.kind not in "iuf"
            or y.dtype.kind not in "iuf"
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise InvalidParameterError(
                "Refit samples must be finite real arrays of the same nonzero length."
            )

        previous = vars(self).copy()

        def restore() -> None:
            vars(self).clear()
            vars(self).update(previous)

        update_label = self._label == self._generated_label()
        try:
            self._curve_to_be_fit = source
            try:
                self._solve()
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                raise PlottingError(
                    f"The curve refit did not succeed ({exc})."
                ) from exc
            if (
                self._y_data.dtype.kind not in "iuf"
                or not np.isfinite(self._y_data).all()
            ):
                raise PlottingError("The refit must produce finite real values.")
            if update_label:
                self._label = self._generated_label()

            session = _bound_session(self)
            if session is not None:
                session.queue_points(
                    self,
                    self._x_data,
                    self._y_data,
                    None,
                    None,
                    show_errorbars=False,
                    restore=restore,
                )
        except BaseException:
            restore()
            raise
        return self

    def set_data(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        *,
        x_error: ArrayLike | None | _KeepError = _KEEP_ERROR,
        y_error: ArrayLike | None | _KeepError = _KEEP_ERROR,
    ) -> Self:
        """Replace the sampled coordinates outside animation."""
        _reject_live_update(
            self, "Call refit() to update a fitted curve during animation."
        )
        return super().set_data(x_data, y_data, x_error=x_error, y_error=y_error)

    @property
    def curve_to_be_fit(self) -> Curve | Scatter:
        return self._curve_to_be_fit

    @curve_to_be_fit.setter
    def curve_to_be_fit(self, curve: Curve | Scatter) -> None:
        _reject_live_update(
            self, "Call refit(source) to change an animated fit's source."
        )
        self._curve_to_be_fit = curve

    @property
    def function(self) -> Callable[[float | np.ndarray], float | np.ndarray]:
        return self._function

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def standard_deviation(self) -> _StandardDeviation:
        return self._standard_deviation

    def __str__(self) -> str:
        """
        Create a string representation of the fit function.
        """
        raise NotImplementedError()

    def _evaluate_scalar(self, x: float) -> float:
        value = self._function(x)
        return float(np.asarray(value).flat[0])

    def get_coordinates_at_x(
        self, x: float, interpolation_method: str = "linear"
    ) -> tuple[float, float]:
        return (x, self._evaluate_scalar(x))

    def create_point_at_x(
        self,
        x: float,
        interpolation_method: str = "linear",
        label: str | None = None,
        face_color: str | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> Point:
        """
        Gets the point on the curve at a given x value.

        Parameters
        ----------
        x : float
            x value of the point.
        interpolation_method : str
            Interpolation method parameter.
            Since fit curves are analytic, this value is ignored.
            Default is ``"linear"``.
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
        marker_style : str
            Style of the point.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the edge of the point.
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
        :class:`~graphinglib.graph_elements.Point` object on the curve at the given x value.
        """
        return Point(
            x,
            self._evaluate_scalar(x),
            label=label,
            face_color=face_color,
            edge_color=edge_color,
            marker_size=marker_size,
            marker_style=marker_style,
            edge_width=line_width,
            alpha=alpha,
        )

    def get_coordinates_at_y(
        self, y: float, interpolation_method: str = "linear"
    ) -> list[tuple[float, float]]:
        return super().get_coordinates_at_y(y, interpolation_method)

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
        Creates the Points on the curve at a given y value.

        Parameters
        ----------
        y : float
            y value of the point.
        interpolation_method : str
            Kind of interpolation to be used.
            Default is "linear".
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
        marker_style : str
            Style of the point.
            Common values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``,
            ``"p"``, ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, ``"_"``,
            ``"P"``, ``"X"``, ``"None"``, ``" "``, and ``""``.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Width of the edge of the point.
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
            List of :class:`~graphinglib.graph_elements.Point` objects on the curve at the given y value.
        """
        coord_pairs = self.get_coordinates_at_y(y, interpolation_method)
        points = [
            Point(
                coord[0],
                coord[1],
                label=label,
                face_color=face_color,
                edge_color=edge_color,
                marker_size=marker_size,
                marker_style=marker_style,
                edge_width=line_width,
                alpha=alpha,
            )
            for coord in coord_pairs
        ]
        return points

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        Axes
        """
        params = {
            "color": self._color,
            "linewidth": self._line_width,
            "linestyle": self._line_style,
            "alpha": self._alpha,
        }
        params = strip_inherit(params)
        (self.handle,) = axes.plot(
            self._x_data,
            self._y_data,
            label=self._label,
            zorder=z_order,
            **params,
        )
        lower = upper = area_fill = None
        error = self._residual_error()
        if error is not None:
            params = {
                "color": self._res_color,
                "linewidth": self._res_line_width,
                "linestyle": self._res_line_style,
                "alpha": self._alpha,
            }
            params = strip_inherit(params)
            (lower,) = axes.plot(
                self._x_data,
                self._y_data - error,
                zorder=z_order,
                **params,
            )
            (upper,) = axes.plot(
                self._x_data,
                self._y_data + error,
                zorder=z_order,
                **params,
            )
        bounds, other = self._fill_between_bounds, self._fill_between_other_curve
        if bounds is not None:
            params = strip_inherit(
                {
                    "alpha": 0.2,
                    "color": (
                        self.handle.get_color()
                        if self._fill_between_color in (None, "same as curve")
                        else self._fill_between_color
                    ),
                }
            )
            if other is None:
                x, y1, y2 = self._x_data, self._y_data, 0
            else:
                x, y1, y2 = self._shared_fill_data(other, bounds)
            area_fill = axes.fill_between(
                x,
                y1,
                y2,
                where=(x >= bounds[0]) & (x <= bounds[1]),
                zorder=z_order - 2,
                **params,
            )

        session = _DRAW_SESSION.get()
        if session is not None:
            line = self.handle
            session.bind_curve(self, line)
            session.bind_legend_source(
                self, line, lambda: self.label, ("_label", "point_data")
            )
            for name, mpl_name in (
                ("_color", "color"),
                ("_line_style", "linestyle"),
                ("_line_width", "linewidth"),
                ("_alpha", "alpha"),
            ):
                session.bind_property(self, name, line, mpl_name)

            if lower is None or upper is None:
                residual_style: dict[str, Any] = {
                    "color": line.get_color(),
                    "linewidth": line.get_linewidth(),
                    "linestyle": "--",
                    "alpha": line.get_alpha(),
                    "zorder": z_order,
                    "visible": False,
                }
                lower, upper = (Line2D([], [], **residual_style) for _ in range(2))
            self._bind_live_residuals(session, axes, line, lower, upper)

            if area_fill is None:
                area_fill = PolyCollection(
                    [],
                    facecolor=line.get_color(),
                    alpha=0.2,
                    zorder=z_order - 2,
                    visible=False,
                )
            self._bind_live_area(session, axes, area_fill)
            self._bind_curve_decoration_style(
                session,
                line,
                (area_fill,),
                "_fill_between_color",
                "facecolor",
                "_color",
            )

    def _bind_live_residuals(
        self,
        session: _AnimationSession,
        axes: plt.Axes,
        line: Line2D,
        lower: Line2D,
        upper: Line2D,
    ) -> None:
        artists = (lower, upper)
        bands = _CurveBands(lower, upper, None)

        def prepare() -> _ArtistEdit:
            error = self._residual_error()
            visible = error is not None
            edits = []
            if visible:
                if lower.axes is None:
                    edits.append(_attach_artists(axes, artists))
                edits.extend(bands.edits(self._x_data, self._y_data, error))
            edits.extend(
                _edit(artist.set_visible, artist.get_visible(), visible)
                for artist in artists
            )
            return _combine_edits(edits)

        session.bind_derived(
            self,
            ("point_data", "_res_curves_to_be_plotted", "_res_sigma_multiplier"),
            prepare,
        )
        for name, field, source in (
            ("_res_color", "color", "_color"),
            ("_res_line_width", "linewidth", "_line_width"),
            ("_res_line_style", "linestyle", "_line_style"),
        ):
            self._bind_curve_decoration_style(
                session, line, artists, name, field, source
            )
        for artist in artists:
            session.bind_property(self, "_alpha", artist, "alpha")

    def get_area_between(
        self,
        x1: float,
        x2: float,
        fill_between: bool = False,
        fill_color: str | Inherit = INHERIT,
        other_curve: Curve | None = None,
    ) -> float:
        """Integrate or shade a fitted curve, using its line color by default."""
        return super().get_area_between(
            x1,
            x2,
            fill_between,
            "same as curve" if is_inherit(fill_color) else fill_color,
            other_curve=other_curve,
        )

    def _residual_error(self) -> np.ndarray | None:
        if not self._res_curves_to_be_plotted:
            return None
        return np.asarray(self._res_sigma_multiplier * np.std(self.get_residuals()))

    def show_residual_curves(
        self,
        sigma_multiplier: float = 1,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
    ) -> None:
        """
        Displays two curves ``"sigma_multiplier"`` standard deviations above and below the fit curve.

        This method can also show or restyle the residual curves during
        animation. Omitted style arguments keep their current settings in a frame.

        Parameters
        ----------
        sigma_multiplier : float
            Distance in standard deviations from the fit curve.
            Default is 1.
        color : str
            Color of the residual curves.
            Default depends on the ``figure_style`` configuration.
        line_width : float
            Line width of the residual curves.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the residual curves.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        if _bound_session(self) is not None:
            if (
                not isinstance(sigma_multiplier, Real)
                or not isfinite(sigma_multiplier)
                or sigma_multiplier < 0
            ):
                raise InvalidParameterError(
                    "The animated residual sigma multiplier must be finite and nonnegative."
                )
            self._res_sigma_multiplier = _stage_live_value(
                self, "_res_sigma_multiplier", sigma_multiplier
            )
            for name, value in (
                ("_res_color", color),
                ("_res_line_width", line_width),
                ("_res_line_style", line_style),
            ):
                if not is_inherit(value):
                    setattr(self, name, _stage_live_value(self, name, value))
            self._res_curves_to_be_plotted = _stage_live_value(
                self, "_res_curves_to_be_plotted", True
            )
            return
        self._res_curves_to_be_plotted = True
        self._res_sigma_multiplier = sigma_multiplier
        self._res_color = color
        self._res_line_width = line_width
        self._res_line_style = line_style

    def hide_residual_curves(self) -> None:
        """Hide both residual curves, including during animation."""
        self._res_curves_to_be_plotted = _stage_live_value(
            self, "_res_curves_to_be_plotted", False
        )

    def get_residuals(self) -> np.ndarray:
        """
        Calculates the residuals of the fit curve.

        Returns
        -------
        residuals : np.ndarray
            Array of residuals.
        """
        y_data = self._function(self._curve_to_be_fit._x_data)
        residuals = y_data - self._curve_to_be_fit._y_data
        return residuals

    def get_Rsquared(self) -> float:
        """
        Calculates the :math:`R^2` value of the fit curve.

        Returns
        -------
        Rsquared : float
            :math:`R^2` value
        """
        y_data = self._curve_to_be_fit._y_data
        total_variance = np.sum((y_data - np.mean(y_data)) ** 2)
        if total_variance == 0:
            # Scale the tolerance to the data's own magnitude, since comparing residuals
            # to an absolute tolerance of 0 makes np.allclose's rtol term vanish. Only
            # fall back to an absolute tolerance when the data is identically zero.
            magnitude = np.max(np.abs(y_data))
            scale = magnitude if magnitude > 0 else 1.0
            is_exact_fit = np.allclose(self.get_residuals(), 0, atol=1e-8 * scale)
            return 1.0 if is_exact_fit else float("nan")
        return 1 - np.sum(self.get_residuals() ** 2) / total_variance

    def copy(self) -> Self:
        return deepcopy(self)


class FitFromPolynomial(GeneralFit):
    """
    Creates a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing curve object using a polynomial fit.

    Fits a polynomial of the form :math:`f(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n` to the given curve. All standard Curve attributes
    and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    degree : int
        Degree of the polynomial fit.
    label : str, optional
        Label to be displayed in the legend.
    color : str
        Color of the :class:`~graphinglib.data_plotting_1d.Curve`.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the :class:`~graphinglib.data_plotting_1d.Curve`.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the :class:`~graphinglib.data_plotting_1d.Curve`.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the :class:`~graphinglib.data_plotting_1d.Curve`.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

    Attributes
    ----------
    coeffs : np.ndarray
        Coefficients of the polynomial fit. The first element is the coefficient of the lowest order term (constant term).
    cov_matrix : np.ndarray
        Covariance matrix of the polynomial fit (using the same order as the coeffs attribute).
    standard_deviation : np.ndarray
        Standard deviation of the coefficients of the polynomial fit (same order as coeffs).
    function : Callable
        Polynomial function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        degree: int,
        label: Optional[str] = None,
        color: str | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> None:
        """
        Creates a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing curve object using a polynomial fit.

        Fits a polynomial of the form :math:`f(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n` to the given curve. All standard Curve attributes
        and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        degree : int
            Degree of the polynomial fit.
        label : str, optional
            Label to be displayed in the legend.
        color : str
            Color of the :class:`~graphinglib.data_plotting_1d.Curve`.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the :class:`~graphinglib.data_plotting_1d.Curve`.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the :class:`~graphinglib.data_plotting_1d.Curve`.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the :class:`~graphinglib.data_plotting_1d.Curve`.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Attributes
        ----------
        coeffs : np.ndarray
            Coefficients of the polynomial fit. The first element is the coefficient of the lowest order term (constant term).
        cov_matrix : np.ndarray
            Covariance matrix of the polynomial fit (using the same order as the coeffs attribute).
        standard_deviation : np.ndarray
            Standard deviation of the coefficients of the polynomial fit (same order as coeffs).
        function : Callable
            Polynomial function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._degree = degree
        self._solve()
        self._color = color
        self._line_width = line_width
        self._set_generated_label(label, prefix="$f(x) = $")
        self._line_style = line_style
        self._alpha = alpha
        self._setup_attributes()

    def _calculate_parameters(self) -> None:
        inversed_coeffs, inversed_cov_matrix = np.polyfit(
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            self._degree,
            cov=True,
        )
        self._coeffs = inversed_coeffs[::-1]
        self._cov_matrix = np.flip(inversed_cov_matrix)
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

    @property
    def coeffs(self) -> np.ndarray:
        return self._coeffs

    @property
    def parameters(self) -> np.ndarray:
        return self._coeffs

    def __str__(self) -> str:
        """
        Creates a string representation of the polynomial function.
        """
        coeff_chunks = []
        power_chunks = []
        ordered_rounded_coeffs = [round(coeff, 3) for coeff in self._coeffs[::-1]]
        for coeff, power in zip(
            ordered_rounded_coeffs, range(len(ordered_rounded_coeffs) - 1, -1, -1)
        ):
            if coeff == 0:
                continue
            coeff_chunks.append(self._format_coeff(coeff))
            power_chunks.append(self._format_power(power))
        if len(coeff_chunks) == 0:
            return "$0$"
        coeff_chunks[0] = coeff_chunks[0].lstrip("+ ")
        return (
            "$"
            + "".join(
                [coeff_chunks[i] + power_chunks[i] for i in range(len(coeff_chunks))]
            )
            + "$"
        )

    @staticmethod
    def _format_coeff(coeff: float) -> str:
        """
        Formats a coefficient to be displayed in the string representation of the polynomial function.
        """
        return " - {0}".format(abs(coeff)) if coeff < 0 else " + {0}".format(coeff)

    @staticmethod
    def _format_power(power: int) -> str:
        """
        Formats a power to be displayed in the string representation of the polynomial function.
        """
        return "x^{0}".format(power) if power != 0 else ""

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a polynomial function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Polynomial function with the parameters of the fit.
        """
        coefficients = tuple(self._coeffs)
        return lambda x: sum(
            coeff * x**exponent for exponent, coeff in enumerate(coefficients)
        )


class FitFromSine(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using a sinusoidal fit.

    Fits a sine function of the form :math:`f(x) = a sin(bx + c) + d` to the given curve. All standard
    :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit (order: amplitude (a), frequency (b), phase (c), vertical shift (d) as written above).
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the curve.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    max_iterations : int
        Maximum number of iterations for the fit.
        Default is 10000.


    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

    Attributes
    ----------
    amplitude : float
        Amplitude of the sine function.
    frequency_rad : float
        Frequency of the sine function in radians.
    phase : float
        Phase of the sine function.
    vertical_shift : float
        Vertical shift of the sine function.
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Sine function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        max_iterations: int = 10000,
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using a sinusoidal fit.

        Fits a sine function of the form :math:`f(x) = a sin(bx + c) + d` to the given curve. All standard
        :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit (order: amplitude (a), frequency (b), phase (c), vertical shift (d) as written above).
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        max_iterations : int
            Maximum number of iterations for the fit.
            Default is 10000.


        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Attributes
        ----------
        amplitude : float
            Amplitude of the sine function.
        frequency_rad : float
            Frequency of the sine function in radians.
        frequency_deg : float
            Frequency of the sine function in degrees.
        phase_rad : float
            Phase of the sine function.
        phase_deg : float
            Phase of the sine function in degrees.
        vertical_shift : float
            Vertical shift of the sine function.
        parameters : np.ndarray
            Parameters of the fit (amplitude, frequency (rad), phase (rad), vertical shift)
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Sine function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._max_iterations = max_iterations
        self._solve()
        self._color = color
        self._set_generated_label(label, prefix="$f(x) = $")
        self._line_width = line_width
        self._line_style = line_style
        self._alpha = alpha
        self._setup_attributes()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @property
    def frequency_rad(self) -> float:
        return self._frequency_rad

    @property
    def frequency_deg(self) -> float:
        return self._frequency_deg

    @property
    def phase_rad(self) -> float:
        return self._phase_rad

    @property
    def phase_deg(self) -> float:
        return self._phase_deg

    @property
    def vertical_shift(self) -> float:
        return self._vertical_shift

    def __str__(self) -> str:
        """
        Creates a string representation of the sine function.
        """
        part1 = rf"{self._amplitude:.3f} \sin({self._frequency_rad:.3f}x"
        part2 = (
            f" + {self._phase_rad:.3f})"
            if self._phase_rad >= 0
            else f" - {abs(self._phase_rad):.3f})"
        )
        part3 = (
            f" + {self._vertical_shift:.3f}"
            if self._vertical_shift >= 0
            else f" - {abs(self._vertical_shift):.3f}"
        )
        return f"${part1 + part2 + part3}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = _run_curve_fit(
            self._sine_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
            maxfev=self._max_iterations,
        )
        self._amplitude, self._frequency_rad, self._phase_rad, self._vertical_shift = (
            self._parameters
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

        # Calculate the frequency and phase in degrees
        self._frequency_deg = np.degrees(self._frequency_rad)
        self._phase_deg = np.degrees(self._phase_rad)

    @staticmethod
    def _sine_func_template(
        x: float | np.ndarray, a: float, b: float, c: float, d: float
    ) -> float | np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return a * np.sin(b * x + c) + d

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a sine function with the parameters of the fit.

        Returns
        -------
        Callable
            Sine function with the parameters of the fit.
        """
        template = self._sine_func_template
        parameters = tuple(self._parameters)
        return lambda x: template(x, *parameters)


class FitFromExponential(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using an exponential fit.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is a, b, c as written above.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the curve.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    max_iterations : int
        Maximum number of iterations for the fit.
        Default is 10000.


    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

    Attributes
    ----------
    parameters : np.ndarray
        Parameters of the fit (same order as guesses).
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Exponential function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        max_iterations: int = 10000,
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`)
        of the form :math:`f(x) = a \\exp(bx + c)` from an existing :class:`~graphinglib.data_plotting_1d.Curve`
        object using an exponential fit.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is a, b, c as written above.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        max_iterations : int
            Maximum number of iterations for the fit.
            Default is 10000.


        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Attributes
        ----------
        parameters : np.ndarray
            Parameters of the fit (same order as guesses).
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Exponential function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._max_iterations = max_iterations
        self._solve()
        self._color = color
        self._set_generated_label(label, prefix="$f(x) = $")
        self._line_width = line_width
        self._line_style = line_style
        self._alpha = alpha
        self._setup_attributes()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def __str__(self) -> str:
        """
        Creates a string representation of the exponential function.
        """
        part1 = rf"{self._parameters[0]:.3f} \exp({self._parameters[1]:.3f}x"
        part2 = (
            f" + {self._parameters[2]:.3f})"
            if self._parameters[2] >= 0
            else f" - {abs(self._parameters[2]):.3f})"
        )
        return f"${part1 + part2}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = _run_curve_fit(
            self._exp_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
            maxfev=self._max_iterations,
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

    @staticmethod
    def _exp_func_template(
        x: float | np.ndarray, a: float, b: float, c: float
    ) -> float | np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return a * np.exp(b * x + c)

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates an exponential function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Exponential function with the parameters of the fit.
        """
        template = self._exp_func_template
        parameters = tuple(self._parameters)
        return lambda x: template(x, *parameters)


class FitFromGaussian(GeneralFit[float | np.floating]):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using a gaussian fit.

    Fits a gaussian function of the form :math:`f(x) = A e^{-\\frac{(x - \\mu)^2}{2 \\sigma^2}}` to the given curve.
    All standard :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is amplitude (A), mean (mu), standard deviation (sigma).
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the curve.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    max_iterations : int
        Maximum number of iterations for the fit.
        Default is 10000.


    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

    Attributes
    ----------
    amplitude : float
        Amplitude of the gaussian function.
    mean : float
        Mean of the gaussian function.
    standard_deviation : float
        Standard deviation of the gaussian function.

        .. warning::

            The ``standard_deviation`` attribute doesn't represent the standard deviation of the fit parameters as it does in the other fit classes. Instead, it represents the standard deviation of the gaussian function (it is one of parameters of the fit). The standard deviation of the fit parameters can be found in the ``standard_deviation_of_fit_params`` attribute.

    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation_of_fit_params : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Gaussian function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        max_iterations: int = 10000,
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using a gaussian fit.

        Fits a gaussian function of the form :math:`f(x) = A e^{-\\frac{(x - \\mu)^2}{2 \\sigma^2}}` to the given curve.
        All standard :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        max_iterations : int
            Maximum number of iterations for the fit.
            Default is 10000.


        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Attributes
        ----------
        amplitude : float
            Amplitude of the gaussian function.
        mean : float
            Mean of the gaussian function.
        standard_deviation : float
            Standard deviation of the gaussian function.
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation_of_fit_params : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Gaussian function with the parameters of the fit.

        Warning
        -------
        The ``standard_deviation`` attribute doesn't represent the standard deviation of the fit parameters as it does in the other fit classes. Instead, it represents the standard deviation of the gaussian function (it is one of parameters of the fit). The standard deviation of the fit parameters can be found in the ``standard_deviation_of_fit_params`` attribute.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._max_iterations = max_iterations
        self._solve()
        self._color = color
        self._set_generated_label(label)
        self._line_width = line_width
        self._line_style = line_style
        self._alpha = alpha
        self._setup_attributes()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def standard_deviation(self) -> float:
        # Unlike the other fit classes, this is the fitted sigma of the gaussian itself,
        # not the standard deviation of the fit parameters (see standard_deviation_of_fit_params).
        return float(self._standard_deviation)

    @property
    def standard_deviation_of_fit_params(self) -> np.ndarray:
        return self._standard_deviation_of_fit_params

    def __str__(self) -> str:
        """
        Creates a string representation of the gaussian function.
        """
        return rf"$\mu = {self._mean:.3f}, \sigma = {self._standard_deviation:.3f}, A = {self._amplitude:.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        guesses = self._guesses
        if guesses is not None:
            # The standard deviation guess must be positive to satisfy the fit's bounds,
            # regardless of the sign the caller happened to guess.
            guesses = _repair_positive_guess(guesses, index=2, number_of_parameters=3)
        self._parameters, self._cov_matrix = _run_curve_fit(
            self._gaussian_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=guesses,
            maxfev=self._max_iterations,
            bounds=([-np.inf, -np.inf, 1e-10], [np.inf, np.inf, np.inf]),
        )
        self._amplitude = self._parameters[0]
        self._mean = self._parameters[1]
        self._standard_deviation = self._parameters[2]
        self._standard_deviation_of_fit_params = np.sqrt(np.diag(self._cov_matrix))

    @staticmethod
    def _gaussian_func_template(
        x: float | np.ndarray, amplitude: float, mean: float, standard_deviation: float
    ) -> float | np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return amplitude * np.exp(-(((x - mean) / standard_deviation) ** 2) / 2)

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a gaussian function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Gaussian function with the parameters of the fit.
        """
        template = self._gaussian_func_template
        parameters = tuple(self._parameters)
        return lambda x: template(x, *parameters)


class FitFromSquareRoot(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using a square root fit.

    Fits a square root function of the form :math:`f(x) = a \\sqrt{x + b} + c` to the given curve. All standard
    :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is a, b, c as written above.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the curve.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    max_iterations : int
        Maximum number of iterations for the fit.
        Default is 10000.


    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

    Attributes
    ----------
    parameters : np.ndarray
        Parameters of the fit (same order as guesses).
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Square root function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        max_iterations: int = 10000,
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using a square root fit.

        Fits a square root function of the form :math:`f(x) = a \\sqrt{x + b} + c` to the given curve. All standard
        :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is a, b, c as written above.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        max_iterations : int
            Maximum number of iterations for the fit.
            Default is 10000.


        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Attributes
        ----------
        parameters : np.ndarray
            Parameters of the fit (same order as guesses).
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Square root function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._max_iterations = max_iterations
        self._solve()
        self._color = color
        self._set_generated_label(label)
        self._line_width = line_width
        self._line_style = line_style
        self._alpha = alpha
        self._setup_attributes()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def __str__(self) -> str:
        """
        Creates a string representation of the square root function.
        """
        return rf"${self._parameters[0]:.3f} \sqrt{{x {'+' if self._parameters[1] > 0 else '-'} {abs(self._parameters[1]):.3f}}} {'+' if self._parameters[2] > 0 else '-'} {abs(self._parameters[2]):.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = _run_curve_fit(
            self._square_root_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
            maxfev=self._max_iterations,
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

    @staticmethod
    def _square_root_func_template(
        x: float | np.ndarray, a: float, b: float, c: float
    ) -> float | np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return a * np.sqrt(x + b) + c

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a square root function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Square root function with the parameters of the fit.
        """
        template = self._square_root_func_template
        parameters = tuple(self._parameters)
        return lambda x: template(x, *parameters)


class FitFromLog(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
    :class:`~graphinglib.data_plotting_1d.Curve` object using a logarithmic fit.

    Fits a logarithmic function of the form :math:`f(x) = a \\log_{base}(x + b) + c` to the given curve. All standard
    :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    log_base : float
        Base of the logarithm.
        Default is e.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is a, b, c as written above.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the curve.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    max_iterations : int
        Maximum number of iterations for the fit.
        Default is 10000.


    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

    Attributes
    ----------
    parameters : np.ndarray
        Parameters of the fit (same order as guesses).
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Logarithmic function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        log_base: float = np.e,
        guesses: Optional[ArrayLike] = None,
        color: str | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        max_iterations: int = 10000,
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing
        :class:`~graphinglib.data_plotting_1d.Curve` object using a logarithmic fit.

        Fits a logarithmic function of the form :math:`f(x) = a \\log_{base}(x + b) + c` to the given curve. All standard
        :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        log_base : float
            Base of the logarithm.
            Default is e.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is a, b, c as written above.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        max_iterations : int
            Maximum number of iterations for the fit.
            Default is 10000.


        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Attributes
        ----------
        parameters : np.ndarray
            Parameters of the fit (same order as guesses).
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Logarithmic function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._log_base = log_base
        self._guesses = guesses
        self._max_iterations = max_iterations
        self._solve()
        self._color = color
        self._set_generated_label(label)
        self._line_width = line_width
        self._line_style = line_style
        self._alpha = alpha
        self._setup_attributes()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def __str__(self) -> str:
        """
        Creates a string representation of the logarithmic function.
        """
        return f"${self._parameters[0]:.3f} log_{self._log_base if self._log_base != np.e else 'e'}(x {'-' if self._parameters[1] < 0 else '+'} {abs(self._parameters[1]):.3f}) {'-' if self._parameters[2] < 0 else '+'} {abs(self._parameters[2]):.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = _run_curve_fit(
            self._log_func_template(),
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
            maxfev=self._max_iterations,
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

    def _log_func_template(
        self,
    ) -> Callable[[float | np.ndarray, float, float, float], float | np.ndarray]:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        log_base = self._log_base
        return lambda x, a, b, c: a * (np.log(x + b) / np.log(log_base)) + c

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a logarithmic function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Logarithmic function with the parameters of the fit.
        """
        template = self._log_func_template()
        parameters = tuple(self._parameters)
        return lambda x: template(x, *parameters)


class FitFromFunction(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from a
    :class:`~graphinglib.data_plotting_1d.Curve` object using an arbitrary function passed as an argument.

    Fits a function of the form :math:`f(x, a, b, c, ...)` to the given curve. All standard
    :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    function : Callable
        Function to fit. The first argument must be the x values, followed by
        fit parameters (``f(x, a, b, c, ...)``), and it must return scalar or
        array-like y values.
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is a, b, c, ...
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the curve.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    max_iterations : int
        Maximum number of iterations for the fit.
        Default is 10000.


    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

    Attributes
    ----------
    parameters : np.ndarray
        Parameters of the fit (same order as guesses).
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        Fitted function with the parameters bound. Accepts scalar or array x values.
    """

    def __init__(
        self,
        function: Callable[..., float | np.ndarray],
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        max_iterations: int = 10000,
    ):
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from a
        :class:`~graphinglib.data_plotting_1d.Curve` object using an arbitrary function passed as an argument.

        Fits a function of the form :math:`f(x, a, b, c, ...)` to the given curve. All standard
        :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        function : Callable
            Function to fit. The first argument must be the x values, followed by
            fit parameters (``f(x, a, b, c, ...)``), and it must return scalar or
            array-like y values.
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is a, b, c, ... as written above.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        max_iterations : int
            Maximum number of iterations for the fit.
            Default is 10000.


        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Attributes
        ----------
        parameters : np.ndarray
            Parameters of the fit (same order as guesses).
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            Fitted function with the parameters bound. Accepts scalar or array x values.
        """
        self._function_template = function
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._color = color
        self._line_width = line_width
        self._line_style = line_style
        self._alpha = alpha
        self._max_iterations = max_iterations

        self._solve()
        self._set_generated_label(label)
        self._setup_attributes()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def __str__(self) -> str:
        """
        Creates a string representation of the fitted function.
        """
        function_name = getattr(
            self._function_template,
            "__name__",
            type(self._function_template).__name__,
        )
        return f"Fit from function: {function_name}"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        self._parameters, self._cov_matrix = _run_curve_fit(
            self._function_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=self._guesses,
            maxfev=self._max_iterations,
        )
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a function with the parameters of the fit.

        Returns
        -------
        function : Callable
            Fitted function with the parameters bound. Accepts scalar or array x values.
        """
        argument_names = list(signature(self._function_template).parameters)[1:]
        args_dict = {
            argument_names[i]: self._parameters[i] for i in range(len(argument_names))
        }
        return cast(
            Callable[[float | np.ndarray], float | np.ndarray],
            partial(self._function_template, **args_dict),
        )


class FitFromFOTF(GeneralFit):
    """
    Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing :class:`~graphinglib.data_plotting_1d.Curve` object using a first order transfer function (FOTF) fit.

    Fits a first order transfer function of the form :math:`f(x) = K\\left(1-e^{-\\frac{t}{\\tau}}\\right)` to the given curve. All standard :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

    Parameters
    ----------
    curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
        The object to be fit.
    label : str, optional
        Label to be displayed in the legend.
    guesses : ArrayLike, optional
        Initial guesses for the parameters of the fit. Order is K, tau.
    color : str
        Color of the curve.
        Default depends on the ``figure_style`` configuration.
    line_width : int
        Line width of the curve.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_style : str
        Line style of the curve.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the curve.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    max_iterations : int
        Maximum number of iterations for the fit.
        Default is 10000.


    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

    Attributes
    ----------
    gain : float
        Gain of the first order transfer function.
    time_constant : float
        Time constant of the first order transfer function.
    cov_matrix : np.ndarray
        Covariance matrix of the parameters of the fit.
    standard_deviation : np.ndarray
        Standard deviation of the parameters of the fit.
    function : Callable
        First order transfer function with the parameters of the fit.
    """

    def __init__(
        self,
        curve_to_be_fit: Curve | Scatter,
        label: Optional[str] = None,
        guesses: Optional[ArrayLike] = None,
        color: str | Inherit = INHERIT,
        line_width: int | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        max_iterations: int = 10000,
    ) -> None:
        """
        Create a curve fit (continuous :class:`~graphinglib.data_plotting_1d.Curve`) from an existing :class:`~graphinglib.data_plotting_1d.Curve` object using a first order transfer function (FOTF) fit.

        Fits a first order transfer function of the form :math:`f(x) = K \\left(1 - e^{-\\frac{t}{\\tau}}\\right)` to the given curve. All standard :class:`~graphinglib.data_plotting_1d.Curve` attributes and methods are available.

        Parameters
        ----------
        curve_to_be_fit : :class:`~graphinglib.data_plotting_1d.Curve` or :class:`~graphinglib.data_plotting_1d.Scatter`
            The object to be fit.
        label : str, optional
            Label to be displayed in the legend.
        guesses : ArrayLike, optional
            Initial guesses for the parameters of the fit. Order is K, tau.
        color : str
            Color of the curve.
            Default depends on the ``figure_style`` configuration.
        line_width : int
            Line width of the curve.
            Typical range is ``0.5`` to ``4``.
            Default depends on the ``figure_style`` configuration.
        line_style : str
            Line style of the curve.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the curve.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        max_iterations : int
            Maximum number of iterations for the fit.
            Default is 10000.


        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).

        Attributes
        ----------
        gain : float
            Gain of the first order transfer function.
        time_constant : float
            Time constant of the first order transfer function.
        cov_matrix : np.ndarray
            Covariance matrix of the parameters of the fit.
        standard_deviation : np.ndarray
            Standard deviation of the parameters of the fit.
        function : Callable
            First order transfer function with the parameters of the fit.
        """
        self._curve_to_be_fit = curve_to_be_fit
        self._guesses = guesses
        self._max_iterations = max_iterations
        self._solve()
        self._color = color
        self._set_generated_label(label)
        self._line_width = line_width
        self._line_style = line_style
        self._alpha = alpha
        self._setup_attributes()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def time_constant(self) -> float:
        return self._time_constant

    def __str__(self) -> str:
        """
        Creates a string representation of the first order transfer function.
        """
        return rf"$K = {self._gain:.3f}, \tau = {self._time_constant:.3f}$"

    def _calculate_parameters(self) -> None:
        """
        Calculates the parameters of the fit.
        """
        guesses = self._guesses
        if guesses is not None:
            # The time constant guess must be positive to satisfy the fit's bounds,
            # regardless of the sign the caller happened to guess.
            guesses = _repair_positive_guess(guesses, index=1, number_of_parameters=2)
        self._parameters, self._cov_matrix = _run_curve_fit(
            self._fotf_func_template,
            self._curve_to_be_fit._x_data,
            self._curve_to_be_fit._y_data,
            p0=guesses,
            maxfev=self._max_iterations,
            bounds=([-np.inf, 1e-10], [np.inf, np.inf]),
        )
        self._gain = self._parameters[0]
        self._time_constant = self._parameters[1]
        self._standard_deviation = np.sqrt(np.diag(self._cov_matrix))

    @staticmethod
    def _fotf_func_template(
        x: float | np.ndarray, gain: float, time_constant: float
    ) -> float | np.ndarray:
        """
        Function to be passed to the ``curve_fit`` function.
        """
        return gain * (1 - np.exp(-x / time_constant))

    def _make_fitted_function(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """
        Creates a first order transfer function with the parameters of the fit.

        Returns
        -------
        function : Callable
            First order transfer function with the parameters of the fit.
        """
        template = self._fotf_func_template
        parameters = tuple(self._parameters)
        return lambda x: template(x, *parameters)
