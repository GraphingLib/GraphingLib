from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any, Literal, Optional, Protocol, Sequence, cast, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure as MPLFigure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from numpy.typing import ArrayLike

from ._animation import (
    _DRAW_SESSION,
    _ArtistEdit,
    _bound_session,
    _edit,
    _reject_live_update,
    _stage_live_value,
)
from .exceptions import (
    IncompatibleArgumentsError,
    InvalidParameterError,
    InvalidParameterTypeError,
    PlottingError,
)
from .inherit import INHERIT, Inherit, Styled, is_inherit, resolve_or, strip_inherit
from .legend_artists import VerticalLineCollection
from .tools import _copy_with_overrides

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@runtime_checkable
class Plottable(Protocol):
    """
    Dummy class for a general plottable object.

    .. attention:: Not to be used directly.

    """

    def copy_with(self, **kwargs) -> Self:
        """
        Returns a deep copy of the Plottable with specified attributes overridden. This is useful when multiple
        properties need to be changed in copies of Plottable objects, as it allows to modify the attributes in a single
        call.

        Parameters
        ----------
        **kwargs
            Public writable properties to override in the copied Plottable. The keys should be property names to
            modify and the values are the new values for those properties.

        Returns
        -------
        Self
            A new instance with the specified attributes overridden.

        Examples
        --------
        Copy an existing Curve and change the color and line_style at the same time::

            curve = Curve(x_data, y_data, color='blue')
            new_curve = curve.copy_with(color='red', line_style='dashed')
        """
        return _copy_with_overrides(self, **kwargs)

    def __deepcopy__(self, memo: dict) -> Self:
        """
        Creates a deep copy of the Plottable instance, intentionally excluding the 'handle' attribute from the copy.
        This avoids issues when copying a Plottable that has been previously drawn and stored.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        excluded_attrs = ["handle"]
        for property_, value in self.__dict__.items():
            if property_ not in excluded_attrs:
                result.__dict__[property_] = deepcopy(value, memo)
        for attr in excluded_attrs:
            if hasattr(self, attr):
                setattr(result, attr, None)
        return result

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        pass


def _bind_reference_lines(
    element: Plottable,
    artists: list[Line2D | LineCollection],
    names: tuple[str, str, str],
    *,
    horizontal: bool,
) -> None:
    session = _DRAW_SESSION.get()
    if session is None:
        return
    position, start, end = names
    first = artists[0]
    if isinstance(first, LineCollection):
        old_paths = first.get_paths()
        count = len(old_paths)

        def prepare(value: Any) -> _ArtistEdit:
            before = [path.vertices for path in first.get_paths()]

            def apply() -> None:
                positions = np.atleast_1d(getattr(element, position))
                low, high = getattr(element, start), getattr(element, end)
                if low is None or high is None or len(positions) != count:
                    raise InvalidParameterError(
                        "An animated reference line collection must retain its count and both bounds."
                    )
                low, high = (
                    np.broadcast_to(low, positions.shape),
                    np.broadcast_to(high, positions.shape),
                )
                x, y = (
                    np.stack((low, high), axis=-1),
                    np.repeat(positions[:, None], 2, axis=1),
                )
                first.set_segments(
                    list(np.stack((x, y) if horizontal else (y, x), axis=-1))
                )

            return _ArtistEdit(apply, lambda: first.set_segments(before))

        for name in names:
            session.bind_update(element, name, prepare)
    else:
        lines = [artist for artist in artists if isinstance(artist, Line2D)]
        getters = [line.get_ydata if horizontal else line.get_xdata for line in lines]
        setters = [line.set_ydata if horizontal else line.set_xdata for line in lines]

        def prepare(value: Any) -> _ArtistEdit:
            positions = np.atleast_1d(value)
            if len(positions) != len(lines):
                raise InvalidParameterError(
                    "An animated reference line must retain its original count."
                )
            before = [get() for get in getters]

            def apply() -> None:
                for setter, coordinate in zip(setters, positions):
                    setter([coordinate, coordinate])

            def undo() -> None:
                for setter, original in zip(setters, before):
                    setter(original)

            return _ArtistEdit(apply, undo)

        session.bind_update(element, position, prepare)


class Hlines(Plottable):
    """
    This class implements simple horizontal lines.

    Parameters
    ----------
    y : ArrayLike
        Vertical positions at which the lines should be plotted.
    x_min : ArrayLike, optional
        Horizontal start position of the lines. Each lines can have a different start.
        If not specified, lines will span the entire axes. Defaults to ``None``.
    x_max : ArrayLike, optional
        Horizontal end position of the lines. Each lines can habe a different end.
        If not specified, lines will span the entire axes. Defaults to ``None``.
    label : str, optional
        Label to be displayed in the legend.
    colors : list[str]
        Colors to use for the lines. One color for every line or a color
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_widths : list[float]
        Line widths to use for the lines. One width for every line or a width
        per line can be specified.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_styles : list[str]
        Line styles to use for the lines. One style for every line or a style
        per line can be specified.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the lines.
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
        y: ArrayLike,
        x_min: Optional[ArrayLike] = None,
        x_max: Optional[ArrayLike] = None,
        label: Optional[str] = None,
        colors: list[str] | str | Inherit = INHERIT,
        line_widths: list[float] | float | Inherit = INHERIT,
        line_styles: list[str] | str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> None:
        self._in_init = True
        self._y: int | float | np.ndarray | None = None
        self._x_min: int | float | np.ndarray | None = None
        self._x_max: int | float | np.ndarray | None = None
        self._colors: Styled[list[str] | str] = INHERIT
        self._line_widths: Styled[list[float] | float] = INHERIT
        self._line_styles: Styled[list[str] | str] = INHERIT
        self.y = y
        self.x_min = x_min
        self.x_max = x_max
        self.label = label
        self.colors = colors
        self.line_widths = line_widths
        self.line_styles = line_styles
        self._alpha = alpha
        self._in_init = False
        self._validate_state()

    def _validate_state(self) -> None:
        if (self._x_min is None) ^ (self._x_max is None):
            raise IncompatibleArgumentsError(
                "x_min and x_max must both be provided, or neither."
            )

        if isinstance(self._y, (int, float)) and isinstance(
            self._colors, (list, np.ndarray)
        ):
            if len(self._colors) > 1:
                raise IncompatibleArgumentsError(
                    "colors must be a single value when there is only one line."
                )
        if isinstance(self._y, (int, float)) and isinstance(
            self._line_styles, (list, np.ndarray)
        ):
            if len(self._line_styles) > 1:
                raise IncompatibleArgumentsError(
                    "line_styles must be a single value when there is only one line."
                )
        if isinstance(self._y, (int, float)) and isinstance(
            self._line_widths, (list, np.ndarray)
        ):
            if len(self._line_widths) > 1:
                raise IncompatibleArgumentsError(
                    "line_widths must be a single value when there is only one line."
                )
        if isinstance(self._y, (list, np.ndarray)):
            if isinstance(self._colors, list) and len(self._y) != len(self._colors):
                raise IncompatibleArgumentsError(
                    "The number of colors must match the number of lines."
                )
            if isinstance(self._line_styles, list) and len(self._y) != len(
                self._line_styles
            ):
                raise IncompatibleArgumentsError(
                    "The number of line styles must match the number of lines."
                )
            if isinstance(self._line_widths, list) and len(self._y) != len(
                self._line_widths
            ):
                raise IncompatibleArgumentsError(
                    "The number of line widths must match the number of lines."
                )

    @property
    def y(self) -> int | float | np.ndarray:
        assert self._y is not None
        return self._y

    @y.setter
    def y(self, y: ArrayLike) -> None:
        value = y if isinstance(y, (int, float)) else np.asarray(y)
        self._y = _stage_live_value(self, "_y", value)
        if not self._in_init:
            self._validate_state()

    @property
    def x_min(self) -> int | float | np.ndarray | None:
        return self._x_min

    @x_min.setter
    def x_min(self, x_min: Optional[ArrayLike]) -> None:
        value = (
            x_min
            if x_min is None or isinstance(x_min, (int, float))
            else np.asarray(x_min)
        )
        self._x_min = _stage_live_value(self, "_x_min", value)
        if not self._in_init:
            self._validate_state()

    @property
    def x_max(self) -> int | float | np.ndarray | None:
        return self._x_max

    @x_max.setter
    def x_max(self, x_max: Optional[ArrayLike]) -> None:
        value = (
            x_max
            if x_max is None or isinstance(x_max, (int, float))
            else np.asarray(x_max)
        )
        self._x_max = _stage_live_value(self, "_x_max", value)
        if not self._in_init:
            self._validate_state()

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = _stage_live_value(self, "_label", label)

    @property
    def colors(self) -> Styled[list[str] | str]:
        return self._colors

    @colors.setter
    def colors(self, colors: Styled[list[str] | str]) -> None:
        _reject_live_update(
            self, "Configure reference-line labels and styles before animating."
        )
        self._colors = colors
        if not self._in_init:
            self._validate_state()

    @property
    def line_widths(self) -> Styled[list[float] | float]:
        return self._line_widths

    @line_widths.setter
    def line_widths(self, line_widths: Styled[list[float] | float]) -> None:
        _reject_live_update(
            self, "Configure reference-line labels and styles before animating."
        )
        self._line_widths = line_widths
        if not self._in_init:
            self._validate_state()

    @property
    def line_styles(self) -> Styled[list[str] | str]:
        return self._line_styles

    @line_styles.setter
    def line_styles(self, line_styles: Styled[list[str] | str]) -> None:
        _reject_live_update(
            self, "Configure reference-line labels and styles before animating."
        )
        self._line_styles = line_styles
        if not self._in_init:
            self._validate_state()

    @property
    def alpha(self) -> float | Inherit:
        return self._alpha

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Hlines` object.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        artists: list[Line2D | LineCollection] = []
        y = self._y
        assert y is not None
        if isinstance(y, np.ndarray) and len(y) > 1:
            if self._x_max is not None and self._x_min is not None:
                x_min = self._x_min
                x_max = self._x_max
                params = {
                    "colors": self._colors,
                    "linestyles": self._line_styles,
                    "linewidths": self._line_widths,
                    "alpha": self._alpha,
                }
                params = strip_inherit(params)
                artists.append(
                    axes.hlines(
                        y,
                        x_min,
                        x_max,
                        zorder=z_order,
                        **params,
                    )
                )
                params.pop("linewidths", None)
            else:
                params = {
                    "color": self._colors,
                    "linestyle": self._line_styles,
                    "linewidth": self._line_widths,
                    "alpha": self._alpha,
                }
                params = strip_inherit(params)
                for i in range(len(y)):
                    artists.append(
                        axes.axhline(
                            float(cast(Any, y)[i]),
                            zorder=z_order,
                            **{
                                k: v if isinstance(v, (int, float, str)) else v[i]
                                for k, v in params.items()
                            },
                        )
                    )
                params.pop("linewidth")
            self.handle = LineCollection(
                [[(0, 0)]] * (len(y) if len(y) <= 3 else 3),
                **params,
            )
        else:
            y_scalar = float(cast(Any, y)[0]) if isinstance(y, np.ndarray) else y
            if self._x_max is not None and self._x_min is not None:
                x_min = self._x_min
                x_max = self._x_max
                params = {
                    "colors": self._colors,
                    "linestyles": self._line_styles,
                    "linewidths": self._line_widths,
                    "alpha": self._alpha,
                }
                params = strip_inherit(params)
                artists.append(
                    axes.hlines(
                        y_scalar,
                        x_min,
                        x_max,
                        zorder=z_order,
                        **params,
                    )
                )
                params.pop("linewidths", None)
            else:
                params = {
                    "color": self._colors,
                    "linestyle": self._line_styles,
                    "linewidth": self._line_widths,
                    "alpha": self._alpha,
                }
                params = strip_inherit(params)
                artists.append(axes.axhline(y_scalar, zorder=z_order, **params))
                params.pop("linewidth")
            if isinstance(y, (int, float)):
                self.handle = LineCollection([[(0, 0)]] * 1, **params)
            else:
                self.handle = LineCollection(
                    [[(0, 0)]] * (len(y) if len(y) <= 3 else 3),
                    **params,
                )
        session = _DRAW_SESSION.get()
        if session is not None:
            session.bind_legend_source(self, self.handle, lambda: self.label)
        _bind_reference_lines(
            self, artists, ("_y", "_x_min", "_x_max"), horizontal=True
        )


class Vlines(Plottable):
    """
    This class implements simple vertical lines.

    Parameters
    ----------
    x : ArrayLike
        Horizontal positions at which the lines should be plotted.
    y_min : ArrayLike, optional
        Vertical start position of the lines. Each line can have a different start.
        If not specified, lines will span the entire axes. Defaults to ``None``.
    y_max : ArrayLike, optional
        Vertical end position of the lines. Each line can habe a different end.
        If not specified, lines will span the entire axes. Defaults to ``None``.
    label : str, optional
        Label to be displayed in the legend.
    colors : list[str]
        Colors to use for the lines. One color for every line or a color
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_widths : list[float]
        Line widths to use for the lines. One width for every line or a width
        per line can be specified.
        Typical range is ``0.5`` to ``4``.
        Default depends on the ``figure_style`` configuration.
    line_styles : list[str]
        Line styles to use for the lines. One style for every line or a style
        per line can be specified.
        Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
        ``"dotted"``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the lines.
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
        x: ArrayLike,
        y_min: Optional[ArrayLike] = None,
        y_max: Optional[ArrayLike] = None,
        label: Optional[str] = None,
        colors: list[str] | str | Inherit = INHERIT,
        line_widths: list[float] | float | Inherit = INHERIT,
        line_styles: list[str] | str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
    ) -> None:
        self._in_init = True
        self._x: int | float | np.ndarray | None = None
        self._y_min: int | float | np.ndarray | None = None
        self._y_max: int | float | np.ndarray | None = None
        self._colors: Styled[list[str] | str] = INHERIT
        self._line_styles: Styled[list[str] | str] = INHERIT
        self._line_widths: Styled[list[float] | float] = INHERIT
        self.x = x
        self.y_min = y_min
        self.y_max = y_max
        self.label = label
        self.colors = colors
        self.line_styles = line_styles
        self.line_widths = line_widths
        self._alpha = alpha
        self._in_init = False
        self._validate_state()

    def _validate_state(self) -> None:
        if isinstance(self._x, (int, float)) and isinstance(
            self._colors, (list, np.ndarray)
        ):
            if len(self._colors) > 1:
                raise IncompatibleArgumentsError(
                    "colors must be a single value when there is only one line."
                )
        if isinstance(self._x, (int, float)) and isinstance(
            self._line_styles, (list, np.ndarray)
        ):
            if len(self._line_styles) > 1:
                raise IncompatibleArgumentsError(
                    "line_styles must be a single value when there is only one line."
                )
        if isinstance(self._x, (int, float)) and isinstance(
            self._line_widths, (list, np.ndarray)
        ):
            if len(self._line_widths) > 1:
                raise IncompatibleArgumentsError(
                    "line_widths must be a single value when there is only one line."
                )
        if isinstance(self._x, (list, np.ndarray)):
            if isinstance(self._colors, list) and len(self._x) != len(self._colors):
                raise IncompatibleArgumentsError(
                    "The number of colors must match the number of lines."
                )
            if isinstance(self._line_styles, list) and len(self._x) != len(
                self._line_styles
            ):
                raise IncompatibleArgumentsError(
                    "The number of line styles must match the number of lines."
                )
            if isinstance(self._line_widths, list) and len(self._x) != len(
                self._line_widths
            ):
                raise IncompatibleArgumentsError(
                    "The number of line widths must match the number of lines."
                )

    @property
    def x(self) -> int | float | np.ndarray:
        assert self._x is not None
        return self._x

    @x.setter
    def x(self, x: ArrayLike) -> None:
        value = x if isinstance(x, (int, float)) else np.asarray(x)
        self._x = _stage_live_value(self, "_x", value)
        if not self._in_init:
            self._validate_state()

    @property
    def y_min(self) -> int | float | np.ndarray | None:
        return self._y_min

    @y_min.setter
    def y_min(self, y_min: Optional[ArrayLike]) -> None:
        value = (
            y_min
            if y_min is None or isinstance(y_min, (int, float))
            else np.asarray(y_min)
        )
        self._y_min = _stage_live_value(self, "_y_min", value)

    @property
    def y_max(self) -> int | float | np.ndarray | None:
        return self._y_max

    @y_max.setter
    def y_max(self, y_max: Optional[ArrayLike]) -> None:
        value = (
            y_max
            if y_max is None or isinstance(y_max, (int, float))
            else np.asarray(y_max)
        )
        self._y_max = _stage_live_value(self, "_y_max", value)

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = _stage_live_value(self, "_label", label)

    @property
    def colors(self) -> Styled[list[str] | str]:
        return self._colors

    @colors.setter
    def colors(self, colors: Styled[list[str] | str]) -> None:
        _reject_live_update(
            self, "Configure reference-line labels and styles before animating."
        )
        self._colors = colors
        if not self._in_init:
            self._validate_state()

    @property
    def line_widths(self) -> Styled[list[float] | float]:
        return self._line_widths

    @line_widths.setter
    def line_widths(self, line_widths: Styled[list[float] | float]) -> None:
        _reject_live_update(
            self, "Configure reference-line labels and styles before animating."
        )
        self._line_widths = line_widths
        if not self._in_init:
            self._validate_state()

    @property
    def line_styles(self) -> Styled[list[str] | str]:
        return self._line_styles

    @line_styles.setter
    def line_styles(self, line_styles: Styled[list[str] | str]) -> None:
        _reject_live_update(
            self, "Configure reference-line labels and styles before animating."
        )
        self._line_styles = line_styles
        if not self._in_init:
            self._validate_state()

    @property
    def alpha(self) -> float | Inherit:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Inherit) -> None:
        _reject_live_update(self, "Configure reference-line styles before animating.")
        self._alpha = alpha

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Vlines` object.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        artists: list[Line2D | LineCollection] = []
        x = self._x
        assert x is not None
        if isinstance(x, np.ndarray) and len(x) > 1:
            if self._y_min is not None and self._y_max is not None:
                y_min = self._y_min
                y_max = self._y_max
                params = {
                    "colors": self._colors,
                    "linestyles": self._line_styles,
                    "linewidths": self._line_widths,
                    "alpha": self._alpha,
                }
                params = strip_inherit(params)
                artists.append(
                    axes.vlines(
                        x,
                        y_min,
                        y_max,
                        zorder=z_order,
                        **params,
                    )
                )
                params.pop("linewidths", None)
            else:
                params = {
                    "color": self._colors,
                    "linestyle": self._line_styles,
                    "linewidth": self._line_widths,
                    "alpha": self._alpha,
                }
                params = strip_inherit(params)
                for i in range(len(x)):
                    artists.append(
                        axes.axvline(
                            float(cast(Any, x)[i]),
                            zorder=z_order,
                            **{
                                k: v if isinstance(v, (int, float, str)) else v[i]
                                for k, v in params.items()
                            },
                        )
                    )
                params.pop("linewidth")
            self.handle = VerticalLineCollection(
                [[(0, 0)]] * (len(x) if len(x) <= 4 else 4),
                **params,
            )
        else:
            x_scalar = float(cast(Any, x)[0]) if isinstance(x, np.ndarray) else x
            if self._y_min is not None and self._y_max is not None:
                y_min = self._y_min
                y_max = self._y_max
                params = {
                    "colors": self._colors,
                    "linestyles": self._line_styles,
                    "linewidths": self._line_widths,
                    "alpha": self._alpha,
                }
                params = strip_inherit(params)
                artists.append(
                    axes.vlines(
                        x_scalar,
                        y_min,
                        y_max,
                        zorder=z_order,
                        **params,
                    )
                )
                params.pop("linewidths", None)
            else:
                params = {
                    "color": self._colors,
                    "linestyle": self._line_styles,
                    "linewidth": self._line_widths,
                    "alpha": self._alpha,
                }
                params = strip_inherit(params)
                artists.append(axes.axvline(x_scalar, zorder=z_order, **params))
                params.pop("linewidth")
            if isinstance(x, (int, float)):
                self.handle = VerticalLineCollection([[(0, 0)]] * 1, **params)
            else:
                self.handle = VerticalLineCollection(
                    [[(0, 0)]] * (len(x) if len(x) <= 4 else 4),
                    **params,
                )
        session = _DRAW_SESSION.get()
        if session is not None:
            session.bind_legend_source(self, self.handle, lambda: self.label)
        _bind_reference_lines(
            self, artists, ("_x", "_y_min", "_y_max"), horizontal=False
        )


class Point(Plottable):
    """
    This class implements a point object.

    The :class:`~graphinglib.graph_elements.Point`
    object can be used to show important coordinates in a plot
    or add a label to some point.

    Parameters
    ----------
    x, y : float
        The x and y coordinates of the :class:`~graphinglib.graph_elements.Point`.
    label : str, optional
        Label to be attached to the :class:`~graphinglib.graph_elements.Point`.
    face_color : str or None
        Face color of the marker.
        Default depends on the ``figure_style`` configuration.
    edge_color : str or None
        Edge color of the marker.
        Default depends on the ``figure_style`` configuration.
    marker_size : float
        Size of the marker.
        Typical range is ``10`` to ``100``.
        Default depends on the ``figure_style`` configuration.
    marker_style : str
        Style of the marker.
        Values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``, ``"p"``,
        ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, and ``"_"``.
        Default depends on the ``figure_style`` configuration.
    edge_width : float
        Edge width of the marker.
        Typical range is ``0`` to ``3``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the point.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    font_size : float
        Font size for the text attached to the marker.
        Typical range is ``8`` to ``20``.
        Default depends on the ``figure_style`` configuration.
    text_color : str
        Color of the text attached to the marker.
        "same as point" uses the color of the point (prioritize edge color, then face color). Default depends on the ``figure_style`` configuration.
    h_align, v_align : str
        Horizontal and vertical alignment of the text attached
        to the :class:`~graphinglib.graph_elements.Point`.
        Horizontal alignment values include ``"left"``, ``"center"``, and ``"right"``. Vertical alignment values
        include ``"bottom"``, ``"baseline"``, ``"center"``, ``"center_baseline"``, and ``"top"``.
        Defaults to bottom left.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        x: float,
        y: float,
        label: Optional[str] = None,
        face_color: Optional[str] | Inherit = INHERIT,
        edge_color: Optional[str] | Inherit = INHERIT,
        marker_size: float | Inherit = INHERIT,
        marker_style: str | Inherit = INHERIT,
        edge_width: float | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        font_size: int | Literal["same as figure"] = "same as figure",
        text_color: str | Inherit = INHERIT,
        h_align: str = "left",
        v_align: str = "bottom",
    ) -> None:
        """
        This class implements a point object.

        The point object can be used to show important coordinates in a plot
        or add a label to some point.

        Parameters
        ----------
        x, y : float
            The x and y coordinates of the :class:`~graphinglib.graph_elements.Point`.
        label : str, optional
            Label to be attached to the :class:`~graphinglib.graph_elements.Point`.
        face_color : str or None
            Face color of the marker.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or None
            Edge color of the marker.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the marker.
            Typical range is ``10`` to ``100``.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the marker.
            Values include ``"."``, ``","``, ``"o"``, ``"v"``, ``"^"``, ``"<"``, ``">"``, ``"s"``, ``"p"``,
            ``"*"``, ``"h"``, ``"H"``, ``"+"``, ``"x"``, ``"D"``, ``"d"``, ``"|"``, and ``"_"``.
            Default depends on the ``figure_style`` configuration.
        edge_width : float
            Edge width of the marker.
            Typical range is ``0`` to ``3``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the point.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        font_size : float
            Font size for the text attached to the marker.
            Typical range is ``8`` to ``20``.
            Default depends on the ``figure_style`` configuration.
        text_color : str
            Color of the text attached to the marker.
            "same as point" uses the color of the point (prioritize edge color, then face color). Default depends on the ``figure_style`` configuration.
        h_align, v_align : str
            Horizontal and vertical alignment of the text attached
            to the :class:`~graphinglib.graph_elements.Point`.
            Horizontal alignment values include ``"left"``, ``"center"``, and ``"right"``. Vertical alignment values
            include ``"bottom"``, ``"baseline"``, ``"center"``, ``"center_baseline"``, and ``"top"``.
            Defaults to bottom left.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        self._validate_coordinate(x)
        self._validate_coordinate(y)
        session = _bound_session(self)
        if session is not None:
            previous = self.coordinates

            def restore() -> None:
                self._x, self._y = previous

            session.queue_property(self, "coordinates", (x, y), restore)
        self._x, self._y = x, y
        self.label = label
        self.face_color = face_color
        self.edge_color = edge_color
        self.marker_size = marker_size
        self.marker_style = marker_style
        self.edge_width = edge_width
        self.alpha = alpha
        self.font_size = font_size
        self.text_color = text_color
        self.h_align = h_align
        self.v_align = v_align
        self._show_coordinates: bool = False

    @staticmethod
    def _validate_coordinate(value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InvalidParameterTypeError(
                "A point's x and y coordinates must each be a single number; got "
                f"{type(value).__name__}."
            )

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._validate_coordinate(x)
        if _bound_session(self) is None:
            self._x = x
        else:
            self.coordinates = x, self._y

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._validate_coordinate(y)
        if _bound_session(self) is None:
            self._y = y
        else:
            self.coordinates = self._x, y

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._label = label

    @property
    def face_color(self) -> Styled[str | None]:
        return self._face_color

    @face_color.setter
    def face_color(self, face_color: Styled[str | None]) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._face_color = face_color

    @property
    def edge_color(self) -> Styled[str | None]:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: Styled[str | None]) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._edge_color = edge_color

    @property
    def marker_size(self) -> float | Inherit:
        return self._marker_size

    @marker_size.setter
    def marker_size(self, marker_size: float | Inherit) -> None:
        self._marker_size = _stage_live_value(self, "_marker_size", marker_size)

    @property
    def marker_style(self) -> Styled[str]:
        return self._marker_style

    @marker_style.setter
    def marker_style(self, marker_style: Styled[str]) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._marker_style = marker_style

    @property
    def edge_width(self) -> float | Inherit:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: float | Inherit) -> None:
        self._edge_width = _stage_live_value(self, "_edge_width", edge_width)

    @property
    def alpha(self) -> float | Inherit:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Inherit) -> None:
        self._alpha = _stage_live_value(self, "_alpha", alpha)

    @property
    def font_size(self) -> float | Literal["same as figure"]:
        return self._font_size

    @font_size.setter
    def font_size(self, font_size: float | Literal["same as figure"]) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._font_size = font_size

    @property
    def text_color(self) -> Styled[str]:
        return self._text_color

    @text_color.setter
    def text_color(self, text_color: Styled[str]) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._text_color = text_color

    @property
    def h_align(self) -> str:
        return self._h_align

    @h_align.setter
    def h_align(self, h_align: str) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._h_align = h_align

    @property
    def v_align(self) -> str:
        return self._v_align

    @v_align.setter
    def v_align(self, v_align: str) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._v_align = v_align

    @property
    def show_coordinates(self) -> bool:
        return self._show_coordinates

    @show_coordinates.setter
    def show_coordinates(self, show_coordinates: bool) -> None:
        _reject_live_update(
            self, "Configure Point labels and annotation styles before animating."
        )
        self._show_coordinates = show_coordinates

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self._x, self._y)

    @coordinates.setter
    def coordinates(self, coordinates: tuple[float, float]) -> None:
        x, y = coordinates
        self._validate_coordinate(x)
        self._validate_coordinate(y)
        session = _bound_session(self)
        if session is not None:
            previous = self.coordinates

            def restore() -> None:
                self._x, self._y = previous

            session.queue_property(self, "coordinates", (x, y), restore)
        self._x, self._y = x, y

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Point` object.
        """
        return deepcopy(self)

    def add_coordinates(self) -> None:
        """
        Displays the coordinates of the :class:`~graphinglib.graph_elements.Point` next to it.
        """
        self.show_coordinates = True

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self._face_color is None and self._edge_color is None:
            raise IncompatibleArgumentsError(
                "A point's face_color and edge_color cannot both be None; set at least "
                "one of them."
            )
        size = self._font_size if self._font_size != "same as figure" else None
        prefix = " " if self._h_align == "left" else ""
        postfix = " " if self._h_align == "right" else ""
        if self._label is not None and not self._show_coordinates:
            point_label = prefix + self._label + postfix
        else:
            point_label = None
        params = {
            "c": self._face_color if self._face_color is not None else "none",
            "edgecolors": self._edge_color if self._edge_color is not None else "none",
            "s": self._marker_size,
            "marker": self._marker_style,
            "linewidths": self._edge_width,
            "alpha": self._alpha,
        }
        params = strip_inherit(params)
        marker = axes.scatter(
            self._x,
            self._y,
            zorder=z_order,
            **params,
        )
        # get text color. if _text_color is "same as point", use the color of the point (prioritize edge color, then face color)
        if self._text_color == "same as point":
            if self._edge_color is not None:
                text_color = self._edge_color
            else:
                text_color = self._face_color
        else:
            text_color = self._text_color
        params = {
            "color": text_color,
            "fontsize": size,
            "horizontalalignment": self._h_align,
            "verticalalignment": self._v_align,
        }
        params = strip_inherit(params)
        label = axes.annotate(
            cast(str, point_label),
            (self._x, self._y),
            zorder=z_order,
            **params,
        )
        if self._show_coordinates:
            prefix = " " if self._h_align == "left" else ""
            postfix = " " if self._h_align == "right" else ""
            if self._label is not None:
                point_label = (
                    prefix
                    + self._label
                    + " : "
                    + f"({self._x:.3f}, {self._y:.3f})"
                    + postfix
                )
            else:
                point_label = prefix + f"({self._x:.3f}, {self._y:.3f})" + postfix
            if self._text_color == "same as point":
                if self._edge_color is not None:
                    text_color = self._edge_color
                else:
                    text_color = self._face_color
            else:
                text_color = self._text_color
            params = {
                "color": text_color,
                "fontsize": size,
                "horizontalalignment": self._h_align,
                "verticalalignment": self._v_align,
            }
            params = strip_inherit(params)
            coordinates = axes.annotate(
                point_label,
                (self._x, self._y),
                zorder=z_order,
                **params,
            )
        session = _DRAW_SESSION.get()
        if session is not None:
            session.bind_property(
                self, "_marker_size", marker, "sizes", lambda value: [value]
            )
            session.bind_property(self, "_edge_width", marker, "linewidth")
            session.bind_property(self, "_alpha", marker, "alpha")
            # Alignment is resolved for each occurrence, including reused Points.
            show_coordinates = self._show_coordinates
            prefix = " " if self._h_align == "left" else ""
            postfix = " " if self._h_align == "right" else ""

            def prepare(position: tuple[float, float]) -> _ArtistEdit:
                if not all(isfinite(coord) for coord in position):
                    raise InvalidParameterError(
                        "Animated Point coordinates must be finite."
                    )
                before = marker.get_offsets()

                def move(artist: Annotation, xy: tuple[float, float]) -> None:
                    artist.xy = xy
                    artist.set_position(xy)

                def update() -> None:
                    marker.set_offsets([position])
                    move(label, position)
                    if show_coordinates:
                        move(coordinates, position)
                        x, y = position
                        caption = f"{self._label} : " if self._label is not None else ""
                        coordinates.set_text(
                            f"{prefix}{caption}({x:.3f}, {y:.3f}){postfix}"
                        )

                def restore() -> None:
                    marker.set_offsets(before)
                    move(label, old_label_position)
                    if show_coordinates:
                        move(coordinates, old_coordinates_position)
                        coordinates.set_text(old_content)

                old_label_position = label.xy
                if show_coordinates:
                    old_coordinates_position = coordinates.xy
                    old_content = coordinates.get_text()
                return _ArtistEdit(update, restore)

            session.bind_update(self, "coordinates", prepare)


@dataclass
class Text(Plottable):
    """
    This class allows text to be plotted.

    It is also possible to attach an arrow to the :class:`~graphinglib.graph_elements.Text`
    with the method :py:meth:`~graphinglib.graph_elements.Text.attach_arrow`
    to point at something of interest in the plot.

    Parameters
    ----------
    x, y : float
        The x and y coordinates at which to plot the :class:`~graphinglib.graph_elements.Text`.
    text : str
        The text to be plotted.
    color : str
        Color of the text.
        Default depends on the ``figure_style`` configuration.
    font_size : float
        Font size of the text.
        Typical range is ``8`` to ``20``.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the text.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Default depends on the ``figure_style`` configuration.
    h_align, v_align : str
        Horizontal and vertical alignment of the text.
        Horizontal alignment values include ``"left"``, ``"center"``, and ``"right"``. Vertical alignment values
        include ``"bottom"``, ``"baseline"``, ``"center"``, ``"center_baseline"``, and ``"top"``.
        Default depends on the ``figure_style`` configuration.
    rotation : float
        Rotation angle of the text in degrees.
        Defaults to 0.
    highlight_color : str, optional
        Color of the background highlight box behind the text.
        If specified, a box will be drawn behind the text.
        Default is ``None`` (no highlight).
    highlight_alpha : float, optional
        Opacity of the highlight box.
        Range is ``0`` (transparent) to ``1`` (opaque).
        Defaults to 1.0.
    highlight_padding : float, optional
        Padding around the text for the highlight box. A value of 0 means no padding.
        Defaults to 0.1.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    _x: float
    _y: float
    _text: str
    _color: str | Inherit = INHERIT
    _font_size: float | Literal["same as figure"] = "same as figure"
    _alpha: float | Inherit = INHERIT
    _h_align: str | Inherit = INHERIT
    _v_align: str | Inherit = INHERIT
    _rotation: float = 0.0
    _highlight_color: Optional[str] = None
    _highlight_alpha: float = 1.0
    _highlight_padding: float = 0.1
    _arrow_pointing_to: Optional[tuple[float, float]] = field(default=None, init=False)

    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        color: str | Inherit = INHERIT,
        font_size: float | Literal["same as figure"] = "same as figure",
        alpha: float | Inherit = INHERIT,
        h_align: str | Inherit = INHERIT,
        v_align: str | Inherit = INHERIT,
        rotation: float = 0.0,
        highlight_color: Optional[str] = None,
        highlight_alpha: float = 1.0,
        highlight_padding: float = 0.1,
    ) -> None:
        """
        This class allows text to be plotted.

        It is also possible to attach an arrow to the :class:`~graphinglib.graph_elements.Text`
        with the method :py:meth:`~graphinglib.graph_elements.Text.attach_arrow`
        to point at something of interest in the plot.

        Parameters
        ----------
        x, y : float
            The x and y coordinates at which to plot the :class:`~graphinglib.graph_elements.Text`.
        text : str
            The text to be plotted.
        color : str
            Color of the text.
            Default depends on the ``figure_style`` configuration.
        font_size : float
            Font size of the text.
            Typical range is ``8`` to ``20``.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the text.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        h_align, v_align : str
            Horizontal and vertical alignment of the text.
            Horizontal alignment values include ``"left"``, ``"center"``, and ``"right"``. Vertical alignment values
            include ``"bottom"``, ``"baseline"``, ``"center"``, ``"center_baseline"``, and ``"top"``.
            Default depends on the ``figure_style`` configuration.
        rotation : float
            Rotation angle of the text in degrees.
            Defaults to 0.
        highlight_color : str, optional
            Color of the background highlight box behind the text.
            If specified, a box will be drawn behind the text.
            Default is ``None`` (no highlight).
        highlight_alpha : float, optional
            Opacity of the highlight box.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Defaults to 1.0.
        highlight_padding : float, optional
            Padding around the text for the highlight box. A value of 0 means no padding.
            Defaults to 0.1.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        self._x = x
        self._y = y
        self._text = text
        self._color = color
        self._font_size = font_size
        self._alpha = alpha
        self._h_align = h_align
        self._v_align = v_align
        self._rotation = rotation
        self._highlight_color = highlight_color
        self._highlight_alpha = highlight_alpha
        self._highlight_padding = highlight_padding
        self._arrow_pointing_to = None

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._set_text_data(x, self._y, self._text)

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._set_text_data(self._x, y, self._text)

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, text: str) -> None:
        self._set_text_data(self._x, self._y, text)

    def _set_text_data(self, x: float, y: float, content: str) -> None:
        session = _bound_session(self)
        if session is not None:
            if not isinstance(content, str):
                raise TypeError("Animated Text content must be a string.")
            if not all(isinstance(value, Real) and isfinite(value) for value in (x, y)):
                raise ValueError(
                    "Animated Text coordinates must be finite real numbers."
                )
            previous = self._x, self._y, self._text

            def restore() -> None:
                self._x, self._y, self._text = previous

            session.queue_text(self, x, y, content, restore)
        self._x, self._y, self._text = x, y, content

    @property
    def color(self) -> Styled[str]:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = _stage_live_value(self, "_color", color)

    @property
    def font_size(self) -> float | Literal["same as figure"]:
        return self._font_size

    @font_size.setter
    def font_size(self, font_size: float | Literal["same as figure"]) -> None:
        self._font_size = _stage_live_value(self, "_font_size", font_size)

    @property
    def alpha(self) -> float | Inherit:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Inherit) -> None:
        self._alpha = _stage_live_value(self, "_alpha", alpha)

    @property
    def h_align(self) -> Styled[str]:
        return self._h_align

    @h_align.setter
    def h_align(self, h_align: str) -> None:
        self._h_align = _stage_live_value(self, "_h_align", h_align)

    @property
    def v_align(self) -> Styled[str]:
        return self._v_align

    @v_align.setter
    def v_align(self, v_align: str) -> None:
        self._v_align = _stage_live_value(self, "_v_align", v_align)

    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: float) -> None:
        self._rotation = _stage_live_value(self, "_rotation", rotation)

    @property
    def highlight_color(self) -> Optional[str]:
        return self._highlight_color

    @highlight_color.setter
    def highlight_color(self, highlight_color: Optional[str]) -> None:
        _reject_live_update(self, "Configure Text highlights before animating.")
        self._highlight_color = highlight_color

    @property
    def highlight_alpha(self) -> float:
        return self._highlight_alpha

    @highlight_alpha.setter
    def highlight_alpha(self, highlight_alpha: float) -> None:
        _reject_live_update(self, "Configure Text highlights before animating.")
        self._highlight_alpha = highlight_alpha

    @property
    def highlight_padding(self) -> float:
        return self._highlight_padding

    @highlight_padding.setter
    def highlight_padding(self, highlight_padding: float) -> None:
        _reject_live_update(self, "Configure Text highlights before animating.")
        self._highlight_padding = highlight_padding

    @property
    def arrow_pointing_to(self) -> Optional[tuple[float, float]]:
        return self._arrow_pointing_to

    @arrow_pointing_to.setter
    def arrow_pointing_to(
        self, arrow_pointing_to: Optional[tuple[float, float]]
    ) -> None:
        self._arrow_pointing_to = _stage_live_value(
            self, "_arrow_pointing_to", arrow_pointing_to
        )

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Text` object.
        """
        return deepcopy(self)

    def add_arrow(
        self,
        points_to: tuple[float, float],
        width: Optional[float] = None,
        shrink: Optional[float] = None,
        head_width: Optional[float] = None,
        head_length: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """
        Adds an arrow pointing from the :class:`~graphinglib.graph_elements.Text`
        to a specified point.

        Parameters
        ----------
        points_to: tuple[float, float]
            Coordinates at which to point.
        width : float, optional
            Arrow width, in points.
            Typical range is ``0.5`` to ``3``.
        shrink : float, optional
            Fraction of the total length of the arrow to shrink from both ends.
            Range is ``0`` to ``0.5``. A value of ``0.5`` means the arrow is no longer visible.
        head_width : float, optional
            Width of the head of the arrow, in points.
            Typical range is ``3`` to ``10``.
        head_length : float, optional
            Length of the head of the arrow, in points.
            Typical range is ``3`` to ``10``.
        alpha : float, optional
            Opacity of the arrow.
            Range is ``0`` (transparent) to ``1`` (opaque).
        """
        session = _bound_session(self)
        if session is not None:
            session.reject_structural_update(
                self, "Configure Text arrows before animating."
            )
        self._arrow_pointing_to = points_to
        self._arrow_properties = {}
        if width is not None:
            self._arrow_properties["width"] = width
        if shrink is not None:
            self._arrow_properties["shrink"] = shrink
        if head_width is not None:
            self._arrow_properties["headwidth"] = head_width
        if head_length is not None:
            self._arrow_properties["headlength"] = head_length
        if alpha is not None:
            self._arrow_properties["alpha"] = alpha

    def _plot_element(self, axes: plt.Axes | MPLFigure, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified target, which can be either an
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_ or a
        `Figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html>`.
        """
        size = self._font_size if self._font_size != "same as figure" else None
        params = {
            "color": self._color,
            "fontsize": size,
            "alpha": self._alpha,
            "horizontalalignment": self._h_align,
            "verticalalignment": self._v_align,
            "rotation": self._rotation,
        }

        # Add highlight/background box if highlight_color is specified
        if self._highlight_color is not None:
            bbox_dict = {
                "boxstyle": f"square,pad={self._highlight_padding}",
                "facecolor": self._highlight_color,
                "edgecolor": "none",
                "alpha": self._highlight_alpha,
            }
            params["bbox"] = bbox_dict

        params = strip_inherit(params)
        artist = axes.text(
            self._x,
            self._y,
            self._text,
            zorder=z_order,
            **params,
        )
        session = _DRAW_SESSION.get()
        if session is not None:
            session.bind_text(self, artist)
            if self._arrow_pointing_to is None:
                for name, mpl_name in (
                    ("_color", "color"),
                    ("_font_size", "fontsize"),
                    ("_alpha", "alpha"),
                    ("_h_align", "horizontalalignment"),
                    ("_v_align", "verticalalignment"),
                    ("_rotation", "rotation"),
                ):
                    session.bind_property(self, name, artist, mpl_name)
        if self._arrow_pointing_to is not None and isinstance(axes, plt.Axes):
            # Build the arrow properties on a local copy: mutating the instance dict
            # here used to leak an unresolved INHERIT color into it, and the arrow
            # was silently skipped when the color was unresolved.
            arrow_properties: dict[str, Any] = dict(self._arrow_properties)
            if not is_inherit(self._color):
                arrow_properties["color"] = self._color
            params = {
                "color": self._color,
                "fontsize": size,
                "horizontalalignment": self._h_align,
                "verticalalignment": self._v_align,
            }
            params = strip_inherit(params)
            params["arrowprops"] = arrow_properties
            annotation = axes.annotate(
                self._text,
                self._arrow_pointing_to,
                xytext=(self._x, self._y),
                zorder=z_order,
                **params,
            )
            if session is not None:
                session.bind_text(self, annotation)

                def prepare_target(target: tuple[float, float] | None) -> _ArtistEdit:
                    if (
                        target is None
                        or len(target) != 2
                        or not all(isfinite(value) for value in target)
                    ):
                        raise InvalidParameterError(
                            "An animated Text arrow needs a finite target."
                        )
                    before = annotation.xy
                    return _ArtistEdit(
                        lambda: setattr(annotation, "xy", target),
                        lambda: setattr(annotation, "xy", before),
                    )

                session.bind_update(self, "_arrow_pointing_to", prepare_target)


@dataclass
class Table(Plottable):
    """
    This class allows to plot a table inside a Figure or MultiFigure.

    The Table object can be used to add raw data to a figure or add supplementary
    information like output parameters for a fit or anyother operation.

    Parameters
    ----------
    cell_text : Sequence[Sequence[str]]
        Text or data to be displayed in the table, as rows of cell values. The shape of the provided data
        determines the number of columns and rows.
    cell_colors : ArrayLike or str, optional
        Colors to apply to the cells' background. Must be a list of colors the same
        shape as the cells.
        Default depends on the ``figure_style`` configuration.
    cell_align : str
        Alignment of the cells' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    col_labels : list[str], optional
        List of labels for the rows of the table. If none are specified, no row labels are displayed.
    col_widths : list[float], optional
        Widths to apply to the columns. Must be a list the same length as the number of columns.
    col_align : str
        Alignment of the column labels' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    col_colors : ArrayLike or str, optional
        Colors to apply to the column labels' background. Must be a list of colors the same
        length as the number of columns.
        Default depends on the ``figure_style`` configuration.
    row_labels : list[str], optional
        List of labels for the rows of the table. If none are specified, no row labels are displayed.
    row_align : str
        Alignment of the row labels' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    row_colors : ArrayLike or str, optional
        Colors to apply to the row labels' background. Must be a list of colors the same
        length as the number of rows.
        Default depends on the ``figure_style`` configuration.
    edge_width : float or str, optional
        Width of the table's edges.
        Typical range is ``0.5`` to ``3``.
        Default depends on the ``figure_style`` configuration.
    edge_color : str, optional
        Color of the table's edges.
        Default depends on the ``figure_style`` configuration.
    text_color : str, optional
        Color of the text in the table.
        Default depends on the ``figure_style`` configuration.
    scaling : tuple[float, float], optional
        Horizontal and vertical scaling factors to apply to the table.
        Defaults to ``(1, 1.5)``.
    location : str
        Position of the table inside the axes. Values are ``"best"``, ``"bottom"``, ``"bottom left"``,
        ``"bottom right"``, ``"center"``, ``"center left"``, ``"center right"``, ``"left"``,
        ``"lower center"``, ``"lower left"``, ``"lower right"``, ``"right"``, ``"top"``, ``"top left"``,
        ``"top right"``, ``"upper center"``, ``"upper left"``, and ``"upper right"``.
        Defaults to ``"best"``.

    Notes
    -----
    Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
    (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
    values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
    """

    def __init__(
        self,
        cell_text: Sequence[Sequence[str]],
        cell_colors: ArrayLike | str | Inherit = INHERIT,
        cell_align: str | Inherit = INHERIT,
        col_labels: Optional[list[str]] = None,
        col_widths: Optional[list[float]] = None,
        col_align: str | Inherit = INHERIT,
        col_colors: ArrayLike | str | Inherit = INHERIT,
        row_labels: Optional[list[str]] = None,
        row_align: str | Inherit = INHERIT,
        row_colors: ArrayLike | str | Inherit = INHERIT,
        edge_width: float | Inherit = INHERIT,
        edge_color: str | Inherit = INHERIT,
        text_color: str | Inherit = INHERIT,
        scaling: tuple[float, float] = (1.0, 1.5),
        location: str = "best",
    ) -> None:
        """
        This class allows to plot a table inside a Figure or MultiFigure.

        The Table object can be used to add raw data to a figure or add supplementary
        information like output parameters for a fit or anyother operation.

        Parameters
        ----------
        cell_text : Sequence[Sequence[str]]
            Text or data to be displayed in the table, as rows of cell values. The shape of the provided data
            determines the number of columns and rows.
        cell_colors : ArrayLike or str, optional
            Colors to apply to the cells' background. Must be a list of colors the same
            shape as the cells.
            Default depends on the ``figure_style`` configuration.
        cell_align : str
            Alignment of the cells' text. Must be one of the following:
            {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
        col_labels : list[str], optional
            List of labels for the rows of the table. If none are specified, no row labels are displayed.
        col_widths : list[float], optional
            Widths to apply to the columns. Must be a list the same length as the number of columns.
        col_align : str
            Alignment of the column labels' text. Must be one of the following:
            {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
        col_colors : ArrayLike or str, optional
            Colors to apply to the column labels' background. Must be a list of colors the same
            length as the number of columns.
            Default depends on the ``figure_style`` configuration
        row_labels : list[str], optional
            List of labels for the rows of the table. If none are specified, no row labels are displayed.
        row_align : str
            Alignment of the row labels' text. Must be one of the following:
            {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
        row_colors : ArrayLike or str, optional
            Colors to apply to the row labels' background. Must be a list of colors the same
            length as the number of rows.
            Default depends on the ``figure_style`` configuration.
        edge_width : float or str, optional
            Width of the table's edges.
            Typical range is ``0.5`` to ``3``.
            Default depends on the ``figure_style`` configuration.
        edge_color : str, optional
            Color of the table's edges.
            Default depends on the ``figure_style`` configuration.
        text_color : str, optional
            Color of the text within the table.
            Default depends on the ``figure_style`` configuration.
        scaling : tuple[float, float], optional
            Horizontal and vertical scaling factors to apply to the table.
            Defaults to ``(1, 1.5)``.
        location : str
            Position of the table inside the axes. Values are ``"best"``, ``"bottom"``, ``"bottom left"``,
            ``"bottom right"``, ``"center"``, ``"center left"``, ``"center right"``, ``"left"``,
            ``"lower center"``, ``"lower left"``, ``"lower right"``, ``"right"``, ``"top"``, ``"top left"``,
            ``"top right"``, ``"upper center"``, ``"upper left"``, and ``"upper right"``.
            Defaults to ``"best"``.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        self._cell_text = cell_text
        self._cell_colors = cell_colors
        self._cell_align = cell_align
        self._col_labels = col_labels
        self._col_widths = col_widths
        self._col_align = col_align
        self._col_colors = col_colors
        self._row_labels = row_labels
        self._row_align = row_align
        self._row_colors = row_colors
        self._edge_width = edge_width
        self._edge_color = edge_color
        self._text_color = text_color
        self._scaling = scaling
        self._location = location

    @property
    def cell_text(self) -> Sequence[Sequence[str]]:
        return self._cell_text

    @cell_text.setter
    def cell_text(self, cell_text: Sequence[Sequence[str]]) -> None:
        self._cell_text = _stage_live_value(self, "_cell_text", cell_text)

    @property
    def cell_colors(self) -> Styled[ArrayLike | str]:
        return self._cell_colors

    @cell_colors.setter
    def cell_colors(self, cell_colors: Styled[ArrayLike | str]) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._cell_colors = cell_colors

    @property
    def cell_align(self) -> Styled[str]:
        return self._cell_align

    @cell_align.setter
    def cell_align(self, cell_align: Styled[str]) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._cell_align = cell_align

    @property
    def col_labels(self) -> list[str] | None:
        return self._col_labels

    @col_labels.setter
    def col_labels(self, col_labels: list[str] | None) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._col_labels = col_labels

    @property
    def col_widths(self) -> list[float] | None:
        return self._col_widths

    @col_widths.setter
    def col_widths(self, col_widths: list[float] | None) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._col_widths = col_widths

    @property
    def col_align(self) -> Styled[str]:
        return self._col_align

    @col_align.setter
    def col_align(self, col_align: Styled[str]) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._col_align = col_align

    @property
    def col_colors(self) -> Styled[ArrayLike | str]:
        return self._col_colors

    @col_colors.setter
    def col_colors(self, col_colors: Styled[ArrayLike | str]) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._col_colors = col_colors

    @property
    def row_labels(self) -> list[str] | None:
        return self._row_labels

    @row_labels.setter
    def row_labels(self, row_labels: list[str] | None) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._row_labels = row_labels

    @property
    def row_align(self) -> Styled[str]:
        return self._row_align

    @row_align.setter
    def row_align(self, row_align: Styled[str]) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._row_align = row_align

    @property
    def row_colors(self) -> Styled[ArrayLike | str]:
        return self._row_colors

    @row_colors.setter
    def row_colors(self, row_colors: Styled[ArrayLike | str]) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._row_colors = row_colors

    @property
    def edge_width(self) -> Styled[float]:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: float) -> None:
        self._edge_width = _stage_live_value(self, "_edge_width", edge_width)
        if _bound_session(self) is None:
            for (i, j), cell in self.handle.get_celld().items():
                cell.set_linewidth(edge_width)

    @property
    def edge_color(self) -> Styled[str]:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = _stage_live_value(self, "_edge_color", edge_color)
        if _bound_session(self) is None:
            for (i, j), cell in self.handle.get_celld().items():
                cell.set_edgecolor(edge_color)

    @property
    def text_color(self) -> Styled[str]:
        return self._text_color

    @text_color.setter
    def text_color(self, text_color: str) -> None:
        self._text_color = _stage_live_value(self, "_text_color", text_color)
        if _bound_session(self) is None:
            for (i, j), cell in self.handle.get_celld().items():
                cell.set_text_props(color=self._text_color)

    @property
    def scaling(self) -> tuple[float, float]:
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: tuple[float, float]) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._scaling = scaling

    @property
    def location(self) -> str:
        return self._location

    @location.setter
    def location(self, location: str) -> None:
        _reject_live_update(
            self, "Configure table dimensions, headers and layout before animating."
        )
        self._location = location

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Table` object.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        params = {
            "cellLoc": self._cell_align,
            "colLoc": self._col_align,
            "rowLoc": self._row_align,
        }
        params = strip_inherit(params)

        cell_colors = resolve_or(self._cell_colors, "white")
        col_colors = resolve_or(self._col_colors, "#bfbfbf")
        row_colors = resolve_or(self._row_colors, "#bfbfbf")

        # Set colors to correct shape if they are strings
        if isinstance(cell_colors, str):
            cell_colors = [[cell_colors] * len(self._cell_text[0])] * len(
                self._cell_text
            )
        if isinstance(col_colors, str):
            col_colors = [col_colors] * len(self._cell_text[0])
        if isinstance(row_colors, str):
            row_colors = [row_colors] * len(self._cell_text)

        self.handle = axes.table(
            cellText=self._cell_text,
            cellColours=cast(Any, cell_colors),
            colLabels=self._col_labels,
            colWidths=self._col_widths,
            colColours=cast(Any, col_colors),
            rowLabels=self._row_labels,
            rowColours=cast(Any, row_colors),
            loc=self._location,
            zorder=z_order,
            **params,
        )
        self.handle.auto_set_font_size(False)
        self.handle.scale(self._scaling[0], self._scaling[1])
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_text_props(color=resolve_or(self._text_color, "black"))
            cell.set_edgecolor(resolve_or(self._edge_color, "black"))
            cell.set_linewidth(resolve_or(self._edge_width, 1))

        session = _DRAW_SESSION.get()
        if session is not None:
            cells = self.handle.get_celld()
            shape = len(self._cell_text), len(self._cell_text[0])

            def prepare(text: Sequence[Sequence[str]]) -> _ArtistEdit:
                rows = tuple(tuple(row) for row in text)
                if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
                    raise InvalidParameterError(
                        "An animated Table must retain its original row and column count."
                    )
                edits = [
                    _edit(
                        cells[row_index + row_offset, col_index].get_text().set_text,
                        cells[row_index + row_offset, col_index].get_text().get_text(),
                        value,
                    )
                    for row_index, row in enumerate(rows)
                    for col_index, value in enumerate(row)
                ]

                def apply() -> None:
                    for edit in edits:
                        edit.apply()

                def undo() -> None:
                    for edit in reversed(edits):
                        edit.undo()

                return _ArtistEdit(apply, undo)

            row_offset = int(self._col_labels is not None)
            session.bind_update(self, "_cell_text", prepare)
            for cell in cells.values():
                session.bind_property(self, "_edge_color", cell, "edgecolor")
                session.bind_property(self, "_edge_width", cell, "linewidth")
                session.bind_property(self, "_text_color", cell.get_text(), "color")


class PlottableAxMethod(Plottable):
    """
    This experimental class allows to call any matplotlib Axes method as a plottable element in a
    :class:`~graphinglib.smart_figure.SmartFigure`. This object can be used to create plot types that have not yet been
    implemented in GraphingLib.

    This class only works with Axes methods that create plottable elements (e.g., ``bar`` or ``pcolormesh``).
    Methods that modify axes properties (e.g., ``set_facecolor``, ``set_title``) are not supported.

    Parameters
    ----------
    meth : str
        Name of the matplotlib Axes method to call. The method will be called as ``axes.meth(*args, **kwargs)``. For
        example, this can be "pcolormesh" or "bar".

        .. warning::
            The provided matplotlib Axes method must accept a ``zorder`` keyword argument to be compatible with this
            class. If not, an exception will be raised when attempting to plot the element.
    *args
        Positional arguments to pass to ``axes.meth``.
    label : str, optional
        Label to be attached to the :class:`~graphinglib.graph_elements.PlottableAxMethod`.
    **kwargs
        Keyword arguments to pass to ``axes.meth``.
    """

    def __init__(self, meth: str, *args, label: Optional[str] = None, **kwargs) -> None:
        self.meth = meth
        self.args = args
        self.kwargs = kwargs
        self.label = label
        self.handle = None

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        try:
            attrs = getattr(axes, self.meth)(*self.args, zorder=z_order, **self.kwargs)
            if isinstance(attrs, list) and len(attrs) > 0:
                self.handle = attrs[0]
        except TypeError as e:
            if "zorder" in str(e):
                try:
                    attrs = getattr(axes, self.meth)(*self.args, **self.kwargs)
                    if isinstance(attrs, list) and len(attrs) > 0:
                        self.handle = attrs[0]
                except Exception as e2:
                    raise PlottingError(
                        f"Failed to call Axes method '{self.meth}' with the provided "
                        "arguments. Please check that all provided arguments are valid "
                        "for the given method."
                    ) from e2
            else:
                raise PlottingError(
                    f"Failed to call Axes method '{self.meth}' with the provided "
                    "arguments. Please check that all provided arguments are valid for "
                    "the given method."
                ) from e
