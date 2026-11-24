from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any, Literal, Optional, Protocol, Sequence, cast, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import GRIDLINE_INTERPOLATION_STEPS
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure as MPLFigure
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.table import Cell
from matplotlib.text import Annotation
from matplotlib.text import Text as TextArtist
from numpy.typing import ArrayLike

from ._animation import (
    _DRAW_SESSION,
    _AnimationSession,
    _ArtistEdit,
    _bound_session,
    _combine_edits,
    _edit,
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


_REFERENCE_LINE_STYLE = {
    "_colors": "color",
    "_line_widths": "linewidth",
    "_line_styles": "linestyle",
    "_alpha": "alpha",
}


def _reference_positions(values: ArrayLike) -> np.ndarray:
    positions = np.atleast_1d(values)
    if positions.ndim != 1:
        raise InvalidParameterError("Reference-line positions must be one-dimensional.")
    return positions


def _reference_line_styles(params: dict[str, Any], count: int) -> list[dict[str, Any]]:
    styles: list[dict[str, Any]] = [{} for _ in range(count)]
    for name, value in params.items():
        if isinstance(value, (list, np.ndarray)):
            if len(value) != count:
                raise InvalidParameterError(
                    "Reference-line styles must match the line count."
                )
            for style, item in zip(styles, value, strict=True):
                style[name] = item
        else:
            for style in styles:
                style[name] = value
    return styles


def _draw_reference_lines(
    element: Hlines | Vlines,
    axes: plt.Axes,
    z_order: int,
    names: tuple[str, str, str],
    *,
    horizontal: bool,
) -> LineCollection:
    position, start, end = names
    coords = _reference_positions(getattr(element, position))
    low, high = getattr(element, start), getattr(element, end)
    params = strip_inherit(
        {field: getattr(element, attr) for attr, field in _REFERENCE_LINE_STYLE.items()}
    )
    if low is not None and high is not None:
        plot = axes.hlines if horizontal else axes.vlines
        collection = plot(
            coords,
            low,
            high,
            zorder=z_order,
            **{
                ("alpha" if field == "alpha" else f"{field}s"): value
                for field, value in params.items()
            },
        )
        _bind_bounded_reference_lines(element, collection, names, horizontal)
    else:
        plot = axes.axhline if horizontal else axes.axvline
        lines = [
            plot(float(coord), zorder=z_order, **style)
            for coord, style in zip(
                coords, _reference_line_styles(params, len(coords)), strict=True
            )
        ]
        _bind_spanning_reference_lines(
            element, axes, lines, position, params, z_order, horizontal
        )

    # Reserve a simple legend swatch even when the data start empty.
    proxy_type = LineCollection if horizontal else VerticalLineCollection
    swatch_count = max(1, min(len(coords), 3 if horizontal else 4))
    legend_style = {
        name: value
        for name, value in params.items()
        if name != "linewidth"
        and (len(coords) or not isinstance(value, (list, np.ndarray)))
    }
    handle = proxy_type([[(0, 0)]] * swatch_count, **legend_style)
    session = _DRAW_SESSION.get()
    if session is not None:
        session.bind_legend_source(element, handle, lambda: element.label)
    return handle


def _bind_bounded_reference_lines(
    element: Hlines | Vlines,
    collection: LineCollection,
    names: tuple[str, str, str],
    horizontal: bool,
) -> None:
    if (draw_session := _DRAW_SESSION.get()) is None:
        return
    session = draw_session

    for attr, field in _REFERENCE_LINE_STYLE.items():
        if attr != "_line_styles":
            session.bind_property(element, attr, collection, field)

    def prepare_dashes(value: Any) -> _ArtistEdit:
        # The public getter returns width-scaled dashes.
        original = cast(Any, collection)._us_linestyles
        return _edit(collection.set_linestyle, original, value)

    session.bind_update(element, "_line_styles", prepare_dashes)
    position, start, end = names

    def prepare_geometry() -> _ArtistEdit:
        element._validate_state()
        if not any(session.has_frame_update(element, name) for name in names):
            return _combine_edits([])
        positions = _reference_positions(getattr(element, position))
        low, high = getattr(element, start), getattr(element, end)
        if low is None or high is None:
            raise InvalidParameterError(
                "An animated bounded reference line must retain both bounds."
            )
        low, high = (
            np.broadcast_to(low, positions.shape),
            np.broadcast_to(high, positions.shape),
        )
        ends = np.stack((low, high), axis=-1)
        across = np.repeat(positions[:, None], 2, axis=1)
        x, y = (ends, across) if horizontal else (across, ends)
        segments = list(np.stack((x, y), axis=-1))
        before = [path.vertices for path in collection.get_paths()]
        return _edit(collection.set_segments, before, segments)

    session.bind_derived(element, (*names, *_REFERENCE_LINE_STYLE), prepare_geometry)


def _bind_spanning_reference_lines(
    element: Hlines | Vlines,
    axes: plt.Axes,
    lines: list[Line2D],
    position: str,
    initial_style: dict[str, Any],
    z_order: int,
    horizontal: bool,
) -> None:
    session = _DRAW_SESSION.get()
    if session is None:
        return
    defaults = {
        "color": plt.rcParams["lines.color"],
        "linewidth": plt.rcParams["lines.linewidth"],
        "linestyle": plt.rcParams["lines.linestyle"],
        "alpha": None,
        **initial_style,
    }
    transform = (
        axes.get_yaxis_transform(which="grid")
        if horizontal
        else axes.get_xaxis_transform(which="grid")
    )
    get_coordinates = Line2D.get_ydata if horizontal else Line2D.get_xdata
    set_coordinates = Line2D.set_ydata if horizontal else Line2D.set_xdata

    def prepare() -> _ArtistEdit:
        element._validate_state()
        positions = _reference_positions(getattr(element, position))
        styles = _reference_line_styles(
            {
                field: resolve_or(getattr(element, attr), defaults[field])
                for attr, field in _REFERENCE_LINE_STYLE.items()
            },
            len(positions),
        )
        original_count = len(lines)
        before = [
            (
                get_coordinates(line),
                line.get_visible(),
                {
                    field: (
                        cast(Any, line)._unscaled_dash_pattern
                        if field == "linestyle" and line.get_linestyle() == "--"
                        else getattr(line, f"get_{field}")()
                    )
                    for field in _REFERENCE_LINE_STYLE.values()
                },
            )
            for line in lines[: len(positions)]
        ]
        previous_visibility = [line.get_visible() for line in lines[len(positions) :]]
        additions = []
        for coord, style in zip(
            positions[original_count:], styles[original_count:], strict=True
        ):
            across, along = [float(coord)] * 2, [0, 1]
            x, y = (along, across) if horizontal else (across, along)
            line = Line2D(x, y, transform=transform, zorder=z_order, **style)
            # Match the projection-aware path of axhline and axvline.
            cast(
                Any, line.get_path()
            )._interpolation_steps = GRIDLINE_INTERPOLATION_STEPS
            additions.append(line)

        def apply() -> None:
            for line, coord, style in zip(lines, positions, styles):
                set_coordinates(line, [float(coord)] * 2)
                line.set(visible=True, **style)
            for line in lines[len(positions) :]:
                line.set_visible(False)
            for line in additions:
                # Adding the artist directly leaves the established data limits alone.
                lines.append(line)
                axes.add_artist(line)

        def undo() -> None:
            added = lines[original_count:]
            if added:
                attached = set(axes.lines)
                for line in reversed(added):
                    if line in attached:
                        line.remove()
            del lines[original_count:]
            for line, (coordinates, visible, style) in zip(lines, before):
                set_coordinates(line, coordinates)
                line.set(visible=visible, **style)
            for line, visible in zip(
                lines[len(positions) :], previous_visibility, strict=True
            ):
                line.set_visible(visible)

        return _ArtistEdit(apply, undo)

    session.bind_derived(element, (position, *_REFERENCE_LINE_STYLE), prepare)


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
        if not self._in_init and _bound_session(self) is None:
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
        if not self._in_init and _bound_session(self) is None:
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
        if not self._in_init and _bound_session(self) is None:
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
        self._colors = _stage_live_value(self, "_colors", colors)
        if not self._in_init and _bound_session(self) is None:
            self._validate_state()

    @property
    def line_widths(self) -> Styled[list[float] | float]:
        return self._line_widths

    @line_widths.setter
    def line_widths(self, line_widths: Styled[list[float] | float]) -> None:
        self._line_widths = _stage_live_value(self, "_line_widths", line_widths)
        if not self._in_init and _bound_session(self) is None:
            self._validate_state()

    @property
    def line_styles(self) -> Styled[list[str] | str]:
        return self._line_styles

    @line_styles.setter
    def line_styles(self, line_styles: Styled[list[str] | str]) -> None:
        self._line_styles = _stage_live_value(self, "_line_styles", line_styles)
        if not self._in_init and _bound_session(self) is None:
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
        self.handle = _draw_reference_lines(
            self, axes, z_order, ("_y", "_x_min", "_x_max"), horizontal=True
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
        if not self._in_init and _bound_session(self) is None:
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
        self._colors = _stage_live_value(self, "_colors", colors)
        if not self._in_init and _bound_session(self) is None:
            self._validate_state()

    @property
    def line_widths(self) -> Styled[list[float] | float]:
        return self._line_widths

    @line_widths.setter
    def line_widths(self, line_widths: Styled[list[float] | float]) -> None:
        self._line_widths = _stage_live_value(self, "_line_widths", line_widths)
        if not self._in_init and _bound_session(self) is None:
            self._validate_state()

    @property
    def line_styles(self) -> Styled[list[str] | str]:
        return self._line_styles

    @line_styles.setter
    def line_styles(self, line_styles: Styled[list[str] | str]) -> None:
        self._line_styles = _stage_live_value(self, "_line_styles", line_styles)
        if not self._in_init and _bound_session(self) is None:
            self._validate_state()

    @property
    def alpha(self) -> float | Inherit:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float | Inherit) -> None:
        self._alpha = _stage_live_value(self, "_alpha", alpha)

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
        self.handle = _draw_reference_lines(
            self, axes, z_order, ("_x", "_y_min", "_y_max"), horizontal=False
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
        self._label = _stage_live_value(self, "_label", label)

    @property
    def face_color(self) -> Styled[str | None]:
        return self._face_color

    @face_color.setter
    def face_color(self, face_color: Styled[str | None]) -> None:
        self._face_color = _stage_live_value(self, "_face_color", face_color)

    @property
    def edge_color(self) -> Styled[str | None]:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: Styled[str | None]) -> None:
        self._edge_color = _stage_live_value(self, "_edge_color", edge_color)

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
        self._marker_style = _stage_live_value(self, "_marker_style", marker_style)

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
        self._font_size = _stage_live_value(self, "_font_size", font_size)

    @property
    def text_color(self) -> Styled[str]:
        return self._text_color

    @text_color.setter
    def text_color(self, text_color: Styled[str]) -> None:
        self._text_color = _stage_live_value(self, "_text_color", text_color)

    @property
    def h_align(self) -> str:
        return self._h_align

    @h_align.setter
    def h_align(self, h_align: str) -> None:
        self._h_align = _stage_live_value(self, "_h_align", h_align)

    @property
    def v_align(self) -> str:
        return self._v_align

    @v_align.setter
    def v_align(self, v_align: str) -> None:
        self._v_align = _stage_live_value(self, "_v_align", v_align)

    @property
    def show_coordinates(self) -> bool:
        return self._show_coordinates

    @show_coordinates.setter
    def show_coordinates(self, show_coordinates: bool) -> None:
        self._show_coordinates = _stage_live_value(
            self, "_show_coordinates", show_coordinates
        )

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

    def _caption(self) -> str:
        text = self._label
        if text is not None and not isinstance(text, str):
            raise InvalidParameterTypeError("A Point label must be a string or None.")
        if self._show_coordinates:
            coordinates = f"({self._x:.3f}, {self._y:.3f})"
            text = coordinates if text is None else f"{text} : {coordinates}"
        if text is None:
            return ""
        prefix = " " if self._h_align == "left" else ""
        suffix = " " if self._h_align == "right" else ""
        return f"{prefix}{text}{suffix}"

    @staticmethod
    def _annotation_color(color: Any, face: Any, edge: Any) -> Any:
        if isinstance(color, str) and color == "same as point":
            if edge is None or isinstance(edge, str) and edge == "face":
                return face
            return edge
        return color

    def _bind_live_point(
        self,
        session: _AnimationSession,
        marker: PathCollection,
        annotation: Annotation,
    ) -> None:
        figure_font_size = plt.rcParams["font.size"]
        scatter_style = plt.rcParams["scatter.marker"]
        initial_style = resolve_or(self._marker_style, scatter_style)
        initial_face = self._face_color
        if is_inherit(initial_face):
            initial_face = marker.get_facecolor()
            if not len(initial_face):
                initial_face = marker.get_edgecolor()
            initial_face = initial_face[0] if len(initial_face) else "none"
        initial_edge = resolve_or(self._edge_color, plt.rcParams["scatter.edgecolors"])
        initial_text_color = resolve_or(self._text_color, annotation.get_color())

        def marker_colors(style: MarkerStyle, face: Any, edge: Any) -> tuple[Any, Any]:
            if face is None and edge is None:
                raise IncompatibleArgumentsError(
                    "A point's face_color and edge_color cannot both be None; set at least one of them."
                )
            face = "none" if face is None else face
            edge = "none" if edge is None else edge
            if style.is_filled():
                return face, edge
            if style.get_fillstyle() == "none":
                return "none", face
            return face, "face"

        active_style = MarkerStyle(initial_style)
        active_colors = marker_colors(active_style, initial_face, initial_edge)
        active = marker.get_paths(), *active_colors

        def prepare_marker() -> _ArtistEdit:
            style = MarkerStyle(resolve_or(self._marker_style, initial_style))
            path = style.get_path().transformed(style.get_transform())
            colors = marker_colors(
                style,
                resolve_or(self._face_color, initial_face),
                resolve_or(self._edge_color, initial_edge),
            )

            def set_marker(state) -> None:
                nonlocal active
                paths, face, edge = state
                marker.set_paths(paths)
                marker.set_facecolor(face)
                marker.set_edgecolor(edge)
                active = state

            return _edit(set_marker, active, ([path], *colors))

        def prepare_text_color() -> _ArtistEdit:
            color = self._annotation_color(
                resolve_or(self._text_color, initial_text_color),
                resolve_or(self._face_color, initial_face),
                resolve_or(self._edge_color, initial_edge),
            )
            return _edit(annotation.set_color, annotation.get_color(), color)

        def prepare_position(position: tuple[float, float]) -> _ArtistEdit:
            if not all(isfinite(coord) for coord in position):
                raise InvalidParameterError(
                    "Animated Point coordinates must be finite."
                )

            def move(xy: tuple[float, float]) -> None:
                annotation.xy = xy
                annotation.set_position(xy)

            return _combine_edits(
                [
                    _edit(marker.set_offsets, marker.get_offsets(), [position]),
                    _edit(move, annotation.xy, position),
                ]
            )

        def prepare_caption() -> _ArtistEdit:
            if not isinstance(self._show_coordinates, (bool, np.bool_)):
                raise InvalidParameterTypeError(
                    "Point.show_coordinates must be a boolean."
                )
            return _edit(annotation.set_text, annotation.get_text(), self._caption())

        session.bind_update(self, "coordinates", prepare_position)
        session.bind_derived(
            self,
            ("coordinates", "_label", "_show_coordinates", "_h_align"),
            prepare_caption,
        )
        session.bind_derived(
            self, ("_marker_style", "_face_color", "_edge_color"), prepare_marker
        )
        session.bind_derived(
            self, ("_text_color", "_face_color", "_edge_color"), prepare_text_color
        )
        session.bind_property(
            self, "_marker_size", marker, "sizes", lambda value: [value]
        )
        session.bind_property(self, "_edge_width", marker, "linewidth")
        session.bind_property(self, "_alpha", marker, "alpha")
        session.bind_property(
            self,
            "_font_size",
            annotation,
            "fontsize",
            lambda value: figure_font_size if value == "same as figure" else value,
        )
        session.bind_property(self, "_h_align", annotation, "horizontalalignment")
        session.bind_property(self, "_v_align", annotation, "verticalalignment")

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
        session = _DRAW_SESSION.get()
        size = self._font_size if self._font_size != "same as figure" else None
        params = {
            "c": self._face_color if self._face_color is not None else "none",
            "edgecolors": self._edge_color if self._edge_color is not None else "none",
            "s": self._marker_size,
            "marker": self._marker_style,
            "linewidths": self._edge_width,
            "alpha": self._alpha,
        }
        marker = axes.scatter(self._x, self._y, zorder=z_order, **strip_inherit(params))
        params = strip_inherit(
            {
                "color": self._annotation_color(
                    self._text_color, self._face_color, self._edge_color
                ),
                "fontsize": size,
                "horizontalalignment": self._h_align,
                "verticalalignment": self._v_align,
            }
        )
        if session is None and self._show_coordinates:
            # Keep the empty label slot created by the static plotting path.
            axes.annotate("", self.coordinates, zorder=z_order, **params)
        label = axes.annotate(
            self._caption(), self.coordinates, zorder=z_order, **params
        )
        if session is not None:
            self._bind_live_point(session, marker, label)


class _LiveTextAnnotation(Annotation):
    """Use one text artist and clip only its optional arrow at the target."""

    def _check_xy(self, renderer=None) -> bool:
        if self.arrow_patch is not None:
            if self.arrowprops is None:
                self.arrow_patch.set_visible(False)
            else:
                self.arrow_patch.set_visible(super()._check_xy(renderer))  # ty: ignore[unresolved-attribute]
        return True

    def get_window_extent(self, renderer=None):
        # The Text extent is independent of the optional arrow and is already
        # in data coordinates before the first render.
        return TextArtist.get_window_extent(self, renderer)


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
        self._arrow_properties: dict[str, float] = {}

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
        self._highlight_color = _stage_live_value(
            self, "_highlight_color", highlight_color
        )

    @property
    def highlight_alpha(self) -> float:
        return self._highlight_alpha

    @highlight_alpha.setter
    def highlight_alpha(self, highlight_alpha: float) -> None:
        self._highlight_alpha = _stage_live_value(
            self, "_highlight_alpha", highlight_alpha
        )

    @property
    def highlight_padding(self) -> float:
        return self._highlight_padding

    @highlight_padding.setter
    def highlight_padding(self, highlight_padding: float) -> None:
        self._highlight_padding = _stage_live_value(
            self, "_highlight_padding", highlight_padding
        )

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
        properties = {
            key: value
            for key, value in (
                ("width", width),
                ("shrink", shrink),
                ("headwidth", head_width),
                ("headlength", head_length),
                ("alpha", alpha),
            )
            if value is not None
        }
        self._arrow_properties = _stage_live_value(
            self, "_arrow_properties", properties
        )
        self.arrow_pointing_to = points_to

    def _bind_live_text(self, session: _AnimationSession, artist: TextArtist) -> None:
        session.bind_text(self, artist)
        figure_font_size = plt.rcParams["font.size"]
        session.bind_property(
            self,
            "_font_size",
            artist,
            "fontsize",
            lambda value: figure_font_size if value == "same as figure" else value,
        )
        for name, mpl_name in (
            ("_color", "color"),
            ("_alpha", "alpha"),
            ("_h_align", "horizontalalignment"),
            ("_v_align", "verticalalignment"),
            ("_rotation", "rotation"),
        ):
            session.bind_property(self, name, artist, mpl_name)

        box = artist.get_bbox_patch()
        assert box is not None
        box.set_visible(self._highlight_color is not None)

        def prepare_highlight() -> _ArtistEdit:
            color, alpha, padding = (
                self._highlight_color,
                self._highlight_alpha,
                self._highlight_padding,
            )
            if not isinstance(padding, Real) or not isfinite(padding) or padding < 0:
                raise InvalidParameterError(
                    "Text highlight padding must be finite and nonnegative."
                )
            return _combine_edits(
                [
                    _edit(
                        box.set_facecolor,
                        box.get_facecolor(),
                        "none" if color is None else color,
                    ),
                    _edit(box.set_alpha, box.get_alpha(), alpha),
                    _edit(
                        box.set_boxstyle, box.get_boxstyle(), f"square,pad={padding}"
                    ),
                    _edit(box.set_visible, box.get_visible(), color is not None),
                ]
            )

        session.bind_derived(
            self,
            ("_highlight_color", "_highlight_alpha", "_highlight_padding"),
            prepare_highlight,
        )

    def _bind_live_text_arrow(
        self, session: _AnimationSession, annotation: _LiveTextAnnotation
    ) -> None:
        initial_color = annotation.get_color()
        patch = annotation.arrow_patch
        assert patch is not None

        def prepare_arrow() -> _ArtistEdit:
            target = self._arrow_pointing_to
            if target is not None and (
                len(target) != 2
                or not all(
                    isinstance(value, Real) and isfinite(value) for value in target
                )
            ):
                raise InvalidParameterError(
                    "An animated Text arrow needs a finite target."
                )
            options = self._arrow_properties
            for name, value in options.items():
                if not isinstance(value, Real) or not isfinite(value) or value < 0:
                    raise InvalidParameterError(
                        "Text arrow options must be finite and nonnegative."
                    )
                if (name == "shrink" and value > 0.5) or (
                    name == "alpha" and value > 1
                ):
                    raise InvalidParameterError(
                        "Text arrow shrink must be at most 0.5 and alpha at most 1."
                    )
            return _combine_edits(
                [
                    _edit(
                        lambda value: setattr(annotation, "xy", value),
                        annotation.xy,
                        (self._x, self._y) if target is None else tuple(target),
                    ),
                    _edit(
                        lambda value: setattr(annotation, "arrowprops", value),
                        annotation.arrowprops,
                        None if target is None else dict(options),
                    ),
                    _edit(
                        patch.set_color,
                        patch.get_facecolor(),
                        resolve_or(self._color, initial_color),
                    ),
                    _edit(patch.set_alpha, patch.get_alpha(), options.get("alpha")),
                    _edit(
                        patch.set_visible,
                        patch.get_visible(),
                        target is not None,
                    ),
                ]
            )

        session.bind_derived(
            self, ("_arrow_pointing_to", "_arrow_properties", "_color"), prepare_arrow
        )

    def _plot_element(self, axes: plt.Axes | MPLFigure, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified target, which can be either an
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_ or a
        `Figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html>`.
        """
        session = _DRAW_SESSION.get()
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
        if self._highlight_color is not None or session is not None:
            bbox_dict = {
                "boxstyle": f"square,pad={self._highlight_padding}",
                "facecolor": "none"
                if self._highlight_color is None
                else self._highlight_color,
                "edgecolor": "none",
                "alpha": self._highlight_alpha,
            }
            params["bbox"] = bbox_dict

        params = strip_inherit(params)
        if session is not None and isinstance(axes, plt.Axes):
            target = self._arrow_pointing_to
            position = self._x, self._y
            arrow_options: dict[str, Any] = dict(self._arrow_properties)
            arrow_options["color"] = resolve_or(self._color, plt.rcParams["text.color"])
            annotation = _LiveTextAnnotation(
                self._text,
                position if target is None else target,
                xytext=position,
                zorder=z_order,
                clip_on=False,
                arrowprops=arrow_options,
                **params,
            )
            annotation.set_transform(axes.transData)
            if target is None:
                annotation.arrowprops = None
                assert annotation.arrow_patch is not None
                annotation.arrow_patch.set_visible(False)
            axes.add_artist(annotation)
            self._bind_live_text(session, annotation)
            self._bind_live_text_arrow(session, annotation)
            return

        artist = axes.text(self._x, self._y, self._text, zorder=z_order, **params)
        if session is not None:
            self._bind_live_text(session, artist)
            return
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
            axes.annotate(
                self._text,
                self._arrow_pointing_to,
                xytext=(self._x, self._y),
                zorder=z_order,
                **params,
            )


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
        List of column labels. Set both these labels and col_colors to None to omit the column header.
    col_widths : list[float], optional
        Widths to apply to the columns. Must be a list the same length as the number of columns.
    col_align : str
        Alignment of the column labels' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    col_colors : ArrayLike, str or None, optional
        Colors to apply to the column labels' background. Must be a list of colors the same
        length as the number of columns.
        Default depends on the ``figure_style`` configuration.
    row_labels : list[str], optional
        List of row labels. Set both these labels and row_colors to None to omit the row header.
    row_align : str
        Alignment of the row labels' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    row_colors : ArrayLike, str or None, optional
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
        col_colors: ArrayLike | str | None | Inherit = INHERIT,
        row_labels: Optional[list[str]] = None,
        row_align: str | Inherit = INHERIT,
        row_colors: ArrayLike | str | None | Inherit = INHERIT,
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
            List of column labels. Set both these labels and col_colors to None to omit the column header.
        col_widths : list[float], optional
            Widths to apply to the columns. Must be a list the same length as the number of columns.
        col_align : str
            Alignment of the column labels' text. Must be one of the following:
            {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
        col_colors : ArrayLike, str or None, optional
            Colors to apply to the column labels' background. Must be a list of colors the same
            length as the number of columns.
            Default depends on the ``figure_style`` configuration
        row_labels : list[str], optional
            List of row labels. Set both these labels and row_colors to None to omit the row header.
        row_align : str
            Alignment of the row labels' text. Must be one of the following:
            {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
        row_colors : ArrayLike, str or None, optional
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
        self._cell_colors = _stage_live_value(self, "_cell_colors", cell_colors)

    @property
    def cell_align(self) -> Styled[str]:
        return self._cell_align

    @cell_align.setter
    def cell_align(self, cell_align: Styled[str]) -> None:
        self._cell_align = _stage_live_value(self, "_cell_align", cell_align)

    @property
    def col_labels(self) -> list[str] | None:
        return self._col_labels

    @col_labels.setter
    def col_labels(self, col_labels: list[str] | None) -> None:
        self._col_labels = _stage_live_value(self, "_col_labels", col_labels)

    @property
    def col_widths(self) -> list[float] | None:
        return self._col_widths

    @col_widths.setter
    def col_widths(self, col_widths: list[float] | None) -> None:
        self._col_widths = _stage_live_value(self, "_col_widths", col_widths)

    @property
    def col_align(self) -> Styled[str]:
        return self._col_align

    @col_align.setter
    def col_align(self, col_align: Styled[str]) -> None:
        self._col_align = _stage_live_value(self, "_col_align", col_align)

    @property
    def col_colors(self) -> Styled[ArrayLike | str | None]:
        return self._col_colors

    @col_colors.setter
    def col_colors(self, col_colors: Styled[ArrayLike | str | None]) -> None:
        self._col_colors = _stage_live_value(self, "_col_colors", col_colors)

    @property
    def row_labels(self) -> list[str] | None:
        return self._row_labels

    @row_labels.setter
    def row_labels(self, row_labels: list[str] | None) -> None:
        self._row_labels = _stage_live_value(self, "_row_labels", row_labels)

    @property
    def row_align(self) -> Styled[str]:
        return self._row_align

    @row_align.setter
    def row_align(self, row_align: Styled[str]) -> None:
        self._row_align = _stage_live_value(self, "_row_align", row_align)

    @property
    def row_colors(self) -> Styled[ArrayLike | str | None]:
        return self._row_colors

    @row_colors.setter
    def row_colors(self, row_colors: Styled[ArrayLike | str | None]) -> None:
        self._row_colors = _stage_live_value(self, "_row_colors", row_colors)

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
        self._scaling = _stage_live_value(self, "_scaling", scaling)

    @property
    def location(self) -> str:
        return self._location

    @location.setter
    def location(self, location: str) -> None:
        self._location = _stage_live_value(self, "_location", location)

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
        # Capture the unscaled height for absolute frame-to-frame size assignments.
        session = _DRAW_SESSION.get()
        base_height = (
            next(iter(self.handle.get_celld().values())).get_height()
            if session is not None
            else 0
        )
        self.handle.auto_set_font_size(False)
        self.handle.scale(self._scaling[0], self._scaling[1])
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_text_props(color=resolve_or(self._text_color, "black"))
            cell.set_edgecolor(resolve_or(self._edge_color, "black"))
            cell.set_linewidth(resolve_or(self._edge_width, 1))

        if session is not None:
            self._bind_table(session, base_height)

    def _bind_table(self, session: _AnimationSession, base_height: float) -> None:
        table = self.handle
        cells = table.get_celld()
        backend_table = cast(Any, table)
        defaults = {
            "_cell_colors": resolve_or(self._cell_colors, "white"),
            "_col_colors": resolve_or(self._col_colors, "#bfbfbf"),
            "_row_colors": resolve_or(self._row_colors, "#bfbfbf"),
            "_cell_align": resolve_or(self._cell_align, "right"),
            "_col_align": resolve_or(self._col_align, "center"),
            "_row_align": resolve_or(self._row_align, "left"),
            "_edge_color": resolve_or(self._edge_color, "black"),
            "_edge_width": resolve_or(self._edge_width, 1),
            "_text_color": resolve_or(self._text_color, "black"),
        }
        offset = int(
            self._col_labels is not None or defaults["_col_colors"] is not None
        )
        # A header changes Matplotlib's row indices. Logical body coordinates
        # and separate header keys keep the same cell alive through that shift.
        pool = {(row - offset, column): cell for (row, column), cell in cells.items()}
        reference_cell = next(iter(cells.values()))
        initial_font_size = reference_cell.get_fontsize()

        def read(name: str):
            return resolve_or(getattr(self, name), defaults[name])

        def header(labels, colors, count, name):
            if labels is None and colors is None:
                return []
            text = [""] * count if labels is None else list(labels)
            palette = (
                ["white" if colors is None else colors] * count
                if colors is None or isinstance(colors, str)
                else list(colors)
            )
            if len(text) != count or len(palette) != count:
                raise InvalidParameterError(
                    f"Animated Table {name} labels and colors must match the final table dimensions."
                )
            return list(zip(text, palette, strict=True))

        def prepare() -> _ArtistEdit:
            grid = tuple(tuple(row) for row in self._cell_text)
            rows = len(grid)
            columns = len(grid[0]) if rows else 0
            if not columns or any(len(row) != columns for row in grid):
                raise InvalidParameterError(
                    "Animated Table cell_text must be a nonempty rectangular grid."
                )
            colors = read("_cell_colors")
            if isinstance(colors, str):
                colors = [[colors] * columns] * rows
            elif len(colors) != rows or any(len(row) != columns for row in colors):
                raise InvalidParameterError(
                    "Animated Table cell_colors must match the final table dimensions."
                )
            row_entries = header(self._row_labels, read("_row_colors"), rows, "row")
            col_entries = header(
                self._col_labels, read("_col_colors"), columns, "column"
            )
            scale = np.asarray(self._scaling, dtype=float)
            widths = np.asarray(
                self._col_widths
                if self._col_widths is not None
                else [1 / columns] * columns,
                dtype=float,
            )
            if (
                scale.shape != (2,)
                or widths.shape != (columns,)
                or not np.isfinite(scale).all()
                or not np.isfinite(widths).all()
                or (scale <= 0).any()
                or (widths <= 0).any()
            ):
                raise InvalidParameterError(
                    "Animated Table widths and scales must be finite, positive, and match the table."
                )
            widths = widths * scale[0]
            height = base_height * scale[1]
            text_color, edge, edge_width = (
                read("_text_color"),
                read("_edge_color"),
                read("_edge_width"),
            )
            column_offset = int(bool(col_entries))
            row_auto = bool(row_entries) and self._row_labels is not None
            row_width = 1e-15 if row_auto else widths[0]
            planned: dict[tuple[int, int], Cell] = {}
            added: dict[tuple[int, int], Cell] = {}
            edits = []

            def plan(key, text, face, align, width):
                row, column = key
                position = (0 if row < 0 else row + column_offset, column)
                cell = pool.get(key)
                if cell is None:
                    cell = Cell((0, 0), width, height, visible_edges=table.edges)
                    cell.set_fontsize(initial_font_size)
                    added[key] = cell
                label = cell.get_text()
                for artist, field, value in (
                    (cell, "width", width),
                    (cell, "height", height),
                    (cell, "facecolor", face),
                    (cell, "edgecolor", edge),
                    (cell, "linewidth", edge_width),
                    (label, "text", text),
                    (label, "color", text_color),
                    (label, "horizontalalignment", align),
                ):
                    edits.append(
                        _edit(
                            getattr(artist, f"set_{field}"),
                            getattr(artist, f"get_{field}")(),
                            value,
                        )
                    )
                planned[position] = cell

            body_align, col_align, row_align = (
                read("_cell_align"),
                read("_col_align"),
                read("_row_align"),
            )
            for row, entries in enumerate(grid):
                for col, text in enumerate(entries):
                    plan(
                        (row, col),
                        text,
                        colors[row][col],
                        body_align,
                        widths[col],
                    )
            for col, (text, face) in enumerate(col_entries):
                plan((-1, col), text, face, col_align, widths[col])
            for row, (text, face) in enumerate(row_entries):
                plan((row, -1), text, face, row_align, row_width)

            previous = dict(cells)
            previous_auto = backend_table._autoColumns.copy()

            def apply_layout() -> None:
                cells.clear()
                for position, cell in planned.items():
                    if cell.get_figure() is None:
                        table[position] = cell
                    else:
                        cells[position] = cell
                pool.update(added)
                backend_table._autoColumns[:] = [
                    col for col in previous_auto if col != -1
                ]
                if row_auto:
                    table.auto_set_column_width(-1)
                table.stale = True

            def undo_layout() -> None:
                cells.clear()
                cells.update(previous)
                for key in added:
                    pool.pop(key, None)
                backend_table._autoColumns[:] = previous_auto
                table.stale = True

            edits.append(_ArtistEdit(apply_layout, undo_layout))
            return _combine_edits(edits)

        session.bind_derived(
            self,
            (
                "_cell_text",
                "_col_labels",
                "_row_labels",
                "_col_widths",
                "_scaling",
                *defaults,
            ),
            prepare,
        )

        def prepare_location(value: str) -> _ArtistEdit:
            if value not in table.codes:
                raise InvalidParameterError(f"Unknown Table location {value!r}.")

            # Matplotlib uses this placement code at draw time and offers no setter.
            def set_location(code: int) -> None:
                backend_table._loc = code
                table.stale = True

            return _edit(set_location, backend_table._loc, table.codes[value])

        session.bind_update(self, "_location", prepare_location)


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
