from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import ArrayLike

from .legend_artists import VerticalLineCollection

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class Plottable(Protocol):
    """
    Dummy class for a general plottable object.

    .. attention:: Not to be used directly.

    """

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        Axes
        """
        pass


class GraphingException(Exception):
    """
    General exception raised for the GraphingLib modules.
    """

    pass


class Hlines:
    """
    This class implements simple horizontal lines.

    Parameters
    ----------
    y : ArrayLike
        Vertical positions at which the lines should be plotted.
    x_min, x_max : ArrayLike
        Horizontal positions at which the lines should start and end. Different
        positions can be specified for each line.
    label : str, optional
        Label to be displayed in the legend.
    colors : list[str]
        Colors to use for the lines. One color for every line or a color
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_widths : list[float]
        Line widths to use for the lines. One width for every line or a width
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_styles : list[str]
        Line styles to use for the lines. One style for every line or a style
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        y: ArrayLike,
        x_min: ArrayLike,
        x_max: ArrayLike,
        label: Optional[str] = None,
        colors: list[str] | str = "default",
        line_widths: list[float] | float = "default",
        line_styles: list[str] | str = "default",
    ) -> None:
        """
        This class implements simple horizontal lines.

        Parameters
        ----------
        y : ArrayLike
            Vertical positions at which the lines should be plotted.
        x_min, x_max : ArrayLike
            Horizontal positions at which the lines should start and end. Different
            positions can be specified for each line.
        label : str, optional
            Label to be displayed in the legend.
        colors : list[str]
            Colors to use for the lines. One color for every line or a color
            per line can be specified.
            Default depends on the ``figure_style`` configuration.
        line_widths : list[float]
            Line widths to use for the lines. One width for every line or a width
            per line can be specified.
            Default depends on the ``figure_style`` configuration.
        line_styles : list[str]
            Line styles to use for the lines. One style for every line or a style
            per line can be specified.
            Default depends on the ``figure_style`` configuration.
        """
        if isinstance(y, (int, float)):
            self._y = y
        elif isinstance(y, (list, np.ndarray)):
            self._y = np.array(y)
        if isinstance(x_min, (int, float)):
            self._x_min = x_min
        elif isinstance(x_min, (list, np.ndarray)):
            self._x_min = np.array(x_min)
        if isinstance(x_max, (int, float)):
            self._x_max = x_max
        elif isinstance(x_max, (list, np.ndarray)):
            self._x_max = np.array(x_max)
        self._label = label
        self._colors = colors
        self._line_widths = line_widths
        self._line_styles = line_styles
        if isinstance(self._y, (int, float)) and isinstance(
            self._colors, (list, np.ndarray)
        ):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self._y, (int, float)) and isinstance(
            self._line_styles, (list, np.ndarray)
        ):
            raise GraphingException(
                "There can't be multiple line styles for a single line!"
            )
        if isinstance(self._y, (int, float)) and isinstance(
            self._line_widths, (list, np.ndarray)
        ):
            raise GraphingException(
                "There can't be multiple line widths for a single line!"
            )
        if (
            isinstance(self._y, (list, np.ndarray))
            and isinstance(self._colors, list)
            and isinstance(self._line_styles, list)
            and isinstance(self._line_widths, list)
        ):
            if (
                len(self._y) != len(self._colors)
                or len(self._y) != len(self._line_styles)
                or len(self._y) != len(self._line_widths)
            ):
                raise GraphingException(
                    "There must be the same number of colors, "
                    + "line styles, line widths and lines!"
                )

    @property
    def y(self) -> ArrayLike:
        return self._y

    @y.setter
    def y(self, y: ArrayLike) -> None:
        self._y = y

    @property
    def x_min(self) -> ArrayLike:
        return self._x_min

    @x_min.setter
    def x_min(self, x_min: ArrayLike) -> None:
        self._x_min = x_min

    @property
    def x_max(self) -> ArrayLike:
        return self._x_max

    @x_max.setter
    def x_max(self, x_max: ArrayLike) -> None:
        self._x_max = x_max

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = label

    @property
    def colors(self) -> list[str] | str:
        return self._colors

    @colors.setter
    def colors(self, colors: list[str] | str) -> None:
        self._colors = colors

    @property
    def line_widths(self) -> list[float] | float:
        return self._line_widths

    @line_widths.setter
    def line_widths(self, line_widths: list[float] | float) -> None:
        self._line_widths = line_widths

    @property
    def line_styles(self) -> list[str] | str:
        return self._line_styles

    @line_styles.setter
    def line_styles(self, line_styles: list[str] | str) -> None:
        self._line_styles = line_styles

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Hlines` object.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        Axes
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if isinstance(self._y, list) and len(self._y) > 1:
            params = {
                "colors": self._colors,
                "linestyles": self._line_styles,
                "linewidths": self._line_widths,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            axes.hlines(
                self._y,
                self._x_min,
                self._x_max,
                zorder=z_order,
                **params,
            )
            params.pop("linewidths")
            self.handle = LineCollection(
                [[(0, 0)]] * (len(self._y) if len(self._y) <= 3 else 3),
                **params,
            )
        else:
            params = {
                "colors": self._colors,
                "linestyles": self._line_styles,
                "linewidths": self._line_widths,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            if isinstance(self._y, (int, float)):
                self.handle = LineCollection(
                    [[(0, 0)]] * 1,
                    **params,
                )
            else:
                self.handle = LineCollection(
                    [[(0, 0)]] * (len(self._y) if len(self._y) <= 3 else 3),
                    **params,
                )
            axes.hlines(
                self._y,
                self._x_min,
                self._x_max,
                zorder=z_order,
                **params,
            )


class Vlines:
    """
    This class implements simple vertical lines.

    Parameters
    ----------
    x : ArrayLike
        Horizontal positions at which the lines should be plotted.
    y_min, y_max : ArrayLike
        Vertical positions at which the lines should start and end. Different
        positions can be specified for each line.
    label : str, optional
        Label to be displayed in the legend.
    colors : list[str]
        Colors to use for the lines. One color for every line or a color
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_widths : list[float]
        Line widths to use for the lines. One width for every line or a width
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    line_styles : list[str]
        Line styles to use for the lines. One style for every line or a style
        per line can be specified.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        x: ArrayLike,
        y_min: ArrayLike,
        y_max: ArrayLike,
        label: Optional[str] = None,
        colors: list[str] | str = "default",
        line_widths: list[float] | float = "default",
        line_styles: list[str] | str = "default",
    ) -> None:
        """
        This class implements simple vertical lines.

        Parameters
        ----------
        x : ArrayLike
            Horizontal positions at which the lines should be plotted.
        y_min, y_max : ArrayLike
            Vertical positions at which the lines should start and end. Different
            positions can be specified for each line.
        label : str, optional
            Label to be displayed in the legend.
        colors : list[str]
            Colors to use for the lines. One color for every line or a color
            per line can be specified.
            Default depends on the ``figure_style`` configuration.
        line_widths : list[float]
            Line widths to use for the lines. One width for every line or a width
            per line can be specified.
            Default depends on the ``figure_style`` configuration.
        line_styles : list[str]
            Line styles to use for the lines. One style for every line or a style
            per line can be specified.
            Default depends on the ``figure_style`` configuration.
        """
        if isinstance(x, (int, float)):
            self._x = x
        elif isinstance(x, (list, np.ndarray)):
            self._x = np.array(x)
        if isinstance(y_min, (int, float)):
            self._y_min = y_min
        elif isinstance(y_min, (list, np.ndarray)):
            self._y_min = np.array(y_min)
        if isinstance(y_max, (int, float)):
            self._y_max = y_max
        elif isinstance(y_max, (list, np.ndarray)):
            self._y_max = np.array(y_max)
        self._label = label
        self._colors = colors
        self._line_styles = line_styles
        self._line_widths = line_widths
        if isinstance(self._x, (int, float)) and isinstance(
            self._colors, (list, np.ndarray)
        ):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self._x, (int, float)) and isinstance(
            self._line_styles, (list, np.ndarray)
        ):
            raise GraphingException(
                "There can't be multiple line styles for a single line!"
            )
        if isinstance(self._x, (int, float)) and isinstance(
            self._line_widths, (list, np.ndarray)
        ):
            raise GraphingException(
                "There can't be multiple line widths for a single line!"
            )
        if (
            isinstance(self._x, (list, np.ndarray))
            and isinstance(self._colors, list)
            and isinstance(self._line_styles, list)
            and isinstance(self._line_widths, list)
        ):
            if (
                len(self._x) != len(self._colors)
                or len(self._x) != len(self._line_styles)
                or len(self._x) != len(self._line_widths)
            ):
                raise GraphingException(
                    "There must be the same number of colors, "
                    + "line styles, line widths and lines!"
                )

    @property
    def x(self) -> ArrayLike:
        return self._x

    @x.setter
    def x(self, x: ArrayLike) -> None:
        self._x = x

    @property
    def y_min(self) -> ArrayLike:
        return self._y_min

    @y_min.setter
    def y_min(self, y_min: ArrayLike) -> None:
        self._y_min = y_min

    @property
    def y_max(self) -> ArrayLike:
        return self._y_max

    @y_max.setter
    def y_max(self, y_max: ArrayLike) -> None:
        self._y_max = y_max

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = label

    @property
    def colors(self) -> list[str] | str:
        return self._colors

    @colors.setter
    def colors(self, colors: list[str] | str) -> None:
        self._colors = colors

    @property
    def line_widths(self) -> list[float] | float:
        return self._line_widths

    @line_widths.setter
    def line_widths(self, line_widths: list[float] | float) -> None:
        self._line_widths = line_widths

    @property
    def line_styles(self) -> list[str] | str:
        return self._line_styles

    @line_styles.setter
    def line_styles(self, line_styles: list[str] | str) -> None:
        self._line_styles = line_styles

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
        if isinstance(self._x, list) and len(self._x) > 1:
            params = {
                "colors": self._colors,
                "linestyles": self._line_styles,
                "linewidths": self._line_widths,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            axes.vlines(
                self._x,
                self._y_min,
                self._y_max,
                zorder=z_order,
                **params,
            )
            params.pop("linewidths")
            self.handle = VerticalLineCollection(
                [[(0, 0)]] * (len(self._x) if len(self._x) <= 4 else 4),
                **params,
            )
        else:
            params = {
                "colors": self._colors,
                "linestyles": self._line_styles,
                "linewidths": self._line_widths,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            self.handle = VerticalLineCollection(
                [[(0, 0)]] * (len(self._x) if len(self._x) <= 4 else 4),
                **params,
            )
            axes.vlines(
                self._x,
                self._y_min,
                self._y_max,
                zorder=z_order,
                **params,
            )


class Point:
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
    color : str or None
        Face color of the marker.
        Default depends on the ``figure_style`` configuration.
    edge_color : str or None
        Edge color of the marker.
        Default depends on the ``figure_style`` configuration.
    marker_size : float
        Size of the marker.
        Default depends on the ``figure_style`` configuration.
    marker_style : str
        Style of the marker.
        Default depends on the ``figure_style`` configuration.
    edge_width : float
        Edge width of the marker.
        Default depends on the ``figure_style`` configuration.
    font_size : float
        Font size for the text attached to the marker.
        Default depends on the ``figure_style`` configuration.
    text_color : str
        Color of the text attached to the marker.
        "same as point" uses the color of the point (prioritize edge color, then face color). Default depends on the ``figure_style`` configuration.
    h_align, v_align : str
        Horizontal and vertical alignment of the text attached
        to the :class:`~graphinglib.graph_elements.Point`.
        Defaults to bottom left.
    """

    def __init__(
        self,
        x: float,
        y: float,
        label: Optional[str] = None,
        color: Optional[str] = "default",
        edge_color: Optional[str] = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        edge_width: float | Literal["default"] = "default",
        font_size: int | Literal["same as figure"] = "same as figure",
        text_color: str = "default",
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
        color : str or None
            Face color of the marker.
            Default depends on the ``figure_style`` configuration.
        edge_color : str or None
            Edge color of the marker.
            Default depends on the ``figure_style`` configuration.
        marker_size : float
            Size of the marker.
            Default depends on the ``figure_style`` configuration.
        marker_style : str
            Style of the marker.
            Default depends on the ``figure_style`` configuration.
        edge_width : float
            Edge width of the marker.
            Default depends on the ``figure_style`` configuration.
        font_size : float
            Font size for the text attached to the marker.
            Default depends on the ``figure_style`` configuration.
        text_color : str
            Color of the text attached to the marker.
            "same as point" uses the color of the point (prioritize edge color, then face color). Default depends on the ``figure_style`` configuration.
        h_align, v_align : str
            Horizontal and vertical alignment of the text attached
            to the :class:`~graphinglib.graph_elements.Point`.
            Defaults to bottom left.
        """
        if not isinstance(x, int | float) or not isinstance(y, int | float):
            raise GraphingException(
                "The x and y coordinates for a point must be a single number each!"
            )
        else:
            self._x = x
            self._y = y
        self._label = label
        self._color = color
        self._edge_color = edge_color
        self._marker_size = marker_size
        self._marker_style = marker_style
        self._edge_width = edge_width
        self._font_size = font_size
        self._text_color = text_color
        self._h_align = h_align
        self._v_align = v_align
        self._show_coordinates: bool = False

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._x = x

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._y = y

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, label: Optional[str]) -> None:
        self._label = label

    @property
    def color(self) -> str | None:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color

    @property
    def edge_color(self) -> str | None:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color

    @property
    def marker_size(self) -> float | Literal["default"]:
        return self._marker_size

    @marker_size.setter
    def marker_size(self, marker_size: float | Literal["default"]) -> None:
        self._marker_size = marker_size

    @property
    def marker_style(self) -> str:
        return self._marker_style

    @marker_style.setter
    def marker_style(self, marker_style: str) -> None:
        self._marker_style = marker_style

    @property
    def edge_width(self) -> float | Literal["default"]:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: float | Literal["default"]) -> None:
        self._edge_width = edge_width

    @property
    def font_size(self) -> float | Literal["same as figure"]:
        return self._font_size

    @font_size.setter
    def font_size(self, font_size: float | Literal["same as figure"]) -> None:
        self._font_size = font_size

    @property
    def text_color(self) -> str:
        return self._text_color

    @text_color.setter
    def text_color(self, text_color: str) -> None:
        self._text_color = text_color

    @property
    def h_align(self) -> str:
        return self._h_align

    @h_align.setter
    def h_align(self, h_align: str) -> None:
        self._h_align = h_align

    @property
    def v_align(self) -> str:
        return self._v_align

    @v_align.setter
    def v_align(self, v_align: str) -> None:
        self._v_align = v_align

    @property
    def show_coordinates(self) -> bool:
        return self._show_coordinates

    @show_coordinates.setter
    def show_coordinates(self, show_coordinates: bool) -> None:
        self._show_coordinates = show_coordinates

    @property
    def coordinates(self) -> tuple[float, float]:
        return (self._x, self._y)

    @coordinates.setter
    def coordinates(self, coordinates: tuple[float, float]) -> None:
        self._x, self._y = coordinates

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.graph_elements.Point` object.
        """
        return deepcopy(self)

    def add_coordinates(self) -> None:
        """
        Displays the coordinates of the :class:`~graphinglib.graph_elements.Point` next to it.
        """
        self._show_coordinates = True

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self._color is None and self._edge_color is None:
            raise GraphingException(
                "Both the face color and edge color of the point can't be None. Set at least one of them."
            )
        size = self._font_size if self._font_size != "same as figure" else None
        prefix = " " if self._h_align == "left" else ""
        postfix = " " if self._h_align == "right" else ""
        if self._label is not None and not self._show_coordinates:
            point_label = prefix + self._label + postfix
        else:
            point_label = None
        params = {
            "c": self._color if self._color is not None else "none",
            "edgecolors": self._edge_color if self._edge_color is not None else "none",
            "s": self._marker_size,
            "marker": self._marker_style,
            "linewidths": self._edge_width,
        }
        params = {k: v for k, v in params.items() if v != "default"}
        axes.scatter(
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
                text_color = self._color
        else:
            text_color = self._text_color
        params = {
            "color": text_color,
            "fontsize": size,
            "horizontalalignment": self._h_align,
            "verticalalignment": self._v_align,
        }
        params = {k: v for k, v in params.items() if v != "default"}
        axes.annotate(
            point_label,
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
                    text_color = self._color
            else:
                text_color = self._text_color
            params = {
                "color": text_color,
                "fontsize": size,
                "horizontalalignment": self._h_align,
                "verticalalignment": self._v_align,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            axes.annotate(
                point_label,
                (self._x, self._y),
                zorder=z_order,
                **params,
            )


@dataclass
class Text:
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
        Default depends on the ``figure_style`` configuration.
    h_align, v_align : str
        Horizontal and vertical alignment of the text.
        Default depends on the ``figure_style`` configuration.
    """

    _x: float
    _y: float
    _text: str
    _color: str = "default"
    _font_size: float | Literal["same as figure"] = "same as figure"
    _h_align: str = "default"
    _v_align: str = "default"
    _arrow_pointing_to: Optional[tuple[float]] = field(default=None, init=False)

    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        color: str = "default",
        font_size: float | Literal["same as figure"] = "same as figure",
        h_align: str = "default",
        v_align: str = "default",
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
            Default depends on the ``figure_style`` configuration.
        h_align, v_align : str
            Horizontal and vertical alignment of the text.
            Default depends on the ``figure_style`` configuration.
        """
        self._x = x
        self._y = y
        self._text = text
        self._color = color
        self._font_size = font_size
        self._h_align = h_align
        self._v_align = v_align
        self._arrow_pointing_to = None

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._x = x

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._y = y

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, text: str) -> None:
        self._text = text

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color

    @property
    def font_size(self) -> float | Literal["same as figure"]:
        return self._font_size

    @font_size.setter
    def font_size(self, font_size: float | Literal["same as figure"]) -> None:
        self._font_size = font_size

    @property
    def h_align(self) -> str:
        return self._h_align

    @h_align.setter
    def h_align(self, h_align: str) -> None:
        self._h_align = h_align

    @property
    def v_align(self) -> str:
        return self._v_align

    @v_align.setter
    def v_align(self, v_align: str) -> None:
        self._v_align = v_align

    @property
    def arrow_pointing_to(self) -> Optional[tuple[float]]:
        return self._arrow_pointing_to

    @arrow_pointing_to.setter
    def arrow_pointing_to(self, arrow_pointing_to: Optional[tuple[float]]) -> None:
        self._arrow_pointing_to = arrow_pointing_to

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
    ) -> None:
        """
        Adds an arrow pointing from the :class:`~graphinglib.graph_elements.Text`
        to a specified point.

        Parameters
        ----------
        points_to: tuple[float, float]
            Coordinates at which to point.
        width : float, optional
            Arrow width.
        shrink : float, optional
            Fraction of the total length of the arrow to shrink from both ends.
            A value of 0.5 means the arrow is no longer visible.
        head_width : float, optional
            Width of the head of the arrow.
        head_length : float, optional
            Length of the head of the arrow.
        """
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

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified Axes
        """
        size = self._font_size if self._font_size != "same as figure" else None
        params = {
            "color": self._color,
            "fontsize": size,
            "horizontalalignment": self._h_align,
            "verticalalignment": self._v_align,
        }
        params = {k: v for k, v in params.items() if v != "default"}
        axes.text(
            self._x,
            self._y,
            self._text,
            zorder=z_order,
            **params,
        )
        if self._arrow_pointing_to is not None:
            self._arrow_properties["color"] = self._color
            params = {
                "color": self._color,
                "fontsize": size,
                "horizontalalignment": self._h_align,
                "verticalalignment": self._v_align,
            }
            params = {k: v for k, v in params.items() if v != "default"}
            if self._color != "default":
                self._arrow_properties["color"] = self._color
                params["arrowprops"] = self._arrow_properties
            axes.annotate(
                self._text,
                self._arrow_pointing_to,
                xytext=(self._x, self._y),
                zorder=z_order,
                **params,
            )


@dataclass
class Table:
    """
    This class allows to plot a table inside a Figure or MultiFigure.

    The Table object can be used to add raw data to a figure or add supplementary
    information like output parameters for a fit or anyother operation.

    Parameters
    ----------
    cell_text : list[str]
        Text or data to be displayed in the table. The shape of the provided data
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
        Default depends on the ``figure_style`` configuration.
    edge_color : str, optional
        Color of the table's edges.
        Default depends on the ``figure_style`` configuration.
    text_color : str, optional
        Color of the text in the table.
        Default depends on the ``figure_style`` configuration.
    scaling : tuple[float], optional
        Horizontal and vertical scaling factors to apply to the table.
        Defaults to ``(1, 1.5)``.
    location : str
        Position of the table inside the axes. Must be one of the following:
        {'best', 'bottom', 'bottom left', 'bottom right', 'center', 'center left', 'center right',
        'left', 'lower center', 'lower left', 'lower right', 'right', 'top', 'top left', 'top right',
        'upper center', 'upper left', 'upper right'}
        Defaults to ``"best"``.
    """

    def __init__(
        self,
        cell_text: list[str],
        cell_colors: ArrayLike | str = "default",
        cell_align: str = "default",
        col_labels: Optional[list[str]] = None,
        col_widths: Optional[list[float]] = None,
        col_align: str = "default",
        col_colors: ArrayLike | str = "default",
        row_labels: Optional[list[str]] = None,
        row_align: str = "default",
        row_colors: ArrayLike | str = "default",
        edge_width: float | Literal["default"] = "default",
        edge_color: str = "default",
        text_color: str = "default",
        scaling: tuple[float, float] = (1.0, 1.5),
        location: str = "best",
    ) -> None:
        """
        This class allows to plot a table inside a Figure or MultiFigure.

        The Table object can be used to add raw data to a figure or add supplementary
        information like output parameters for a fit or anyother operation.

        Parameters
        ----------
        cell_text : list[str]
            Text or data to be displayed in the table. The shape of the provided data
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
            Default depends on the ``figure_style`` configuration.
        edge_color : str, optional
            Color of the table's edges.
            Default depends on the ``figure_style`` configuration.
        text_color : str, optional
            Color of the text within the table.
            Default depends on the ``figure_style`` configuration.
        scaling : tuple[float], optional
            Horizontal and vertical scaling factors to apply to the table.
            Defaults to ``(1, 1.5)``.
        location : str
            Position of the table inside the axes. Must be one of the following:
            {'best', 'bottom', 'bottom left', 'bottom right', 'center', 'center left', 'center right',
            'left', 'lower center', 'lower left', 'lower right', 'right', 'top', 'top left', 'top right',
            'upper center', 'upper left', 'upper right'}
            Defaults to ``"best"``.
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
    def cell_text(self) -> list[str]:
        return self._cell_text

    @cell_text.setter
    def cell_text(self, cell_text: list[str]) -> None:
        self._cell_text = cell_text

    @property
    def cell_colors(self) -> ArrayLike | str:
        return self._cell_colors

    @cell_colors.setter
    def cell_colors(self, cell_colors: list) -> None:
        self._cell_colors = cell_colors

    @property
    def cell_align(self) -> str:
        return self._cell_align

    @cell_align.setter
    def cell_align(self, cell_align: str) -> None:
        self._cell_align = cell_align

    @property
    def col_labels(self) -> list[str]:
        return self._col_labels

    @col_labels.setter
    def col_labels(self, col_labels: list[str]) -> None:
        self._col_labels = col_labels

    @property
    def col_widths(self) -> list[float]:
        return self._col_widths

    @col_widths.setter
    def col_widths(self, col_widths: list[float]) -> None:
        self._col_widths = col_widths

    @property
    def col_align(self) -> str:
        return self._col_align

    @col_align.setter
    def col_align(self, col_align: str) -> None:
        self._col_align = col_align

    @property
    def col_colors(self) -> ArrayLike | str:
        return self._col_colors

    @col_colors.setter
    def col_colors(self, col_colors: list) -> None:
        self._col_colors = col_colors

    @property
    def row_labels(self) -> list[str]:
        return self._row_labels

    @row_labels.setter
    def row_labels(self, row_labels: list[str]) -> None:
        self._row_labels = row_labels

    @property
    def row_align(self) -> str:
        return self._row_align

    @row_align.setter
    def row_align(self, row_align: str) -> None:
        self._row_align = row_align

    @property
    def row_colors(self) -> ArrayLike | str:
        return self._row_colors

    @row_colors.setter
    def row_colors(self, row_colors: list) -> None:
        self._row_colors = row_colors

    @property
    def edge_width(self) -> float:
        return self._edge_width

    @edge_width.setter
    def edge_width(self, edge_width: float) -> None:
        self._edge_width = edge_width
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_linewidth(self._edge_width)

    @property
    def edge_color(self) -> str:
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color: str) -> None:
        self._edge_color = edge_color
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_edgecolor(self._edge_color)

    @property
    def text_color(self) -> str:
        return self._text_color

    @text_color.setter
    def text_color(self, text_color: str) -> None:
        self._text_color = text_color
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_text_props(color=self._text_color)

    @property
    def scaling(self) -> tuple[float]:
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: tuple[float]) -> None:
        self._scaling = scaling

    @property
    def location(self) -> str:
        return self._location

    @location.setter
    def location(self, location: str) -> None:
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
        params = {k: v for k, v in params.items() if v != "default"}

        # Set colors to correct shape if they are strings
        if isinstance(self._cell_colors, str):
            self._cell_colors = [[self._cell_colors] * len(self._cell_text[0])] * len(
                self._cell_text
            )
        if isinstance(self._col_colors, str):
            self._col_colors = [self._col_colors] * len(self._cell_text[0])
        if isinstance(self._row_colors, str):
            self._row_colors = [self._row_colors] * len(self._cell_text)

        self.handle = axes.table(
            cellText=self._cell_text,
            cellColours=self._cell_colors,
            colLabels=self._col_labels,
            colWidths=self._col_widths,
            colColours=self._col_colors,
            rowLabels=self._row_labels,
            rowColours=self._row_colors,
            loc=self._location,
            zorder=z_order,
            **params,
        )
        self.handle.auto_set_font_size(False)
        self.handle.scale(self._scaling[0], self._scaling[1])
        for (i, j), cell in self.handle.get_celld().items():
            cell.set_text_props(color=self._text_color)
            cell.set_edgecolor(self._edge_color)
            cell.set_linewidth(self._edge_width)
