from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import ArrayLike

from .legend_artists import VerticalLineCollection


class Plottable(Protocol):
    """
    Dummy class for a general plottable object.

    .. attention:: Not to be used directly.

    """

    def _plot_element(self, axes: plt.Axes, z_order: int):
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
        line_styles : list[str]
            Line styles to use for the lines. One style for every line or a style
            per line can be specified.
            Default depends on the ``figure_style`` configuration.
        """
        if isinstance(y, (int, float)):
            self.y = y
        elif isinstance(y, (list, np.ndarray)):
            self.y = np.array(y)
        if isinstance(x_min, (int, float)):
            self.x_min = x_min
        elif isinstance(x_min, (list, np.ndarray)):
            self.x_min = np.array(x_min)
        if isinstance(x_max, (int, float)):
            self.x_max = x_max
        elif isinstance(x_max, (list, np.ndarray)):
            self.x_max = np.array(x_max)
        self.label = label
        self.colors = colors
        self.line_styles = line_styles
        if isinstance(self.y, (int, float)) and isinstance(
            self.colors, (list, np.ndarray)
        ):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self.y, (int, float)) and isinstance(
            self.line_styles, (list, np.ndarray)
        ):
            raise GraphingException(
                "There can't be multiple linestyles for a single line!"
            )
        if (
            isinstance(self.y, (list, np.ndarray))
            and isinstance(self.colors, list)
            and isinstance(self.line_styles, list)
        ):
            if len(self.y) != len(self.colors) or len(self.y) != len(self.line_styles):
                raise GraphingException(
                    "There must be the same number of colors, "
                    + "linestyles and lines!"
                )

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        Axes
        """
        if isinstance(self.y, list) and len(self.y) > 1:
            axes.hlines(
                self.y,
                self.x_min,
                self.x_max,
                colors=self.colors,
                linestyles=self.line_styles,
                zorder=z_order,
            )
            self.handle = LineCollection(
                [[(0, 0)]] * (len(self.y) if len(self.y) <= 3 else 3),
                color=self.colors,
                linestyle="solid",
            )
        else:
            self.handle = axes.hlines(
                self.y,
                self.x_min,
                self.x_max,
                colors=self.colors,
                linestyles=self.line_styles,
                zorder=z_order,
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
        line_styles : list[str]
            Line styles to use for the lines. One style for every line or a style
            per line can be specified.
            Default depends on the ``figure_style`` configuration.
        """
        if isinstance(x, (int, float)):
            self.x = x
        elif isinstance(x, (list, np.ndarray)):
            self.x = np.array(x)
        if isinstance(y_min, (int, float)):
            self.y_min = y_min
        elif isinstance(y_min, (list, np.ndarray)):
            self.y_min = np.array(y_min)
        if isinstance(y_max, (int, float)):
            self.y_max = y_max
        elif isinstance(y_max, (list, np.ndarray)):
            self.y_max = np.array(y_max)
        self.label = label
        self.colors = colors
        self.line_styles = line_styles
        if isinstance(self.x, (int, float)) and isinstance(
            self.colors, (list, np.ndarray)
        ):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self.x, (int, float)) and isinstance(
            self.line_styles, (list, np.ndarray)
        ):
            raise GraphingException(
                "There can't be multiple linestyles for a single line!"
            )
        if (
            isinstance(self.x, (list, np.ndarray))
            and isinstance(self.colors, list)
            and isinstance(self.line_styles, list)
        ):
            if len(self.x) != len(self.colors) or len(self.x) != len(self.line_styles):
                raise GraphingException(
                    "There must be the same number of colors, "
                    + "linestyles and lines!"
                )

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        Axes
        """
        if isinstance(self.x, list) and len(self.x) > 1:
            axes.vlines(
                self.x,
                self.y_min,
                self.y_max,
                colors=self.colors,
                linestyles=self.line_styles,
                zorder=z_order,
            )
            self.handle = VerticalLineCollection(
                [[(0, 0)]] * (len(self.x) if len(self.x) <= 4 else 4),
                color=self.colors,
                linestyle="solid",
            )
        else:
            self.handle = axes.vlines(
                self.x,
                self.y_min,
                self.y_max,
                colors=self.colors,
                linestyles=self.line_styles,
                zorder=z_order,
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
    color : str
        Face color of the marker.
        Default depends on the ``figure_style`` configuration.
    edge_color : str
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
        Defaults to `"k"` (black).
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
        color: str = "default",
        edge_color: str = "default",
        marker_size: float | Literal["default"] = "default",
        marker_style: str = "default",
        edge_width: float | Literal["default"] = "default",
        font_size: int | Literal["same as figure"] = "same as figure",
        text_color: str = "k",
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
        color : str
            Face color of the marker.
            Default depends on the ``figure_style`` configuration.
        edge_color : str
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
            Defaults to ``"k"``.
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
            self.x = x
            self.y = y
        self.label = label
        self.color = color
        self.edge_color = edge_color
        self.marker_size = marker_size
        self.marker_style = marker_style
        self.edge_width = edge_width
        self.font_size = font_size
        self.text_color = text_color
        self.h_align = h_align
        self.v_align = v_align
        self._show_coordinates: bool = False

    def add_coordinates(self) -> None:
        """
        Displays the coordinates of the :class:`~graphinglib.graph_elements.Point` next to it.
        """
        self._show_coordinates = True

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        Axes
        """
        size = self.font_size if self.font_size != "same as figure" else None
        prefix = " " if self.h_align == "left" else ""
        postfix = " " if self.h_align == "right" else ""
        if self.label is not None and not self._show_coordinates:
            point_label = prefix + self.label + postfix
        else:
            point_label = None
        axes.scatter(
            self.x,
            self.y,
            c=self.color,
            edgecolors=self.edge_color,
            s=self.marker_size,
            marker=self.marker_style,
            linewidths=self.edge_width,
            zorder=z_order,
        )
        axes.annotate(
            point_label,
            (self.x, self.y),
            color=self.text_color,
            fontsize=size,
            horizontalalignment=self.h_align,
            verticalalignment=self.v_align,
            zorder=z_order,
        )
        if self._show_coordinates:
            prefix = " " if self.h_align == "left" else ""
            postfix = " " if self.h_align == "right" else ""
            if self.label is not None:
                point_label = (
                    prefix
                    + self.label
                    + " : "
                    + f"({self.x:.3f}, {self.y:.3f})"
                    + postfix
                )
            else:
                point_label = prefix + f"({self.x:.3f}, {self.y:.3f})" + postfix
            axes.annotate(
                point_label,
                (self.x, self.y),
                color=self.text_color,
                fontsize=size,
                horizontalalignment=self.h_align,
                verticalalignment=self.v_align,
                zorder=z_order,
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

    x: float
    y: float
    text: str
    color: str = "default"
    font_size: float | Literal["same as figure"] = "same as figure"
    h_align: str = "default"
    v_align: str = "default"
    _arrow_pointing_to: Optional[tuple[float]] = field(default=None, init=False)

    def attach_arrow(
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
        self.width = width
        self.shrink = shrink
        self.arrow_properties = {}
        if width is not None:
            self.arrow_properties["width"] = width
        if shrink is not None:
            self.arrow_properties["shrink"] = shrink
        if head_width is not None:
            self.arrow_properties["headwidth"] = head_width
        if head_length is not None:
            self.arrow_properties["headlength"] = head_length

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        Axes
        """
        size = self.font_size if self.font_size != "same as figure" else None
        axes.text(
            self.x,
            self.y,
            self.text,
            horizontalalignment=self.h_align,
            verticalalignment=self.v_align,
            color=self.color,
            fontsize=size,
            zorder=z_order,
        )
        if self._arrow_pointing_to is not None:
            self.arrow_properties["color"] = self.color
            axes.annotate(
                self.text,
                self._arrow_pointing_to,
                xytext=(self.x, self.y),
                color=self.color,
                arrowprops=self.arrow_properties,
                fontsize=size,
                horizontalalignment=self.h_align,
                verticalalignment=self.v_align,
                zorder=z_order,
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
    cell_colors : list, optional
        Colors to apply to the cells' background. Must be a list of colors the same
        shape as the cells. If none are specified, no color is applied.
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
    col_colors : list, optional
        Colors to apply to the column labels' background. Must be a list of colors the same
        length as the number of columns. If none are specified, no color is applied.
    row_labels : list[str], optional
        List of labels for the rows of the table. If none are specified, no row labels are displayed.
    row_align : str
        Alignment of the row labels' text. Must be one of the following:
        {'left', 'center', 'right'}. Default depends on the ``figure_style`` configuration.
    row_colors : list, optional
        Colors to apply to the row labels' background. Must be a list of colors the same
        length as the number of rows. If none are specified, no color is applied.
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

    cell_text: list[str]
    cell_colors: Optional[list] = None
    cell_align: str = "default"
    col_labels: Optional[list[str]] = None
    col_widths: Optional[list[float]] = None
    col_align: str = "default"
    col_colors: Optional[list] = None
    row_labels: Optional[list[str]] = None
    row_align: str = "default"
    row_colors: Optional[list] = None
    scaling: tuple[float] = (1, 1.5)
    location: str = "best"

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified Axes
        """
        table = axes.table(
            cellText=self.cell_text,
            cellColours=self.cell_colors,
            cellLoc=self.cell_align,
            colLabels=self.col_labels,
            colWidths=self.col_widths,
            colLoc=self.col_align,
            colColours=self.col_colors,
            rowLabels=self.row_labels,
            rowLoc=self.row_align,
            rowColours=self.row_colors,
            loc=self.location,
            zorder=z_order,
        )
        table.auto_set_font_size(False)
        table.scale(self.scaling[0], self.scaling[1])
