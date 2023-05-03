from dataclasses import dataclass

import numpy as np
from matplotlib.collections import LineCollection

from .legend_artists import VerticalLineCollection


class GraphingException(Exception):
    """
    General exception raised for the GraphingLib module.
    """

    pass


class Hlines:
    """
    One or multiple horizontal lines.

    Use lists or ndarrays to specify different parameters for each line or simply enter the desired value (float or string) if all lines are the same or if there is only one line.

    Parameters
    ----------
        y: float, list or ndarray
            The y position of each line.
        xmin: float, list or ndarray
            The x values which represent the start of each line.
        xmax: float, list or ndarray
            The x values which represent the end of each line.
        label: string
            The name of the set of lines which will appear in the legend.
        colors: str, list or ndarray
            The color of each line.
        line_styles: str, list or ndarray
            The line style of each line.
    """

    def __init__(
        self,
        y: list | np.ndarray,
        xmin: list | np.ndarray,
        xmax: list | np.ndarray,
        label: str,
        colors: list[str] | str = "default",
        line_styles: list[str] | str = "default",
    ):
        self.y = y
        self.xmin = xmin
        self.xmax = xmax
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

    def plot_element(self, axes):
        if isinstance(self.y, list) and len(self.y) > 1:
            axes.hlines(
                self.y,
                self.xmin,
                self.xmax,
                colors=self.colors,
                linestyles=self.line_styles,
            )
            self.handle = LineCollection(
                [[(0, 0)]] * (len(self.y) if len(self.y) <= 3 else 3),
                color=self.colors,
                linestyle="solid",
            )
        else:
            self.handle = axes.hlines(
                self.y,
                self.xmin,
                self.xmax,
                colors=self.colors,
                linestyles=self.line_styles,
            )


class Vlines:
    """
    One or multiple horizontal lines.

    Use lists or ndarrays to specify different parameters for each line or simply enter the desired value (float or string) if all lines are the same or if there is only one line.

    Parameters
    ----------
        x: float, list or ndarray
            The x position of each line.
        ymin: float, list or ndarray
            The y values which represent the start of each line.
        ymax: float, list or ndarray
            The y values which represent the end of each line.
        label: string
            The name of the lines which will appear in the legend.
        colors: str, list or ndarray
            The color of each line.
        line_styles: str, list or ndarray
            The line style of each line.
    """

    def __init__(
        self,
        x: list | np.ndarray,
        ymin: list | np.ndarray,
        ymax: list | np.ndarray,
        label: str,
        colors: list[str] | str = "default",
        line_styles: list[str] | str = "default",
    ):
        self.x = x
        self.ymin = ymin
        self.ymax = ymax
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

    def plot_element(self, axes):
        if isinstance(self.x, list) and len(self.x) > 1:
            axes.vlines(
                self.x,
                self.ymin,
                self.ymax,
                colors=self.colors,
                linestyles=self.line_styles,
            )
            self.handle = VerticalLineCollection(
                [[(0, 0)]] * (len(self.x) if len(self.x) <= 4 else 4),
                color=self.colors,
                linestyle="solid",
            )
        else:
            self.handle = axes.vlines(
                self.x,
                self.ymin,
                self.ymax,
                colors=self.colors,
                linestyles=self.line_styles,
            )


class Point:
    """
    A single point which can be labeled and customized.

    Parameters
    ----------
    x: float
        The x coordinate of the point.
    y: float
        The y coordinate of the point.
    label: str
        The name of the point which will appear in the legend.
    color: str
        The fill color of the marker.
    edge_color: str
        The edge color of the marker.
    marker_size: float
        The size of the marker.
    marker_style: str
        The shape of the marker. See https://matplotlib.org/stable/api/markers_api.html for options.
    lines_to_axis: bool
        If True, dotted lines will be drawn from the point to both axes.
    show_coordinates: bool
        If True, the coordinates of the point will be displayed near the point.
    """

    def __init__(
        self,
        x: int | float,
        y: int | float,
        label: str = "",
        color: str = "default",
        edge_color: str = "default",
        marker_size: int | float = "default",
        marker_style: str = "default",
        lines_to_axis: bool = True,
        show_coordinates: bool = False,
    ):
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
        self.lines_to_axis = lines_to_axis
        self.show_coordinates = show_coordinates

    def plot_element(self, axes):
        axes.scatter(
            self.x,
            self.y,
            c=self.color,
            edgecolors=self.edge_color,
            s=self.marker_size,
            marker=self.marker_style,
        )
        if self.lines_to_axis:
            self.add_lines_to_axis(axes)
        if self.show_coordinates:
            self.add_coordinates(axes)

    def add_lines_to_axis(self, axes):
        axes.hlines(
            self.y, axes.get_xlim()[0], self.x, linestyle=":", color="k", zorder=0
        )
        axes.vlines(
            self.x, axes.get_ylim()[0], self.y, linestyle=":", color="k", zorder=0
        )

    def add_coordinates(self):
        raise NotImplementedError


@dataclass
class Text:
    """
    A text object to be displayed on a graph.

    Parameters
    ----------
    x: float
        The x coordinate of the text.
    y: float
        The y coordinate of the text.
    text: str
        The text to be displayed on the graph.
    color: str, optional
        The color of the text.
    size: float, optional
        The size of the text.
    add_arrow: bool, optional
        If True, an arrow will be drawn from the text to the specified coordinates.
    arrow_pointing_to: tuple
        The end coordinates of the arrow. Add_arrow must be True.
    """

    x: int | float
    y: int | float
    text: str
    color: str = "k"
    size: int | float = 10
    add_arrow: bool = False
    arrow_pointing_to: tuple[float] = None

    def plot_element(self, axes):
        axes.text(self.x, self.y, self.text)
        if self.add_arrow:
            self.add_arrow_pointing_to(self.arrow_pointing_to)

    def add_arrow_pointing_to(self, points_to):
        raise NotImplementedError
