from dataclasses import dataclass

import numpy as np
from .legend_artists import VerticalLineCollection
from matplotlib.collections import LineCollection


class GraphingException(Exception):
    """
    General exception raised for the GraphingLib module.
    """
    pass


class Hlines():
    """
    Horizontal lines.
    """
    def __init__(self, y: list | np.ndarray, xmin: list | np.ndarray, xmax: list | np.ndarray,
                 label: str, colors: list[str] | str = "default", line_styles: list[str] | str = "default"):
        self.y = y
        self.xmin = xmin
        self.xmax = xmax
        self.label = label
        self.colors = colors
        self.line_styles = line_styles
        if isinstance(self.y, (int, float)) and isinstance(self.colors, (list, np.ndarray)):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self.y, (int, float)) and isinstance(self.line_styles, (list, np.ndarray)):
            raise GraphingException("There can't be multiple linestyles for a single line!")
        if isinstance(self.y, (list, np.ndarray)) and isinstance(self.colors, list)\
                                                    and isinstance(self.line_styles, list):
            if len(self.y) != len(self.colors) or len(self.y) != len(self.line_styles):
                raise GraphingException("There must be the same number of colors, "
                                        + "linestyles and lines!")

    def plot_curve(self, axes):
        if isinstance(self.y, list) and len(self.y) > 1:
            axes.hlines(
                self.y,
                self.xmin,
                self.xmax,
                colors=self.colors,
                linestyles=self.line_styles
            )
            self.handle = LineCollection(
                [[(0,0)]] * (len(self.y) if len(self.y) <= 3 else 3),
                color=self.colors,
                linestyle='solid'
            )
        else:
            self.handle = axes.hlines(
                self.y,
                self.xmin,
                self.xmax,
                colors=self.colors,
                linestyles=self.line_styles
            )


class Vlines():
    """
    Vertical lines.
    """
    def __init__(self, x: list | np.ndarray, ymin: list | np.ndarray, ymax: list | np.ndarray,
                 label: str, colors: list[str] | str = "default", line_styles: list[str] | str = "default"):
        self.x = x
        self.ymin = ymin
        self.ymax = ymax
        self.label = label
        self.colors = colors
        self.line_styles = line_styles
        if isinstance(self.x, (int, float)) and isinstance(self.colors, (list, np.ndarray)):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self.x, (int, float)) and isinstance(self.line_styles, (list, np.ndarray)):
            raise GraphingException("There can't be multiple linestyles for a single line!")
        if isinstance(self.x, (list, np.ndarray)) and isinstance(self.colors, list)\
                                                    and isinstance(self.line_styles, list):
            if len(self.x) != len(self.colors) or len(self.x) != len(self.line_styles):
                raise GraphingException("There must be the same number of colors, "
                                        + "linestyles and lines!")

    def plot_curve(self, axes):
        if isinstance(self.x, list) and len(self.x) > 1:
            axes.vlines(
                self.x,
                self.ymin,
                self.ymax,
                colors=self.colors,
                linestyles=self.line_styles
            )
            self.handle = VerticalLineCollection(
                [[(0,0)]] * (len(self.x) if len(self.x) <= 4 else 4),
                color=self.colors,
                linestyle='solid'
            )
        else:
            self.handle = axes.vlines(
                self.x,
                self.ymin,
                self.ymax,
                colors=self.colors,
                linestyles=self.line_styles
            )


class Point():
    """
    A general point object.
    """
    def __init__(self, x: int | float, y: int | float, label: str = "", color: str = "default",
                 edge_color: str = "default", marker_size: int | float = "default",
                 marker_style: str = "default", line_width: int | float = "default",
                 lines_to_axis: bool = True, show_coordinates: bool = False):
        if not isinstance(x, int|float) or not isinstance(y, int|float):
            raise GraphingException("The x and y coordinates for a point must be a single number each!")
        else:
            self.x = x
            self.y = y
        self.label = label
        self.color = color
        self.edge_color = edge_color
        self.marker_size = marker_size
        self.marker_style = marker_style
        self.line_width = line_width
        self.lines_to_axis = lines_to_axis
        self.show_coordinates = show_coordinates

    def plot_curve(self, axes):
        axes.scatter(self.x, self.y, c=self.color, edgecolors=self.edge_color, s=self.marker_size,
                     marker=self.marker_style, linewidths=self.line_width)
        if self.lines_to_axis:
            self.add_lines_to_axis(axes)
        if self.show_coordinates:
            self.add_coordinates(axes)

    def add_lines_to_axis(self, axes):
        axes.hlines(self.y, axes.get_xlim()[0], self.x, linestyle=':', color='k', zorder=0)
        axes.vlines(self.x, axes.get_ylim()[0], self.y, linestyle=':', color='k', zorder=0)

    def add_coordinates(self):
        raise NotImplementedError


@dataclass
class Text():
    """
    A text object to be displayed on a graph.
    """
    x: int | float
    y: int | float
    text: str
    color: str = "k"
    size: int | float = 10
    add_arrow: bool = False
    arrow_pointing_to: tuple[float] = None

    def plot_curve(self, axes):
        axes.text(self.x, self.y, self.text)
        if self.add_arrow:
            self.add_arrow_pointing_to(self.arrow_pointing_to)

    def add_arrow_pointing_to(self, points_to):
        raise NotImplementedError
