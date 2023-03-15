"""
# GraphingLib

Provides a simpler way to generate graphs with Matplotlib and an inclusion of certain Scipy
functions to simplify the process of analysing data.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerPatch
from matplotlib.colors import to_rgba
from Legend_artists import *
from matplotlib.collections import LineCollection
from typing import Callable


class GraphingException(Exception):
    """
    General exception raised for the GraphingLib module.
    """
    pass


@dataclass
class Curve:
    """
    A general continuous curve.
    """
    xdata: list | np.ndarray
    ydata: list | np.ndarray
    color: str
    label: str
    
    @classmethod
    def from_function(cls, func: Callable, xmin: float, xmax: float,
                    color: str, label: str, number_of_points: int=500):
        xdata = np.linspace(xmin, xmax, number_of_points)
        ydata = func(xdata)
        return cls(xdata, ydata, color, label)
        
    def set_color(self, color: str or list[str]):
        self.color = color

    def plot_curve(self, axes: plt.Axes):
        self.handle, = axes.plot(
            self.xdata,
            self.ydata,
            color=self.color,
            label=self.label
        )


class Scatter(Curve):
    """
    A general scatter plot.
    """
    def plot_curve(self, axes: plt.Axes):
        self.handle = axes.scatter(
            self.xdata,
            self.ydata,
            color=self.color,
            label=self.label
        )


class Dashed(Curve):
    """
    A dashed curve derived from the Curve object.
    """
    def plot_curve(self, axes: plt.Axes):
        self.handle, = axes.plot(
            self.xdata,
            self.ydata,
            color=self.color,
            label=self.label,
            linestyle='--'
        )


@dataclass
class Histogram:
    """
    A histogram plot with minor changes to the lable icon.
    """
    xdata: list | np.ndarray
    bins: int
    label: str
    face_color: str = "silver"
    edge_color: str = "k"
    hist_type: str = "stepfilled"
    alpha: float = 1.0
    line_width: int | float = 2

    def plot_curve(self, axes: plt.Axes):
        self.handle = Polygon(
            np.array([[0, 2, 2, 3, 3, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 0]]).T,
            facecolor=to_rgba(self.face_color, self.alpha),
            edgecolor=to_rgba(self.edge_color, 1),
            linewidth=1
        )
        axes.hist(
            self.xdata,
            bins=self.bins,
            facecolor=to_rgba(self.face_color, self.alpha),
            edgecolor=to_rgba(self.edge_color, 1),
            label=self.label,
            histtype=self.hist_type,
            linewidth=self.line_width
        )


@dataclass
class Hlines():
    """
    Horizontal lines.
    """
    def __init__(self, y: list | np.ndarray, xmin: list | np.ndarray, xmax: list | np.ndarray,
                    label: str, colors: list[str] | str =None, linestyles: list[str] | str ='solid'):
        self.y = y
        self.xmin = xmin
        self.xmax = xmax
        self.label = label
        self.colors = colors
        self.linestyles = linestyles
        if isinstance(self.y, (int, float)) and isinstance(self.colors, (list, np.ndarray)):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self.y, (int, float)) and isinstance(self.linestyles, (list, np.ndarray)):
            raise GraphingException("There can't be multiple linestyles for a single line!")
        if isinstance(self.y, (list, np.ndarray)) and isinstance(self.colors, list)\
                                                    and isinstance(self.linestyles, list):
            if len(self.y) != len(self.colors) or len(self.y) != len(self.linestyles):
                raise GraphingException("There must be the same number of colors, "
                                        + "linestyles and lines!")

    def plot_curve(self, axes):
        if isinstance(self.y, list) and len(self.y) > 1:
            axes.hlines(
                self.y,
                self.xmin,
                self.xmax,
                colors=self.colors,
                linestyles=self.linestyles
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
                linestyles=self.linestyles
            )


class Vlines():
    """
    Vertical lines.
    """
    def __init__(self, x: list | np.ndarray, ymin: list | np.ndarray, ymax: list | np.ndarray,
                    label: str, colors: list[str] | str =None, linestyles: list[str] | str ='solid'):
        self.x = x
        self.ymin = ymin
        self.ymax = ymax
        self.label = label
        self.colors = colors
        self.linestyles = linestyles
        if isinstance(self.x, (int, float)) and isinstance(self.colors, (list, np.ndarray)):
            raise GraphingException("There can't be multiple colors for a single line!")
        if isinstance(self.x, (int, float)) and isinstance(self.linestyles, (list, np.ndarray)):
            raise GraphingException("There can't be multiple linestyles for a single line!")
        if isinstance(self.x, (list, np.ndarray)) and isinstance(self.colors, list)\
                                                    and isinstance(self.linestyles, list):
            if len(self.x) != len(self.colors) or len(self.x) != len(self.linestyles):
                raise GraphingException("There must be the same number of colors, "
                                        + "linestyles and lines!")

    def plot_curve(self, axes):
        if isinstance(self.x, list) and len(self.x) > 1:
            axes.vlines(
                self.x,
                self.ymin,
                self.ymax,
                colors=self.colors,
                linestyles=self.linestyles
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
                linestyles=self.linestyles
            )


class Figure:
    """
    A general Matplotlib figure.
    """
    def __init__(self, size: tuple = (10, 7)):
        self.figure, self.axes = plt.subplots(figsize=size)
        self.curves = []
        self.labels = []
        self.handles = []

    def add_curve(self, curve: Curve):
        """
        Adds a Curve object to the figure.
        """
        self.curves.append(curve)
        self.labels.append(curve.label)

    def generate_figure(self, legend=True, test=False):
        if self.curves:
            for curve in self.curves:
                curve.plot_curve(self.axes)
                self.handles.append(curve.handle)
            if legend:
                self.axes.legend(
                    handles=self.handles,
                    labels=self.labels,
                    handleheight=1.3,
                    handler_map={
                        Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                        LineCollection: HandlerMultipleLines(),
                        VerticalLineCollection: HandlerMultipleVerticalLines()
                    }
                )
            if not test:
                plt.tight_layout()
                plt.show()
        else:
            raise GraphingException("No curves to be plotted!")
