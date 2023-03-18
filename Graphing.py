"""
# GraphingLib

Provides a simpler way to generate graphs with Matplotlib and an inclusion of certain Scipy
functions to simplify the process of analysing data.
"""

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from Legend_artists import *
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon


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
    number_of_bins: int
    label: str
    face_color: str = "silver"
    edge_color: str = "k"
    hist_type: str = "stepfilled"
    alpha: float = 1.0
    line_width: int | float = 2
    normalize: bool = True
    show_pdf: str = None
    
    def __post_init__(self):
        self.mean = np.mean(self.xdata)
        self.standard_deviation = np.std(self.xdata)
        parameters = np.histogram(self.xdata, bins=self.number_of_bins, density=self.normalize)
        self.bin_heights, bin_edges = parameters[0], parameters[1]
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        self.bin_width = bin_width
        self.bin_centers = bin_centers
        self.bin_edges = bin_edges
        self.label = self.label + f' : $\mu$ = {self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}'
    
    def normal_normalized(self, x): 
        return (1 / (self.standard_deviation*np.sqrt(2*np.pi))) * np.exp(-0.5 * (((x-self.mean)/self.standard_deviation)**2))
    
    def normal_not_normalized(self, x):
        return sum(self.bin_heights) * self.bin_width * self.normal_normalized(x)

    def plot_curve(self, axes: plt.Axes):
        self.handle = Polygon(
            np.array([[0, 2, 2, 3, 3, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 0]]).T,
            facecolor=to_rgba(self.face_color, self.alpha),
            edgecolor=to_rgba(self.edge_color, 1),
            linewidth=1
        )
        axes.hist(
            self.xdata,
            bins=self.number_of_bins,
            facecolor=to_rgba(self.face_color, self.alpha),
            edgecolor=to_rgba(self.edge_color, 1),
            label=self.label,
            histtype=self.hist_type,
            linewidth=self.line_width,
            density=self.normalize
        )
        if self.show_pdf in ['normal', 'gaussian']:
            normal = self.normal_normalized if self.normalize else self.normal_not_normalized
            num_of_points = 500
            xdata = np.linspace(self.bin_edges[0], self.bin_edges[-1], num_of_points)
            ydata = normal(xdata)
            axes.plot(xdata, ydata, color=self.edge_color, label=self.label)
            curve_max_y = normal(self.mean)
            curve_std_y = normal(self.mean + self.standard_deviation)
            plt.vlines([self.mean - self.standard_deviation,
                        self.mean,
                        self.mean + self.standard_deviation],
                        [0, 0, 0],
                        [curve_std_y, curve_max_y, curve_std_y],
                        linestyles=['dashed'],
                        colors=['k', 'r', 'k'])


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
    def __init__(self, x_label: str = 'x axis', y_label: str = 'y axis', size: tuple = (10, 7), legend_is_boxed: bool = True):
        self.figure, self.axes = plt.subplots(figsize=size)
        self.curves = []
        self.labels = []
        self.handles = []
        self.x_axis_name = x_label
        self.y_axis_name = y_label
        self.legend_is_boxed = legend_is_boxed

    def add_curve(self, curve: Curve):
        """
        Adds a Curve object to the figure.
        """
        self.curves.append(curve)
        self.labels.append(curve.label)

    def generate_figure(self, legend=True, test=False):
        self.axes.set_xlabel(self.x_axis_name)
        self.axes.set_ylabel(self.y_axis_name)
        
        if self.curves:
            for curve in self.curves:
                curve.plot_curve(self.axes)
                self.handles.append(curve.handle)
            if legend:
                try:
                    self.axes.legend(
                        handles=self.handles,
                        labels=self.labels,
                        handleheight=1.3,
                        handler_map={
                            Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                            LineCollection: HandlerMultipleLines(),
                            VerticalLineCollection: HandlerMultipleVerticalLines()
                        },
                        frameon=self.legend_is_boxed,
                        draggable=True
                    )
                except :
                    self.axes.legend(
                        handles=self.handles,
                        labels=self.labels,
                        handleheight=1.3,
                        handler_map={
                            Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                            LineCollection: HandlerMultipleLines(),
                            VerticalLineCollection: HandlerMultipleVerticalLines()
                        },
                        frameon=self.legend_is_boxed
                    )
            if not test:
                plt.tight_layout()
                plt.show()
        else:
            raise GraphingException("No curves to be plotted!")
