from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from .legend_artists import *
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
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
    label: str
    color: str = "default"
    line_width: int = "default"
    
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
            linewidth=self.line_width,
            label=self.label
        )

@dataclass
class Scatter(Curve):
    """
    A general scatter plot.
    """
    marker_size: float = "default"
    edge_color: str = "default"
    marker_style: str = "default"

    def plot_curve(self, axes: plt.Axes):
        self.handle = axes.scatter(
            self.xdata,
            self.ydata,
            color=self.color,
            edgecolors=self.edge_color,
            s=self.marker_size,
            marker=self.marker_style,
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
            linewidth=self.line_width,
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
    face_color: str = "default"
    edge_color: str = "default"
    hist_type: str = "default"
    alpha: float = "default"
    line_width: int | float = "default"
    normalize: bool = "default"
    show_pdf: str = "default"
    show_params: bool = "default"
    
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
        self.create_label()
    
    def create_label(self):
        lab = self.label
        if self.label and self.show_params:
            lab += ' :\n'
        if self.show_params:
            lab += f'$\mu$ = {self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}'
        self.label = lab
    
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


@dataclass
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


