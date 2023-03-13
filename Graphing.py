from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerPatch


class GraphingException(Exception):
    pass


class Figure():
    def __init__(self, size: tuple=(10,7)):
        self.figure, self.axes = plt.subplots(figsize=size)
        self.curves = []
        self.labels = []
        self.handles = []
    
    def add_curve(self, curve: list):
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
                    handler_map={Polygon:HandlerPatch(patch_func=histogram_legend_artist)}
                    )
            if not test:
                plt.tight_layout()
                plt.show()
        else:
            raise GraphingException('No curves to be plotted!')


@dataclass
class Curve():
    xdata: list | np.ndarray
    ydata: list | np.ndarray
    color: str
    label: str
    
    def set_color(self, color: str or list[str]):
        self.color = color
    
    def plot_curve(self, axes: plt.Axes):
        self.handle, = axes.plot(self.xdata, self.ydata, color=self.color, label=self.label)


class Scatter(Curve):
    def plot_curve(self, axes: plt.Axes):
        self.handle = axes.scatter(self.xdata, self.ydata, color=self.color, label=self.label)


class Dashed(Curve):
    pass


@dataclass
class Histogram():
    xdata: list | np.ndarray
    bins: int
    color: str
    label: str

    def plot_curve(self, axes: plt.Axes):
        xy = np.array([[0,2,2,3,3,1,1,0,0], [0,0,1,1,2,2,3,3,0]]).T
        self.handle = Polygon(xy, facecolor='silver', edgecolor='k', linewidth=1)
        axes.hist(self.xdata, bins=self.bins, color=self.color, label=self.label)

def histogram_legend_artist(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    xy = np.array([[0,0,1,1,2,2,3,3,4,4,0], [0,4,4,2.5,2.5,5,5,1.5,1.5,0,0]]).T
    xy[:,0] = width * xy[:,0] / 4 + xdescent
    xy[:,1] = height * xy[:,1] / 5 - ydescent
    patch = Polygon(xy, fc='silver', ec='k')
    return patch
