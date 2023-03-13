"""
# GraphingLib

Provides a simpler way to generate graphs with Matplotlib and an inclusion of certain Scipy
functions to simplify the process of analysing data.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerPatch, HandlerLineCollection
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from Legend_artists import histogram_legend_artist


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
    '''
    A dashed curve derived from the Curve object.
    '''
    def plot_curve(self, axes: plt.Axes):
        self.handle, = axes.plot(
            self.xdata,
            self.ydata,
            color=self.color,
            label=self.label,
            linestyle='--'
        )


class FitFromPolynomial(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object.
    """
    
    def __init__(self, curve_to_be_fit: Curve, degree: int, color: str, label: str):
        self.curve_to_be_fit = curve_to_be_fit
        self.coeffs = np.polyfit(self.curve_to_be_fit.xdata, self.curve_to_be_fit.ydata, degree)[::-1]
        self.function = self.get_polynomial_function()
        self.color = color
        self.label = label
        
    def __str__(self):
        coeff_chunks = []
        power_chunks = []
        ordered_rounded_coeffs = [round(coeff, 3) for coeff in self.coeffs[::-1]]
        for coeff, power in zip(ordered_rounded_coeffs, range(len(ordered_rounded_coeffs) - 1, -1, -1)):
            if coeff == 0:
                continue
            coeff_chunks.append(self.format_coeff(coeff))
            power_chunks.append(self.format_power(power))
        coeff_chunks[0] = coeff_chunks[0].lstrip("+ ")
        return 'f(x) = '+''.join([coeff_chunks[i] + power_chunks[i] for i in range(len(coeff_chunks))])

    @staticmethod
    def format_coeff(coeff):
        return " - {0}".format(abs(coeff)) if coeff < 0 else " + {0}".format(coeff)

    @staticmethod
    def format_power(power):
        return 'x^{0}'.format(power) if power != 0 else ''

    def get_polynomial_function(self):
        """
        Returns a linear function using the given coefficients.
        """
        return lambda x: sum(coeff * x**exponent for exponent, coeff in enumerate(self.coeffs))
    
    def plot_curve(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points)
        ydata = self.function(xdata)
        self.handle, = axes.plot(xdata, ydata, color=self.color, label=self.label)


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
        xy = np.array([[0, 2, 2, 3, 3, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 0]]).T
        self.handle = Polygon(
            xy,
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
class HLines():
    pass


@dataclass
class VLines():
    pass


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
                    handler_map={
                        Polygon: HandlerPatch(patch_func=histogram_legend_artist)#,
                        # LineCollection: HandlerLineCollection()
                    }
                )
            if not test:
                plt.tight_layout()
                plt.show()
        else:
            raise GraphingException("No curves to be plotted!")
