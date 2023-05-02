from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon


@dataclass
class Curve:
    """A general continuous curve.

    Creates a Curve object which can be plotted on a Figure. Optional parameters which are left blank will be filled according to the default parameters of the chosen ``figure_style`` (choose when creating the `graphinglib.Figure` object).

    Parameters
    ----------
        xdata: list or ndarray
            An array of x values for the curve.
        ydata: list or ndarray
            An array of y values for the curve.
        label: str
            The name of the curve which will appear on the legend.
        color: str, optional
            The color of the curve.
        line_width: float, optional
            The width of the curve.
    """

    xdata: list | np.ndarray
    ydata: list | np.ndarray
    label: str
    color: str = "default"
    line_width: int = "default"

    @classmethod
    def from_function(
        cls,
        func: Callable,
        xmin: float,
        xmax: float,
        label: str,
        color: str = "default",
        line_width: int = "default",
        number_of_points: int = 500,
    ):
        """Creates a Curve object from a function instead of from a list of x and y values.

        Optional parameters which are left blank will be filled according to the default parameters of the chosen ``figure_style`` (choose when creating the `graphinglib.Figure` object).

        Arguments
        ---------
            func: Callable
                The function to be plotted. Must be a function of x only.
            xmin: float
                The curve will be plotted starting at this x value.
            xmax: float
                The curve will be plotted until this x value is reached.
            label: str
                The name of the curve which will appear on the legend.
            color: str, optional
                The color of the curve. Standard matplotlib colors are available.
            line_width: int, optional
                The width of the curve.
            number_of_points: int
                The number of sample points. Defaults to 500.

        Returns
        -------
            curve: Curve
                The curve object with the specified parameters.
        """
        xdata = np.linspace(xmin, xmax, number_of_points)
        ydata = func(xdata)
        return cls(xdata, ydata, label, color, line_width)

    def set_color(self, color: str or list[str]):
        self.color = color

    def plot_element(self, axes: plt.Axes):
        (self.handle,) = axes.plot(
            self.xdata,
            self.ydata,
            color=self.color,
            linewidth=self.line_width,
            label=self.label,
        )


@dataclass
class Scatter(Curve):
    """
    A general scatter plot.

    Creates a Scatter object which can be plotted on a Figure. Optional parameters which are left blank will be filled according to the default parameters of the chosen ``figure_style`` (choose when creating the `graphinglib.Figure` object).

    Parameters
    ----------
        xdata: list or ndarray
            An array of x values for the curve.
        ydata: list or ndarray
            An array of y values for the curve.
        label: str
            The name of the curve which will appear on the legend.
        color: str, optional
            The color of the curve. Standard matplotlib colors are available.
        marker_size: float, optional
            The size of the marker used to draw the individual points.
        edge_color: str, optional
            The color of the marker's edges.
        marker_style: str, optional
            The type of marker. See https://matplotlib.org/stable/api/markers_api.html for options.
    """

    marker_size: float = "default"
    edge_color: str = "default"
    marker_style: str = "default"

    def plot_element(self, axes: plt.Axes):
        self.handle = axes.scatter(
            self.xdata,
            self.ydata,
            color=self.color,
            edgecolors=self.edge_color,
            s=self.marker_size,
            marker=self.marker_style,
            label=self.label,
        )


class Dashed(Curve):
    """
    A general dashed curve.

    Creates a Dashed object which can be plotted on a Figure. Optional parameters which are left blank will be filled according to the default parameters of the chosen ``figure_style`` (choose when creating the `graphinglib.Figure` object).

    Parameters
    ----------
        xdata: list or ndarray
            An array of x values for the curve.
        ydata: list or ndarray
            An array of y values for the curve.
        label: str
            The name of the curve which will appear on the legend.
        color: str, optional
            The color of the curve. Standard matplotlib colors are available.
    """

    def plot_element(self, axes: plt.Axes):
        (self.handle,) = axes.plot(
            self.xdata,
            self.ydata,
            color=self.color,
            linewidth=self.line_width,
            label=self.label,
            linestyle="--",
        )


@dataclass
class Histogram:
    """
    A histogram plot created from a one dimensional array.

    The mean and standard deviation of the data can be accessed with::

        hist = gl.Histogram(xdata=data, number_of_bins=30, label="Histogram of results")
        mean = hist.mean
        std = hist.standard_deviation

    Parameters
    ----------
        xdata: list or ndarray
            An array of values to be plotted.
        number_of_bins: int
            The number of bins that the data will be sorted in.
        label: str
            The name of the histogram which will appear on the legend.
        face_color: str, optional
            The color of the histogram's fill.
        edge_color: str, optional
            The color of the histogram's edges.
        hist_type: str, optional
            Options are 'bar', 'step', and 'stepfilled'. 'bar' is a standard bar graph, step is just the outline of the histgram, and stepfilled is the outline with a fill.
        alpha: float, optional
            Value between 0 and 1 which determines the opacity of the fill.
        line_width: float, optional
            The width of the histogram's edges.
        normalize: bool, optional
            If True, the total area of the histogram will be equal to 1. This takes into account the width of the bars.
        show_pdf: str, optional
            If 'normal' or 'gaussian', a normal probability density function with the data's mean and standard deviation will be overlayed onto the histogram.
        show_params: bool, optional
            If True, the histogram's mean and standard deviation will be shown in the legend.
    """

    xdata: list | np.ndarray
    number_of_bins: int
    label: str
    face_color: str = "default"
    edge_color: str = "default"
    hist_type: str = "default"
    alpha: float = "default"
    line_width: float = "default"
    normalize: bool = "default"
    show_pdf: str = "default"
    show_params: bool = "default"

    def __post_init__(self):
        self.mean = np.mean(self.xdata)
        self.standard_deviation = np.std(self.xdata)
        parameters = np.histogram(
            self.xdata, bins=self.number_of_bins, density=self.normalize
        )
        self.bin_heights, bin_edges = parameters[0], parameters[1]
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[1:] - bin_width / 2
        self.bin_width = bin_width
        self.bin_centers = bin_centers
        self.bin_edges = bin_edges
        self.create_label()

    @classmethod
    def plot_residuals_from_fit(
        cls,
        fit,
        number_of_bins: int,
        label: str,
        face_color: str = "default",
        edge_color: str = "default",
        hist_type: str = "default",
        alpha: int = "default",
        line_width: int = "default",
        normalize: bool = "default",
        show_pdf: str = "default",
        show_params: bool = "default",
    ):
        """Creates a histogram of the residuals of a curve fit.

        Parameters
        ----------
            fit: Fit
                Any fit object.
            number_of_bins: int
                The number of bins that the data will be sorted in.
            label: str
                The name of the histogram which will appear on the legend.
            face_color: str, optional
                The color of the histogram's fill.
            edge_color: str, optional
                The color of the histogram's edges.
            hist_type: str, optional
                Options are 'bar', 'step', and 'stepfilled'. 'bar' is a standard bar graph, step is just the outline of the histgram, and stepfilled is the outline with a fill.
            alpha: float, optional
                Value between 0 and 1 which determines the opacity of the fill.
            line_width: float, optional
                The width of the histogram's edges.
            normalize: bool, optional
                If True, the total area of the histogram will be equal to 1. This takes into account the width of the bars.
            show_pdf: str, optional
                If 'normal' or 'gaussian', a normal probability density function with the data's mean and standard deviation will be overlayed onto the histogram.
            show_params: bool, optional
                If True, the histogram's mean and standard deviation will be shown in the legend.
        """
        residuals = fit.calculate_residuals()
        return cls(
            residuals,
            number_of_bins,
            label,
            face_color,
            edge_color,
            hist_type,
            alpha,
            line_width,
            normalize,
            show_pdf,
            show_params,
        )

    def create_label(self):
        lab = self.label
        if self.label and self.show_params:
            lab += " :\n"
        if self.show_params:
            lab += f"$\mu$ = {self.mean:.3f}, $\sigma$ = {self.standard_deviation:.3f}"
        self.label = lab

    def normal_normalized(self, x):
        return (1 / (self.standard_deviation * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (((x - self.mean) / self.standard_deviation) ** 2)
        )

    def normal_not_normalized(self, x):
        return sum(self.bin_heights) * self.bin_width * self.normal_normalized(x)

    def plot_element(self, axes: plt.Axes):
        self.handle = Polygon(
            np.array([[0, 2, 2, 3, 3, 1, 1, 0, 0], [0, 0, 1, 1, 2, 2, 3, 3, 0]]).T,
            facecolor=to_rgba(self.face_color, self.alpha),
            edgecolor=to_rgba(self.edge_color, 1),
            linewidth=1,
        )
        axes.hist(
            self.xdata,
            bins=self.number_of_bins,
            facecolor=to_rgba(self.face_color, self.alpha),
            edgecolor=to_rgba(self.edge_color, 1),
            label=self.label,
            histtype=self.hist_type,
            linewidth=self.line_width,
            density=self.normalize,
        )
        if self.show_pdf in ["normal", "gaussian"]:
            normal = (
                self.normal_normalized if self.normalize else self.normal_not_normalized
            )
            num_of_points = 500
            xdata = np.linspace(self.bin_edges[0], self.bin_edges[-1], num_of_points)
            ydata = normal(xdata)
            axes.plot(xdata, ydata, color=self.edge_color, label=self.label)
            curve_max_y = normal(self.mean)
            curve_std_y = normal(self.mean + self.standard_deviation)
            plt.vlines(
                [
                    self.mean - self.standard_deviation,
                    self.mean,
                    self.mean + self.standard_deviation,
                ],
                [0, 0, 0],
                [curve_std_y, curve_max_y, curve_std_y],
                linestyles=["dashed"],
                colors=["k", "r", "k"],
            )
