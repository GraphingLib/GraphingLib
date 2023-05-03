from fileinput import filename

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon

from .data_plotting_1d import *
from .file_manager import *
from .graph_elements import *
from .legend_artists import *


class Figure:
    """
    A general Matplotlib figure.

    A Figure object always has a ``figure_style`` which determines the style parameters of the figure itself and all the elements and curves it contains. If a style parameter is specified for either the figure or an element, then it overrides the default set by the ``figure_style`` for that parameter.

    Parameters
    ----------
        x_label: str
            The label of the figure's x axis.
        y_label: str
            The label of the figure's y axis.
        size: tuple
            The size of the figure.
        log_scale_x: bool
            If True, the x axis will be logarithmic.
        log_scale_y: bool
            If True, the y axis will be logarithmic.
        legend_is_boxed: bool
            If True, the legend will have an outline.
        ticks_are_in: bool
            If True, the tick marks for both axes will be inside the figure as opposed to outside.
        figure_style: str
            The name of the desired default style .yml file. Currently available defaults are "plain" and "weird". Custom styles can be created by creating your own .yaml file with your own custom name and adding them to the default_styles folder.
    """

    def __init__(
        self,
        x_label: str = "x axis",
        y_label: str = "y axis",
        size: tuple = "default",
        log_scale_x: bool = "default",
        log_scale_y: bool = "default",
        legend_is_boxed: bool = "default",
        ticks_are_in: bool = "default",
        figure_style: str = "plain",
    ):
        self.figure_style = figure_style
        file_loader = FileLoader(self.figure_style)
        self.default_params = file_loader.load()
        size = size if size != "default" else self.default_params["Figure"]["size"]
        self.size = size
        legend_is_boxed = (
            legend_is_boxed
            if legend_is_boxed != "default"
            else self.default_params["Figure"]["boxed_legend"]
        )
        ticks_are_in = (
            ticks_are_in
            if ticks_are_in != "default"
            else self.default_params["Figure"]["ticks_are_in"]
        )
        log_scale_x = (
            log_scale_x
            if log_scale_x != "default"
            else self.default_params["Figure"]["log_scale_x"]
        )
        log_scale_y = (
            log_scale_y
            if log_scale_y != "default"
            else self.default_params["Figure"]["log_scale_y"]
        )
        self.figure, self.axes = plt.subplots(figsize=self.size)
        self.elements = []
        self.labels = []
        self.handles = []
        self.x_axis_name = x_label
        self.y_axis_name = y_label
        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        self.legend_is_boxed = legend_is_boxed
        self.ticks_are_in = ticks_are_in

    def add_element(self, *curves: Curve | Hlines | Vlines | Histogram):
        """
        Adds a Curve or other object to the figure.

        Parameters
        ----------
            curves : Curve
                The curve or other object to be added. Multiple objects can added at once as separate parameters.
        """
        for curve in curves:
            self.elements.append(curve)
            try:
                self.labels.append(curve.label)
            except AttributeError:
                pass

    def prepare_figure(self, legend=True):
        self.axes.set_xlabel(self.x_axis_name)
        self.axes.set_ylabel(self.y_axis_name)
        if self.log_scale_x:
            self.axes.set_xscale("log")
        if self.log_scale_y:
            self.axes.set_yscale("log")
        if self.ticks_are_in:
            self.axes.tick_params(axis="both", direction="in", which="both")
        if self.elements:
            for curve in self.elements:
                self.fill_in_missing_params(curve)
                curve.plot_element(self.axes)
                try:
                    self.handles.append(curve.handle)
                except AttributeError:
                    continue
            if legend:
                try:
                    self.axes.legend(
                        handles=self.handles,
                        labels=self.labels,
                        handleheight=1.3,
                        handler_map={
                            Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                            LineCollection: HandlerMultipleLines(),
                            VerticalLineCollection: HandlerMultipleVerticalLines(),
                        },
                        frameon=self.legend_is_boxed,
                        draggable=True,
                    )
                except:
                    self.axes.legend(
                        handles=self.handles,
                        labels=self.labels,
                        handleheight=1.3,
                        handler_map={
                            Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                            LineCollection: HandlerMultipleLines(),
                            VerticalLineCollection: HandlerMultipleVerticalLines(),
                        },
                        frameon=self.legend_is_boxed,
                    )
            else:
                self.axes.legend(draggable=True, frameon=self.legend_is_boxed)
        else:
            raise GraphingException("No curves to be plotted!")

    def generate_figure(self, legend=True, test=False):
        """Prepares and displays the Matplotlib figure.

        Parameters
        ----------
            legend: bool, optional
                If True, displays the legend using the labels of each object in the figure.
            test: bool, optional
                If True, doesn't display the figure, just prepares it.
        """
        self.prepare_figure(legend=legend)
        if not test:
            plt.tight_layout()
            plt.show()

    def save_figure(self, file_name: str, legend=True):
        """Saves the figure to a file.

        Parameters
        ----------
            file_name: str
                The name of the file with the extension. Ex: "figure_1.png", "figure_2.pdf", "figure_3.jpg"
            legend: bool, optional
                If True, displays the legend using the labels of each object in the figure.
        """
        self.prepare_figure(legend=legend)
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches="tight")

    def fill_in_missing_params(self, curve):
        object_type = type(curve).__name__
        for property, value in vars(curve).items():
            if (type(value) == str) and (value == "default"):
                curve.__dict__[property] = self.default_params[object_type][property]
