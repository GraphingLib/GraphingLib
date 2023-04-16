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
    """
    def __init__(self, x_label: str = 'x axis', y_label: str = 'y axis', size: tuple = "default",
                    log_scale_x: bool = "default", log_scale_y: bool = "default", legend_is_boxed: bool = "default", ticks_are_in: bool = "default", figure_style: str = 'plain'):
        self.figure_style = figure_style
        file_loader = FileLoader(self.figure_style)
        self.default_params = file_loader.load()
        size = size if size != "default" else self.default_params["Figure"]["size"]
        self.size = size
        legend_is_boxed = legend_is_boxed if legend_is_boxed != "default" else self.default_params["Figure"]["boxed_legend"]
        tick_are_in = ticks_are_in if ticks_are_in != "default" else self.default_params["Figure"]["ticks_are_in"]
        log_scale_x = log_scale_x if log_scale_x != "default" else self.default_params["Figure"]["log_scale_x"]
        log_scale_y = log_scale_y if log_scale_y != "default" else self.default_params["Figure"]["log_scale_y"]
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
        Adds a Curve object to the figure.
        """
        for curve in curves:
            self.elements.append(curve)
            try:
                self.labels.append(curve.label)
            except AttributeError:
                pass


    def generate_figure(self, legend=True, test=False):
        self.axes.set_xlabel(self.x_axis_name)
        self.axes.set_ylabel(self.y_axis_name)
        if self.log_scale_x:
            self.axes.set_xscale('log')
        if self.log_scale_y:
            self.axes.set_yscale('log')
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
            else:
                self.axes.legend(draggable=True, frameon=self.legend_is_boxed)
            if not test:
                plt.tight_layout()
                plt.show()
        else:
            raise GraphingException("No curves to be plotted!")


    def fill_in_missing_params(self, curve):
        object_type = type(curve).__name__
        for property, value in vars(curve).items():
            if (type(value) == str) and (value == "default"):
                curve.__dict__[property] = self.default_params[object_type][property]
