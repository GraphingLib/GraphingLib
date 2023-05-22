import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib import rcParamsDefault

from .data_plotting_1d import *
from .file_manager import *
from .graph_elements import *
from .legend_artists import *


class Figure:
    """
    A general Matplotlib figure.
    """

    def __init__(
        self,
        x_label: str = "x axis",
        y_label: str = "y axis",
        size: tuple = "default",
        x_lim: tuple = None,
        y_lim: tuple = None,
        log_scale_x: bool = "default",
        log_scale_y: bool = "default",
        legend_is_boxed: bool = "default",
        ticks_are_in: bool = "default",
        figure_style: str = "plain",
        use_latex: bool = False,
        font_size: int = 12,
    ) -> None:
        if use_latex:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.size": font_size + 3,
                }
            )
        else:
            plt.rcParams.update(rcParamsDefault)
            plt.rcParams["font.size"] = font_size
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
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        self.legend_is_boxed = legend_is_boxed
        self.ticks_are_in = ticks_are_in

    def add_element(self, *elements: Plottable) -> None:
        """
        Adds a Curve object to the figure.
        """
        for element in elements:
            self.elements.append(element)
            try:
                self.labels.append(element.label)
            except AttributeError:
                pass

    def prepare_figure(self, legend: bool = True) -> None:
        self.axes.set_xlabel(self.x_axis_name)
        self.axes.set_ylabel(self.y_axis_name)
        if self.x_lim:
            self.axes.set_xlim(*self.x_lim)
        if self.y_lim:
            self.axes.set_ylim(*self.y_lim)
        if self.log_scale_x:
            self.axes.set_xscale("log")
        if self.log_scale_y:
            self.axes.set_yscale("log")
        if self.ticks_are_in:
            self.axes.tick_params(axis="both", direction="in", which="both")
        if not self.labels:
            legend = False
        if self.elements:
            z_order = 0
            for curve in self.elements:
                self.fill_in_missing_params(curve)
                curve.plot_element(self.axes, z_order)
                try:
                    self.handles.append(curve.handle)
                except AttributeError:
                    continue
                z_order += 2
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
            raise GraphingException("No curves to be plotted!")

    def display(self, legend: bool = True) -> None:
        self.prepare_figure(legend=legend)
        plt.tight_layout()
        plt.show()

    def save_figure(self, file_name: str, legend: bool = True) -> None:
        self.prepare_figure(legend=legend)
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches="tight")

    def fill_in_missing_params(self, element: Plottable) -> None:
        object_type = type(element).__name__
        for property, value in vars(element).items():
            if (type(value) == str) and (value == "default"):
                if self.default_params[object_type][property] == "same as curve":
                    element.__dict__["errorbars_color"] = self.default_params[
                        object_type
                    ]["color"]
                    element.__dict__["errorbars_line_width"] = self.default_params[
                        object_type
                    ]["line_width"]
                    element.__dict__["cap_thickness"] = self.default_params[
                        object_type
                    ]["line_width"]
                elif self.default_params[object_type][property] == "same as scatter":
                    element.__dict__["errorbars_color"] = self.default_params[
                        object_type
                    ]["face_color"]
                else:
                    element.__dict__[property] = self.default_params[object_type][
                        property
                    ]
