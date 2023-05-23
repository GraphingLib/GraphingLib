from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon

from .file_manager import FileLoader
from .graph_elements import GraphingException, Plottable
from .legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)


class Figure:
    """
    A general Matplotlib figure.
    """

    def __init__(
        self,
        x_label: str = "x axis",
        y_label: str = "y axis",
        size: tuple = "default",
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
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
        self._figure_style = figure_style
        file_loader = FileLoader(self._figure_style)
        self._default_params = file_loader.load()
        size = size if size != "default" else self._default_params["Figure"]["size"]
        self._size = size
        legend_is_boxed = (
            legend_is_boxed
            if legend_is_boxed != "default"
            else self._default_params["Figure"]["boxed_legend"]
        )
        ticks_are_in = (
            ticks_are_in
            if ticks_are_in != "default"
            else self._default_params["Figure"]["ticks_are_in"]
        )
        log_scale_x = (
            log_scale_x
            if log_scale_x != "default"
            else self._default_params["Figure"]["log_scale_x"]
        )
        log_scale_y = (
            log_scale_y
            if log_scale_y != "default"
            else self._default_params["Figure"]["log_scale_y"]
        )
        self._figure, self._axes = plt.subplots(figsize=self._size)
        self._elements = []
        self._labels = []
        self._handles = []
        self._x_axis_name = x_label
        self._y_axis_name = y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self._log_scale_x = log_scale_x
        self._log_scale_y = log_scale_y
        self._legend_is_boxed = legend_is_boxed
        self._ticks_are_in = ticks_are_in

    def add_element(self, *elements: Plottable) -> None:
        """
        Adds a Curve object to the figure.
        """
        for element in elements:
            self._elements.append(element)
            try:
                if element.label is not None:
                    self._labels.append(element.label)
            except AttributeError:
                pass

    def _prepare_figure(self, legend: bool = True) -> None:
        self._axes.set_xlabel(self._x_axis_name)
        self._axes.set_ylabel(self._y_axis_name)
        if self._x_lim:
            self._axes.set_xlim(*self._x_lim)
        if self._y_lim:
            self._axes.set_ylim(*self._y_lim)
        if self._log_scale_x:
            self._axes.set_xscale("log")
        if self._log_scale_y:
            self._axes.set_yscale("log")
        if self._ticks_are_in:
            self._axes.tick_params(axis="both", direction="in", which="both")
        if not self._labels:
            legend = False
        if self._elements:
            z_order = 0
            for element in self._elements:
                self._fill_in_missing_params(element)
                element._plot_element(self._axes, z_order)
                try:
                    if element._label is not None:
                        self._handles.append(element._handle)
                except AttributeError:
                    continue
                z_order += 2
            if legend:
                try:
                    self._axes.legend(
                        handles=self._handles,
                        labels=self._labels,
                        handleheight=1.3,
                        handler_map={
                            Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                            LineCollection: HandlerMultipleLines(),
                            VerticalLineCollection: HandlerMultipleVerticalLines(),
                        },
                        frameon=self._legend_is_boxed,
                        draggable=True,
                    )
                except:
                    self._axes.legend(
                        handles=self._handles,
                        labels=self._labels,
                        handleheight=1.3,
                        handler_map={
                            Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                            LineCollection: HandlerMultipleLines(),
                            VerticalLineCollection: HandlerMultipleVerticalLines(),
                        },
                        frameon=self._legend_is_boxed,
                    )
        else:
            raise GraphingException("No curves to be plotted!")

    def display(self, legend: bool = True) -> None:
        self._prepare_figure(legend=legend)
        plt.tight_layout()
        plt.show()

    def save_figure(self, file_name: str, legend: bool = True) -> None:
        self._prepare_figure(legend=legend)
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches="tight")

    def _fill_in_missing_params(self, element: Plottable) -> None:
        object_type = type(element).__name__
        for property, value in vars(element).items():
            if (type(value) == str) and (value == "default"):
                if self._default_params[object_type][property] == "same as curve":
                    element.__dict__["errorbars_color"] = self._default_params[
                        object_type
                    ]["color"]
                    element.__dict__["errorbars_line_width"] = self._default_params[
                        object_type
                    ]["line_width"]
                    element.__dict__["cap_thickness"] = self._default_params[
                        object_type
                    ]["line_width"]
                elif self._default_params[object_type][property] == "same as scatter":
                    element.__dict__["errorbars_color"] = self._default_params[
                        object_type
                    ]["face_color"]
                else:
                    element.__dict__[property] = self._default_params[object_type][
                        property
                    ]
