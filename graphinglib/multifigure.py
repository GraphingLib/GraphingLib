from typing import Literal, Optional

import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes

from .file_manager import FileLoader
from .graph_elements import GraphingException, Plottable
from .legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)


class Subfigure:
    """
    A single plot inside a multifigure.

    placement: (row, col, rowspan, colspan)
    """

    def __init__(
        self,
        placement: tuple[int, int, int, int],
        x_label: str = "x axis",
        y_label: str = "y axis",
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        figure_style: str = "plain",
        log_scale_x: bool | Literal["default"] = "default",
        log_scale_y: bool | Literal["default"] = "default",
        show_grid: bool | Literal["default"] = "default",
        legend_is_boxed: bool | Literal["default"] = "default",
        ticks_are_in: bool | Literal["default"] = "default",
    ):
        self.x_axis_name = x_label
        self.y_axis_name = y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.placement = placement
        file_loader = FileLoader(figure_style)
        self.default_params = file_loader.load()
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
        show_grid = (
            show_grid
            if show_grid != "default"
            else self.default_params["Figure"]["show_grid"]
        )
        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        self.legend_is_boxed = legend_is_boxed
        self.ticks_are_in = ticks_are_in
        self.grid_is_set = False
        self._elements: list[Plottable] = []
        self._labels: list[str | None] = []
        self._handles = []

    def add_element(self, *elements: Plottable) -> None:
        """
        Adds a Plottable element to the subfigure.
        """
        for element in elements:
            self._elements.append(element)
            try:
                if element.label is not None:
                    self._labels.append(element.label)
            except AttributeError:
                pass

    def _prepare_subfigure(self, grid: GridSpec, legend: bool = True) -> Axes:
        self._axes = plt.subplot(
            grid.new_subplotspec(
                (self.placement[0], self.placement[1]),
                rowspan=self.placement[2],
                colspan=self.placement[3],
            )
        )
        self._axes.set_xlabel(self.x_axis_name)
        self._axes.set_ylabel(self.y_axis_name)
        if self.x_lim:
            self._axes.set_xlim(*self.x_lim)
        if self.y_lim:
            self._axes.set_ylim(*self.y_lim)
        if self.log_scale_x:
            self._axes.set_xscale("log")
        if self.log_scale_y:
            self._axes.set_yscale("log")
        if self.ticks_are_in:
            self._axes.tick_params(axis="both", direction="in", which="both")
        if self.grid_is_set:
            self._axes.grid(
                which="major",
                linestyle=self.grid_line_style,
                linewidth=self.grid_line_width,
                color=self.grid_color,
                alpha=self.grid_alpha,
            )
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
                        frameon=self.legend_is_boxed,
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
                        frameon=self.legend_is_boxed,
                    )
        else:
            raise GraphingException("No curves to be plotted!")
        return self._labels, self._handles

    def _fill_in_missing_params(self, element: Plottable) -> None:
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

    def set_grid(
        self,
        line_width: float | Literal["default"] = "default",
        line_style: str = "default",
        color: str = "default",
        alpha: float | Literal["default"] = "default",
    ) -> None:
        self.grid_line_style = (
            line_style
            if line_style != "default"
            else self.default_params["Figure"]["grid_line_style"]
        )
        self.grid_line_width = (
            line_width
            if line_width != "default"
            else self.default_params["Figure"]["grid_line_width"]
        )
        self.grid_color = (
            color if color != "default" else self.default_params["Figure"]["grid_color"]
        )
        self.grid_alpha = (
            alpha if alpha != "default" else self.default_params["Figure"]["grid_alpha"]
        )
        self.grid_is_set = True


class Multifigure:
    """
    The container for multiple plots.
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        size: tuple[float, float] | Literal["default"] = "default",
        title: Optional[str] = None,
        figure_style: str = "plain",
        use_latex: bool = False,
        font_size: int = 12,
        legend_is_boxed: bool | Literal["default"] = "default",
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.title = title
        self.figure_style = figure_style
        file_loader = FileLoader(figure_style)
        self.default_params = file_loader.load()
        size = size if size != "default" else self.default_params["Figure"]["size"]
        self.size = size
        legend_is_boxed = (
            legend_is_boxed
            if legend_is_boxed != "default"
            else self.default_params["Figure"]["boxed_legend"]
        )
        self.legend_is_boxed = legend_is_boxed
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
        self._subfigures = []

    def add_subfigure(
        self,
        placement: tuple[int, int],
        x_label: str = "x axis",
        y_label: str = "y axis",
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        log_scale_x: bool | Literal["default"] = "default",
        log_scale_y: bool | Literal["default"] = "default",
        show_grid: bool | Literal["default"] = "default",
        legend_is_boxed: bool | Literal["default"] = "default",
        ticks_are_in: bool | Literal["default"] = "default",
    ) -> Subfigure:
        if placement[0] >= self.size[0] or placement[1] >= self.size[1]:
            raise GraphingException(
                "The placement value must be inside the size of the Multifigure."
            )
        if placement[0] < 0 or placement[1] < 0:
            raise GraphingException("The placement value cannot be negative.")
        new_subfigure = Subfigure(
            placement,
            x_label,
            y_label,
            x_lim,
            y_lim,
            self.figure_style,
            log_scale_x,
            log_scale_y,
            show_grid,
            legend_is_boxed,
            ticks_are_in,
        )
        self._subfigures.append(new_subfigure)
        return new_subfigure

    def _prepare_multifigure(self, legend: bool = False) -> None:
        self._figure = plt.figure(layout="constrained", figsize=self.size)
        multifigure_grid = GridSpec(self.num_rows, self.num_cols, figure=self._figure)
        subfigures_legend = True if not legend else False
        labels, handles = [], []
        for subfigure in self._subfigures:
            subfigure_labels, subfigure_handles = subfigure._prepare_subfigure(
                multifigure_grid, legend=subfigures_legend
            )
            labels += subfigure_labels
            handles += subfigure_handles
        if legend:
            try:
                self._figure.legend(
                    handles=handles,
                    labels=labels,
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
                self._figure.legend(
                    handles=handles,
                    labels=labels,
                    handleheight=1.3,
                    handler_map={
                        Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                        LineCollection: HandlerMultipleLines(),
                        VerticalLineCollection: HandlerMultipleVerticalLines(),
                    },
                    frameon=self.legend_is_boxed,
                )
        self._figure.suptitle(self.title)

    def display(self, legend: bool = False) -> None:
        self._prepare_multifigure(legend=legend)
        plt.show()

    def save_figure(self, file_name: str, legend: bool = True) -> None:
        self._prepare_multifigure(legend=legend)
        plt.savefig(file_name, bbox_inches="tight")

    def _fill_in_missing_params(self, element: Plottable) -> None:
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
