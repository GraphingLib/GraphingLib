from typing import Literal, Optional

import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import empty

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
        self.x_label = x_label
        self.y_label = y_label
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
        ax = plt.subplot(
            grid.new_subplotspec(
                (self.placement[0], self.placement[1]),
                rowspan=self.placement[2],
                colspan=self.placement[3],
            )
        )


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
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.title = title
        self.figure_style = figure_style
        file_loader = FileLoader(figure_style)
        self.default_params = file_loader.load()
        size = size if size != "default" else self.default_params["Figure"]["size"]
        self.size = size
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

    def _prepare_multifigure(self, legend: bool = True) -> None:
        self._figure = plt.figure(layout="constrained")
        multifigure_grid = GridSpec(self.num_rows, self.num_cols, figure=self._figure)
        for subfigure in self._subfigures:
            subfigure._prepare_subfigure(multifigure_grid)

    def display(self, legend: bool = True) -> None:
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