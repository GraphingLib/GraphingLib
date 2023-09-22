from typing import Literal, Optional
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from numpy import empty

from .file_manager import FileLoader
from .graph_elements import GraphingException, Plottable
from .legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)


@dataclass
class Subfigure:
    """
    A single plot inside a multifigure.
    """

    x_label: str = "x axis"
    y_label: str = "y axis"
    x_lim: Optional[tuple[float, float]] = None
    y_lim: Optional[tuple[float, float]] = None
    log_scale_x: bool | Literal["default"] = "default"
    log_scale_y: bool | Literal["default"] = "default"
    show_grid: bool | Literal["default"] = "default"
    legend_is_boxed: bool | Literal["default"] = "default"
    ticks_are_in: bool | Literal["default"] = "default"
    placement: Optional[tuple[int, int]] = None


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
        self._subfigures = empty((self.num_rows, self.num_cols)).astype(Subfigure)

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
        placement = [i - 1 for i in placement]
        new_subfigure = Subfigure(
            x_label,
            y_label,
            x_lim,
            y_lim,
            log_scale_x,
            log_scale_y,
            show_grid,
            legend_is_boxed,
            ticks_are_in,
            placement=placement,
        )
        self._subfigures[placement[0], placement[1]] = new_subfigure
        return new_subfigure

    def _prepare_multifigure(self) -> None:
        pass

    def display(self, legend: bool = True) -> None:
        # self._prepare_figure(legend=legend)
        plt.tight_layout()
        plt.show()

    def save_figure(self, file_name: str, legend: bool = True) -> None:
        # self._prepare_figure(legend=legend)
        plt.tight_layout()
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
