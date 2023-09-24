from typing import Literal, Optional

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
    This class implements a general figure object.

    Parameters
    ----------
    x_label, y_label : str
        The indentification for the x-axis and y-axis.
        Defaults to `"x axis"` and `"y axis"`.
    x_lim, y_lim : tuple[float, float], optional
        The limits for the x-axis and y-axis.
    size : tuple[float, float]
        Overall size of the figure.
        Default depends on the figure style configuration.
    log_scale_x, log_scale_y : bool
        Whether or not to set the scale of the x- or y-axis to logaritmic scale.
        Default depends on the figure style configuration.
    show_grid : bool
        Wheter or not to show the grid.
        Default depends on the figure style configuration.
    legend_is_boxed : bool
        Wheter or not to display the legend inside a box.
        Default depends on the figure style configuration.
    ticks_are_in : bool
        Wheter or not to display the axis ticks inside the axis.
        Default depends on the figure style configuration.
    figure_style : str
        The figure style to use for the figure.
    use_latex : bool
        Wheter or not to use LaTeX to render text and math symbols in the figure.
        Defaults to `False`.
        Requires a LaTeX distribution.
    font_size : int
        Font size used to render the text and math symbols in the figure.
        Defaults to `12`.
    """

    def __init__(
        self,
        x_label: str = "x axis",
        y_label: str = "y axis",
        size: tuple[float, float] | Literal["default"] = "default",
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        log_scale_x: bool | Literal["default"] = "default",
        log_scale_y: bool | Literal["default"] = "default",
        show_grid: bool | Literal["default"] = "default",
        legend_is_boxed: bool | Literal["default"] = "default",
        ticks_are_in: bool | Literal["default"] = "default",
        figure_style: str = "plain",
        use_latex: bool = False,
        font_size: int = 12,
    ) -> None:
        """
        This class implements a general figure object.

        Parameters
        ----------
        x_label, y_label : str
            The indentification for the x-axis and y-axis.
            Defaults to `"x axis"` and `"y axis"`.
        x_lim, y_lim : tuple[float, float], optional
            The limits for the x-axis and y-axis.
        size : tuple[float, float]
            Overall size of the figure.
            Default depends on the figure style configuration.
        log_scale_x, log_scale_y : bool
            Whether or not to set the scale of the x- or y-axis to logaritmic scale.
            Default depends on the figure style configuration.
        show_grid : bool
            Wheter or not to show the grid.
            Default depends on the figure style configuration.
        legend_is_boxed : bool
            Wheter or not to display the legend inside a box.
            Default depends on the figure style configuration.
        ticks_are_in : bool
            Wheter or not to display the axis ticks inside the axis.
            Default depends on the figure style configuration.
        figure_style : str
            The figure style to use for the figure.
        use_latex : bool
            Wheter or not to use LaTeX to render text and math symbols in the figure.
            Defaults to `False`.
            Requires a LaTeX distribution.
        font_size : int
            Font size used to render the text and math symbols in the figure.
            Defaults to `12`.
        """
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
        file_loader = FileLoader(figure_style)
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
        show_grid = (
            show_grid
            if show_grid != "default"
            else self.default_params["Figure"]["show_grid"]
        )
        self._elements: list[Plottable] = []
        self._labels: list[str | None] = []
        self._handles = []
        self.x_axis_name = x_label
        self.y_axis_name = y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        self.legend_is_boxed = legend_is_boxed
        self.ticks_are_in = ticks_are_in
        if show_grid:
            self.set_grid()

    def add_element(self, *elements: Plottable) -> None:
        """
        Adds a Plottable element to the figure.

        Parameters
        ----------
        elements : Plottable
            Elements to plot in the figure.
        """
        for element in elements:
            self._elements.append(element)
            try:
                if element.label is not None:
                    self._labels.append(element.label)
            except AttributeError:
                pass

    def _prepare_figure(self, legend: bool = True) -> None:
        """
        Prepares the figure to be displayed.
        """
        self._figure, self._axes = plt.subplots(figsize=self.size)
        try:
            self._axes.grid(
                which="major",
                linestyle=self.grid_line_style,
                linewidth=self.grid_line_width,
                color=self.grid_color,
                alpha=self.grid_alpha,
            )
        except:
            pass
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

    def display(self, legend: bool = True) -> None:
        """
        Displays the figure.

        Parameters
        ----------
        legend : bool
            Wheter or not to display the legend. The legend is always set to be
            draggable.
            Defaults to `True`.
        """
        self._prepare_figure(legend=legend)
        plt.tight_layout()
        plt.show()

    def save_figure(self, file_name: str, legend: bool = True) -> None:
        """
        Saves the figure.

        Parameters
        ----------
        legend : bool
            Wheter or not to display the legend.
            Defaults to `True`.
        """
        self._prepare_figure(legend=legend)
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches="tight")

    def _fill_in_missing_params(self, element: Plottable) -> None:
        """
        Fills in the missing parameters from the specified figure style.
        """
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
        """
        Sets the grid in the figure.

        Parameters
        ----------
        line_width : float
            Width of the lines forming the grid.
            Default depends on the figure style configuration.
        line_style : str
            Line style of the lines forming the grid.
            Default depends on the figure style configuration.
        color : str
            Color of the lines forming the grid.
            Default depends on the figure style configuration.
        alpha : float
            Opacity of the lines forming the grid.
            Default depends on the figure style configuration.
        """
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
