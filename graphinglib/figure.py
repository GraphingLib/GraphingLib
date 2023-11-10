from typing import Literal, Optional

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon

from .file_manager import FileLoader, FileUpdater
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
        Defaults to ``"x axis"`` and ``"y axis"``.
    x_lim, y_lim : tuple[float, float], optional
        The limits for the x-axis and y-axis.
    size : tuple[float, float]
        Overall size of the figure.
        Default depends on the ``figure_style`` configuration.
    log_scale_x, log_scale_y : bool
        Whether or not to set the scale of the x- or y-axis to logaritmic scale.
        Default depends on the ``figure_style`` configuration.
    show_grid : bool
        Whether or not to show the grid.
        Default depends on the ``figure_style`` configuration.
    figure_style : str
        The figure style to use for the figure.
    color_cycle : list[str]
        List of colors applied to the elements cyclically if none is provided.
        Default depends on the ``figure_style`` configuration.
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
        figure_style: str = "plain",
        color_cycle: list[str] | Literal["default"] = "default",
    ) -> None:
        """
        This class implements a general figure object.

        Parameters
        ----------
        x_label, y_label : str
            The indentification for the x-axis and y-axis.
            Defaults to ``"x axis"`` and ``"y axis"``.
        x_lim, y_lim : tuple[float, float], optional
            The limits for the x-axis and y-axis.
        size : tuple[float, float]
            Overall size of the figure.
            Default depends on the ``figure_style`` configuration.
        log_scale_x, log_scale_y : bool
            Whether or not to set the scale of the x- or y-axis to logaritmic scale.
            Default depends on the ``figure_style`` configuration.
        show_grid : bool
            Whether or not to show the grid.
            Default depends on the ``figure_style`` configuration.
        figure_style : str
            The figure style to use for the figure.
        color_cycle : list[str]
            List of colors applied to the elements cyclically if none is provided.
            Default depends on the ``figure_style`` configuration.
        """
        self.figure_style = figure_style
        self.size = size
        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        self.show_grid = show_grid
        self.color_cycle = color_cycle
        self._elements: list[Plottable] = []
        self._labels: list[str | None] = []
        self._handles = []
        self.x_axis_name = x_label
        self.y_axis_name = y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self._rc_dict = {}
        self._user_rc_dict = {}

    def add_element(self, *elements: Plottable) -> None:
        """
        Adds a :class:`~graphinglib.graph_elements.Plottable` element to the :class:`~graphinglib.figure.Figure`.

        Parameters
        ----------
        elements : :class:`~graphinglib.graph_elements.Plottable`
            Elements to plot in the :class:`~graphinglib.figure.Figure`.
        """
        for element in elements:
            self._elements.append(element)

    def _prepare_figure(self, legend: bool = True) -> None:
        """
        Prepares the :class:`~graphinglib.figure.Figure` to be displayed.
        """
        try:
            file_loader = FileLoader(self.figure_style)
            self.default_params = file_loader.load()
            self._fill_in_rc_params()
        except FileNotFoundError:
            # set the style use matplotlib style
            try:
                plt.style.use(self.figure_style)
                file_loader = FileLoader("plain")
                self.default_params = file_loader.load()
            except OSError:
                raise GraphingException(
                    f"The figure style {self.figure_style} was not found. Please choose a different style."
                )

        figure_params_to_reset = self._fill_in_missing_params(self)

        self._figure, self._axes = plt.subplots(figsize=self.size)

        if self.show_grid:
            self._axes.grid(True)
        else:
            self._axes.grid(False)

        self.color_cycle = cycler(color=self.color_cycle)
        self._axes.set_prop_cycle(self.color_cycle)
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
        if self._elements:
            z_order = 1
            for element in self._elements:
                params_to_reset = self._fill_in_missing_params(element)
                element._plot_element(self._axes, z_order)
                self._reset_params_to_default(element, params_to_reset)
                try:
                    if element.label is not None:
                        self._handles.append(element.handle)
                        self._labels.append(element.label)
                except AttributeError:
                    continue
                z_order += 2
            if not self._labels:
                legend = False
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
                    )
        else:
            raise GraphingException("No curves to be plotted!")
        self._reset_params_to_default(self, figure_params_to_reset)
        self._handles = []
        self._labels = []
        self._rc_dict = {}
        plt.rcParams.update(plt.rcParamsDefault)

    def display(self, legend: bool = True) -> None:
        """
        Displays the :class:`~graphinglib.figure.Figure`.

        Parameters
        ----------
        legend : bool
            Wheter or not to display the legend. The legend is always set to be
            draggable.
            Defaults to ``True``.
        """
        self._prepare_figure(legend=legend)
        plt.tight_layout()
        plt.show()

    def save_figure(self, file_name: str, legend: bool = True) -> None:
        """
        Saves the :class:`~graphinglib.figure.Figure`.

        Parameters
        ----------
        legend : bool
            Wheter or not to display the legend.
            Defaults to ``True``.
        """
        self._prepare_figure(legend=legend)
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches="tight")
        plt.close()

    def _fill_in_missing_params(self, element: Plottable) -> list[str]:
        """
        Fills in the missing parameters from the specified ``figure_style``.
        """
        params_to_reset = []
        object_type = type(element).__name__
        tries = 0
        curve_defaults = {
            "errorbars_color": "color",
            "errorbars_line_width": "line_width",
            "cap_thickness": "line_width",
            "fill_under_color": "color",
        }
        while tries < 2:
            try:
                for property, value in vars(element).items():
                    if (type(value) == str) and (value == "default"):
                        params_to_reset.append(property)
                        default_value = self.default_params[object_type][property]
                        if default_value == "same as curve":
                            setattr(
                                element,
                                property,
                                getattr(element, curve_defaults[property]),
                            )
                        elif default_value == "same as scatter":
                            element.errorbars_color = getattr(element, "face_color")
                        else:
                            setattr(element, property, default_value)
                break
            except KeyError as e:
                tries += 1
                if tries >= 2:
                    raise GraphingException(
                        f"There was an error auto updating your {self.figure_style} style file following the recent GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub page. In the meantime, you can manually add the following parameter to your {self.figure_style} style file:\n {e.args[0]}"
                    )
                file_updater = FileUpdater(self.figure_style)
                file_updater.update()
                file_loader = FileLoader(self.figure_style)
                self.default_params = file_loader.load()
        return params_to_reset

    def _reset_params_to_default(
        self, element: Plottable, params_to_reset: list[str]
    ) -> None:
        """
        Resets the parameters that were set to default in the _fill_in_missing_params method.
        """
        for param in params_to_reset:
            setattr(
                element,
                param,
                "default",
            )

    def customize_visual_style(
        self,
        rc_params_dict: dict[str, str | float] | Literal["default"] = "default",
    ):
        """
        Customize the visual style of the :class:`~graphinglib.figure.Figure`.

        Any rc parameter that is not specified in the dictionary will be set to the default value for the specified ``figure_style``.

        Parameters
        ----------
        rc_params_dict : dict[str, str | float]
            Dictionary of rc parameters to update.
            Defaults depends on the ``figure_style`` configuration.
        """
        self._user_rc_dict = rc_params_dict

    def _fill_in_rc_params(self):
        """
        Fills in the missing rc parameters from the specified ``figure_style``.
        """
        params = self.default_params["rc_params"]
        for property, value in params.items():
            # add to rc_dict if not already in there
            if property not in self._rc_dict:
                self._rc_dict[property] = value
        all_rc_params = {**self._rc_dict, **self._user_rc_dict}
        plt.rcParams.update(all_rc_params)
