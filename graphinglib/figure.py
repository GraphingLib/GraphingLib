from shutil import which
from typing import Literal, Optional
from warnings import warn

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon

from .file_manager import (
    FileLoader,
    FileUpdater,
    get_default_style,
)
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
    title: str, optional
        The title of the figure.
    log_scale_x, log_scale_y : bool
        Whether or not to set the scale of the x- or y-axis to logaritmic scale.
        Default depends on the ``figure_style`` configuration.
    show_grid : bool
        Whether or not to show the grid.
        Default depends on the ``figure_style`` configuration.
    remove_axes : bool
        Whether or not to show the axes. Useful for adding tables or text to
        the subfigure. Defaults to ``False``.
    aspect_ratio : float, str
        The aspect ratio of the axis scaling. Can be either "equal", "auto" or a float.
        Defaults to "auto".
    figure_style : str
        The figure style to use for the figure.
    """

    def __init__(
        self,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        size: tuple[float, float] | Literal["default"] = "default",
        title: Optional[str] = None,
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        log_scale_x: bool | Literal["default"] = "default",
        log_scale_y: bool | Literal["default"] = "default",
        show_grid: bool | Literal["default"] = "default",
        remove_axes: bool = False,
        aspect_ratio: float | str = "auto",
        figure_style: str = "default",
    ) -> None:
        """
        This class implements a general figure object.

        Parameters
        ----------
        x_label, y_label : str, optional
            The indentification for the x-axis and y-axis.
            Defaults to ``"x axis"`` and ``"y axis"``.
        x_lim, y_lim : tuple[float, float], optional
            The limits for the x-axis and y-axis.
        size : tuple[float, float]
            Overall size of the figure.
            Default depends on the ``figure_style`` configuration.
        title: str, optional
            The title of the figure.
        log_scale_x, log_scale_y : bool
            Whether or not to set the scale of the x- or y-axis to logaritmic scale.
            Default depends on the ``figure_style`` configuration.
        show_grid : bool
            Whether or not to show the grid.
            Default depends on the ``figure_style`` configuration.
        remove_axes : bool
            Whether or not to show the axes. Useful for adding tables or text to
            the subfigure. Defaults to ``False``.
        aspect_ratio : float, str
            The aspect ratio of the axis scaling. Can be either "equal", "auto" or a float.
            Defaults to "auto".
        figure_style : str
            The figure style to use for the figure.
            Default can be set using ``gl.set_default_style()``.
        """
        self._figure_style = figure_style
        self._size = size
        self._title = title
        self._log_scale_x = log_scale_x
        self._log_scale_y = log_scale_y
        if show_grid == "default":
            self._show_grid = "unchanged"
        else:
            self._show_grid = show_grid
        self._elements: list[Plottable] = []
        self._labels: list[str | None] = []
        self._handles = []
        self._x_axis_name = x_label
        self._y_axis_name = y_label
        self._x_lim = x_lim
        self._y_lim = y_lim
        self._rc_dict = {}
        self._user_rc_dict = {}
        self._custom_ticks = False
        self._remove_axes = remove_axes
        self._twin_x_axis = None
        self._twin_y_axis = None

        if isinstance(aspect_ratio, str):
            if aspect_ratio not in ["equal", "auto"]:
                raise GraphingException(
                    "Aspect ratio must be either 'equal', 'auto' or a float."
                )
        elif isinstance(aspect_ratio, float):
            if aspect_ratio <= 0:
                raise GraphingException("Aspect ratio must be a positive float.")
        self._aspect_ratio = aspect_ratio

    @property
    def figure_style(self) -> str:
        return self._figure_style

    @figure_style.setter
    def figure_style(self, value: str):
        self._figure_style = value

    @property
    def size(self) -> tuple[float, float]:
        return self._size

    @size.setter
    def size(self, value: tuple[float, float]):
        self._size = value

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str):
        self._title = value

    @property
    def log_scale_x(self) -> bool:
        return self._log_scale_x

    @log_scale_x.setter
    def log_scale_x(self, value: bool):
        self._log_scale_x = value

    @property
    def log_scale_y(self) -> bool:
        return self._log_scale_y

    @log_scale_y.setter
    def log_scale_y(self, value: bool):
        self._log_scale_y = value

    @property
    def show_grid(self) -> bool:
        return self._show_grid

    @show_grid.setter
    def show_grid(self, value: bool):
        self._show_grid = value

    @property
    def x_axis_name(self) -> str:
        return self._x_axis_name

    @x_axis_name.setter
    def x_axis_name(self, value: str):
        self._x_axis_name = value

    @property
    def y_axis_name(self) -> str:
        return self._y_axis_name

    @y_axis_name.setter
    def y_axis_name(self, value: str):
        self._y_axis_name = value

    @property
    def x_lim(self) -> tuple[float, float]:
        return self._x_lim

    @x_lim.setter
    def x_lim(self, value: tuple[float, float]):
        self._x_lim = value

    @property
    def y_lim(self) -> tuple[float, float]:
        return self._y_lim

    @y_lim.setter
    def y_lim(self, value: tuple[float, float]):
        self._y_lim = value

    @property
    def remove_axes(self) -> bool:
        return self._remove_axes

    @remove_axes.setter
    def remove_axes(self, value: bool):
        self._remove_axes = value

    @property
    def aspect_ratio(self) -> float | str:
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, value: float | str):
        self._aspect_ratio = value

    def add_elements(self, *elements: Plottable) -> None:
        """
        Adds one or more :class:`~graphinglib.graph_elements.Plottable` elements to the :class:`~graphinglib.figure.Figure`.

        Parameters
        ----------
        elements : :class:`~graphinglib.graph_elements.Plottable`
            Elements to plot in the :class:`~graphinglib.figure.Figure`.
        """
        for element in elements:
            self._elements.append(element)

    def _prepare_figure(
        self,
        legend: bool = True,
        legend_loc: str = None,
        legend_cols: int = 1,
        axes: plt.Axes = None,
        default_params: dict = None,
        is_matplotlib_style: bool = False,
    ):
        """
        Prepares the :class:`~graphinglib.figure.Figure` to be displayed.
        """

        if default_params is not None:
            self._default_params = default_params
            is_a_subfigure = default_params.get("is_a_subfigure", False)
            if not is_a_subfigure:
                self._fill_in_rc_params()
            figure_params_to_reset = self._fill_in_missing_params(self)
        else:
            if self._figure_style == "default":
                self._figure_style = get_default_style()
            try:
                file_loader = FileLoader(self._figure_style)
                self._default_params = file_loader.load()
                self._fill_in_rc_params()
            except FileNotFoundError:
                # set the style use matplotlib style
                try:
                    is_matplotlib_style = True
                    if self._figure_style == "matplotlib":
                        # set the style to default
                        plt.style.use("default")
                    else:
                        plt.style.use(self._figure_style)
                    file_loader = FileLoader("plain")
                    self._default_params = file_loader.load()
                except OSError:
                    raise GraphingException(
                        f"The figure style {self._figure_style} was not found. Please choose a different style."
                    )
            figure_params_to_reset = self._fill_in_missing_params(self)

        if axes is not None:
            self._axes = axes
            if self._title is not None:
                self._axes.set_title(self._title, fontdict={"fontsize": "medium"})
        else:
            self._figure, self._axes = plt.subplots(
                figsize=self._size, layout="constrained"
            )
            if self._title is not None:
                self._axes.set_title(self._title)

        if self._show_grid == "unchanged":
            pass
        elif self._show_grid:
            self._axes.grid(True)
        else:
            self._axes.grid(False)

        self._axes.set_xlabel(self._x_axis_name)
        self._axes.set_ylabel(self._y_axis_name)
        self._axes.set_aspect(self._aspect_ratio)
        if self._custom_ticks:
            if self._xticks:
                self._axes.set_xticks(self._xticks, self._xticklabels)
            if self._xticklabels_rotation:
                self._axes.tick_params("x", labelrotation=self._xticklabels_rotation)
            if self._yticks:
                self._axes.set_yticks(self._yticks, self._yticklabels)
            if self._yticklabels_rotation:
                self._axes.tick_params("y", labelrotation=self._yticklabels_rotation)
        if self._x_lim:
            self._axes.set_xlim(*self._x_lim)
        if self._y_lim:
            self._axes.set_ylim(*self._y_lim)
        if self._log_scale_x:
            self._axes.set_xscale("log")
        if self._log_scale_y:
            self._axes.set_yscale("log")
        if self._remove_axes:
            self._axes.axis("off")
            warn("Axes on this figure have been removed.")
        if self._twin_x_axis:
            labels, handles = self._twin_x_axis._prepare_twin_axis(
                self._axes,
                is_matplotlib_style,
                self._default_params,
                self._figure_style,
            )
            self._handles += handles
            self._labels += labels
        if self._twin_y_axis:
            labels, handles = self._twin_y_axis._prepare_twin_axis(
                self._axes,
                is_matplotlib_style,
                self._default_params,
                self._figure_style,
            )
            self._handles += handles
            self._labels += labels
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        num_cycle_colors = len(cycle_colors)
        if self._elements:
            z_order = 2
            for index, element in enumerate(self._elements):
                params_to_reset = []
                if not is_matplotlib_style:
                    params_to_reset = self._fill_in_missing_params(element)
                element._plot_element(
                    self._axes,
                    z_order,
                    cycle_color=cycle_colors[index % num_cycle_colors],
                )
                if not is_matplotlib_style:
                    self._reset_params_to_default(element, params_to_reset)
                try:
                    if element.label is not None:
                        self._handles.append(element.handle)
                        self._labels.append(element.label)
                except AttributeError:
                    continue
                z_order += 5
            if not self._labels:
                legend = False
            if legend:
                if legend_loc is not None and "outside" in legend_loc:
                    outside_coords = {
                        "outside upper center": (0.5, 1),
                        "outside center right": (1, 0.5),
                        "outside lower center": (0.5, -0.1),
                    }
                    outside_keyword = {
                        "outside upper center": "lower center",
                        "outside center right": "center left",
                        "outside lower center": "upper center",
                    }
                    legend_params = {
                        "loc": outside_keyword[legend_loc],
                        "bbox_to_anchor": outside_coords[legend_loc],
                    }
                else:
                    legend_params = {"loc": legend_loc}
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
                        **legend_params,
                        ncols=legend_cols,
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
                        **legend_params,
                        ncols=legend_cols,
                    )
        else:
            raise GraphingException("No curves to be plotted!")
        self._reset_params_to_default(self, figure_params_to_reset)
        temp_handles = self._handles
        temp_labels = self._labels
        self._handles = []
        self._labels = []
        self._rc_dict = {}
        return temp_labels, temp_handles

    def show(
        self, legend: bool = True, legend_loc: str = "best", legend_cols: int = 1
    ) -> None:
        """
        Displays the :class:`~graphinglib.figure.Figure`.

        Parameters
        ----------
        legend : bool
            Whether or not to display the legend. The legend is always set to be
            draggable.
            Defaults to ``True``.
        legend_loc : str
            Legend location keyword. Any value in {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center"} or {"outside upper center", "outside center right",
            "outside lower center"}.
            Defaults to "best".
        legend_cols : int
            Number of columns in the legend.
            Defaults to 1.
        """
        self._prepare_figure(
            legend=legend, legend_loc=legend_loc, legend_cols=legend_cols
        )
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

    def save(
        self,
        file_name: str,
        legend: bool = True,
        legend_loc: str = "best",
        legend_cols: int = 1,
        dpi: Optional[int] = None,
    ) -> None:
        """
        Saves the :class:`~graphinglib.figure.Figure`.

        Parameters
        ----------
        file_name : str
            The name of the file to save the figure to (including the file extension).
        legend : bool
            Wheter or not to display the legend.
            Defaults to ``True``.
        legend_loc : str
            Legend location keyword. Any value in {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center"} or {"outside upper center", "outside center right",
            "outside lower center"}.
            Defaults to "best".
        legend_cols : int
            Number of columns in the legend.
            Defaults to 1.
        dpi : int
            The resolution of the saved figure. Only used for raster formats (e.g. PNG, JPG, etc.).
            Default depends on the ``figure_style`` configuration.
        """
        self._prepare_figure(
            legend=legend, legend_loc=legend_loc, legend_cols=legend_cols
        )
        if dpi is not None:
            plt.savefig(file_name, bbox_inches="tight", dpi=dpi)
        else:
            plt.savefig(file_name, bbox_inches="tight")
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)

    def _fill_in_missing_params(self, element: Plottable) -> list[str]:
        """
        Fills in the missing parameters from the specified ``figure_style``.
        """
        params_to_reset = []
        object_type = type(element).__name__
        tries = 0
        while tries < 2:
            try:
                for property, value in vars(element).items():
                    if (type(value) == str) and (value == "default"):
                        params_to_reset.append(property)
                        default_value = self._default_params[object_type][property]
                        setattr(element, property, default_value)
                break
            except KeyError as e:
                tries += 1
                if tries >= 2:
                    raise GraphingException(
                        f"There was an error auto updating your {self._figure_style} style file following the recent GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub page. In the meantime, you can manually add the following parameter to your {self._figure_style} style file:\n {e.args[0]}"
                    )
                file_updater = FileUpdater(self._figure_style)
                file_updater.update()
                file_loader = FileLoader(self._figure_style)
                self._default_params = file_loader.load()
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

    def set_rc_params(
        self,
        rc_params_dict: dict[str, str | float] = {},
        reset: bool = False,
    ):
        """
        Customize the visual style of the :class:`~graphinglib.figure.Figure`.

        Any rc parameter that is not specified in the dictionary will be set to the default value for the specified ``figure_style``.

        Parameters
        ----------
        rc_params_dict : dict[str, str | float]
            Dictionary of rc parameters to update.
            Defaults to empty dictionary.
        reset : bool
            Whether or not to reset the rc parameters to the default values for the specified ``figure_style``.
            Defaults to ``False``.
        """
        if reset:
            self._user_rc_dict = {}
        for property, value in rc_params_dict.items():
            self._user_rc_dict[property] = value

    def set_visual_params(
        self,
        reset: bool = False,
        figure_face_color: str | None = None,
        axes_face_color: str | None = None,
        axes_edge_color: str | None = None,
        axes_label_color: str | None = None,
        axes_line_width: float | None = None,
        color_cycle: list[str] | None = None,
        x_tick_color: str | None = None,
        y_tick_color: str | None = None,
        legend_face_color: str | None = None,
        legend_edge_color: str | None = None,
        font_family: str | None = None,
        font_size: float | None = None,
        font_weight: str | None = None,
        text_color: str | None = None,
        use_latex: bool | None = None,
        grid_line_style: str | None = None,
        grid_line_width: float | None = None,
        grid_color: str | None = None,
        grid_alpha: float | None = None,
    ):
        """
        Customize the visual style of the :class:`~graphinglib.figure.Figure`.

        Any parameter that is not specified (None) will be set to the default value for the specified ``figure_style``.

        Parameters
        ----------
        reset : bool
            Whether or not to reset the rc parameters to the default values for the specified ``figure_style``.
            Defaults to ``False``.
        figure_face_color : str
            The color of the figure face.
            Defaults to ``None``.
        axes_face_color : str
            The color of the axes face.
            Defaults to ``None``.
        axes_edge_color : str
            The color of the axes edge.
            Defaults to ``None``.
        axes_label_color : str
            The color of the axes labels.
            Defaults to ``None``.
        axes_line_width : float
            The width of the axes lines.
            Defaults to ``None``.
        color_cycle : list[str]
            A list of colors to use for the color cycle.
            Defaults to ``None``.
        x_tick_color : str
            The color of the x-axis ticks.
            Defaults to ``None``.
        y_tick_color : str
            The color of the y-axis ticks.
            Defaults to ``None``.
        legend_face_color : str
            The color of the legend face.
            Defaults to ``None``.
        legend_edge_color : str
            The color of the legend edge.
            Defaults to ``None``.
        font_family : str
            The font family to use.
            Defaults to ``None``.
        font_size : float
            The font size to use.
            Defaults to ``None``.
        font_weight : str
            The font weight to use.
            Defaults to ``None``.
        text_color : str
            The color of the text.
            Defaults to ``None``.
        use_latex : bool
            Whether or not to use latex.
            Defaults to ``None``.
        grid_line_style : str
            The style of the grid lines.
            Defaults to ``None``.
        grid_line_width : float
            The width of the grid lines.
            Defaults to ``None``.
        grid_color : str
            The color of the grid lines.
            Defaults to ``None``.
        grid_alpha : float
            The alpha of the grid lines.
            Defaults to ``None``.
        """
        if color_cycle is not None:
            color_cycle = plt.cycler(color=color_cycle)

        rc_params_dict = {
            "figure.facecolor": figure_face_color,
            "axes.facecolor": axes_face_color,
            "axes.edgecolor": axes_edge_color,
            "axes.labelcolor": axes_label_color,
            "axes.linewidth": axes_line_width,
            "axes.prop_cycle": color_cycle,
            "xtick.color": x_tick_color,
            "ytick.color": y_tick_color,
            "legend.facecolor": legend_face_color,
            "legend.edgecolor": legend_edge_color,
            "font.family": font_family,
            "font.size": font_size,
            "font.weight": font_weight,
            "text.color": text_color,
            "text.usetex": use_latex,
            "grid.linestyle": grid_line_style,
            "grid.linewidth": grid_line_width,
            "grid.color": grid_color,
            "grid.alpha": grid_alpha,
        }
        rc_params_dict = {
            key: value for key, value in rc_params_dict.items() if value is not None
        }
        self.set_rc_params(rc_params_dict, reset=reset)

    def _fill_in_rc_params(self):
        """
        Fills in the missing rc parameters from the specified ``figure_style``.
        """
        params = self._default_params["rc_params"]
        for property, value in params.items():
            # add to rc_dict if not already in there
            if (property not in self._rc_dict) and (property not in self._user_rc_dict):
                self._rc_dict[property] = value
        all_rc_params = {**self._rc_dict, **self._user_rc_dict}
        try:
            if all_rc_params["text.usetex"] and which("latex") is None:
                all_rc_params["text.usetex"] = False
        except KeyError:
            pass
        plt.rcParams.update(all_rc_params)

    def set_ticks(
        self,
        xticks: Optional[list[float]] = None,
        xticklabels: Optional[list[str]] = None,
        xticklabels_rotation: Optional[float] = None,
        yticks: Optional[list[float]] = None,
        yticklabels: Optional[list[str]] = None,
        yticklabels_rotation: Optional[float] = None,
    ):
        """
        Sets custom [x/y]ticks and [x/y]ticks' labels.

        ..note::
            [x/y]ticks and [x/y]ticks' labels can be omited as long as labels are provided for
            specified ticks.

        Parameters
        ----------
        xticks : list[float], optional
            Tick positions for the x axis.
        xticklabels : list[str], optional
            Tick labels for the x axis.
        xticklabels_rotation : float, optional
            Rotation value for xtick labels.
        yticks : list[float], optional
            Tick positions for the y axis.
        yticklabels : list[str], optional
            Tick labels for the y axis.
        yticklabels_rotation : float, optional
            Rotation value for ytick labels.
        """
        self._custom_ticks = True
        self._xticks = xticks
        self._xticklabels = xticklabels
        self._xticklabels_rotation = xticklabels_rotation
        self._yticks = yticks
        self._yticklabels = yticklabels
        self._yticklabels_rotation = yticklabels_rotation
        if self._xticks or self._yticks:
            if self._yticks and not self._yticklabels:
                raise GraphingException(
                    "Ticks position and corresponding labels must both be specified for the y axis."
                )
            if self._xticks and not self._xticklabels:
                raise GraphingException(
                    "Ticks position and corresponding labels must both be specified for the x axis."
                )

    def create_twin_axis(
        self,
        is_y: bool = True,
        label: str = None,
        log_scale: bool = False,
        axis_lim: Optional[tuple[float, float]] = None,
    ) -> "TwinAxis":
        """
        Creates a twin axis for the :class:`~graphinglib.figure.Figure` object.

        Parameters
        ----------
        is_y : bool
            If ``True``, the twin axis will be a y-axis, otherwise it will be an x-axis.
        label : str
            The identification label for the twin axis.
        log_scale : bool
            Whether or not to set the scale of the twin axis to logaritmic scale.
            Defaults to ``False``.
        axis_lim : tuple[float, float], optional
            The limits for the axis.

        Returns
        -------
        :class:`~graphinglib.figure.TwinAxis`
            The created twin axis.
        """
        if self._remove_axes:
            raise GraphingException(
                "Axis in this figure were removed, therefore twin-axis can't be added."
            )
        twin = TwinAxis(is_y, label, log_scale, axis_lim)
        if is_y:
            self._twin_y_axis = twin
        else:
            self._twin_x_axis = twin
        return twin


class TwinAxis:
    """
    This class implements a twin axis for the :class:`~graphinglib.figure.Figure` class.

    Behaves like a :class:`~graphinglib.figure.Figure` object, but is not meant to be used on its own.
    Elements can be added to the twin axis using the :meth:`~graphinglib.figure.TwinAxis.add_element` method,
    the visual style can be customized using the :meth:`~graphinglib.figure.TwinAxis.customize_visual_style` method,
    and tick labels can be customized using the :meth:`~graphinglib.figure.TwinAxis.set_ticks` method.

    Parameters
    ----------
    is_y : bool
        If ``True``, the twin axis will be a y-axis, otherwise it will be an x-axis.
    label : str
        The identification for the twin axis.
    log_scale : bool
        Whether or not to set the scale of the twin axis to logaritmic scale.
        Defaults to ``False``.
    axis_lim : tuple[float, float], optional
            The limits for the axis.
    """

    def __init__(
        self,
        is_y: bool = True,
        label: Optional[str] = None,
        log_scale: bool = False,
        axis_lim: Optional[tuple[float, float]] = None,
    ):
        """
        This class implements a twin axis for the :class:`~graphinglib.figure.Figure` class.

        Behaves like a :class:`~graphinglib.figure.Figure` object, but is not meant to be used on its own.
        Elements can be added to the twin axis using the :meth:`~graphinglib.figure.TwinAxis.add_element` method,
        the visual style can be customized using the :meth:`~graphinglib.figure.TwinAxis.customize_visual_style` method,
        and tick labels can be customized using the :meth:`~graphinglib.figure.TwinAxis.set_ticks` method.

        Parameters
        ----------
        is_y : bool
            If ``True``, the twin axis will be a y-axis, otherwise it will be an x-axis.
        label : str, optional
            The identification for the twin axis.
        log_scale : bool
            Whether or not to set the scale of the twin axis to logaritmic scale.
            Defaults to ``False``.
        axis_lim : tuple[float, float], optional
            The limits for the axis.
        """
        self._is_y = is_y
        self._label = label
        self._log_scale = log_scale
        self._elements: list[Plottable] = []
        self._custom_ticks = False
        self._labels: list[str | None] = []
        self._handles = []
        self._figure_style = None
        self._default_params = None
        self._tick_color = None
        self._axes_label_color = None
        self._axes_edge_color = None
        self._axis_lim = axis_lim

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    @property
    def log_scale(self) -> bool:
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value: bool):
        self._log_scale = value

    @property
    def axis_lim(self) -> tuple[float, float]:
        return self._axis_lim

    def _prepare_twin_axis(
        self,
        fig_axes: plt.Axes,
        is_matplotlib_style: bool = False,
        default_params: dict = None,
        figure_style: str = "default",
    ):
        """
        Prepares the :class:`~graphinglib.figure.TwinAxis` to be displayed.
        """
        self._default_params = default_params
        self._figure_style = (
            figure_style if figure_style != "default" else get_default_style()
        )
        if self._is_y:
            self._axes = fig_axes.twinx()
            self._axes.set_ylabel(self._label)
            if self._axis_lim:
                self._axes.set_ylim(*self._axis_lim)
        else:
            self._axes = fig_axes.twiny()
            self._axes.set_xlabel(self._label)
            if self._axis_lim:
                self._axes.set_xlim(*self._axis_lim)
        if self._is_y:
            if self._tick_color:
                self._axes.tick_params(axis="y", colors=self._tick_color)
            if self._axes_label_color:
                self._axes.yaxis.label.set_color(self._axes_label_color)
            if self._axes_edge_color:
                self._axes.spines["right"].set_color(self._axes_edge_color)
        else:
            if self._tick_color:
                self._axes.tick_params(axis="x", colors=self._tick_color)
            if self._axes_label_color:
                self._axes.xaxis.label.set_color(self._axes_label_color)
            if self._axes_edge_color:
                self._axes.spines["top"].set_color(self._axes_edge_color)
        if self._custom_ticks:
            if self._ticks:
                if self._is_y:
                    self._axes.set_yticks(self._ticks, self._ticklabels)
                    if self._ticklabels_rotation:
                        self._axes.tick_params(
                            "y", labelrotation=self._ticklabels_rotation
                        )
                else:
                    self._axes.set_xticks(self._ticks, self._ticklabels)
                    if self._ticklabels_rotation:
                        self._axes.tick_params(
                            "x", labelrotation=self._ticklabels_rotation
                        )
        if self._log_scale:
            if self._is_y:
                self._axes.set_yscale("log")
            else:
                self._axes.set_xscale("log")
        z_order = 1
        for element in self._elements:
            params_to_reset = []
            if not is_matplotlib_style:
                params_to_reset = self._fill_in_missing_params(element)
            element._plot_element(self._axes, z_order)
            if not is_matplotlib_style:
                self._reset_params_to_default(element, params_to_reset)
            try:
                if element.label is not None:
                    self._handles.append(element.handle)
                    self._labels.append(element.label)
            except AttributeError:
                continue
            z_order += 2
        temp_handles = self._handles
        temp_labels = self._labels
        self._handles = []
        self._labels = []
        self._rc_dict = {}
        return temp_labels, temp_handles

    def set_ticks(
        self,
        ticks: list[float],
        ticklabels: list[str],
        ticklabels_rotation: Optional[float] = None,
    ):
        """
        Sets custom ticks and labels for the twin axis.

        Parameters
        ----------
        ticks : list[float], optional
            Tick positions for the axis.
        ticklabels : list[str], optional
            Tick labels for the axis.
        ticklabels_rotation : float, optional
            Rotation value for the tick labels.
        """
        if not ticks or not ticklabels:
            raise GraphingException(
                "Ticks position and corresponding labels must both be specified for the twin axis."
            )
        if len(ticks) != len(ticklabels):
            raise GraphingException(
                f"Number of ticks ({len(ticks)}) and number of tick labels ({len(ticklabels)}) must be the same."
            )
        self._custom_ticks = True
        self._ticks = ticks
        self._ticklabels = ticklabels
        self._ticklabels_rotation = ticklabels_rotation

    def add_elements(self, *elements: Plottable) -> None:
        """
        Adds one or more :class:`~graphinglib.graph_elements.Plottable` elements to the :class:`~graphinglib.figure.Figure`.

        Parameters
        ----------
        elements : :class:`~graphinglib.graph_elements.Plottable`
            Elements to plot in the :class:`~graphinglib.figure.Figure`.
        """
        for element in elements:
            self._elements.append(element)

    def set_visual_params(
        self,
        axes_label_color: str | None = None,
        tick_color: str | None = None,
        axes_edge_color: str | None = None,
    ):
        """
        Customize the visual style of the :class:`~graphinglib.figure.Figure`.

        Any parameter that is not specified (None) will be set to the default value for the specified ``figure_style``.

        Parameters
        ----------
        axes_edge_color : str
            The color of the axes edge.
            Defaults to ``None``.
        axes_label_color : str
            The color of the axes labels.
            Defaults to ``None``.
        tick_color : str
            The color of the axis ticks.
            Defaults to ``None``.
        """
        self._axes_label_color = axes_label_color
        self._tick_color = tick_color
        self._axes_edge_color = axes_edge_color

    def _fill_in_missing_params(self, element: Plottable) -> list[str]:
        """
        Fills in the missing parameters from the specified ``figure_style``.
        """
        params_to_reset = []
        object_type = type(element).__name__
        tries = 0
        curve_defaults = {
            "_errorbars_color": "_color",
            "_errorbars_line_width": "_line_width",
            "_cap_thickness": "_line_width",
            "_fill_under_color": "_color",
        }
        while tries < 2:
            try:
                for property, value in vars(element).items():
                    if (type(value) == str) and (value == "default"):
                        params_to_reset.append(property)
                        default_value = self._default_params[object_type][property]
                        if default_value == "same as curve":
                            setattr(
                                element,
                                property,
                                getattr(element, curve_defaults[property]),
                            )
                        elif default_value == "same as scatter":
                            element.errorbars_color = getattr(element, "_face_color")
                        else:
                            setattr(element, property, default_value)
                break
            except KeyError as e:
                tries += 1
                if tries >= 2:
                    raise GraphingException(
                        f"There was an error auto updating your {self._figure_style} style file following the recent GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub page. In the meantime, you can manually add the following parameter to your {self._figure_style} style file:\n {e.args[0]}"
                    )
                file_updater = FileUpdater(self._figure_style)
                file_updater.update()
                file_loader = FileLoader(self._figure_style)
                self._default_params = file_loader.load()
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
