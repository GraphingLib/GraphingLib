from __future__ import annotations
from shutil import which
from typing import Literal, Optional, Any
from warnings import warn
from string import ascii_lowercase
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation
from matplotlib.figure import SubFigure

from graphinglib.file_manager import (
    FileLoader,
    FileUpdater,
    get_default_style,
)
from graphinglib.graph_elements import GraphingException, Plottable
from graphinglib.legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)

import numpy as np
from numpy.typing import ArrayLike

class SmartFigure:
    def __init__(
        self,
        num_rows: int = 1,
        num_cols: int = 1,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        size: tuple[float, float] | Literal["default"] = "default",
        title: Optional[str] = None,
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        log_scale_x: bool = False,
        log_scale_y: bool = False,
        remove_axes: bool = False,
        aspect_ratio: float | str = "auto",
        remove_x_ticks: bool = False,
        remove_y_ticks: bool = False,
        reference_labels: bool = True,
        reflabel_loc: str = "outside",
        width_padding: float = None,
        height_padding: float = None,
        width_ratios: ArrayLike = None,
        height_ratios: ArrayLike = None,
        share_x: bool = False,
        share_y: bool = False,
        projection: str | Any = None,
        figure_style: str = "default",
        elements: Optional[list[Plottable]] = [],
    ) -> None:
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._x_label = x_label
        self._y_label = y_label
        self._size = size
        self._title = title
        self._x_lim = x_lim
        self._y_lim = y_lim
        self._log_scale_x = log_scale_x
        self._log_scale_y = log_scale_y
        self._remove_axes = remove_axes
        self._aspect_ratio = aspect_ratio
        self._remove_x_ticks = remove_x_ticks
        self._remove_y_ticks = remove_y_ticks
        self._reference_labels = reference_labels
        self._reflabel_loc = reflabel_loc
        self._width_padding = width_padding
        self._height_padding = height_padding
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios
        self._share_x = share_x
        self._share_y = share_y
        self._projection = projection
        self._figure_style = figure_style

        self._elements = {}
        for i, element in enumerate(elements):
            if isinstance(element, (Plottable, list, SmartFigure)):
                self._elements[self._keys_to_slices(divmod(i, self._num_cols))] = element
            else:
                raise GraphingException(f"Invalid element type: {type(element).__name__}")

        self._figure = None
        self._reference_label_i = None

        self._custom_ticks = False
        self._xticks = None
        self._xticklabels = None
        self._xticklabels_rotation = None
        self._xtick_spacing = None
        self._yticks = None
        self._yticklabels = None
        self._yticklabels_rotation = None
        self._ytick_spacing = None
        self._minor_xticks = None
        self._minor_xtick_spacing = None
        self._minor_yticks = None
        self._minor_ytick_spacing = None

        self._show_grid = False
        self._grid_visible_x = None
        self._grid_which_x = None
        self._grid_visible_y = None
        self._grid_which_y = None

        self._rc_dict = {}
        self._user_rc_dict = {}
        self._default_params = {}

    def __len__(self) -> int:
        return len(self._elements)
    
    def __setitem__(self, key: tuple[slice | int], element: Plottable | list[Plottable] | SmartFigure) -> None:
        key_ = self._keys_to_slices(key)
        if element is None:
            if key_ in self._elements.keys():
                del self._elements[key_]
        else:
            self._elements[key_] = element

    def __getitem__(self, key: tuple[slice | int]) -> Plottable | list[Plottable] | SmartFigure:
        key_ = self._keys_to_slices(key)
        return self._elements.get(key_, None)

    @property
    def _ordered_elements(self) -> OrderedDict:
        return OrderedDict(sorted(self._elements.items(), key=lambda item: (item[0][0].start, item[0][1].start)))

    @staticmethod
    def _keys_to_slices(keys: tuple[slice | int]) -> tuple[slice]:
        new_slices = [k if isinstance(k, slice) else slice(k, k+1, None) for k in keys]   # convert int -> slice
        new_slices = [s if s.start is not None else slice(0, s.stop) for s in new_slices] # convert starting None -> 0
        return tuple(new_slices)

    def _get_reflabel_translation(
            self,
    ) -> ScaledTranslation:
        if self._reflabel_loc == "outside":
            return ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
        elif self._reflabel_loc == "inside":
            return ScaledTranslation(10 / 72, -15 / 72, self._figure.dpi_scale_trans)
        else:
            raise ValueError("Invalid reference label location. Please specify either 'inside' or 'outside'.")

    def show(
        self,
    ) -> None:
        self._initialize_parent_smart_figure()
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

    def save(
        self,
        file_name: str,
        dpi: Optional[int] = None,
        transparent: bool = False,
    ) -> None:
        self._initialize_parent_smart_figure()
        plt.savefig(
            file_name,
            bbox_inches="tight",
            dpi=dpi if dpi is not None else "figure",
            transparent=transparent,
        )
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)

    def _initialize_parent_smart_figure(
        self,
    ) -> None:
        if self._figure_style == "default":
            self._figure_style = get_default_style()
        try:
            file_loader = FileLoader(self._figure_style)
            self._default_params = file_loader.load()
            is_matplotlib_style = False
        except FileNotFoundError:
            is_matplotlib_style = True
            try:
                if self._figure_style == "matplotlib":
                    plt.style.use("default")
                else:
                    plt.style.use(self._figure_style)
                file_loader = FileLoader("plain")
                self._default_params = file_loader.load()
            except OSError:
                raise GraphingException(
                    f"The figure style {self._figure_style} was not found. Please choose a different style."
                )

        multi_figure_params_to_reset = self._fill_in_missing_params(self)
        self._fill_in_rc_params(is_matplotlib_style)

        self._figure = plt.figure(constrained_layout=True, figsize=self._size)
        self._figure.set_constrained_layout_pads(w_pad=0, h_pad=0)
        self._reference_label_i = 0
        main_gridspec = self._prepare_figure(self._default_params, is_matplotlib_style)

        self._reset_params_to_default(self, multi_figure_params_to_reset)
        self._rc_dict = {}

        # Create an artificial axis to add padding around the figure
        ax_dummy = self._figure.add_subplot(main_gridspec[:, :])
        ax_dummy.xaxis.grid(False)
        ax_dummy.yaxis.grid(False)
        ax_dummy.set_facecolor((0, 0, 0, 0))
        ax_dummy.set_zorder(-1)
        ax_dummy.set_navigate(False)
        ax_dummy.tick_params(colors=(0,0,0,0), axis="both", direction="in", labelright=True, labeltop=True, labelsize=0)
        ax_dummy.spines["top"].set_visible(False)
        ax_dummy.spines["right"].set_visible(False)
        ax_dummy.spines["left"].set_visible(False)
        ax_dummy.spines["bottom"].set_visible(False)
        ax_dummy.set_xticks([0.5])
        ax_dummy.set_yticks([0.5])
        ax_dummy.set_xticklabels([" "])
        ax_dummy.set_yticklabels([" "])

    def _prepare_figure(
        self,
        default_params: dict = None,
        is_matplotlib_style: bool = False,
    ) -> GridSpec:
        sub_rcs = self._user_rc_dict
        plt.rcParams.update(sub_rcs)
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        num_cycle_colors = len(cycle_colors)

        gridspec = self._figure.add_gridspec(
            self._num_rows,
            self._num_cols,
            wspace=self._width_padding,
            hspace=self._height_padding,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
        )
        # Plottable and subfigure plotting
        ax = None   # keep track of the last plt.Axes object, needed for sharing axes
        for (rows, cols), element in self._ordered_elements.items():
            if isinstance(element, SmartFigure):
                subfig = self._figure.add_subfigure(gridspec[rows, cols])
                element._figure = subfig        # associates the current subfigure with the nested SmartFigure
                element._reference_label_i = self._reference_label_i
                default_params_copy = default_params.copy()
                default_params_copy.update(is_a_subfigure=True)
                default_params_copy["Figure"]["_figure_style"] = self._figure_style
                element._prepare_figure(default_params_copy, is_matplotlib_style)
                self._reference_label_i = element._reference_label_i

                self._fill_in_rc_params(is_matplotlib_style)

            elif isinstance(element, (Plottable, list)):
                current_elements = element if isinstance(element, list) else [element]
                subfig = self._figure.add_subfigure(gridspec[rows, cols])
                ax = subfig.add_subplot(
                    sharex=ax if self._share_x else None,
                    sharey=ax if self._share_y else None,
                    projection=self._projection,
                )
                self._default_params = default_params
                figure_params_to_reset = self._fill_in_missing_params(self)

                # Plotting loop
                z_order = 2
                for index, current_element in enumerate(current_elements):
                    if isinstance(current_element, Plottable):
                        params_to_reset = []
                        if not is_matplotlib_style:
                            params_to_reset = self._fill_in_missing_plottable_params(current_element)
                        current_element._plot_element(
                            ax,
                            z_order,
                            cycle_color=cycle_colors[index % num_cycle_colors],
                        )
                        if not is_matplotlib_style:
                            self._reset_params_to_default(current_element, params_to_reset)
                        z_order += 5
                    else:
                        raise GraphingException(f"Unsupported element type: {type(current_element).__name__}")

                # Add reference label
                if self._reference_labels and (len(self) > 1 or isinstance(self._figure, SubFigure)):
                    ax.text(
                        0,
                        1,
                        ascii_lowercase[self._reference_label_i] + ")",
                        transform=ax.transAxes + self._get_reflabel_translation(),
                    )
                    self._reference_label_i += 1

                # If axes are shared, manually remove ticklabels from unnecessary plots
                if self._share_x and rows.start != (self._num_rows - 1):
                    ax.tick_params(labelbottom=False)
                if self._share_y and cols.start != 0:
                    ax.tick_params(labelleft=False)

                # Remove ticks
                if self._remove_x_ticks:
                    ax.get_xaxis().set_visible(False)
                if self._remove_y_ticks:
                    ax.get_yaxis().set_visible(False)

                # Axes limits
                if self._x_lim:
                    ax.set_xlim(*self._x_lim)
                if self._y_lim:
                    ax.set_ylim(*self._y_lim)

                # Logarithmic scale
                if self._log_scale_x:
                    ax.set_xscale("log")
                if self._log_scale_y:
                    ax.set_yscale("log")

                # Remove axes
                if self._remove_axes:
                    ax.axis("off")

                ax.set_aspect(self._aspect_ratio)

                # Customize ticks
                if self._custom_ticks:
                    if self._xticks:
                        ax.set_xticks(self._xticks, self._xticklabels)
                    if self._xticklabels_rotation:
                        ax.tick_params("x", labelrotation=self._xticklabels_rotation)
                    if self._xtick_spacing:
                        ax.xaxis.set_major_locator(
                            ticker.MultipleLocator(self._xtick_spacing)
                        )
                    if self._yticks:
                        ax.set_yticks(self._yticks, self._yticklabels)
                    if self._yticklabels_rotation:
                        ax.tick_params("y", labelrotation=self._yticklabels_rotation)
                    if self._ytick_spacing:
                        ax.yaxis.set_major_locator(
                            ticker.MultipleLocator(self._ytick_spacing)
                        )
                    if self._minor_xticks:
                        ax.set_xticks(self._minor_xticks, minor=True)
                    if self._minor_xtick_spacing:
                        ax.xaxis.set_minor_locator(
                            ticker.MultipleLocator(self._minor_xtick_spacing)
                        )
                    if self._minor_yticks:
                        ax.set_yticks(self._minor_yticks, minor=True)
                    if self._minor_ytick_spacing:
                        ax.yaxis.set_minor_locator(
                            ticker.MultipleLocator(self._minor_ytick_spacing)
                        )

                # Customize grid
                if self._show_grid:
                    ax.grid(self._grid_visible_x, self._grid_which_x, "x")
                    ax.grid(self._grid_visible_y, self._grid_which_y, "y")

                self._reset_params_to_default(self, figure_params_to_reset)

            elif element is not None:
                raise GraphingException(f"Unsupported element type in list: {type(element).__name__}")

        # Axes labels
        if self._num_cols == 1 and self._num_rows == 1:
            if ax is not None:  # makes sure an element was plotted and that an axis was created
                ax.set_xlabel(self._x_label)
                ax.set_ylabel(self._y_label)
        else:
            suplabel_params = {
                "fontsize" : plt.rcParams["font.size"],
                "color" : plt.rcParams["axes.labelcolor"],
                "fontweight" : plt.rcParams["font.weight"],
            }
            self._figure.supxlabel(self._x_label, **suplabel_params)
            self._figure.supylabel(self._y_label, **suplabel_params)

        # Title
        if self._title:
            self._figure.suptitle(self._title, fontdict={"fontsize": "medium"}, fontweight=plt.rcParams["font.weight"])

        return gridspec

    def _fill_in_missing_params(self, element: SmartFigure) -> list[str]:
        """
        Fills in the missing parameters from the specified ``figure_style``.
        """
        params_to_reset = []
        object_type = type(element).__name__
        for property, value in vars(element).items():
            if isinstance(value, str) and (value == "default") and not (property == "_figure_style"):
                params_to_reset.append(property)
                element.__dict__[property] = self._default_params[object_type][property]
        return params_to_reset

    def _fill_in_missing_plottable_params(self, element: Plottable) -> list[str]:
        """
        Fills in the missing parameters for a plottable from the specified ``figure_style``.
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
                        f"There was an error auto updating your {self._figure_style} style file following the recent "
                         "GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub"
                         "page. In the meantime, you can manually add the following parameter to your "
                        f"{self._figure_style} style file:\n {e.args[0]}"
                    )
                file_updater = FileUpdater(self._figure_style)
                file_updater.update()
                file_loader = FileLoader(self._figure_style)
                self._default_params = file_loader.load()
        return params_to_reset

    def _reset_params_to_default(
        self, element: Plottable | SmartFigure, params_to_reset: list[str]
    ) -> None:
        """
        Resets the parameters that were set to default in the _fill_in_missing_params method.
        """
        for param in params_to_reset:
            setattr(element, param, "default")

    def _fill_in_rc_params(self, is_matplotlib_style: bool = False) -> None:
        """
        Fills in and sets the missing rc parameters from the specified ``figure_style``.
        If ``is_matplotlib_style`` is ``True``, the rc parameters are reset to the default values for the specified 
        ``figure_style``. If ``is_matplotlib_style`` is ``False``, the rc parameters are updated with the missing
        parameters from the specified ``figure_style``. In both cases, the rc parameters are then updated with the
        user-specified parameters.
        """
        if is_matplotlib_style:
            if self._figure_style == "matplotlib":
                plt.style.use("default")
            else:
                plt.style.use(self._figure_style)
            plt.rcParams.update(self._user_rc_dict)
        else:
            params = self._default_params["rc_params"]
            for property, value in params.items():
                # add to rc_dict if not already in there
                if (property not in self._rc_dict) and (
                    property not in self._user_rc_dict
                ):
                    self._rc_dict[property] = value
            all_rc_params = {**self._rc_dict, **self._user_rc_dict}
            try:
                if all_rc_params["text.usetex"] and which("latex") is None:
                    all_rc_params["text.usetex"] = False
            except KeyError:
                pass
            plt.rcParams.update(all_rc_params)

    def set_rc_params(
        self,
        rc_params_dict: dict[str, str | float] = {},
        reset: bool = False,
    ) -> None:
        """
        Customize the visual style of the :class:`~graphinglib.smart_figure.SmartFigure`.

        Any rc parameter that is not specified in the dictionary will be set to the default value for the specified
        ``figure_style``.

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
        # legend_face_color: str | None = None,
        # legend_edge_color: str | None = None,
        font_family: str | None = None,
        font_size: float | None = None,
        font_weight: str | None = None,
        text_color: str | None = None,
        use_latex: bool | None = None,
    ) -> None:
        """
        Customize the visual style of the :class:`~graphinglib.smart_figure.SmartFigure`.

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
        # legend_face_color : str
        #     The color of the legend face.
        #     Defaults to ``None``.
        # legend_edge_color : str
        #     The color of the legend edge.
        #     Defaults to ``None``.
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
            # "legend.facecolor": legend_face_color,
            # "legend.edgecolor": legend_edge_color,
            "font.family": font_family,
            "font.size": font_size,
            "font.weight": font_weight,
            "text.color": text_color,
            "text.usetex": use_latex,
        }
        rc_params_dict = {
            key: value for key, value in rc_params_dict.items() if value is not None
        }
        self.set_rc_params(rc_params_dict, reset=reset)

    def set_ticks(
        self,
        xticks: Optional[list[float]] = None,
        xticklabels: Optional[list[str]] = None,
        xticklabels_rotation: Optional[float] = None,
        xtick_spacing: Optional[float] = None,
        yticks: Optional[list[float]] = None,
        yticklabels: Optional[list[str]] = None,
        yticklabels_rotation: Optional[float] = None,
        ytick_spacing: Optional[float] = None,
        minor_xticks: Optional[list[float]] = None,
        minor_xtick_spacing: Optional[float] = None,
        minor_yticks: Optional[list[float]] = None,
        minor_ytick_spacing: Optional[float] = None,
    ) -> None:
        """
        Sets custom ticks and ticks labels.

        Parameters
        ----------
        xticks : list[float], optional
            Tick positions for the x axis. If a value is specified, the xtick_spacing parameter cannot be specified.
        xticklabels : list[str], optional
            Tick labels for the x axis. If a value is specified, the xticks parameter must also be specified.
        xticklabels_rotation : float, optional
            Rotation value for xtick labels.
        xtick_spacing : float, optional
            Spacing between ticks on the x axis. If a value is specified, the xticks parameter cannot be specified.
        yticks : list[float], optional
            Tick positions for the y axis. If a value is specified, the ytick_spacing parameter cannot be specified.
        yticklabels : list[str], optional
            Tick labels for the y axis. If a value is specified, the yticks parameter must also be specified.
        yticklabels_rotation : float, optional
            Rotation value for ytick labels.
        ytick_spacing : float, optional
            Spacing between ticks on the y axis. If a value is specified, the yticks parameter cannot be specified.
        minor_xticks : list[float], optional
            Minor tick positions for the x axis. If a value is specified, minor_the xtick_spacing parameter cannot be 
            specified.
        minor_xtick_spacing : float, optional
            Spacing between minor ticks on the x axis. If a value is specified, the minor_xticks parameter cannot be
            specified.
        minor_yticks : list[float], optional
            Minor tick positions for the y axis. If a value is specified, the minor_ytick_spacing parameter cannot be 
            specified.
        minor_ytick_spacing : float, optional
            Spacing between minor ticks on the y axis. If a value is specified, the minor_yticks parameter cannot be
            specified.
        """
        if any([
            self._xticklabels and not self._xticks,
            self._yticklabels and not self._yticks,
        ]):
            raise GraphingException("Ticks position must be specified when ticks labels are specified")
        
        if any([
            self._xticks and self._xtick_spacing, 
            self._yticks and self._ytick_spacing,
            self._minor_xticks and self._minor_xtick_spacing, 
            self._minor_yticks and self._minor_ytick_spacing,
        ]):
            raise GraphingException("Tick spacing and tick positions cannot be set simultaneously")

        self._custom_ticks = True
        self._xticks = xticks
        self._xticklabels = xticklabels
        self._xticklabels_rotation = xticklabels_rotation
        self._xtick_spacing = xtick_spacing
        self._yticks = yticks
        self._yticklabels = yticklabels
        self._yticklabels_rotation = yticklabels_rotation
        self._ytick_spacing = ytick_spacing
        self._minor_xticks = minor_xticks
        self._minor_xtick_spacing = minor_xtick_spacing
        self._minor_yticks = minor_yticks
        self._minor_ytick_spacing = minor_ytick_spacing

    def set_grid(
        self,
        visible_x: bool = True,
        visible_y: bool = True,
        which_x: Literal["both", "major", "minor"] = "both",
        which_y: Literal["both", "major", "minor"] = "both",
        color: str = "default",
        alpha: float | Literal["default"] = "default",
        line_style: str = "default",
        line_width: float | Literal["default"] = "default",
    ) -> None:
        """
        Sets the grid parameters for the figure.

        Parameters
        ----------
        visible_x : bool, optional
            If ``True``, sets the x-axis grid visible. Defaults to ``True``.
        visible_y : bool, optional
            If ``True``, sets the y-axis grid visible. Defaults to ``True``.
        which_x : {"both", "major", "minor"}, optional
            Sets whether both, only major or only minor grid lines are shown for the
            x-axis. Defaults to ``"both"``.
        which_y : {"both", "major", "minor"}, optional
            Sets whether both, only major or only minor grid lines are shown for the
            y-axis. Defaults to ``"both"``.
        color : str, optional
            sets the color of the grid lines.
            Default depends on the ``figure_style`` configuration.
        alpha : float, optional
            Sets the alpha value for the grid lines.
            Default depends on the ``figure_style`` configuration.
        line_style : str, optional
            Sets the line style of the grid lines.
            Default depends on the ``figure_style`` configuration.
        line_width : float, optional
            Sets the line width of the grid lines.
            Default depends on the ``figure_style`` configuration.
        """
        self._show_grid = True
        self._grid_visible_x = visible_x
        self._grid_visible_y = visible_y
        self._grid_which_x = which_x
        self._grid_which_y = which_y
        rc_params_dict = {
            "grid.color": color,
            "grid.alpha": alpha,
            "grid.linestyle": line_style,
            "grid.linewidth": line_width,
        }
        rc_params_dict = {k: v for k, v in rc_params_dict.items() if v is not "default"}
        self.set_rc_params(rc_params_dict)
