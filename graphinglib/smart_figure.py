from __future__ import annotations
from shutil import which
from typing import Literal, Optional, Any
from warnings import warn
from string import ascii_lowercase
from collections import OrderedDict
from typing import Self
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import ScaledTranslation
from matplotlib.figure import SubFigure
from matplotlib.axes import Axes

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
        global_reference_label: bool = False,
        reflabel_loc: str = "outside",
        width_padding: float = None,
        height_padding: float = None,
        width_ratios: ArrayLike = None,
        height_ratios: ArrayLike = None,
        share_x: bool = False,
        share_y: bool = False,
        projection: str | Any = None,
        general_legend: bool = False,
        legend_loc: str | tuple = "best",
        legend_cols: int = 1,
        show_legend: bool = True,
        figure_style: str = "default",
        elements: Optional[list[list[Plottable]]] = [],
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
        self._global_reference_label = global_reference_label
        self._reflabel_loc = reflabel_loc
        self._width_padding = width_padding
        self._height_padding = height_padding
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios
        self._share_x = share_x
        self._share_y = share_y
        self._projection = projection
        self._general_legend = general_legend
        self._legend_loc = legend_loc
        self._legend_cols = legend_cols
        self._show_legend = show_legend
        self._figure_style = figure_style

        self._elements = {}
        if elements:
            if num_rows == 1 and num_cols == 1 and not isinstance(elements[0], list):
                elements = [elements]
            for i, element in enumerate(elements):
                if isinstance(element, (Plottable, list, SmartFigure)):
                    if isinstance(element, Plottable):
                        element = [element]
                    self._elements[self._keys_to_slices(divmod(i, self._num_cols))] = element
                elif element is not None:
                    raise GraphingException(f"Invalid element type: {type(element).__name__}")

        self._figure = None
        self._gridspec = None
        self._reference_label_i = None

        self._x_ticks = None
        self._x_tick_labels = None
        self._x_tick_labels_rotation = None
        self._x_tick_spacing = None
        self._y_ticks = None
        self._y_tick_labels = None
        self._y_tick_labels_rotation = None
        self._y_tick_spacing = None
        self._minor_x_ticks = None
        self._minor_x_tick_spacing = None
        self._minor_y_ticks = None
        self._minor_y_tick_spacing = None
        self._tick_params = {"x major": {}, "y major": {}, "x minor": {}, "y minor": {}}

        self._show_grid = False
        self._grid_visible_x = None
        self._grid_visible_y = None
        self._grid_show_on_top = None
        self._grid_which_x = None
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
                self._elements.pop(key_)
        else:
            self._elements[key_] = element

    def __getitem__(self, key: tuple[slice | int]) -> Plottable | list[Plottable] | SmartFigure:
        key_ = self._keys_to_slices(key)
        return self._elements.get(key_, [])

    def copy(self) -> Self:
        return deepcopy(self)

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
        target: Axes | SubFigure,
    ) -> ScaledTranslation:
        if isinstance(target, Axes):
            if self._reflabel_loc == "outside":
                return ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
            elif self._reflabel_loc == "inside":
                return ScaledTranslation(10 / 72, -15 / 72, self._figure.dpi_scale_trans)
            else:
                raise ValueError("Invalid reference label location. Please specify either 'inside' or 'outside'.")

        elif isinstance(target, SubFigure):
            return ScaledTranslation(7 / 72, -10 / 72, self._figure.dpi_scale_trans)
        else:
            raise ValueError("Target must be either an Axes or SubFigure instance.")
    
    def add_elements(self, *elements: Plottable) -> None:
        """
        Adds one or more :class:`~graphinglib.graph_elements.Plottable` elements to the 
        :class:`~graphinglib.smart_figure.SmartFigure`. This convenience method is equivalent to using __setitem__, but
        only works if the SmartFigure contains a single plot (1x1).

        Parameters
        ----------
        elements : :class:`~graphinglib.graph_elements.Plottable`
            Elements to plot in the :class:`~graphinglib.smart_figure.SmartFigure`.
        """
        if self._num_rows != 1 or self._num_cols != 1:
            raise GraphingException("The add_elements() method only works for 1x1 SmartFigures.")
        self[0,0] += elements

    def show(
        self,
    ) -> None:
        self._initialize_parent_smart_figure()

        # Create an artificial axis to add padding around the figure
        # This is needed because the figure is created with set_constrained_layout_pads creating 0 padding
        ax_dummy = self._figure.add_subplot(self._gridspec[:, :])
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

        if all([
            self._show_legend,
            self._general_legend,
            self._legend_loc is not None and "outside" in self._legend_loc
        ]):
            warn("The general legend location is set to 'outside' and matplotlib windows may not be able to show it "
                 "properly. Consider using inline figures in a jupyter notebook or saving the figure to a file instead "
                 "to get the full figure.")
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
        self._prepare_figure(self._default_params, is_matplotlib_style)

        self._reset_params_to_default(self, multi_figure_params_to_reset)
        self._rc_dict = {}

    def _prepare_figure(
        self,
        default_params: dict = None,
        is_matplotlib_style: bool = False,
        make_legend: bool = True,
    ) -> tuple[list[str], list[Any]]:
        sub_rcs = self._user_rc_dict
        plt.rcParams.update(sub_rcs)
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        num_cycle_colors = len(cycle_colors)

        self._gridspec = self._figure.add_gridspec(
            self._num_rows,
            self._num_cols,
            wspace=self._width_padding,
            hspace=self._height_padding,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
        )

        if self._global_reference_label:
            self._create_ref_label(self._figure)
            self._figure.suptitle(" ")     # Create a blank title to reserve space

        # Plottable and subfigure plotting
        ax = None   # keep track of the last plt.Axes object, needed for sharing axes
        labels, handles = [], []
        for (rows, cols), element in self._ordered_elements.items():
            if isinstance(element, SmartFigure):
                subfig = self._figure.add_subfigure(self._gridspec[rows, cols])
                element._figure = subfig        # associates the current subfigure with the nested SmartFigure
                element._reference_label_i = self._reference_label_i
                default_params_copy = default_params.copy()
                default_params_copy.update(is_a_subfigure=True)
                default_params_copy["Figure"]["_figure_style"] = self._figure_style
                element_labels, element_handles = element._prepare_figure(
                    default_params=default_params_copy, 
                    is_matplotlib_style=is_matplotlib_style,
                    make_legend=(not self._general_legend),
                )
                self._reference_label_i = element._reference_label_i
                labels += element_labels
                handles += element_handles

                self._fill_in_rc_params(is_matplotlib_style)

            elif isinstance(element, (Plottable, list)):
                current_elements = element if isinstance(element, list) else [element]
                subfig = self._figure.add_subfigure(self._gridspec[rows, cols])
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
                        try:
                            if current_element.label is not None:
                                labels.append(current_element.label)
                                handles.append(current_element.handle)
                        except AttributeError:
                            continue
                        z_order += 5
                    elif current_element is not None:
                        raise GraphingException(f"Unsupported element type: {type(current_element).__name__}")

                # Add reference label
                if self._reference_labels and (len(self) > 1 or isinstance(self._figure, SubFigure)):
                    self._create_ref_label(ax)

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
                if self._x_ticks:
                    ax.set_xticks(self._x_ticks, self._x_tick_labels)
                ax.tick_params(axis="x", which="major", **self._tick_params["x major"])
                if self._x_tick_spacing:
                    ax.xaxis.set_major_locator(
                        ticker.MultipleLocator(self._x_tick_spacing)
                    )
                if self._y_ticks:
                    ax.set_yticks(self._y_ticks, self._y_tick_labels)
                ax.tick_params(axis="y", which="major", **self._tick_params["y major"])
                if self._y_tick_spacing:
                    ax.yaxis.set_major_locator(
                        ticker.MultipleLocator(self._y_tick_spacing)
                    )
                if self._minor_x_ticks:
                    ax.set_xticks(self._minor_x_ticks, minor=True)
                ax.tick_params(axis="x", which="minor", **self._tick_params["x minor"])
                if self._minor_x_tick_spacing:
                    ax.xaxis.set_minor_locator(
                        ticker.MultipleLocator(self._minor_x_tick_spacing)
                    )
                if self._minor_y_ticks:
                    ax.set_yticks(self._minor_y_ticks, minor=True)
                ax.tick_params(axis="y", which="minor", **self._tick_params["y minor"])
                if self._minor_y_tick_spacing:
                    ax.yaxis.set_minor_locator(
                        ticker.MultipleLocator(self._minor_y_tick_spacing)
                    )

                # Customize grid
                if self._show_grid:
                    ax.grid(self._grid_visible_x, which=self._grid_which_x, axis="x")
                    ax.grid(self._grid_visible_y, which=self._grid_which_y, axis="y")
                    if self._grid_show_on_top:
                        ax.set_axisbelow(False)

                # Axes legend
                if not self._general_legend and make_legend and labels:
                    if self._show_legend:
                        legend_params = self._get_legend_params(labels, handles, -0.1)
                        try:
                            _legend = ax.legend(
                                draggable=True,
                                **legend_params,
                            )
                        except:
                            _legend = ax.legend(
                                **legend_params,
                            )
                        _legend.set_zorder(10000)
                    labels, handles = [], []

                # Axes title (if the geometry is 1x1)
                if self._title and (self._num_cols == 1 and self._num_rows == 1):
                    ax.set_title(self._title)

                self._reset_params_to_default(self, figure_params_to_reset)

            elif element is not None:
                raise GraphingException(f"Unsupported element type in list: {type(element).__name__}")

        # Axes labels
        if self._num_cols == 1 and self._num_rows == 1:
            if ax is not None:  # makes sure an element was plotted and that an axis was created
                if self._x_label is not None:       # some projection axes can not deal with empty labels
                    ax.set_xlabel(self._x_label)
                if self._y_label is not None:
                    ax.set_ylabel(self._y_label)
        else:
            suplabel_params = {
                "fontsize" : plt.rcParams["font.size"],
                "color" : plt.rcParams["axes.labelcolor"],
                "fontweight" : plt.rcParams["font.weight"],
            }
            self._figure.supxlabel(self._x_label, **suplabel_params)
            self._figure.supylabel(self._y_label, **suplabel_params)

        # Title (if the geometry is not 1x1)
        if self._title and not (self._num_cols == 1 and self._num_rows == 1):
            self._figure.suptitle(self._title)

        # General legend
        if self._general_legend and labels:     # making a general legend is priorized over make_legend=False
            if self._show_legend:
                legend_params = self._get_legend_params(labels, handles, 0)
                try:
                    _legend = self._figure.legend(
                        **legend_params,
                        draggable=True,
                    )
                except:
                    _legend = self._figure.legend(
                        **legend_params,
                    )
                _legend.set_zorder(10000)
            return [], []
        else:
            return labels, handles

    def _create_ref_label(
        self,
        target: Axes | SubFigure
    ) -> None:
        if isinstance(target, Axes):
            trans = target.transAxes
        elif isinstance(target, SubFigure):
            trans = target.transSubfigure
        else:
            raise ValueError("Target must be either Axes or SubFigure.")
        target.text(
            0,
            1,
            ascii_lowercase[self._reference_label_i] + ")",
            transform=trans + self._get_reflabel_translation(target)
        )
        self._reference_label_i += 1

    def _get_legend_params(
        self,
        labels: list[str],
        handles: list[Any],
        outside_lower_center_y_offset: float,
    ) -> dict[str, Any]:
        legend_params = {
            "handles" : handles,
            "labels" : labels,
            "handleheight" : 1.3,
            "handler_map" : {
                Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                LineCollection: HandlerMultipleLines(),
                VerticalLineCollection: HandlerMultipleVerticalLines(),
            },
            "ncols" : self._legend_cols,
        }
        if self._legend_loc is not None and "outside" in self._legend_loc:
            outside_coords = {
                "outside upper center": (0.5, 1),
                "outside center right": (1, 0.5),
                "outside lower center": (0.5, outside_lower_center_y_offset),
                "outside center left": (0, 0.5),
            }
            outside_keyword = {
                "outside upper center": "lower center",
                "outside center right": "center left",
                "outside lower center": "upper center",
                "outside center left": "center right",
            }
            legend_params.update({
                "loc": outside_keyword[self._legend_loc],
                "bbox_to_anchor": outside_coords[self._legend_loc],
            })
        else:
            legend_params.update({"loc": self._legend_loc})
        return legend_params

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
        legend_face_color: str | None = None,
        legend_edge_color: str | None = None,
        legend_font_size: float | None = None,
        legend_handle_length: float | None = None,
        font_family: str | None = None,
        font_size: float | None = None,
        font_weight: str | None = None,
        title_font_size: float | None = None,
        title_font_weight: str | None = None,
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
        legend_face_color : str
            The color of the legend face.
            Defaults to ``None``.
        legend_edge_color : str
            The color of the legend edge.
            Defaults to ``None``.
        legend_font_size : float
            The font size of the legend.
            Defaults to ``None``.
        legend_handle_length : float
            The length of the legend handles.
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
        title_font_size : float
            The font size of the title.
            Defaults to ``None``.
        title_font_weight : str
            The font weight of the title.
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
            "legend.facecolor": legend_face_color,
            "legend.edgecolor": legend_edge_color,
            "legend.fontsize": legend_font_size,
            "legend.handlelength": legend_handle_length,
            "font.family": font_family,
            "font.size": font_size,
            "font.weight": font_weight,
            "axes.titlesize": title_font_size,
            "figure.titlesize": title_font_size,
            "axes.titleweight": title_font_weight,
            "figure.titleweight": title_font_weight,
            "text.color": text_color,
            "text.usetex": use_latex,
        }
        rc_params_dict = {
            key: value for key, value in rc_params_dict.items() if value is not None
        }
        self.set_rc_params(rc_params_dict, reset=reset)

    def set_ticks(
        self,
        x_ticks: Optional[list[float]] = None,
        x_tick_labels: Optional[list[str]] = None,
        x_tick_spacing: Optional[float] = None,
        y_ticks: Optional[list[float]] = None,
        y_tick_labels: Optional[list[str]] = None,
        y_tick_spacing: Optional[float] = None,
        minor_x_ticks: Optional[list[float]] = None,
        minor_x_tick_spacing: Optional[float] = None,
        minor_y_ticks: Optional[list[float]] = None,
        minor_y_tick_spacing: Optional[float] = None,
    ) -> None:
        """
        Sets custom ticks and ticks labels.

        Parameters
        ----------
        x_ticks : list[float], optional
            Tick positions for the x axis. If a value is specified, the x_tick_spacing parameter must be None.
        x_tick_labels : list[str], optional
            Tick labels for the x axis. If a value is specified, the x_ticks parameter must also be given.
        x_tick_spacing : float, optional
            Spacing between ticks on the x axis. If a value is specified, the x_ticks parameter must be None.
        y_ticks : list[float], optional
            Tick positions for the y axis. If a value is specified, the y_tick_spacing parameter must be None.
        y_tick_labels : list[str], optional
            Tick labels for the y axis. If a value is specified, the y_ticks parameter must also be given.
        y_tick_spacing : float, optional
            Spacing between ticks on the y axis. If a value is specified, the y_ticks parameter must be None.
        minor_x_ticks : list[float], optional
            Minor tick positions for the x axis. If a value is specified, minor_the x_tick_spacing parameter must be 
            None.
        minor_x_tick_spacing : float, optional
            Spacing between minor ticks on the x axis. If a value is specified, the minor_x_ticks parameter must be
            None.
        minor_y_ticks : list[float], optional
            Minor tick positions for the y axis. If a value is specified, the minor_y_tick_spacing parameter must be 
            None.
        minor_y_tick_spacing : float, optional
            Spacing between minor ticks on the y axis. If a value is specified, the minor_y_ticks parameter must be
            None.
        """
        if any([
            x_tick_labels and not x_ticks,
            y_tick_labels and not y_ticks,
        ]):
            raise GraphingException("Ticks position must be specified when ticks labels are specified")
        
        if any([
            x_ticks and x_tick_spacing, 
            y_ticks and y_tick_spacing,
            minor_x_ticks and minor_x_tick_spacing, 
            minor_y_ticks and minor_y_tick_spacing,
        ]):
            raise GraphingException("Tick spacing and tick positions cannot be set simultaneously")

        self._x_ticks = x_ticks
        self._x_tick_labels = x_tick_labels
        self._x_tick_spacing = x_tick_spacing
        self._y_ticks = y_ticks
        self._y_tick_labels = y_tick_labels
        self._y_tick_spacing = y_tick_spacing
        self._minor_x_ticks = minor_x_ticks
        self._minor_x_tick_spacing = minor_x_tick_spacing
        self._minor_y_ticks = minor_y_ticks
        self._minor_y_tick_spacing = minor_y_tick_spacing
    
    def set_tick_params(
        self,
        axis: Optional[Literal["x", "y", "both"]] = "both",
        which: Optional[Literal["major", "minor", "both"]] = "major",
        reset: Optional[bool] = False,
        direction: Optional[Literal["in", "out", "inout"]] = None,
        length: Optional[float] = None,
        width: Optional[float] = None,
        color: Optional[str] = None,
        pad: Optional[float] = None,
        label_size: Optional[float | str] = None,
        label_color: Optional[str] = None,
        label_rotation: Optional[float] = None,
        draw_bottom_tick: Optional[bool] = None,
        draw_top_tick: Optional[bool] = None,
        draw_left_tick: Optional[bool] = None,
        draw_right_tick: Optional[bool] = None,
        draw_bottom_label: Optional[bool] = None,
        draw_top_label: Optional[bool] = None,
        draw_left_label: Optional[bool] = None,
        draw_right_label: Optional[bool] = None,
    ) -> None:
        new_tick_params = {
            "direction": direction,
            "length": length,
            "width": width,
            "color": color,
            "pad": pad,
            "labelsize": label_size,
            "labelcolor": label_color,
            "labelrotation": label_rotation,
            "bottom": draw_bottom_tick,
            "top": draw_top_tick,
            "left": draw_left_tick,
            "right": draw_right_tick,
            "labelbottom": draw_bottom_label,
            "labeltop": draw_top_label,
            "labelleft": draw_left_label,
            "labelright": draw_right_label
        }
        for axis_i in [axis] if axis != "both" else ["x", "y"]:
            for which_i in [which] if which != "both" else ["major", "minor"]:
                if reset:
                    self._tick_params[f"{axis_i} {which_i}"] = {}
                for param, value in new_tick_params.items():
                    if value is not None:
                        self._tick_params[f"{axis_i} {which_i}"][param] = value

    def set_grid(
        self,
        visible_x: bool = True,
        visible_y: bool = True,
        show_on_top: bool = False,
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
        show_on_top : bool, optional
            If ``True``, sets the grid lines to be shown on top of the plot elements. Defaults to ``False``.
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
        self._grid_show_on_top = show_on_top
        self._grid_which_x = which_x
        self._grid_which_y = which_y
        rc_params_dict = {
            "grid.color": color,
            "grid.alpha": alpha,
            "grid.linestyle": line_style,
            "grid.linewidth": line_width,
        }
        rc_params_dict = {k: v for k, v in rc_params_dict.items() if v != "default"}
        self.set_rc_params(rc_params_dict)
