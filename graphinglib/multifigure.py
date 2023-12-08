from string import ascii_lowercase
from typing import Literal, Optional
from warnings import warn

import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.transforms import ScaledTranslation

from .file_manager import FileLoader, FileUpdater
from .graph_elements import GraphingException, Plottable, Text
from .legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)


class SubFigure:
    """
    This class implements the individual plots added inside the
    :class:`~graphinglib.multifigure.MultiFigure` object.

    .. attention::

        This class is not meant to be used directly by the user. Instead, it is used in
        conjunction with the :class:`~graphinglib.multifigure.MultiFigure` class.

    Parameters
    ----------
    row_start : int
        The row where to set the upper-left corner of the SubFigure.
    col_start : int
        The column where to set the upper-left corner of the SubFigure.
    row_span : int
        The number of rows spanned by the SubFigure.
    col_span : int
        The number of columns spanned by the SubFigure.
    x_label, y_label : str
        The indentification for the x-axis and y-axis.
        Defaults to ``"x axis"`` and ``"y axis"``.
    x_lim, y_lim : tuple[float, float], optional
        The limits for the x-axis and y-axis.
    figure_style : str
        The figure style to use for the figure.
    add_reference_label : bool
        Whether or not to add a reference label to the SubFigure.
        Defaults to ``True``.
    log_scale_x, log_scale_y : bool
        Whether or not to set the scale of the x- or y-axis to logaritmic scale.
        Default depends on the ``figure_style`` configuration.
    show_grid : bool
        Wheter or not to show the grid.
        Default depends on the ``figure_style`` configuration.
    remove_axes : bool
        Whether or not to show the axes. Useful for adding tables or text to
        the subfigure. Defaults to ``False``.
    """

    def __init__(
        self,
        row_start: int,
        col_start: int,
        row_span: int,
        col_span: int,
        x_label: str = "x axis",
        y_label: str = "y axis",
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        figure_style: str = "plain",
        add_reference_label: bool = True,
        log_scale_x: bool | Literal["default"] = "default",
        log_scale_y: bool | Literal["default"] = "default",
        show_grid: bool | Literal["default"] = "default",
        remove_axes: bool = False,
    ):
        """
        This class implements the individual plots added inside the
        :class:`~graphinglib.multifigure.MultiFigure` object.

        .. attention::

            This class is not meant to be used directly by the user. Instead, it is used in
            conjunction with the :class:`~graphinglib.multifigure.MultiFigure` class.

        Parameters
        ----------
        row_start : int
            The row where to set the upper-left corner of the SubFigure.
        col_start : int
            The column where to set the upper-left corner of the SubFigure.
        row_span : int
            The number of rows spanned by the SubFigure.
        col_span : int
            The number of columns spanned by the SubFigure.
        x_label, y_label : str
            The indentification for the x-axis and y-axis.
            Defaults to ``"x axis"`` and ``"y axis"``.
        x_lim, y_lim : tuple[float, float], optional
            The limits for the x-axis and y-axis.
        figure_style : str
            The figure style to use for the figure.
        add_reference_label : bool
            Whether or not to add a reference label to the SubFigure.
            Defaults to ``True``.
        log_scale_x, log_scale_y : bool
            Whether or not to set the scale of the x- or y-axis to logaritmic scale.
            Default depends on the ``figure_style`` configuration.
        show_grid : bool
            Wheter or not to show the grid.
            Default depends on the ``figure_style`` configuration.
        remove_axes : bool
            Whether or not to show the axes. Useful for adding tables or text to
            the subfigure. Defaults to ``False``.
        """
        self.x_axis_name = x_label
        self.y_axis_name = y_label
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.row_start, self.col_start = row_start, col_start
        self.row_span, self.col_span = row_span, col_span
        self.figure_style = figure_style
        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        if show_grid == "default":
            self.show_grid = "unchanged"
        else:
            self.show_grid = show_grid
        self.add_reference_label = add_reference_label
        self.remove_axes = remove_axes
        self._elements: list[Plottable] = []
        self._labels: list[str | None] = []
        self._handles = []

    def add_element(self, *elements: Plottable) -> None:
        """
        Adds a :class:`~graphinglib.graph_elements.Plottable` element to the
        :class:`~graphinglib.multifigure.SubFigure`.

        Parameters
        ----------
        elements : :class:`~graphinglib.graph_elements.Plottable`
            Elements to plot in the :class:`~graphinglib.multifigure.SubFigure`.
        """
        for element in elements:
            self._elements.append(element)

    def _prepare_SubFigure(
        self,
        grid: GridSpec,
        transformation: ScaledTranslation,
        reference_label: str,
        legend: bool = True,
    ) -> Axes:
        """
        Prepares the :class:`~graphinglib.multifigure.SubFigure` to be displayed.
        """
        try:
            file_loader = FileLoader(self.figure_style)
            self.default_params = file_loader.load()
            is_matplotlib_style = False
        except FileNotFoundError:
            try:
                if self.figure_style == "matplotlib":
                    plt.style.use("default")
                else:
                    plt.style.use(self.figure_style)
                file_loader = FileLoader("plain")
                self.default_params = file_loader.load()
                is_matplotlib_style = True
            except OSError:
                raise GraphingException(
                    f"The figure style {self.figure_style} was not found. Please choose a different style."
                )
        figure_params_to_reset = self._fill_in_missing_params(self)

        self._axes = plt.subplot(
            grid.new_subplotspec(
                (self.row_start, self.col_start),
                rowspan=self.row_span,
                colspan=self.col_span,
            )
        )
        if self.add_reference_label:
            self._axes.text(
                0,
                1,
                reference_label,
                transform=self._axes.transAxes + transformation,
            )
        if self.show_grid == "unchanged":
            pass
        elif self.show_grid:
            self._axes.grid(True)
        else:
            self._axes.grid(False)
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
        if self.remove_axes:
            self._axes.axis("off")
            warn(
                "Axes on SubFigure placed at ({},{},{},{}) have been removed.".format(
                    self.row_start, self.col_start, self.row_span, self.col_span
                )
            )
        if self._elements:
            z_order = 12
            for element in self._elements:
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
        temp_labels, temp_handles = self._labels, self._handles
        self._labels, self._handles = [], []
        return temp_labels, temp_handles

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
                    if type(value) == str and value == "default":
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
                        f"There was an error auto updating your {self.figure_style} style file following the recent GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub page. In the meantime, you can manually add the following parameter to your {self.figure_style} style file:\n {object_type}.{e.args[0]}"
                    )
                file_updater = FileUpdater(self.figure_style)
                print(object_type, e.args[0])
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
            setattr(element, param, "default")


class MultiFigure:
    """
    This class implements the "canvas" on which multiple plots are displayed.

    The canvas consists of a grid of a specified size on which the
    :class:`~graphinglib.multifigure.SubFigure` objects are displayed.

    Parameters
    ----------
    num_rows, num_cols : int
        Number of rows and columns for the grid. These parameters determine the
        the number of "squares" on which a plot can be placed.

        .. note::
            Note that a single plot can span multiple squares.
            See :py:meth:`~graphinglib.multifigure.MultiFigure.add_SubFigure`.

    size : tuple[float, float]
        Overall size of the figure.
        Default depends on the ``figure_style`` configuration.
    title : str, optional
        General title of the figure.
    reference_labels : bool
        Wheter or not to add reference labels to the SubFigures.
        Defaults to ``True``.

        .. note::
            The reference labels are in the form of "a)", "b)", etc. and are used to refer
            to a particular SubFigure in a caption accompanying the MultiFigure.

    reflabel_loc : str
        Location of the reference labels of the SubFigures. Either "inside" or "outside".
        Defaults to "outside".
    figure_style : str
        The figure style to use for the figure.
        Defaults to "plain".
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        size: tuple[float, float] | Literal["default"] = "default",
        title: Optional[str] = None,
        reference_labels: bool = True,
        reflabel_loc: str = "outside",
        figure_style: str = "plain",
    ) -> None:
        """
        This class implements the "canvas" on which multiple plots are displayed.

        The canvas consists of a grid of a specified size on which the
        :class:`~graphinglib.multifigure.SubFigure` objects are displayed.

        Parameters
        ----------
        num_rows, num_cols : int
            Number of rows and columns for the grid. These parameters determine the
            the number of "squares" on which a plot can be placed.

            .. note::
                Note that a single plot can span multiple squares.
                See :py:meth:`~graphinglib.multifigure.MultiFigure.add_SubFigure`.

        size : tuple[float, float]
            Overall size of the figure.
            Default depends on the ``figure_style`` configuration.
        title : str, optional
            General title of the figure.
        reference_labels : bool
            Wheter or not to add reference labels to the SubFigures.
            Defaults to ``True``.

            .. note::
                The reference labels are in the form of "a)", "b)", etc. and are used to refer
                to a particular SubFigure in a caption accompanying the MultiFigure.

        reflabel_loc : str
            Location of the reference labels of the SubFigures. Either "inside" or "outside".
            Defaults to "outside".
        figure_style : str
            The figure style to use for the figure.
            Defaults to "plain".
        """
        if type(num_rows) != int or type(num_cols) != int:
            raise TypeError("The number of rows and columns must be integers.")
        if num_rows < 1 or num_cols < 1:
            raise ValueError("The number of rows and columns must be greater than 0.")
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.title = title
        self.reference_labels = reference_labels
        self.reflabel_loc = reflabel_loc
        self.figure_style = figure_style
        self.size = size
        self._SubFigures = []
        self._rc_dict = {}
        self._user_rc_dict = {}

    def add_SubFigure(
        self,
        row_start: int,
        col_start: int,
        row_span: int,
        col_span: int,
        x_label: str = "x axis",
        y_label: str = "y axis",
        x_lim: Optional[tuple[float, float]] = None,
        y_lim: Optional[tuple[float, float]] = None,
        log_scale_x: bool | Literal["default"] = "default",
        log_scale_y: bool | Literal["default"] = "default",
        show_grid: bool | Literal["default"] = "default",
        remove_axes: bool = False,
    ) -> SubFigure:
        """
        Adds a :class:`~graphinglib.multifigure.SubFigure` to a :class:`~graphinglib.multifigure.MultiFigure`.

        Parameters
        ----------
        row_start : int
            The row where to set the upper-left corner of the SubFigure.
        col_start : int
            The column where to set the upper-left corner of the SubFigure.
        row_span : int
            The number of rows spanned by the SubFigure.
        col_span : int
            The number of columns spanned by the SubFigure.
        x_label, y_label : str
            The indentification for the x-axis and y-axis.
            Defaults to ``"x axis"`` and ``"y axis"``.
        x_lim, y_lim : tuple[float, float], optional
            The limits for the x-axis and y-axis.
        log_scale_x, log_scale_y : bool
            Whether or not to set the scale of the x- or y-axis to logaritmic scale.
            Default depends on the ``figure_style`` configuration.
        show_grid : bool
            Wheter or not to show the grid.
            Default depends on the ``figure_style`` configuration.
        remove_axes : bool
            Whether or not to show the axes. Useful for adding tables or text to
            the subfigure. Defaults to ``False``.

        Returns
        -------
        new_SubFigure : :class:`~graphinglib.multifigure.SubFigure`
            :class:`~graphinglib.multifigure.SubFigure` to be added to the :class:`~graphinglib.multifigure.MultiFigure`.
        """

        if type(row_start) != int or type(col_start) != int:
            raise TypeError("The placement values must be integers.")
        if row_start < 0 or col_start < 0:
            raise ValueError("The placement values cannot be negative.")
        if type(row_span) != int or type(col_span) != int:
            raise TypeError("The span values must be integers.")
        if row_span < 1 or col_span < 1:
            raise ValueError("The span values must be greater than 0.")
        if row_start + row_span > self.num_rows or col_start + col_span > self.num_cols:
            raise ValueError(
                "The placement values and span values must be inside the size of the MultiFigure."
            )
        new_SubFigure = SubFigure(
            row_start,
            col_start,
            row_span,
            col_span,
            x_label,
            y_label,
            x_lim,
            y_lim,
            self.figure_style,
            self.reference_labels,
            log_scale_x,
            log_scale_y,
            show_grid,
            remove_axes,
        )
        self._SubFigures.append(new_SubFigure)
        return new_SubFigure

    def _prepare_MultiFigure(
        self,
        general_legend: bool = False,
        legend_loc: str = "outside lower center",
        legend_cols: int = 1,
    ) -> None:
        """
        Prepares the :class:`~graphinglib.multifigure.MultiFigure` to be displayed.
        """
        try:
            file_loader = FileLoader(self.figure_style)
            self.default_params = file_loader.load()
            self._fill_in_rc_params()
        except FileNotFoundError:
            try:
                if self.figure_style == "matplotlib":
                    plt.style.use("default")
                else:
                    plt.style.use(self.figure_style)
                file_loader = FileLoader("plain")
                self.default_params = file_loader.load()
            except OSError:
                raise GraphingException(
                    f"The figure style {self.figure_style} was not found. Please choose a different style."
                )

        multi_figure_params_to_reset = self._fill_in_missing_params(self)

        self._figure = plt.figure(layout="constrained", figsize=self.size)
        MultiFigure_grid = GridSpec(self.num_rows, self.num_cols, figure=self._figure)

        if self.reflabel_loc == "outside":
            trans = ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
        elif self.reflabel_loc == "inside":
            trans = ScaledTranslation(10 / 72, -15 / 72, self._figure.dpi_scale_trans)
        else:
            raise ValueError("Invalid location parameter")

        SubFigures_legend = True if not general_legend else False

        labels, handles = [], []
        for i, SubFigure in enumerate(self._SubFigures):
            SubFigure.figure_style = self.figure_style
            SubFigure_labels, SubFigure_handles = SubFigure._prepare_SubFigure(
                MultiFigure_grid,
                transformation=trans,
                reference_label=ascii_lowercase[i] + ")",
                legend=SubFigures_legend,
            )
            labels += SubFigure_labels
            handles += SubFigure_handles
        if general_legend:
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
                    draggable=True,
                    loc=legend_loc,
                    ncols=legend_cols,
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
                    loc=legend_loc,
                    ncols=legend_cols,
                )
        self._figure.suptitle(self.title)
        self._reset_params_to_default(self, multi_figure_params_to_reset)
        self._rc_dict = {}

    def display(
        self,
        general_legend: bool = False,
        legend_loc: str = "outside lower center",
        legend_cols: int = 1,
    ) -> None:
        """
        Displays the :class:`~graphinglib.multifigure.MultiFigure`.

        Parameters
        ----------
        general_legend : bool
            Wheter or not to display a overall legend for the :class:`~graphinglib.multifigure.MultiFigure` containing
            the labels for every :class:`~graphinglib.multifigure.SubFigure` in it. Note that enabling this option will
            disable the individual legends for every :class:`~graphinglib.multifigure.SubFigure`.
            Defaults to ``False``.
        legend_loc : str
            The location of the legend in the MultiFigure. Possible placement keywords are: for vertical placement: ``{"upper", "center", "lower"}``, for horizontal placement: ``{"left", "center", "right"}``. The keyword ``"outside"`` can be added to put the legend outside of the axes. Defaults to ``"outside lower center"``.
        legend_cols : int
            Number of colums in which to arange the legend items. Defaults to 1.
        """
        self._prepare_MultiFigure(
            general_legend=general_legend,
            legend_loc=legend_loc,
            legend_cols=legend_cols,
        )
        plt.show()
        plt.rcParams.update(rcParamsDefault)

    def save_figure(
        self,
        file_name: str,
        general_legend: bool = False,
        legend_loc: str = "outside lower center",
        legend_cols: int = 1,
    ) -> None:
        """
        Saves the :class:`~graphinglib.multifigure.MultiFigure`.

        Parameters
        ----------
        file_name : str
            File name or path at which to save the figure.
        general_legend : bool
            Wheter or not to display a overall legend for the :class:`~graphinglib.multifigure.MultiFigure` containing
            the labels for every :class:`~graphinglib.multifigure.SubFigure` in it. Note that enabling this option will
            disable the individual legends for every :class:`~graphinglib.multifigure.SubFigure`.
            Defaults to ``False``.
        legend_loc : str
            The location of the legend in the MultiFigure. Possible placement keywords are: for vertical placement: ``{"upper", "center", "lower"}``, for horizontal placement: ``{"left", "center", "right"}``. The keyword ``"outside"`` can be added to put the legend outside of the axes. Defaults to ``"outside lower center"``.
        legend_cols : int
            Number of colums in which to arange the legend items. Defaults to 1.
        """
        self._prepare_MultiFigure(
            general_legend=general_legend,
            legend_loc=legend_loc,
            legend_cols=legend_cols,
        )
        plt.savefig(file_name, bbox_inches="tight")
        plt.close()
        plt.rcParams.update(rcParamsDefault)

    def _fill_in_missing_params(self, element: Plottable) -> list[str]:
        """
        Fills in the missing parameters from the specified ``figure_style``.
        """
        params_to_reset = []
        object_type = type(element).__name__
        for property, value in vars(element).items():
            if (type(value) == str) and (value == "default"):
                params_to_reset.append(property)
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
        return params_to_reset

    def _reset_params_to_default(
        self, element: Plottable, params_to_reset: list[str]
    ) -> None:
        """
        Resets the parameters that were set to default in the _fill_in_missing_params method.
        """
        for param in params_to_reset:
            setattr(element, param, "default")

    def update_rc_params(
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

    def customize_visual_style(
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
        self.update_rc_params(rc_params_dict, reset=reset)

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
