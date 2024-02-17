from shutil import which
from string import ascii_lowercase
from typing import Literal, Optional

import matplotlib.pyplot as plt
from matplotlib import rcParamsDefault
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.transforms import ScaledTranslation

from graphinglib.file_manager import FileLoader
from graphinglib.graph_elements import GraphingException, Plottable
from graphinglib.legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)

from .figure import Figure

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class MultiFigure:
    """
    This class implements the "canvas" on which multiple plots are displayed.

    The canvas consists of a grid of a specified size on which the
    :class:`~graphinglib.figure.Figure` objects are displayed.

    Parameters
    ----------
    num_rows, num_cols : int
        Number of rows and columns for the grid. These parameters determine the
        the number of "squares" on which a plot can be placed.

        .. note::
            Note that a single plot can span multiple squares.
            See :py:meth:`~graphinglib.multifigure.MultiFigure.add_SubFigure`.

    size : tuple[float, float]
        Overall size of the multifigure.
        Default depends on the ``figure_style`` configuration.
    title : str, optional
        General title of the figure.
    reference_labels : bool
        Whether or not to add reference labels to the SubFigures.
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

        The canvas consists of a grid of a specified size on which the individual
        :class:`~graphinglib.figure.Figure` objects are displayed.

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
        self._sub_figures = []
        self._rc_dict = {}
        self._user_rc_dict = {}

    @classmethod
    def from_row(
        cls,
        figures: list[Figure],
        size: tuple[float, float] | Literal["default"] = "default",
        title: Optional[str] = None,
        reference_labels: bool = True,
        reflabel_loc: str = "outside",
        figure_style: str = "plain",
    ) -> Self:
        """Creates a MultiFigure with the specified :class:`~graphinglib.figure.Figure` objects in a horizontal configuration.

        Parameters
        ----------
        figures : list[Figure]
            The :class:`~graphinglib.figure.Figure` objects to add to the MultiFigure, from left to right.
        size : tuple[float, float]
            Overall size of the figure.
            Default depends on the ``figure_style`` configuration.
        title : str, optional
            Title of the MultiFigure.
            Defaults to ``None``.
        reference_labels : bool
            Whether or not to add reference labels to the SubFigures.
            Defaults to ``True``.
        reflabel_loc : str
            Location of the reference labels of the SubFigures. Either "inside" or "outside".
            Defaults to "outside".
        figure_style : str
            The figure style to use for the figure.
            Defaults to "plain".

        Returns
        -------
        A new MultiFigure object.
        """
        multi_fig = cls(
            num_rows=1,
            num_cols=len(figures),
            size=size,
            title=title,
            reference_labels=reference_labels,
            reflabel_loc=reflabel_loc,
            figure_style=figure_style,
        )
        for i, figure in enumerate(figures):
            multi_fig.add_figure(figure, 0, i, 1, 1)
        return multi_fig

    @classmethod
    def from_stack(
        cls,
        figures: list[Figure],
        size: tuple[float, float] | Literal["default"] = "default",
        title: Optional[str] = None,
        reference_labels: bool = True,
        reflabel_loc: str = "outside",
        figure_style: str = "plain",
    ) -> Self:
        """Creates a MultiFigure with the specified :class:`~graphinglib.figure.Figure` objects in a vertical configuration.

        Parameters
        ----------
        figures : list[Figure]
            The :class:`~graphinglib.figure.Figure` objects to add to the MultiFigure, from top to bottom.
        size : tuple[float, float]
            Overall size of the figure.
            Default depends on the ``figure_style`` configuration.
        title : str, optional
            Title of the MultiFigure.
            Defaults to ``None``.
        reference_labels : bool
            Whether or not to add reference labels to the SubFigures.
            Defaults to ``True``.
        reflabel_loc : str
            Location of the reference labels of the SubFigures. Either "inside" or "outside".
            Defaults to "outside".
        figure_style : str
            The figure style to use for the figure.
            Defaults to "plain".

        Returns
        -------
        A new MultiFigure object.
        """
        multi_fig = cls(
            num_rows=len(figures),
            num_cols=1,
            size=size,
            title=title,
            reference_labels=reference_labels,
            reflabel_loc=reflabel_loc,
            figure_style=figure_style,
        )
        for i, figure in enumerate(figures):
            multi_fig.add_figure(figure, i, 0, 1, 1)
        return multi_fig

    @classmethod
    def from_grid(
        cls,
        figures: list[Figure],
        dimensions: tuple[int, int],
        size: tuple[float, float] | Literal["default"] = "default",
        title: Optional[str] = None,
        reference_labels: bool = True,
        reflabel_loc: str = "outside",
        figure_style: str = "plain",
    ) -> Self:
        """Creates a MultiFigure with the specified :class:`~graphinglib.figure.Figure` objects in a grid configuration.

        Parameters
        ----------
        figures : list[Figure]
            The :class:`~graphinglib.figure.Figure` objects to add to the MultiFigure, from top-left to bottom-right.
        dimensions : tuple[int, int]
            The number of rows and columns of the grid (product should equal the number of figures).
        size : tuple[float, float]
            Overall size of the figure.
            Default depends on the ``figure_style`` configuration.
        title : str, optional
            Title of the MultiFigure.
            Defaults to ``None``.
        reference_labels : bool
            Whether or not to add reference labels to the SubFigures.
            Defaults to ``True``.
        reflabel_loc : str
            Location of the reference labels of the SubFigures. Either "inside" or "outside".
            Defaults to "outside".
        figure_style : str
            The figure style to use for the figure.
            Defaults to "plain".

        Returns
        -------
        A new MultiFigure object.
        """
        num_rows, num_cols = dimensions
        if num_rows * num_cols < len(figures):
            raise ValueError(
                f"The product of the dimensions ({num_rows} x {num_cols}) must be greater than or equal to the number of figures ({len(figures)})."
            )
        multi_fig = cls(
            num_rows=num_rows,
            num_cols=num_cols,
            size=size,
            title=title,
            reference_labels=reference_labels,
            reflabel_loc=reflabel_loc,
            figure_style=figure_style,
        )
        for i, figure in enumerate(figures):
            row = i // num_cols
            col = i % num_cols
            multi_fig.add_figure(figure, row, col, 1, 1)
        return multi_fig

    def add_figure(
        self,
        figure: Figure,
        row_start: int,
        col_start: int,
        row_span: int,
        col_span: int,
    ) -> None:
        """
        Adds a :class:`~graphinglib.figure.Figure` to a :class:`~graphinglib.multifigure.MultiFigure`.

        Parameters
        ----------
        figure : Figure
            The :class:`~graphinglib.figure.Figure` to add to the MultiFigure.
        row_start : int
            The row where to set the upper-left corner of the SubFigure.
        col_start : int
            The column where to set the upper-left corner of the SubFigure.
        row_span : int
            The number of rows spanned by the SubFigure.
        col_span : int
            The number of columns spanned by the SubFigure.
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
        # Add location and span to the SubFigure (create new attributes)
        figure.row_start = row_start
        figure.col_start = col_start
        figure.row_span = row_span
        figure.col_span = col_span
        self._sub_figures.append(figure)

    def show(
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
            Whether or not to display an overall legend for the :class:`~graphinglib.multifigure.MultiFigure` containing
            the labels of every :class:`~graphinglib.Figure.Figure` inside it. Note that enabling this option will
            disable the individual legends for every :class:`~graphinglib.multifigure.SubFigure`.
            Defaults to ``False``.
        legend_loc : str
            The location of the legend in the MultiFigure. Possible placement keywords are: for vertical placement: ``{"upper", "center", "lower"}``, for horizontal placement: ``{"left", "center", "right"}``. The keyword ``"outside"`` can be added to put the legend outside of the axes. Defaults to ``"outside lower center"``.
        legend_cols : int
            Number of colums in which to arrange the legend items. Defaults to 1.
        """
        self._prepare_multi_figure(
            general_legend=general_legend,
            legend_loc=legend_loc,
            legend_cols=legend_cols,
        )
        plt.show()
        plt.rcParams.update(rcParamsDefault)

    def save(
        self,
        file_name: str,
        general_legend: bool = False,
        legend_loc: str = "outside lower center",
        legend_cols: int = 1,
    ) -> None:
        """
        Saves the :class:`~graphinglib.multifigure.MultiFigure` to a file.

        Parameters
        ----------
        file_name : str
            File name or path at which to save the figure.
        general_legend : bool
            Whether or not to display an overall legend for the :class:`~graphinglib.multifigure.MultiFigure` containing
            the labels of every :class:`~graphinglib.figure.Figure` inside it. Note that enabling this option will
            disable the individual legends for every :class:`~graphinglib.figure.Figure`.
            Defaults to ``False``.
        legend_loc : str
            The location of the legend in the MultiFigure. Possible placement keywords are: for vertical placement: ``{"upper", "center", "lower"}``, for horizontal placement: ``{"left", "center", "right"}``. The keyword ``"outside"`` can be added to put the legend outside of the axes. Defaults to ``"outside lower center"``.
        legend_cols : int
            Number of colums in which to arrange the legend items. Defaults to 1.
        """
        self._prepare_multi_figure(
            general_legend=general_legend,
            legend_loc=legend_loc,
            legend_cols=legend_cols,
        )
        plt.savefig(file_name, bbox_inches="tight")
        plt.close()
        plt.rcParams.update(rcParamsDefault)

    def _prepare_multi_figure(
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
            is_matplotlib_style = False
        except FileNotFoundError:
            is_matplotlib_style = True
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

        self._fill_in_rc_params(is_matplotlib_style)
        self._figure = plt.figure(layout="constrained", figsize=self.size)
        MultiFigure_grid = GridSpec(self.num_rows, self.num_cols, figure=self._figure)

        if self.reflabel_loc == "outside":
            trans = ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
        elif self.reflabel_loc == "inside":
            trans = ScaledTranslation(10 / 72, -15 / 72, self._figure.dpi_scale_trans)
        else:
            raise ValueError(
                "Invalid reference label location. Please specify either 'inside' or 'outside'."
            )

        sub_figures_do_legend = True if not general_legend else False

        labels, handles = [], []
        for i, sub_figure in enumerate(self._sub_figures):
            self._fill_in_rc_params(is_matplotlib_style)
            sub_figure_labels, sub_figure_handles = self._prepare_sub_figure(
                sub_figure,
                MultiFigure_grid,
                transformation=trans,
                reference_label=ascii_lowercase[i] + ")",
                legend=sub_figures_do_legend,
                is_matplotlib_style=is_matplotlib_style,
            )
            labels += sub_figure_labels
            handles += sub_figure_handles
        self._fill_in_rc_params(is_matplotlib_style)
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

    def _prepare_sub_figure(
        self,
        sub_figure: Figure,
        grid: GridSpec,
        transformation: ScaledTranslation,
        reference_label: str,
        legend: bool,
        is_matplotlib_style: bool,
    ):
        """
        Prepares a single subfigure.
        """
        sub_rcs = sub_figure._user_rc_dict
        plt.rcParams.update(sub_rcs)
        axes = plt.subplot(
            grid.new_subplotspec(
                (sub_figure.row_start, sub_figure.col_start),
                rowspan=sub_figure.row_span,
                colspan=sub_figure.col_span,
            )
        )
        if self.reference_labels:
            axes.text(
                0,
                1,
                reference_label,
                transform=axes.transAxes + transformation,
            )
        default_params_copy = self.default_params.copy()
        default_params_copy.update(is_a_subfigure=True)
        labels, handles = sub_figure._prepare_figure(
            legend=legend,
            axes=axes,
            default_params=default_params_copy,
            is_matplotlib_style=is_matplotlib_style,
        )
        return labels, handles

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

    def _fill_in_rc_params(self, is_matplotlib_style: bool = False) -> None:
        """
        Fills in and sets the missing rc parameters from the specified ``figure_style``.
        If ``is_matplotlib_style`` is ``True``, the rc parameters are reset to the default values for the specified ``figure_style``.
        If ``is_matplotlib_style`` is ``False``, the rc parameters are updated with the missing parameters from the specified ``figure_style``.
        In both cases, the rc parameters are then updated with the user-specified parameters.
        """
        if is_matplotlib_style:
            if self.figure_style == "matplotlib":
                plt.style.use("default")
            else:
                plt.style.use(self.figure_style)
            plt.rcParams.update(self._user_rc_dict)
        else:
            params = self.default_params["rc_params"]
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
        Customize the visual style of the :class:`~graphinglib.multifigure.MultiFigure`.

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
    ) -> None:
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
