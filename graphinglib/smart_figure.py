from __future__ import annotations as _annotations_

from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from fractions import Fraction
from functools import partial
from inspect import iscoroutinefunction, signature
from logging import warning
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from shutil import which
from string import ascii_lowercase
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Self,
    TypeVar,
    Union,
    cast,
)

from .inherit import INHERIT, Inherit, is_inherit, resolved, strip_inherit

try:  # Optional dependency: astropy
    from astropy.units import Quantity
    from astropy.wcs import WCS

    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False
    WCS = cast(Any, type("WCSPlaceholder", (), {}))
    Quantity = cast(Any, type("QuantityPlaceholder", (), {}))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, writers
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure, SubFigure
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.projections import get_projection_names
from matplotlib.transforms import ScaledTranslation
from numpy.typing import ArrayLike

from ._animation import (
    _DRAW_SESSION,
    _AnimationSession,
    _ArtistEdit,
    _axis_removal,
    _bound_session,
    _combine_edits,
    _edit,
    _reject_live_update,
    _stage_live_value,
    _subplot_visibility_entries,
    _tick_removal,
)
from .exceptions import (
    IncompatibleArgumentsError,
    InvalidOperationError,
    InvalidParameterError,
    InvalidParameterTypeError,
    LayoutError,
    StyleFileError,
    StyleNotFoundError,
    UnsupportedFeatureError,
)
from .file_manager import FileLoader, FileUpdater, get_default_style, get_styles
from .graph_elements import Plottable, Text
from .legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    LegendElement,
    VerticalLineCollection,
    histogram_legend_artist,
)
from .tools import _copy_with_overrides, _require_optional_dependency

T = TypeVar("T")
ListOrItem = Union[T, list[T]]
_GRID_RC = ("grid.color", "grid.alpha", "grid.linestyle", "grid.linewidth")
_LIVE_VISUAL_RC = (
    "figure.facecolor",
    "axes.facecolor",
    "axes.edgecolor",
    "axes.linewidth",
)
_SPINE_NAMES = ("right", "left", "top", "bottom")
_SUBPLOT_TEXT = {
    "_x_label": "_sub_x_labels",
    "_y_label": "_sub_y_labels",
    "_title": "_subtitles",
}


def _require_astropy(feature: str = "this feature") -> None:
    """Raise a clear error when an astro-extra feature is used without the optional dependency installed."""
    _require_optional_dependency(_ASTROPY_AVAILABLE, feature, "astro", "Astropy")


HAS_ASTROPY = _ASTROPY_AVAILABLE


def _reject_animated_layout(element: object) -> None:
    _reject_live_update(
        element, "Configure figure layout and axis properties before animating."
    )


class SmartFigure:
    """
    This class implements a figure object for plotting :class:`~graphinglib.Plottable` elements.

    It allows for the creation of complex figures recursively, where each :class:`~graphinglib.SmartFigure` can contain
    other :class:`~graphinglib.SmartFigure` objects. The class supports a variety of customization options as well as
    the ability to use styles and themes for consistent visual appearance across different figures. A SmartFigure can
    either be used directly as a single plot containing :class:`~graphinglib.Plottable` objects, or as a layout that
    arranges child :class:`~graphinglib.SmartFigure` objects on a grid. The idea behind this class is that every
    SmartFigure contains a single x_label, y_label, title, projection, etc. and that nested SmartFigures can be
    inserted into the main SmartFigure to create complex figures with more parameters.

    Parameters
    ----------
    num_rows, num_cols : int, optional
        Number of rows and columns for the base grid. These parameters determine the number of "squares" on which the
        plots can be placed.
        Defaults to ``1``.
    x_label, y_label : str, optional
        Labels for the x and y axes of the figure.
    size : tuple[float, float], optional
        Overall size of the figure. Note that this option is useless if the SmartFigure is nested inside another
        SmartFigure, as the size is then determined by the parent SmartFigure and the available space.
        Figure size is in inches; typical width is ``4`` to ``12`` and typical height is ``3`` to ``8``.
        Default depends on the ``figure_style`` configuration.
    title : str, optional
        General title of the figure.
    x_lim, y_lim : tuple[float, float] | list[tuple[float, float]], optional
        Limits for the x and y axes of the figure. This can be given as a single value or a list of values to apply to
        each subplot.
    sub_x_labels, sub_y_labels : Iterable[str | None], optional
        Labels for the x and y axes of each subfigure, respectively. This is only useful for figures that are not a
        single subplot and when each subfigure needs its own x and y labels. This prevents the creation of nested
        :class:`~graphinglib.SmartFigure` objects for each subfigure only to set the x and y labels. This list cannot
        be longer than the number of subfigures drawn by the SmartFigure and None values can be used to skip specific
        subfigures.
    subtitles : Iterable[str | None], optional
        Labels for the subtitles of each subfigure, respectively. Similarly to `sub_x_labels` and `sub_y_labels`, this
        allows to set subtitles for each subfigure without needing to create nested
        :class:`~graphinglib.SmartFigure` objects. It is only useful for figures that are not a single subplot and when
        each subfigure needs its own subtitle. This list cannot be longer than the number of subfigures drawn by the
        SmartFigure and None values can be used to skip specific subfigures.
    log_scale_x, log_scale_y : bool | list[bool | None], optional
        Whether to use a logarithmic scale for the x and y axes, respectively. This can be given as a single value or
        a list of values to apply to each subplot.
        Defaults to ``False``.
    remove_axes : bool | list[bool | None], optional
        Whether to remove the axes from the figure. This can be given as a single boolean or a list of booleans to apply
        to each subplot.
        Defaults to ``False``.
    aspect_ratio : float | Literal["auto", "equal"] | list[float | Literal["auto", "equal"]], optional
        Aspect ratio of the figure. If set to "auto", the aspect ratio is determined automatically to fill the available
        space. If set to "equal", the aspect ratio is set to 1:1. If set to a float, the aspect ratio represents the
        ratio of the height to the width of the data. This can be given as a single value or a list of values to apply
        to each subplot.
        Values are ``"equal"``, ``"auto"``, or a positive float. Defaults to ``"auto"``.

        .. warning::
            This parameter must not be confused with the `box_aspect_ratio` parameter, which is the aspect ratio of the
            box containing the elements. The `aspect_ratio` parameter is the aspect ratio of the data itself, which
            does not change the size of the plot but rather how the data is displayed within the plot.

    box_aspect_ratio : float | list[float], optional
        Aspect ratio of the box containing the elements, i.e. the ratio of the height to the width of the plot. This can
        be given as a single value or a list of values to apply to each subplot.

        .. warning::
            This parameter must not be confused with the `aspect_ratio` parameter, which is the aspect ratio of the
            data itself. The `box_aspect_ratio` parameter changes the size of the plot, which does not affect the
            figure's axes.

    remove_x_ticks, remove_y_ticks : bool | list[bool | None], optional
        Whether to remove the x and y ticks from the figure, respectively. This can be given as a single value or a list
        of values to apply to each subplot.
        Defaults to ``False``.
    invert_x_axis, invert_y_axis : bool | list[bool | None], optional
        Whether to invert the x and y axes, respectively. Use a single value or a list aligned with the subplots.
        During animation, a list entry of ``None`` leaves that subplot alone.
        Defaults to ``False``.
    reference_labels : bool | list[bool], optional
        Whether or not to add reference labels to the subfigures. If set to ``True``, each subfigure will be labeled
        alphabetically in the form of "a)", "b)", etc. This can be given as a single value or a list of values to apply
        to each subplot.
        Defaults to ``True``.

        .. note::
            For nested figures, each subfigure controls its own reference labels. This means that if a nested
            SmartFigure turns off reference labels, the plots in it will not be labeled, even if the parent SmartFigure
            has reference labels turned on.

    global_reference_label : bool, optional
        Whether to use a single reference label for the entire figure instead of individual labels for each subfigure.
        If set to ``True``, the reference label will be placed in the top left corner of the global SmartFigure. This is
        useful for labeling the entire figure rather than individual subfigures.
        Defaults to ``False``.

        .. warning::
            As the global reference label is placed more left than the reference label, this forces the horizontal shift
            of the axes, which may lead to overlapping between axes. Consider modifying the `size` or `width_padding`
            parameters to avoid this issue.
    reference_labels_loc : Literal["inside", "outside"] | tuple[float, float] | list, optional
        Location of the reference labels of the SubFigures, which can be either ``"inside"``, ``"outside"``, or a tuple of
        (x, y) relative coordinates to the top-left corner of each subfigure. This can be given as a single value or a
        list of values to apply to each subplot.
        Defaults to ``"outside"``.
    width_padding, height_padding : float, optional
        Padding between the subfigures in the x and y directions, respectively. The default value of ``None`` results in
        a default small amount of padding. This may be set to 0 to completely remove the space between subfigures, but
        note that axes labels may need to be removed to delete additional space.
    width_ratios, height_ratios : ArrayLike, optional
        Ratios of the widths and heights of the subfigures, respectively. These ratios determine how much space each
        column and row of subfigures will take up in the overall figure. The length of these arrays must match the
        number of columns and rows, respectively. By default, all subfigures are given equal space.
    share_x, share_y : bool, optional
        Whether to share the x and y axes between subfigures, respectively. This means that all subfigures will have
        the same x and y limits, and the ticks will be shared as well. This is useful for comparing data across
        subfigures.

        .. note::
            Sharing axes only works for plots drawn directly in the SmartFigure. If you insert an existing nested
            SmartFigure, that nested figure keeps its own axes sharing settings.

    projection : Any | list[Any], optional
        Projection type for the subfigures. This can be a string of a matplotlib projection (e.g., "polar") or an object
        capable of creating a projection (e.g. astropy.wcs.WCS). This can be given as a single value or a list of values
        to apply to each subplot.

        .. note::
            3D projections are not supported at the moment.

    general_legend : bool, optional
        Whether to create a general legend for the entire figure. If set to ``True``, a single legend will be created
        to regroup all the legends from the subplots. If set to ``False``, all subplots will have their own legend. If
        nested SmartFigures set this parameter to ``False``, their legend is added to the parent's general legend.
        However, if a nested SmartFigure sets its general legend to ``True``, it will be created separately and will not
        be added to the parent's general legend.
        Defaults to ``False``.
    legend_loc : str | tuple | list[str | tuple], optional
        Location of the legend. This can be a string (e.g., "upper right") or a tuple of (x, y) relative coordinates.
        The supported string locations are: {"upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center", "center", "outside upper center",
        "outside center right", "outside lower center", "outside center left"}. Additionally, only if ``general_legend``
        is set to ``False``, the legend location can also be set to "best". This option can be given as a single value
        or a list of values to apply to each subplot.
        Defaults to ``"best"`` if ``general_legend`` is set to ``False``, otherwise it defaults to ``"lower center"``.

        .. warning::
            If ``general_legend`` is set to ``True`` and the legend location is set to a position containing "outside",
            the legend may not be displayed correctly in some matplotlib backends. In such cases, it is recommended to
            use inline figures in a Jupyter notebook or save the figure to a file to ensure proper display of the
            legend outside the figure.

    legend_cols : int | list[int], optional
        Number of columns to display the labels in the legend. This is only used if the legend is displayed. This can be
        given as a single value or a list of values to apply to each subplot.
        Defaults to ``1``.
    show_legend : bool | list[bool | None], optional
        Whether to show the legend for the figure. This allows to easily toggle the visibility of the legend. This can
        be given as a single value or a list of values to apply to each subplot.
        Defaults to ``True``.
    twin_x_axis, twin_y_axis : SmartTwinAxis, optional
        Twin axes for the x and y axes, respectively. This allows to attach additional axes to the main axes of the
        SmartFigure, which can be useful for displaying additional information or data on the same plot without
        cluttering the main axes. The twin axes can be used to plot additional data with different scales or units. See
        the :class:`~graphinglib.SmartTwinAxis` class for more details on how to use twin axes and the
        :meth:`~graphinglib.SmartFigure.create_twin_axis` method for wrapping the creation of twin axes.
    figure_style : str, optional
        The figure style to use for the figure. The default style can be set using ``gl.set_default_style()``.
        Defaults to ``"default"``.
    elements : Plottable | Iterable[Plottable | SmartFigure | None] | Iterable[Iterable[Plottable | None]], optional
        The elements to plot in the figure.
        For a standalone :class:`~graphinglib.SmartFigure`:class:`~graphinglib.SmartFigure`, this can be a single
        :class:`~graphinglib.Plottable` or an iterable of :class:`~graphinglib.Plottable` objects to plot in the figure.
        For a :class:`~graphinglib.SmartFigure` with multiple cells, passing a single :class:`~graphinglib.Plottable`
        will add it in the first cell of the figure. If an iterable is provided for a figure with multiple cells,
        the items are assigned to the cells from left-to-right, top-to-bottom. Each item can be a child
        :class:`~graphinglib.SmartFigure`, a :class:`~graphinglib.Plottable`, an iterable of :class:`~graphinglib.Plottable`
        objects, or ``None``. The iterable can be shorter than the total number of cells, in which case the remaining cells
        are left empty. If a :class:`~graphinglib.SmartFigure` is provided, it occupies exactly one cell in the parent figure.
        Its ``num_rows`` and ``num_cols`` only describe its internal layout.

        .. note::
            The ``elements`` parameter does not use a child :class:`~graphinglib.SmartFigure`'s shape to determine how
            much space it occupies in the parent. If you want a bare :class:`~graphinglib.Plottable`, an iterable of
            Plottables, or a child :class:`~graphinglib.SmartFigure` to span multiple subplots in the parent, you
            should use the __setitem__ method instead.
            For example, to add an element spanning the complete first row , use ``fig[0,:] = element``.

    annotations : Iterable[Text], optional
        Text annotations to add on the figure. The provided Text elements must have figure-relative coordinates, i.e.
        in the range [0, 1] for both x and y. This allows to add annotations that are not tied to a specific subplot,
        for example, to add general notes or labels on the figure.
    """

    def __init__(
        self,
        num_rows: int = 1,
        num_cols: int = 1,
        x_label: str | None = None,
        y_label: str | None = None,
        size: tuple[float, float] | Inherit = INHERIT,
        title: str | None = None,
        x_lim: ListOrItem[tuple[float | None, float | None] | None] = None,
        y_lim: ListOrItem[tuple[float | None, float | None] | None] = None,
        sub_x_labels: Iterable[str | None] | None = None,
        sub_y_labels: Iterable[str | None] | None = None,
        subtitles: Iterable[str | None] | None = None,
        log_scale_x: bool | list[bool | None] = False,
        log_scale_y: bool | list[bool | None] = False,
        remove_axes: bool | list[bool | None] = False,
        aspect_ratio: ListOrItem[float | Literal["auto", "equal"]] = "auto",
        box_aspect_ratio: ListOrItem[float | None] = None,
        remove_x_ticks: bool | list[bool | None] = False,
        remove_y_ticks: bool | list[bool | None] = False,
        invert_x_axis: bool | list[bool | None] = False,
        invert_y_axis: bool | list[bool | None] = False,
        reference_labels: ListOrItem[bool] = True,
        global_reference_label: bool = False,
        reference_labels_loc: ListOrItem[
            Literal["inside", "outside"] | tuple[float, float]
        ] = "outside",
        width_padding: float | None = None,
        height_padding: float | None = None,
        width_ratios: ArrayLike | None = None,
        height_ratios: ArrayLike | None = None,
        share_x: bool = False,
        share_y: bool = False,
        projection: ListOrItem[Any | None] = None,
        general_legend: bool = False,
        legend_loc: ListOrItem[str | tuple | None] = None,
        legend_cols: ListOrItem[int] = 1,
        show_legend: bool | list[bool | None] = True,
        twin_x_axis: SmartTwinAxis | None = None,
        twin_y_axis: SmartTwinAxis | None = None,
        figure_style: str | Inherit = INHERIT,
        elements: Plottable
        | Iterable[Plottable | SmartFigure | None | Iterable[Plottable | None]] = [],
        annotations: Iterable[Text] | None = None,
    ) -> None:
        self._mode: Literal["leaf", "container"] = "leaf"
        self._leaf_elements: list[Plottable] = []
        self._children: OrderedDict[tuple[slice, slice], SmartFigure] = OrderedDict()
        self._is_auto_child = False
        self._flatten_in_parent = False
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.x_label = x_label
        self.y_label = y_label
        self.size = size
        self.title = title
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.sub_x_labels = sub_x_labels
        self.sub_y_labels = sub_y_labels
        self.subtitles = subtitles
        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        self.remove_axes = remove_axes
        self.aspect_ratio = aspect_ratio
        self.box_aspect_ratio = box_aspect_ratio
        self.remove_x_ticks = remove_x_ticks
        self.remove_y_ticks = remove_y_ticks
        self.invert_x_axis = invert_x_axis
        self.invert_y_axis = invert_y_axis
        self.reference_labels = reference_labels
        self.global_reference_label = global_reference_label
        self.reference_labels_loc = reference_labels_loc
        self.width_padding = width_padding
        self.height_padding = height_padding
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.share_x = share_x
        self.share_y = share_y
        self.projection = projection
        self.general_legend = general_legend
        self.legend_loc = legend_loc
        self.legend_cols = legend_cols
        self.show_legend = show_legend
        self.twin_x_axis = twin_x_axis
        self.twin_y_axis = twin_y_axis
        self.figure_style = figure_style
        if isinstance(elements, Plottable):
            elements = [elements]
        self.elements = elements
        self.annotations = annotations

        self._figure: Figure | SubFigure | None = None
        self._gridspec: GridSpec | None = None
        self._reference_label_i: int | None = None
        self._animation: FuncAnimation | None = None

        self._ticks: dict[str, Any] = {}
        self._tick_params: dict[str, dict[str, Any]] = {
            "x major": {},
            "y major": {},
            "x minor": {},
            "y minor": {},
        }
        self._pad_params = {}
        self._reference_labels_params = {}

        self._x_axis_label_color = "k"
        self._y_axis_label_color = "k"

        self._show_grid: bool | list[bool | None] | None = None
        self._grid = {}

        self.hide_custom_legend_elements = False
        self.hide_default_legend_elements = False
        self._custom_legend_handles = []
        self._custom_legend_labels = []

        self._hidden_spines = None
        self._user_rc_dict: dict[str, Any] = {}
        self._default_params = {}
        self._subplot_p: dict[
            str, list[Any]
        ] = {}  # used to store the ListOrItem parameters that can be different for each subplot

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @num_rows.setter
    def num_rows(self, value: int) -> None:
        _reject_animated_layout(self)
        if not isinstance(value, int):
            raise InvalidParameterTypeError("num_rows must be an integer.")
        if value < 1:
            raise InvalidParameterError("num_rows must be greater than 0.")
        should_promote = False
        # Check if the number of rows is being reduced and conflicts with existing elements
        try:
            if self._num_rows > value:
                removed_rows = list(range(value, self._num_rows))
                for pos, element in self._children.items():
                    if (pos[0].stop - 1) in removed_rows and element:
                        raise InvalidOperationError(
                            "Cannot remove rows from the SmartFigure when there are elements in "
                            "them. Please remove the elements first."
                        )
            should_promote = (
                self._mode == "leaf"
                and self._num_rows * self._num_cols == 1
                and value * self._num_cols > 1
            )
        except AttributeError:
            # The figure is being created, so the _num_rows attribute is not yet set
            pass
        if should_promote:
            self._promote_leaf_to_container()
        self._num_rows = value

    @property
    def num_cols(self) -> int:
        return self._num_cols

    @num_cols.setter
    def num_cols(self, value: int) -> None:
        _reject_animated_layout(self)
        if not isinstance(value, int):
            raise InvalidParameterTypeError("num_cols must be an integer.")
        if value < 1:
            raise InvalidParameterError("num_cols must be greater than 0.")
        should_promote = False
        # Check if the number of rows is being reduced and conflicts with existing elements
        try:
            if self._num_cols > value:
                removed_cols = list(range(value, self._num_cols))
                for pos, element in self._children.items():
                    if (pos[1].stop - 1) in removed_cols and element:
                        raise InvalidOperationError(
                            "Cannot remove cols from the SmartFigure when there are elements in "
                            "them. Please remove the elements first."
                        )
            should_promote = (
                self._mode == "leaf"
                and self._num_rows * self._num_cols == 1
                and self._num_rows * value > 1
            )
        except AttributeError:
            # The figure is being created, so the _num_cols attribute is not yet set
            pass
        if should_promote:
            self._promote_leaf_to_container()
        self._num_cols = value

    @property
    def shape(self) -> tuple[int, int]:
        return self._num_rows, self._num_cols

    @shape.setter
    def shape(self, value: tuple[int, int]) -> None:
        _reject_animated_layout(self)
        self.num_rows, self.num_cols = value

    @property
    def x_label(self) -> str | None:
        return self._x_label

    @x_label.setter
    def x_label(self, value: str | None) -> None:
        self._x_label = _stage_live_value(self, "_x_label", value)

    @property
    def y_label(self) -> str | None:
        return self._y_label

    @y_label.setter
    def y_label(self, value: str | None) -> None:
        self._y_label = _stage_live_value(self, "_y_label", value)

    @property
    def size(self) -> tuple[float, float] | Inherit:
        return self._size

    @size.setter
    def size(self, value: tuple[float, float] | Inherit):
        _reject_animated_layout(self)
        if is_inherit(value):
            self._size = value
            return
        if not isinstance(value, tuple):
            raise InvalidParameterTypeError("size must be a tuple or 'default'.")
        if len(value) != 2:
            raise InvalidParameterError("size must be a tuple of length 2.")
        if value[0] <= 0 or value[1] <= 0:
            raise InvalidParameterError("size values must be greater than 0.")
        self._size = value

    @property
    def title(self) -> str | None:
        return self._title

    @title.setter
    def title(self, value: str | None) -> None:
        self._title = _stage_live_value(self, "_title", value)

    @property
    def x_lim(self) -> ListOrItem[tuple[float | None, float | None] | None]:
        return self._x_lim

    @x_lim.setter
    def x_lim(
        self, value: ListOrItem[tuple[float | None, float | None] | None]
    ) -> None:
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if not isinstance(v, tuple):
                    raise InvalidParameterTypeError("x_lim must be a tuple.")
                if len(v) != 2:
                    raise InvalidParameterError("x_lim must be a tuple of length 2.")
        self._x_lim = _stage_live_value(self, "_x_lim", value)

    @property
    def y_lim(self) -> ListOrItem[tuple[float | None, float | None] | None]:
        return self._y_lim

    @y_lim.setter
    def y_lim(
        self, value: ListOrItem[tuple[float | None, float | None] | None]
    ) -> None:
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if not isinstance(v, tuple):
                    raise InvalidParameterTypeError("y_lim must be a tuple.")
                if len(v) != 2:
                    raise InvalidParameterError("y_lim must be a tuple of length 2.")
        self._y_lim = _stage_live_value(self, "_y_lim", value)

    @property
    def sub_x_labels(self) -> Iterable[str | None] | None:
        return self._sub_x_labels

    @sub_x_labels.setter
    def sub_x_labels(self, value: Iterable[str | None] | None) -> None:
        self._sub_x_labels = self._stage_subplot_text("_sub_x_labels", value)

    @property
    def sub_y_labels(self) -> Iterable[str | None] | None:
        return self._sub_y_labels

    @sub_y_labels.setter
    def sub_y_labels(self, value: Iterable[str | None] | None) -> None:
        self._sub_y_labels = self._stage_subplot_text("_sub_y_labels", value)

    @property
    def subtitles(self) -> Iterable[str | None] | None:
        return self._subtitles

    @subtitles.setter
    def subtitles(self, value: Iterable[str | None] | None) -> None:
        self._subtitles = self._stage_subplot_text("_subtitles", value)

    def _stage_subplot_text(
        self, name: str, value: Iterable[str | None] | None
    ) -> str | list[str | None] | None:
        if value is not None and not isinstance(value, str):
            if not isinstance(value, Iterable):
                raise InvalidParameterTypeError(
                    f"{name[1:]} must be an iterable of strings or None."
                )
            value = value if isinstance(value, list) else list(value)
            if _bound_session(self) is not None:
                if len(value) > len(self._ordered_elements):
                    raise InvalidParameterError(
                        f"Too many {name[1:]} entries for this SmartFigure."
                    )
                if any(
                    item is not None and not isinstance(item, str) for item in value
                ):
                    raise InvalidParameterTypeError(
                        f"{name[1:]} entries must be strings or None."
                    )
        return _stage_live_value(self, name, value)

    def _stage_view_flag(
        self, name: str, value: bool | list[bool | None]
    ) -> bool | list[bool | None]:
        valid = (
            all(flag is None or isinstance(flag, bool) for flag in value)
            if isinstance(value, list)
            else isinstance(value, bool)
        )
        if not valid:
            raise InvalidParameterTypeError(
                f"{name} must be a bool or a list of bool or None."
            )
        return _stage_live_value(self, f"_{name}", value)

    @property
    def log_scale_x(self) -> bool | list[bool | None]:
        return self._log_scale_x

    @log_scale_x.setter
    def log_scale_x(self, value: bool | list[bool | None]) -> None:
        self._log_scale_x = self._stage_view_flag("log_scale_x", value)

    @property
    def log_scale_y(self) -> bool | list[bool | None]:
        return self._log_scale_y

    @log_scale_y.setter
    def log_scale_y(self, value: bool | list[bool | None]) -> None:
        self._log_scale_y = self._stage_view_flag("log_scale_y", value)

    @property
    def remove_axes(self) -> bool | list[bool | None]:
        return self._remove_axes

    @remove_axes.setter
    def remove_axes(self, value: bool | list[bool | None]) -> None:
        self._remove_axes = self._stage_view_flag("remove_axes", value)

    @property
    def aspect_ratio(self) -> ListOrItem[float | Literal["auto", "equal"]]:
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, value: ListOrItem[float | Literal["auto", "equal"]]) -> None:
        _reject_animated_layout(self)
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, (float, int)) and v != "auto" and v != "equal":
                raise InvalidParameterTypeError(
                    "aspect_ratio must be a float, 'auto' or 'equal'."
                )
            if isinstance(v, (float, int)) and v <= 0:
                raise InvalidParameterError("aspect_ratio must be greater than 0.")
        self._aspect_ratio = value

    @property
    def box_aspect_ratio(self) -> ListOrItem[float | None]:
        return self._box_aspect_ratio

    @box_aspect_ratio.setter
    def box_aspect_ratio(self, value: ListOrItem[float | None]) -> None:
        _reject_animated_layout(self)
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if not isinstance(v, (float, int)):
                    raise InvalidParameterTypeError(
                        "box_aspect_ratio must be a number."
                    )
                if v <= 0:
                    raise InvalidParameterError(
                        "box_aspect_ratio must be greater than 0."
                    )
        self._box_aspect_ratio = value

    @property
    def remove_x_ticks(self) -> bool | list[bool | None]:
        return self._remove_x_ticks

    @remove_x_ticks.setter
    def remove_x_ticks(self, value: bool | list[bool | None]) -> None:
        self._remove_x_ticks = self._stage_view_flag("remove_x_ticks", value)

    @property
    def remove_y_ticks(self) -> bool | list[bool | None]:
        return self._remove_y_ticks

    @remove_y_ticks.setter
    def remove_y_ticks(self, value: bool | list[bool | None]) -> None:
        self._remove_y_ticks = self._stage_view_flag("remove_y_ticks", value)

    @property
    def invert_x_axis(self) -> bool | list[bool | None]:
        return self._invert_x_axis

    @invert_x_axis.setter
    def invert_x_axis(self, value: bool | list[bool | None]) -> None:
        self._invert_x_axis = self._stage_view_flag("invert_x_axis", value)

    @property
    def invert_y_axis(self) -> bool | list[bool | None]:
        return self._invert_y_axis

    @invert_y_axis.setter
    def invert_y_axis(self, value: bool | list[bool | None]) -> None:
        self._invert_y_axis = self._stage_view_flag("invert_y_axis", value)

    @property
    def reference_labels(self) -> ListOrItem[bool]:
        return self._reference_labels

    @reference_labels.setter
    def reference_labels(self, value: ListOrItem[bool]) -> None:
        _reject_animated_layout(self)
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise InvalidParameterTypeError("reference_labels must be a bool.")
        self._reference_labels = value

    @property
    def global_reference_label(self) -> bool:
        return self._global_reference_label

    @global_reference_label.setter
    def global_reference_label(self, value: bool) -> None:
        _reject_animated_layout(self)
        if not isinstance(value, bool):
            raise InvalidParameterTypeError("global_reference_label must be a bool.")
        self._global_reference_label = value

    @property
    def reference_labels_loc(
        self,
    ) -> ListOrItem[Literal["inside", "outside"] | tuple[float, float]]:
        return self._reference_labels_loc

    @reference_labels_loc.setter
    def reference_labels_loc(
        self, value: ListOrItem[Literal["inside", "outside"] | tuple[float, float]]
    ) -> None:
        _reject_animated_layout(self)
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, tuple):
                if len(v) != 2:
                    raise InvalidParameterError(
                        "If reference_labels_loc is a tuple, it must be of length 2."
                    )
            elif v not in ["inside", "outside"]:
                raise InvalidParameterError(
                    "reference_labels_loc must be either 'inside' or 'outside'."
                )
        self._reference_labels_loc = value

    @property
    def width_padding(self) -> float | None:
        return self._width_padding

    @width_padding.setter
    def width_padding(self, value: float | None) -> None:
        _reject_animated_layout(self)
        if value is not None:
            if not isinstance(value, (float, int)):
                raise InvalidParameterTypeError("width_padding must be a number.")
            if value < 0:
                raise InvalidParameterError(
                    "width_padding must be greater than or equal to 0."
                )
        self._width_padding = value

    @property
    def height_padding(self) -> float | None:
        return self._height_padding

    @height_padding.setter
    def height_padding(self, value: float | None) -> None:
        _reject_animated_layout(self)
        if value is not None:
            if not isinstance(value, (float, int)):
                raise InvalidParameterTypeError("height_padding must be a number.")
            if value < 0:
                raise InvalidParameterError(
                    "height_padding must be greater than or equal to 0."
                )
        self._height_padding = value

    @property
    def width_ratios(self) -> ArrayLike | None:
        return self._width_ratios

    @width_ratios.setter
    def width_ratios(self, value: ArrayLike | None) -> None:
        _reject_animated_layout(self)
        if value is not None:
            if not hasattr(value, "__len__"):
                raise InvalidParameterTypeError("width_ratios must be an ArrayLike.")
            ratios = cast(Sequence[float], value)
            if not all(isinstance(x, (float, int)) for x in ratios):
                raise InvalidParameterTypeError(
                    "width_ratios must contain only numbers."
                )
            if len(ratios) != self._num_cols:
                raise InvalidParameterError(
                    "width_ratios must have the same length as num_cols."
                )
        self._width_ratios = value

    @property
    def height_ratios(self) -> ArrayLike | None:
        return self._height_ratios

    @height_ratios.setter
    def height_ratios(self, value: ArrayLike | None) -> None:
        _reject_animated_layout(self)
        if value is not None:
            if not hasattr(value, "__len__"):
                raise InvalidParameterTypeError("height_ratios must be an ArrayLike.")
            ratios = cast(Sequence[float], value)
            if not all(isinstance(x, (float, int)) for x in ratios):
                raise InvalidParameterTypeError(
                    "height_ratios must contain only numbers."
                )
            if len(ratios) != self._num_rows:
                raise InvalidParameterError(
                    "height_ratios must have the same length as num_rows."
                )
        self._height_ratios = value

    @property
    def share_x(self) -> bool:
        return self._share_x

    @share_x.setter
    def share_x(self, value: bool) -> None:
        _reject_animated_layout(self)
        if not isinstance(value, bool):
            raise InvalidParameterTypeError("share_x must be a bool.")
        self._share_x = value

    @property
    def share_y(self) -> bool:
        return self._share_y

    @share_y.setter
    def share_y(self, value: bool) -> None:
        _reject_animated_layout(self)
        if not isinstance(value, bool):
            raise InvalidParameterTypeError("share_y must be a bool.")
        self._share_y = value

    @property
    def projection(self) -> ListOrItem[Any | None]:
        return self._projection

    @projection.setter
    def projection(self, value: ListOrItem[Any | None]) -> None:
        _reject_animated_layout(self)
        valid_projections = get_projection_names()
        if "3d" in valid_projections:
            valid_projections.remove("3d")
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if isinstance(v, str):
                    if v == "3d":
                        raise UnsupportedFeatureError("3D projection is not supported.")
                    if v not in valid_projections:
                        raise InvalidParameterError(
                            f"projection must be one of {valid_projections} or a valid object."
                        )
                elif isinstance(v, WCS):
                    raise UnsupportedFeatureError(
                        "WCS projection should be used with the SmartFigureWCS object."
                    )
        self._projection = value
        if hasattr(self, "_children"):
            self._sync_auto_child_projections()

    @property
    def general_legend(self) -> bool:
        return self._general_legend

    @general_legend.setter
    def general_legend(self, value: bool) -> None:
        _reject_animated_layout(self)
        if not isinstance(value, bool):
            raise InvalidParameterTypeError("general_legend must be a bool.")
        self._general_legend = value

    @property
    def legend_loc(self) -> ListOrItem[str | tuple | None]:
        return self._legend_loc

    @legend_loc.setter
    def legend_loc(self, value: ListOrItem[str | tuple | None]) -> None:
        _reject_animated_layout(self)
        choices = [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
            "outside upper center",
            "outside center right",
            "outside lower center",
            "outside center left",
        ]
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if isinstance(v, str):
                    if v not in choices:
                        raise IncompatibleArgumentsError(
                            f"legend_loc must be one of {choices}."
                        )
                    if self._general_legend and v == "best":
                        raise IncompatibleArgumentsError(
                            "legend_loc cannot be 'best' when general_legend is True."
                        )
                elif isinstance(v, tuple):
                    if len(v) != 2:
                        raise InvalidParameterError(
                            "legend_loc must be a string or a tuple of length 2."
                        )
                else:
                    raise InvalidParameterTypeError(
                        "legend_loc must be a string or tuple."
                    )
        self._legend_loc = value

    @property
    def legend_cols(self) -> ListOrItem[int]:
        return self._legend_cols

    @legend_cols.setter
    def legend_cols(self, value: ListOrItem[int]) -> None:
        _reject_animated_layout(self)
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, int):
                raise InvalidParameterTypeError("legend_cols must be an integer.")
            if v < 1:
                raise InvalidParameterError("legend_cols must be greater than 0.")
        self._legend_cols = value

    @property
    def show_legend(self) -> bool | list[bool | None]:
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value: bool | list[bool | None]) -> None:
        if (
            self._general_legend
            and isinstance(value, list)
            and _bound_session(self) is not None
        ):
            raise IncompatibleArgumentsError(
                "A general legend requires a single show_legend boolean."
            )
        self._show_legend = self._stage_view_flag("show_legend", value)

    @property
    def twin_x_axis(self) -> SmartTwinAxis | None:
        return self._twin_x_axis

    @twin_x_axis.setter
    def twin_x_axis(self, value: SmartTwinAxis | None) -> None:
        _reject_animated_layout(self)
        if value is not None:
            if not self.is_single_subplot:
                raise InvalidOperationError(
                    "Twin axes can only be created for single subplot SmartFigures."
                )
            if not isinstance(value, SmartTwinAxis):
                raise InvalidParameterTypeError(
                    "twin_x_axis must be a SmartTwinAxis instance."
                )
        self._twin_x_axis = value

    @property
    def twin_y_axis(self) -> SmartTwinAxis | None:
        return self._twin_y_axis

    @twin_y_axis.setter
    def twin_y_axis(self, value: SmartTwinAxis | None) -> None:
        _reject_animated_layout(self)
        if value is not None:
            if not self.is_single_subplot:
                raise InvalidOperationError(
                    "Twin axes can only be created for single subplot SmartFigures."
                )
            if not isinstance(value, SmartTwinAxis):
                raise InvalidParameterTypeError(
                    "twin_y_axis must be a SmartTwinAxis instance."
                )
        self._twin_y_axis = value

    @property
    def figure_style(self) -> str | Inherit:
        return self._figure_style

    @figure_style.setter
    def figure_style(self, value: str | Inherit) -> None:
        _reject_animated_layout(self)
        if not isinstance(value, str) and not is_inherit(value):
            raise InvalidParameterTypeError("figure_style must be a string or INHERIT.")
        available_styles = [INHERIT, "matplotlib"] + get_styles(matplotlib=True)
        if value not in available_styles:
            raise InvalidParameterError(
                f"figure_style must be one of {available_styles}."
            )
        self._figure_style = value

    @property
    def elements(self) -> list[Plottable] | list[SmartFigure | None]:
        if self._mode == "leaf":
            return self._leaf_elements

        dense: list[SmartFigure | None] = [None] * (self._num_rows * self._num_cols)
        for (rows, cols), child in self._iter_child_items():
            dense[rows.start * self._num_cols + cols.start] = child
        return dense

    @elements.setter
    def elements(
        self,
        value: (
            Plottable
            | Iterable[Plottable | SmartFigure | None | Iterable[Plottable | None]]
        ),
    ) -> None:
        """
        Sets the elements of the SmartFigure with the same rules as the constructor.

        Assigning a single :class:`~graphinglib.Plottable` or an iterable of Plottables replaces the contents of the
        SmartFigure's single plot.
        Assigning a flat iterable containing :class:`~graphinglib.SmartFigure` objects, Plottables, iterables of
        Plottables, or ``None`` fills the figure's grid from left-to-right, top-to-bottom and rebuilds its subplot
        layout from scratch.

        For adding elements instead of replacing them, use the :meth:`~graphinglib.SmartFigure.add_elements` or the
        :meth:`~graphinglib.SmartFigure.__setitem__` methods.
        """
        _reject_animated_layout(self)
        if isinstance(value, SmartFigure):
            raise InvalidParameterTypeError(
                "Leaf elements cannot be assigned a SmartFigure directly."
            )

        if isinstance(value, Plottable):
            self._ensure_leaf_mode()
            self._leaf_elements = [value]
            return

        if not isinstance(value, Iterable):
            raise InvalidParameterTypeError(
                "elements must be a Plottable or an iterable."
            )

        value_list = list(value)
        if self._should_use_container_elements_setter(value_list):
            self._set_container_elements(value_list)
        else:
            self._ensure_leaf_mode()
            self._leaf_elements = self._normalize_leaf_rhs(value_list)

    @property
    def annotations(self) -> Iterable[Text] | None:
        return self._annotations

    @annotations.setter
    def annotations(self, value: Iterable[Text] | None) -> None:
        _reject_animated_layout(self)
        if value is not None:
            if not isinstance(value, Iterable) or not all(
                isinstance(t, Text) for t in value
            ):
                raise InvalidParameterTypeError(
                    "annotations must be an iterable of Text elements."
                )
        self._annotations = value

    @property
    def show_grid(self) -> bool | list[bool | None]:
        """
        Whether to show the grid lines on the figure. The figure style supplies
        the default grid. Use :meth:`~graphinglib.SmartFigure.set_grid` to configure it.
        """
        return self._show_grid if self._show_grid is not None else False

    @show_grid.setter
    def show_grid(self, value: bool | list[bool | None]) -> None:
        self._show_grid = self._stage_view_flag("show_grid", value)

    @property
    def hide_custom_legend_elements(self) -> bool:
        """
        Whether to hide custom legend elements. This is useful if a custom legend was previously created using the
        :meth:`~graphinglib.SmartFigure.set_custom_legend` method and you want to hide these elements. Each SmartFigure
        controls its own custom legend elements, so if this property is set to True in a nested SmartFigure, the custom
        legend elements will be hidden even if the parent SmartFigure attempts to create a general legend. However, both
        the nested and parent SmartFigures need to set this property to False to display the custom elements of a nested
        SmartFigure in a global general legend.

        .. note::
            Custom legend elements can only be plotted if the :attr:`~graphinglib.SmartFigure.general_legend` property
            is set to ``True`` or if the SmartFigure is a single subplot. This is because custom legend elements are
            associated with the figure as a whole, and not with individual subplots.
        """
        return self._hide_custom_legend_elements

    @hide_custom_legend_elements.setter
    def hide_custom_legend_elements(self, value: bool) -> None:
        _reject_animated_layout(self)
        if not isinstance(value, bool):
            raise InvalidParameterTypeError(
                "hide_custom_legend_elements must be a bool."
            )
        self._hide_custom_legend_elements = value

    @property
    def hide_default_legend_elements(self) -> ListOrItem[bool]:
        """
        Whether to hide default legend elements. This is useful if a custom legend was previously created using the
        :meth:`~graphinglib.SmartFigure.set_custom_legend` method and you want to hide the default labels created with
        each :class:`~graphinglib.Plottable` element's label. Each SmartFigure controls its own default legend elements,
        so if this property is set to True in a nested SmartFigure, the default elements will be hidden even if the
        parent SmartFigure attempts to create a general legend. However, both the nested and parent SmartFigures need to
        set this property to False to display the default elements of a nested SmartFigure in a global general legend.

        .. warning::
            The use of this property for simply toggling the visibility of the legend is discouraged. Instead, use the
            :meth:`~graphinglib.SmartFigure.show_legend` property to show or hide all the legend elements. This should
            only be used if a custom legend was created.
        """
        return self._hide_default_legend_elements

    @hide_default_legend_elements.setter
    def hide_default_legend_elements(self, value: ListOrItem[bool]) -> None:
        _reject_animated_layout(self)
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise InvalidParameterTypeError(
                    "hide_default_legend_elements must be a bool."
                )
        self._hide_default_legend_elements = value

    @property
    def is_single_subplot(self) -> bool:
        """
        Whether the SmartFigure currently behaves as a single plot.

        This is useful to determine if the SmartFigure can directly host Plottables and twin axes or if it currently
        behaves as a layout containing child SmartFigures.

        .. note::
            This property is used to verify if custom legend elements can be added to the SmartFigure even if the
            :attr:`~graphinglib.SmartFigure.general_legend` is set to ``False``.
        """
        return self._mode == "leaf"

    def __len__(self) -> int:
        """
        Gives the number of immediate contents in the :class:`~graphinglib.SmartFigure`.

        For a SmartFigure used as a single plot, this is the number of Plottables in that plot.
        For a SmartFigure used as a layout, this is the number of immediate child SmartFigures.
        """
        if self._mode == "leaf":
            return len(self._leaf_elements)
        return len(self._children)

    def __setitem__(
        self,
        key: int | slice | tuple[int | slice, ...],
        element: Plottable | Iterable[Plottable | None] | SmartFigure | None,
    ) -> None:
        """
        Assigns a Plottable, a list of Plottable objects, or a SmartFigure to a specified position in the SmartFigure.
        The indexing follows classical 2D numpy-like indexing, where the first element corresponds to the row and the
        second element corresponds to the column.

        Parameters
        ----------
        key : int | slice | tuple[int | slice, ...]
            The key specifying the location(s) in the SmartFigure to assign the element(s). If a tuple of ints is
            provided, the element is placed in the corresponding square of the grid, following classical 2D numpy-like
            indexing. If slices are provided, the element can span multiple squares in the grid. If ``num_rows`` or
            ``num_cols`` is set to 1, the key can be a single int or slice. Otherwise, the key must be a two-tuple.
        element : Plottable | Iterable[Plottable | None] | SmartFigure | None
            The element(s) to assign. Must be a Plottable, an iterable of Plottable objects, or a SmartFigure. If a
            Plottable or an iterable of Plottables is provided, the selected area becomes a child plot containing those
            elements. If ``None``, any child overlapping with the specified key will be removed.

            .. note::
                - SmartFigures used as a single plot do not support subplot assignment. To use ``__setitem__``, first
                  create a layout by setting ``num_rows`` or ``num_cols`` larger than 1.
                - You can access and modify multi-cell child figures by indexing any cell they occupy.
                - Setting a cell to None will delete the entire child figure occupying that cell.
                - Assigning new Plottables to a cell with an existing child figure replaces that child figure's plotted
                  elements while preserving its span.
                - You can add elements to an existing child plot using the ``+=`` operator.
                - If the requested slice overlaps with multiple different child figures, a LayoutError is raised.

        Examples
        --------
        Create a SmartFigure with 2 rows and 2 columns, and assign Plottables to specific subplots::

            fig = SmartFigure(num_rows=2, num_cols=2)
            fig[0, 0] = gl.Curve(x, y)
            fig[0, 1] = [gl.Scatter(x, y), gl.Text(1, 1, "text")]
            fig[1, :] = gl.Histogram(x, n_bins)

        Now we have a 2x2 SmartFigure with the following layout::

            +------------+------------+
            | 0,0        | 0,1        |
            | Curve      | Scatter    |
            |            | Text       |
            +------------+------------+
            | 1,0          1,1        |
            | Histogram               |
            +-------------------------+

        We can add elements using the ``+=`` operator and remove them using ``None``. Notice that we can access
        the multi-cell Histogram by indexing any cell it occupies::

            fig[0, 0] += [gl.Curve(x2, y2)]
            fig[0, 1] = None
            fig[1, 0] = None  # This deletes the Histogram even though it spans both cells
            # Or equivalently:
            # fig[1, :] = None

        Which will result in the following layout::

            +------------+------------+
            | 0,0        | 0,1        |
            | Curve      |            |
            | Curve      |            |
            +------------+------------+
            | 1,0        | 1,1        |
            |            |            |
            +------------+------------+

        We can also insert a nested SmartFigure. If it overlaps with existing elements, they will be replaced::

            subfigure = SmartFigure(num_rows=2, num_cols=1)
            subfigure.add_elements(gl.Heatmap(data1), gl.Heatmap(data2))
            fig[0, 1] = subfigure  # Placed in the top-right cell

        Which will lead to the following layout::

            +------------+------------+
            | 0,0        | Heatmap    |
            | Curve      +------------+
            | Curve      | Heatmap    |
            +------------+------------+
            | 1,0        | 1,1        |
            |            |            |
            +------------+------------+
        """
        _reject_animated_layout(self)
        if self._mode == "leaf":
            raise LayoutError(
                "SmartFigures used as a single plot do not support subplot assignment. "
                "Increase num_rows or num_cols first to turn the SmartFigure into a layout."
            )

        key_ = self._validate_and_normalize_key(key)
        overlapping = self._get_overlapping_elements(key_)

        if element is None:
            if len(overlapping) == 0:
                return
            if len(overlapping) > 1:
                raise LayoutError(
                    f"The requested slice {key} overlaps with multiple subfigures. "
                    f"Cannot delete multiple subfigures at once. Please delete each subfigure separately."
                )
            self._children.pop(overlapping[0][0])
            return

        if isinstance(element, SmartFigure):
            if len(overlapping) > 1:
                raise LayoutError(
                    f"The requested slice {key} overlaps with multiple subfigures. "
                    f"Cannot assign a new element to a position that overlaps with multiple subfigures. "
                    f"Please remove the overlapping subfigures first or use a more specific slice."
                )
            if len(overlapping) == 1:
                existing_key, existing_child = overlapping[0]
                if element is existing_child and existing_key == key_:
                    return
                self._children.pop(existing_key)
            element._flatten_in_parent = False
            element._is_auto_child = False
            self._children[key_] = element
            self._children = self._ordered_children()
            return

        normalized = self._normalize_leaf_rhs(element)
        if len(overlapping) == 0:
            self._children[key_] = self._make_auto_child(normalized)
            changed_span = key_
            changed_child = self._children[key_]
        elif len(overlapping) == 1:
            existing_key, existing_child = overlapping[0]
            existing_child._ensure_leaf_mode()
            existing_child._leaf_elements = normalized
            self._children.move_to_end(existing_key)
            changed_span = existing_key
            changed_child = existing_child
        else:
            raise LayoutError(
                f"The requested slice {key} overlaps with multiple subfigures. "
                f"Cannot assign a new element to a position that overlaps with multiple subfigures. "
                f"Please remove the overlapping subfigures first or use a more specific slice."
            )
        self._children = self._ordered_children()
        self._sync_auto_child_projection(changed_span, changed_child)

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> SmartFigure:
        """
        Gives the child SmartFigure at the specified key in the SmartFigure. This can be used to modify or extract
        directly a child figure in a SmartFigure used as a layout. The indexing follows classical 2D numpy-like indexing,
        where the first element corresponds to the row and the second element corresponds to the column.

        Parameters
        ----------
        key : int | slice | tuple[int | slice, ...]
            The key specifying the location(s) in the SmartFigure to access. If a tuple of ints is provided, the child
            figure is accessed in the corresponding square of the grid, following classical 2D numpy-like indexing. If
            slices are provided, a child figure spanning multiple squares in the grid can be retrieved. If ``num_rows`` or
            ``num_cols`` is set to 1, the key can be a single int or slice to index into the single row or column.
            Otherwise, the key must be a two-tuple.

            .. note::
                If a child figure spans multiple cells, you can access it by indexing any cell it occupies. For
                example, if a child figure spans ``[0, :]`` (entire first row), you can access it via ``fig[0, :]``,
                ``fig[0, 0]``, or ``fig[0, 1]`` (assuming there are at least 2 columns). If the requested slice
                overlaps with multiple different child figures, a LayoutError is raised. SmartFigures used as a
                single plot do not support subplot indexing.

        Returns
        -------
        SmartFigure
            The child SmartFigure at the specified key.
        """
        if self._mode == "leaf":
            raise LayoutError("Leaf SmartFigures do not support subplot indexing.")
        span, child = self._get_selected_child(
            self._validate_and_normalize_key(key), key
        )
        self._sync_auto_child_projection(span, child)
        return child

    def __iter__(self) -> Iterator[Plottable | SmartFigure]:
        """
        Iterates over the immediate contents of the SmartFigure.

        SmartFigures used as a single plot yield their Plottables in insertion order. SmartFigures used as a layout
        yield their immediate child SmartFigures in grid order from top-left to bottom-right.
        """
        if self._mode == "leaf":
            yield from self._leaf_elements
            return
        for _, child in self._iter_child_items():
            yield child

    def __deepcopy__(self, memo: dict) -> Self:
        """
        Creates a deep copy of the SmartFigure instance, intentionally excluding the '_figure' and '_gridspec'
        attributes from the copy. These attributes are matplotlib objects and are not duplicated to avoid issues with
        copying live figure state.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        excluded_attrs = ["_figure", "_gridspec", "_animation"]
        for property_, value in self.__dict__.items():
            if property_ not in excluded_attrs:
                result.__dict__[property_] = deepcopy(value, memo)
        for attr in excluded_attrs:
            setattr(result, attr, None)
        return result

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.SmartFigure` object.
        """
        return deepcopy(self)

    def copy_with(self, **kwargs) -> Self:
        """
        Returns a deep copy of the SmartFigure with specified attributes overridden. This is useful when including
        SmartFigures in other SmartFigures, as it allows to modify the attributes in a single call.

        Parameters
        ----------
        **kwargs
            Public writable properties to override in the copied SmartFigure. The keys should be property names to
            modify and the values are the new values for those properties.

        Returns
        -------
        SmartFigure
            A new SmartFigure instance with the specified attributes overridden.

        Examples
        --------
        Copy an existing SmartFigure to remove the x and y labels::

            fig2 = fig1.copy_with(x_label=None, y_label=None)
        """
        new_copy = _copy_with_overrides(self, **kwargs)
        if "figure_style" in kwargs and "elements" not in kwargs:
            new_copy._reset_stylable_elements_to_default()
        return new_copy

    def _reset_stylable_elements_to_default(self) -> None:
        style_name = self._figure_style
        if is_inherit(style_name):
            style_name = get_default_style()
        try:
            defaults = FileLoader(style_name).load()
        except FileNotFoundError:
            return

        for element in self._iter_all_plottables_recursive():
            object_type = type(element).__name__
            for property_ in defaults.get(object_type, {}):
                if hasattr(element, property_):
                    setattr(element, property_, INHERIT)

    def _iter_all_plottables_recursive(self) -> Iterator[Plottable]:
        if self._mode == "leaf":
            yield from self._leaf_elements
            return
        for child in self._children.values():
            yield from child._iter_all_plottables_recursive()

    @property
    def _ordered_elements(self) -> OrderedDict:
        if self._mode == "leaf":
            if not self._leaf_elements:
                return OrderedDict()
            return OrderedDict({(slice(0, 1), slice(0, 1)): list(self._leaf_elements)})

        ordered = OrderedDict()
        for span, child in self._ordered_children().items():
            if child._flatten_in_parent:
                ordered[span] = list(child._leaf_elements)
            else:
                ordered[span] = child
        return ordered

    def _validate_and_normalize_key(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> tuple[slice, slice]:
        """
        Validates and normalizes the key for indexing into the SmartFigure. This method ensures that the key is
        either a single integer, a slice, or a tuple of integers/slices. It also checks for out-of-bounds indices and
        raises appropriate exceptions if the key is invalid. The returned key is always a tuple of two slices without
        None or negative indices..

        Parameters
        ----------
        key : int | slice | tuple[int | slice, ...]
            The key to validate and normalize.

        Returns
        -------
        tuple[slice, slice]
            The normalized key as a two-tuple of slices.
        """
        key_parts = key if isinstance(key, tuple) else (key,)

        # 1D SmartFigures
        if self._num_rows == 1 or self._num_cols == 1:
            if len(key_parts) == 1:
                row_key, col_key = (
                    (0, key_parts[0]) if self._num_rows == 1 else (key_parts[0], 0)
                )
            elif len(key_parts) == 2:
                row_key, col_key = key_parts
            else:
                raise InvalidParameterError(
                    "Key must be 1D (int or slice) or 2D with one zero index for 1D SmartFigure."
                )

        # 2D SmartFigures
        else:
            if len(key_parts) != 2:
                raise InvalidParameterError("2D indexing must use a tuple of length 2.")
            row_key, col_key = key_parts

        return (
            self._normalize_axis_key(row_key, self._num_rows, 0),
            self._normalize_axis_key(col_key, self._num_cols, 1),
        )

    @staticmethod
    def _normalize_axis_key(key: int | slice, axis_size: int, axis_index: int) -> slice:
        if isinstance(key, int):
            normalized_key = key + axis_size if key < 0 else key
            if not (0 <= normalized_key < axis_size):
                raise IndexError(
                    f"Index {key} out of bounds for axis {axis_index} with size {axis_size}."
                )
            return slice(normalized_key, normalized_key + 1, None)
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            start = start + axis_size if start < 0 else start
            stop = key.stop if key.stop is not None else axis_size
            stop = stop + axis_size if stop < 0 else stop
            if start < 0 or stop > axis_size:
                raise IndexError(
                    f"{key} out of bounds for axis {axis_index} with size {axis_size}."
                )
            if start >= stop:
                raise IndexError(
                    f"{key} for axis {axis_index} must have stop larger than start."
                )
            if key.step is not None:
                raise InvalidParameterError(
                    f"{key} step for axis {axis_index} must be None."
                )
            return slice(start, stop, None)
        raise InvalidParameterTypeError(
            f"Key element {key} for axis {axis_index} must be an int or a slice."
        )

    @staticmethod
    def _is_iterable_of_plottables(item: Any) -> bool:
        """
        Checks if the given item is an iterable of Plottable elements or None. This is used to determine if the item can
        be added to the SmartFigure as a list of elements.

        Parameters
        ----------
        item : Any
            The item to check.

        Returns
        -------
        bool
            True if the item is an iterable of Plottable elements or None, False otherwise.
        """
        return (
            not isinstance(item, (str, bytes, SmartFigure))
            and isinstance(item, Iterable)
            and all(isinstance(el, (Plottable, type(None))) for el in item)
        )

    def _get_overlapping_elements(
        self, key: tuple[slice, slice]
    ) -> list[tuple[tuple[slice, slice], SmartFigure]]:
        """
        Finds all child SmartFigures that overlap with the specified key range.

        Parameters
        ----------
        key : tuple[slice, slice]
            The key range to check for overlaps, as a tuple of two slices.

        Returns
        -------
        list[tuple[tuple[slice, slice], SmartFigure]]
            A list of tuples, each containing the key and child SmartFigure that overlaps with the specified range.
        """
        overlapping = []
        row_slice, col_slice = key

        for existing_key, element in self._children.items():
            existing_row, existing_col = existing_key

            # Check if there's any overlap
            row_overlap = not (
                row_slice.stop <= existing_row.start
                or row_slice.start >= existing_row.stop
            )
            col_overlap = not (
                col_slice.stop <= existing_col.start
                or col_slice.start >= existing_col.stop
            )

            if row_overlap and col_overlap:
                overlapping.append((existing_key, element))

        return overlapping

    def add_elements(
        self,
        *elements: Plottable | SmartFigure | None | Iterable[Plottable | None],
    ) -> Self:
        """
        Adds one or more :class:`~graphinglib.Plottable` objects to the current SmartFigure.

        If the SmartFigure is used as a single plot, the elements are appended to that plot.
        If the SmartFigure is used as a layout, the elements are applied to the grid from left to right, then top to
        bottom. Existing child plots are appended to, and empty positions are filled by creating new child plots as
        needed. This method is equivalent to using the :meth:`~graphinglib.SmartFigure.__setitem__` method for single
        cells, but can only add elements spanning single subplots.

        Parameters
        ----------
        elements : Plottable | Iterable[Plottable]
            Elements to plot in the :class:`~graphinglib.SmartFigure`. Iterables of
            :class:`~graphinglib.Plottable` objects can be provided to add multiple elements to the same plot.
            If the SmartFigure is used as a single plot, all provided elements are added to that plot. If the
            SmartFigure is used as a layout, the given elements are applied to the grid from left to right, then
            top to bottom. Existing child plots are appended to, and empty positions are filled by creating new child
            plots as needed. If ``None`` elements are present, the corresponding target is skipped.

            .. note::
                This method can create new 1x1 child plots when filling an existing layout, but it does not create
                multi-cell spans. If you want to add elements that span multiple subplots, you should use the
                :meth:`~graphinglib.SmartFigure.__setitem__` method instead.

        Returns
        -------
        Self
            For convenience, the same :class:`~graphinglib.SmartFigure` with the added elements.

        See Also
        --------
        :meth:`~graphinglib.SmartFigure.__setitem__`
            For more information on how to use the ``__setitem__`` method to add elements that span multiple columns or
            rows to the :class:`~graphinglib.SmartFigure`.
        """
        _reject_animated_layout(self)
        if self._mode == "leaf":
            for element in elements:
                if element is None:
                    continue
                if isinstance(element, Plottable):
                    self._leaf_elements.append(element)
                else:
                    self._leaf_elements.extend(self._normalize_leaf_rhs(element))
            return self

        max_cells = self._num_rows * self._num_cols
        if len(elements) > max_cells:
            raise InvalidParameterError(
                "Too many elements provided for the number of cells in the SmartFigure."
            )

        for index, element in enumerate(elements):
            if element is None:
                continue
            if isinstance(element, SmartFigure):
                raise InvalidParameterTypeError(
                    "Container add_elements does not accept SmartFigures."
                )
            key = self._dense_index_to_key(index)
            overlapping = self._get_overlapping_elements(key)

            if len(overlapping) > 1:
                raise LayoutError(
                    "Cannot add elements to a cell that overlaps with multiple different subfigures."
                )

            if len(overlapping) == 0:
                self._children[key] = self._make_auto_child(element)
                self._children = self._ordered_children()
                self._sync_auto_child_projection(key, self._children[key])
                continue

            _, child = overlapping[0]
            if not child.is_single_subplot:
                raise LayoutError(
                    "add_elements can only append to child SmartFigures that are used as a single plot."
                )
            child += element
        return self

    def __add__(self, other):
        result = self.copy()
        result += other
        return result

    def __iadd__(self, other):
        _reject_animated_layout(self)
        if self._mode == "leaf":
            self._leaf_elements.extend(self._normalize_leaf_rhs(other))
            return self

        if isinstance(other, SmartFigure) or not isinstance(other, Iterable):
            raise InvalidParameterTypeError(
                "Container SmartFigure += expects a dense iterable of Plottables or iterables of Plottables."
            )

        values = list(other)
        dense_children: list[SmartFigure | None] = [None] * (
            self._num_rows * self._num_cols
        )
        for (rows, cols), child in self._iter_child_items():
            dense_children[rows.start * self._num_cols + cols.start] = child
        for index, value in enumerate(values):
            if index >= len(dense_children) or value is None:
                continue
            child = dense_children[index]
            if child is None:
                continue
            if isinstance(value, SmartFigure):
                raise InvalidParameterTypeError(
                    "Container SmartFigure += does not accept SmartFigures."
                )
            child += value
        return self

    def _ensure_leaf_mode(self) -> None:
        self._mode = "leaf"
        self._children = OrderedDict()

    def _ensure_container_mode(self) -> None:
        if self._mode == "container":
            return
        self._mode = "container"
        self._leaf_elements = []

    def _promote_leaf_to_container(self) -> None:
        if self._mode != "leaf":
            return
        child = None
        if (
            self._leaf_elements
            or self._twin_x_axis is not None
            or self._twin_y_axis is not None
        ):
            child = self.copy()
            child._num_rows = 1
            child._num_cols = 1
            child._mode = "leaf"
            child._children = OrderedDict()
            child._is_auto_child = True
            child._flatten_in_parent = True
        self._mode = "container"
        self._leaf_elements = []
        self._twin_x_axis = None
        self._twin_y_axis = None
        self._children = OrderedDict()
        if child is not None:
            self._children[(slice(0, 1), slice(0, 1))] = child

    def _ordered_children(self) -> OrderedDict[tuple[slice, slice], SmartFigure]:
        return OrderedDict(
            sorted(
                self._children.items(),
                key=lambda item: (item[0][0].start, item[0][1].start),
            )
        )

    def _iter_child_items(self) -> Iterator[tuple[tuple[slice, slice], SmartFigure]]:
        yield from self._ordered_children().items()

    def _make_auto_child(
        self, value: Plottable | Iterable[Plottable | None]
    ) -> SmartFigure:
        if isinstance(self, SmartFigureWCS):
            projection = (
                self._projection[0]
                if isinstance(self._projection, list)
                else self._projection
            )
            child = self.__class__(projection=projection)
        else:
            child = SmartFigure()
        child._is_auto_child = True
        child._flatten_in_parent = True
        child.elements = self._normalize_leaf_rhs(value)
        return child

    def _sync_auto_child_projection(
        self, span: tuple[slice, slice], child: SmartFigure
    ) -> None:
        if not child._is_auto_child:
            return
        projection = self._projection
        if isinstance(projection, list):
            ordered_spans = [
                existing_span for existing_span, _ in self._iter_child_items()
            ]
            try:
                index = ordered_spans.index(span)
            except ValueError:
                index = 0
            if not projection:
                return
            projection = projection[min(index, len(projection) - 1)]
        if projection is None and isinstance(child, SmartFigureWCS):
            return
        child.projection = projection

    def _sync_auto_child_projections(self) -> None:
        for span, child in self._iter_child_items():
            self._sync_auto_child_projection(span, child)

    def _normalize_leaf_rhs(self, value: Any) -> list[Plottable]:
        if isinstance(value, Plottable):
            return [value]
        if isinstance(value, (str, bytes, SmartFigure)) or not isinstance(
            value, Iterable
        ):
            raise InvalidParameterTypeError(
                "Leaf contents must be Plottables or iterables of Plottables."
            )
        elements: list[Plottable] = []
        for element in value:
            if element is None:
                continue
            if not isinstance(element, Plottable):
                raise InvalidParameterTypeError(
                    "Leaf contents must be Plottables or iterables of Plottables."
                )
            elements.append(element)
        return elements

    def _get_selected_child(
        self, key: tuple[slice, slice], original_key: Any = None
    ) -> tuple[tuple[slice, slice], SmartFigure]:
        if key in self._children:
            return key, self._children[key]

        overlapping = self._get_overlapping_elements(key)
        if len(overlapping) == 0:
            raise LayoutError(
                f"The requested slice {original_key if original_key is not None else key} does not select a subfigure."
            )
        if len(overlapping) > 1:
            raise LayoutError(
                f"The requested slice {original_key if original_key is not None else key} overlaps with multiple subfigures. "
                "Cannot return a single element. Please use a more specific slice that matches only one subfigure."
            )
        return overlapping[0]

    def _should_use_container_elements_setter(self, value_list: list[Any]) -> bool:
        if any(isinstance(value, SmartFigure) for value in value_list):
            return True
        if self._num_rows * self._num_cols > 1:
            for value in value_list:
                if value is None:
                    return True
                if SmartFigure._is_iterable_of_plottables(value):
                    return True
            return len(value_list) <= self._num_rows * self._num_cols
        return False

    def _dense_index_to_key(self, index: int) -> tuple[slice, slice]:
        row, col = divmod(index, self._num_cols)
        return (slice(row, row + 1), slice(col, col + 1))

    def _set_container_elements(self, value_list: list[Any]) -> None:
        self._ensure_container_mode()
        self._children = OrderedDict()
        dense = value_list + [None] * (
            self._num_rows * self._num_cols - len(value_list)
        )
        occupied: set[tuple[int, int]] = set()

        for index, value in enumerate(dense[: self._num_rows * self._num_cols]):
            row, col = divmod(index, self._num_cols)
            if (row, col) in occupied:
                if value is not None:
                    raise LayoutError(
                        "Dense elements cannot assign a value to a cell already covered by a spanning child."
                    )
                continue
            if value is None:
                continue

            if isinstance(value, SmartFigure):
                row_span, col_span = 1, 1
                child = value
                child._flatten_in_parent = False
                child._is_auto_child = False
            else:
                row_span, col_span = 1, 1
                child = self._make_auto_child(value)

            if row + row_span > self._num_rows or col + col_span > self._num_cols:
                raise LayoutError(
                    "Child SmartFigure does not fit in the target dense layout."
                )

            span = (slice(row, row + row_span), slice(col, col + col_span))
            for covered_row in range(row, row + row_span):
                for covered_col in range(col, col + col_span):
                    if (covered_row, covered_col) in occupied:
                        raise LayoutError(
                            "Dense elements contain overlapping SmartFigure spans."
                        )
                    occupied.add((covered_row, covered_col))
            self._children[span] = child

        self._children = self._ordered_children()
        self._sync_auto_child_projections()

    def show(
        self,
        fullscreen: bool = False,
    ) -> Self:
        """
        Plots and displays the :class:`~graphinglib.SmartFigure`. The :meth:`~graphinglib.SmartFigure.save` method is
        recommended to see properly what the figure looks like, as the display may not show the full figure or the
        appropriate spacings in some cases.

        .. warning::
            If the SmartFigure contains a general legend and the legend location is set to an "outside" position, it may
            not be displayed correctly in matplotlib windows. Inline figures in a Jupyter notebook or saving the figure
            to a file using the :meth:`~graphinglib.SmartFigure.save` method are recommended to get the figure properly
            displayed.

        Parameters
        ----------
        fullscreen : bool, optional
            If True, the figure will be displayed in fullscreen mode.
            Defaults to ``False``.

        Returns
        -------
        Self
            The same SmartFigure instance, allowing for method chaining.
        """
        self._initialize_parent_smart_figure()
        assert self._figure is not None and self._gridspec is not None

        # Create an artificial axis to add padding around the figure
        # This is needed because the figure is created with h_pad=0 and w_pad=0 creating 0 padding
        ax_dummy = self._figure.add_subplot(self._gridspec[:, :], frameon=False)
        ax_dummy.grid(False)
        ax_dummy.set_facecolor((0, 0, 0, 0))
        ax_dummy.set_zorder(-1)
        ax_dummy.set_navigate(False)
        ax_dummy.tick_params(
            colors=(0, 0, 0, 0),
            axis="both",
            direction="in",
            labelright=True,
            labeltop=True,
            labelsize=0.01,
        )
        ax_dummy.set_xticks([0.5])
        ax_dummy.set_yticks([0.5])
        ax_dummy.set_xticklabels(["."])
        ax_dummy.set_yticklabels(["."])

        if all(
            [
                self._show_legend,
                self._general_legend,
                self._legend_loc is not None and "outside" in self._legend_loc,
            ]
        ):
            warning(
                "The general legend location is set to an 'outside' position and matplotlib windows may not be "
                "able to show it properly. Consider using inline figures in a jupyter notebook or saving the "
                "figure to a file instead to get the full figure."
            )

        if fullscreen:
            fig_manager = plt.get_current_fig_manager()
            assert fig_manager is not None
            fig_manager.full_screen_toggle()

        plt.show()
        if not any(
            plt.fignum_exists(num) for num in plt.get_fignums()
        ):  # check if the parameters can be reset
            plt.rcParams.update(plt.rcParamsDefault)
            self._figure = None
            self._gridspec = None
        return self

    def save(
        self,
        file_name: str | PdfPages,
        dpi: int | None = None,
        transparent: bool = False,
        split_pdf: bool = False,
    ) -> Self:
        """
        Saves the :class:`~graphinglib.SmartFigure` to a file.

        Parameters
        ----------
        file_name : str | PdfPages
            The name of the file or the :class:`~matplotlib.backends.backend_pdf.PdfPages` object to save the figure to.
            The file extension determines the format (e.g., .png, .pdf). If a
            :class:`~matplotlib.backends.backend_pdf.PdfPages` object is provided, the figure will be saved to that PDF
            file instead. Note that the provided PdfPages object must be opened by the user, preferably using a context
            manager (``with PdfPages(...) as pdf:``).
        dpi : int, optional
            The resolution in dots per inch. If None, the figure's DPI is used.
        transparent : bool, optional
            Whether to save the figure with a transparent background. A file extension that supports transparency (e.g.
            .png) should be used.
            Defaults to ``False``.
        split_pdf : bool, optional
            Whether to save each subplot of the SmartFigure as a separate page in a PDF file.
            Defaults to ``False``.

        Returns
        -------
        Self
            The same SmartFigure instance, allowing for method chaining.
        """

        def recursive_save(pdf_file: PdfPages) -> None:
            """Recursively saves each element of a SmartFigure to a separate page in the provided PdfPages object."""
            for element in self._ordered_elements.values():
                if isinstance(element, (Plottable, list)):
                    leaf_elements = element if isinstance(element, list) else [element]
                    subfig = self.copy_with(
                        elements=leaf_elements,
                        num_rows=1,
                        num_cols=1,
                    )
                elif isinstance(element, SmartFigure):
                    subfig = element
                subfig.save(pdf_file, dpi, transparent)

        self._initialize_parent_smart_figure()

        if not isinstance(file_name, PdfPages) and split_pdf:
            if not file_name.endswith(".pdf"):
                dot_pos = file_name.rfind(".")
                if dot_pos == -1:  # no extension
                    file_name += ".pdf"
                else:  # wrong extension
                    file_name = f"{file_name[:dot_pos]}.pdf"
                warning(
                    "File extension was changed to '.pdf' to allow for splitting the figure into PdfPages."
                )

        if split_pdf:
            if isinstance(file_name, PdfPages):
                recursive_save(file_name)
            else:
                pdf_file = PdfPages(file_name)
                with pdf_file as pdf:
                    recursive_save(pdf)

        else:
            save_kwargs = {
                "bbox_inches": "tight",
                "dpi": dpi if dpi is not None else "figure",
                "transparent": transparent,
            }
            if isinstance(file_name, PdfPages):
                file_name.savefig(self._figure, **save_kwargs)
            else:
                plt.savefig(file_name, **save_kwargs)

        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)
        self._figure = None
        self._gridspec = None
        return self

    def show_animation(
        self,
        update_function: Callable[[int], Any],
        number_of_frames: int,
        interval: int = 100,
    ) -> Self:
        """
        Animates and displays the :class:`~graphinglib.SmartFigure`.

        Parameters
        ----------
        update_function : callable
            Function that sets plot data for a zero-based frame index. Playback can request an index again.
        number_of_frames : int
            The number of frames in the animation.
        interval : int
            The time between each frame in milliseconds.
            Defaults to ``100``.

        Returns
        -------
        Self
            The same SmartFigure instance, allowing for method chaining.
        """
        self._validate_animation_parameters(update_function, number_of_frames, interval)
        with plt.rc_context():
            figure, animation, session = self._create_animation(
                update_function, number_of_frames, interval, interactive=True
            )

            def on_close(event: Any) -> None:
                figure.canvas.mpl_disconnect(close_id)
                self._finish_animation(figure, animation, session, close=False)

            close_id = figure.canvas.mpl_connect("close_event", on_close)
            try:
                figure.canvas.draw()
                plt.show()
            except BaseException:
                self._finish_animation(figure, animation, session)
                raise
            finally:
                if not plt.fignum_exists(figure.number):
                    figure.canvas.mpl_disconnect(close_id)
                    self._finish_animation(figure, animation, session, close=False)
        return self

    def save_animation(
        self,
        file_name: str | Path,
        update_function: Callable[[int], Any],
        number_of_frames: int,
        interval: int = 100,
        dpi: int | None = None,
        fps: float | None = None,
        writer: str | None = None,
        transparent: bool = False,
    ) -> Self:
        """
        Animates the :class:`~graphinglib.SmartFigure` and saves it to a file.

        Parameters
        ----------
        file_name : str | pathlib.Path
            The output file path ending in ``.gif`` or ``.mp4``.
        update_function : callable
            Function that sets plot data for a zero-based frame index.
        number_of_frames : int
            The number of frames in the animation.
        interval : int
            The time between each frame in milliseconds.
            Defaults to ``100``.
        dpi : int, optional
            Resolution used when rendering frames. If None, the figure DPI is used.
        fps : float, optional
            Frames per second for the saved animation. If None, this is inferred from ``interval``.
        writer : str, optional
            The writer matching the output format, ``"pillow"`` for GIF or ``"ffmpeg"`` for MP4.
            The writer is inferred from the extension when omitted.
        transparent : bool, optional
            Whether to save with transparent background.
            Defaults to ``False``.

        Returns
        -------
        Self
            The same SmartFigure instance, allowing for method chaining.
        """
        self._validate_animation_parameters(update_function, number_of_frames, interval)
        save_fps: float | Fraction = fps if fps is not None else 1000 / interval
        self._validate_animation_number("fps", save_fps)
        if dpi is not None:
            self._validate_animation_number("dpi", dpi)
        destination = Path(file_name)
        default_writer = {".gif": "pillow", ".mp4": "ffmpeg"}.get(
            destination.suffix.lower()
        )
        if default_writer is None:
            raise InvalidParameterError(
                "Animation output must have a .gif or .mp4 extension."
            )
        if writer is None:
            writer = default_writer
        if not isinstance(writer, str):
            raise InvalidParameterTypeError(
                "writer must be the name of a Matplotlib animation writer."
            )
        if not writers.is_available(writer):
            raise InvalidOperationError(
                f"The Matplotlib animation writer {writer!r} is unavailable."
            )
        if writer != default_writer:
            raise InvalidParameterError(
                "Use the Pillow writer for GIF output or the FFmpeg writer for MP4 output."
            )
        if writer == "pillow" and fps is None:
            # PillowWriter truncates 1000 / fps to milliseconds. Preserve exact GIF delays
            # such as 120 ms when the inferred frame rate is a repeating decimal.
            save_fps = Fraction(1000) / Fraction(float(interval))

        with plt.rc_context():
            figure, animation, session = self._create_animation(
                update_function, number_of_frames, interval
            )
            temporary_path = None
            try:
                with NamedTemporaryFile(
                    dir=destination.parent, suffix=destination.suffix, delete=False
                ) as output:
                    temporary_path = Path(output.name)
                try:
                    animation.save(
                        temporary_path,
                        writer=writer,
                        fps=save_fps,  # ty: ignore[invalid-argument-type]  # Writers accept fractional rates.
                        dpi=dpi,
                        savefig_kwargs={"transparent": transparent},
                    )
                except BaseException as writer_error:
                    # Pillow can fail while finishing an empty file and mask frame zero's error.
                    if (
                        session.failure is not None
                        and session.failure is not writer_error
                    ):
                        raise session.failure from None
                    raise
                temporary_path.replace(destination)
            finally:
                try:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
                finally:
                    self._finish_animation(figure, animation, session)
        return self

    def _create_animation(
        self,
        update_function: Callable[[int], Any],
        number_of_frames: int,
        interval: int,
        *,
        interactive: bool = False,
    ) -> tuple[Figure, FuncAnimation, _AnimationSession]:
        if self._animation is not None:
            raise InvalidOperationError(
                "Close the current animation before starting another one."
            )

        _reject_live_update(
            self, "Finish the active animation before rendering this figure again."
        )
        session = _AnimationSession()
        session.mark_present(self)
        token = _DRAW_SESSION.set(session)
        try:
            self._initialize_parent_smart_figure()
        except BaseException:
            session.clear()
            raise
        finally:
            _DRAW_SESSION.reset(token)
        figure = self._figure
        assert isinstance(figure, Figure)

        def draw_frame(index: int) -> None:
            try:
                with plt.rc_context(rc=frame_style):
                    session.draw_frame(update_function, index)
            except BaseException:
                if interactive:
                    self._finish_animation(figure, animation, session)
                raise

        try:
            figure.set_layout_engine("none")
            session.apply_initial_visibility()
            for axes in figure.axes:
                axes.set_autoscale_on(False)
            frame_style = plt.rcParams.copy()
            animation = FuncAnimation(
                figure,
                draw_frame,
                frames=number_of_frames,
                interval=interval,
                init_func=lambda: None,
                blit=False,
                cache_frame_data=False,
            )
        except BaseException:
            session.clear()
            plt.close(figure)
            self._figure = None
            self._gridspec = None
            raise
        self._animation = animation
        return figure, animation, session

    def _finish_animation(
        self,
        figure: Figure,
        animation: FuncAnimation,
        session: _AnimationSession,
        *,
        close: bool = True,
    ) -> None:
        if self._animation is not animation:
            return
        if animation.event_source is not None:
            animation.event_source.stop()
        session.clear()
        self._animation = None
        if self._figure is figure:
            self._figure = None
            self._gridspec = None
        if close:
            plt.close(figure)

    @classmethod
    def _validate_animation_parameters(
        cls, update_function: Callable[[int], Any], number_of_frames: int, interval: int
    ) -> None:
        cls._validate_animation_update_function(update_function)
        if isinstance(number_of_frames, bool) or not isinstance(
            number_of_frames, Integral
        ):
            raise InvalidParameterTypeError("number_of_frames must be an integer.")
        if number_of_frames <= 0:
            raise InvalidParameterError("number_of_frames must be positive.")
        cls._validate_animation_number("interval", interval)

    @staticmethod
    def _validate_animation_number(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidParameterTypeError(f"{name} must be a real number.")
        if value <= 0 or not isfinite(value):
            raise InvalidParameterError(f"{name} must be finite and positive.")

    @staticmethod
    def _validate_animation_update_function(
        update_function: Callable[[int], Any],
    ) -> None:
        if not callable(update_function) or iscoroutinefunction(update_function):
            raise InvalidParameterTypeError(
                "The update function must be a synchronous callable."
            )

        try:
            signature(update_function).bind(0)
        except (TypeError, ValueError) as exc:
            raise InvalidOperationError(
                "The update function must take an integer argument representing the frame number."
            ) from exc

    def _initialize_parent_smart_figure(
        self,
    ) -> None:
        """
        Initializes the parent :class:`~graphinglib.SmartFigure` for plotting. This method initializes the appropriate
        figure style, parameters and matplotlib figure and calls the :meth:`~graphinglib.SmartFigure._prepare_figure`
        method.
        """
        _reject_live_update(
            self, "Finish the animation before rendering this figure again."
        )
        if self._animation is not None:
            raise InvalidOperationError(
                "Close the current animation before rendering this figure again."
            )
        if is_inherit(self._figure_style):
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
                    plt.style.use(resolved(self._figure_style))
                file_loader = FileLoader("plain")
                self._default_params = file_loader.load()
            except OSError:
                raise StyleNotFoundError(
                    f"The figure style {self._figure_style} was not found. Please choose a different style."
                )

        parent_figure_params_to_reset = self._fill_in_missing_params(
            self
        )  # Fill "default" parameters
        animation_grid_defaults = None
        if _DRAW_SESSION.get() is not None:
            style = (
                plt.rcParams
                if is_matplotlib_style
                else self._default_params["rc_params"]
            )
            animation_grid_defaults = {
                key: style.get(key, plt.rcParams[key]) for key in _GRID_RC
            }

        self._default_params["rc_params"].update(
            self._user_rc_dict
        )  # Custom rc parameters supersede the defaults
        self._fill_in_rc_params(is_matplotlib_style)

        figure = None
        try:
            figure = plt.figure(constrained_layout=True, figsize=resolved(self._size))
            self._figure = figure
            layout_engine = figure.get_layout_engine()
            assert isinstance(layout_engine, ConstrainedLayoutEngine)
            if _DRAW_SESSION.get() is None:
                layout_engine.set(w_pad=0, h_pad=0)
            self._reference_label_i = self._reference_labels_params.get(
                "start_index", 0
            )
            if animation_grid_defaults is None:
                self._prepare_figure(is_matplotlib_style)
            else:
                self._prepare_figure(
                    is_matplotlib_style, animation_grid_defaults=animation_grid_defaults
                )
            figure.canvas.draw()
            self._align_shared_x_spines()
        except BaseException:
            if figure is not None:
                plt.close(figure)
                self._figure = None
                self._gridspec = None
            raise
        finally:
            self._reset_params_to_default(self, parent_figure_params_to_reset)
            self._default_params = {}

    def _prepare_figure(
        self,
        is_matplotlib_style: bool = False,
        make_legend: bool = True,
        *,
        animation_parent: tuple[SmartFigure, int] | None = None,
        animation_grid_defaults: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, list[str | Any]]]:
        """
        Prepares the figure for plotting. This method sets up the figure, axes, and any other necessary elements
        before plotting the elements. It also handles the creation of legends and reference labels. If nested
        SmartFigures are present, they are prepared by calling this method recursively.

        Parameters
        ----------
        is_matplotlib_style : bool, optional
            Whether the figure style is a matplotlib style, which allows the use of the plt.style.use function. This
            argument is passed to the :meth:`~graphinglib.SmartFigure._fill_in_rc_params` method, and determines if
            missing plottable parameters should be filled in.
            Defaults to ``False``.
        make_legend : bool, optional
            Whether to create a legend for the figure. This parameter is set to ``False`` when the parent SmartFigure
            is generating a general legend for all subfigures, and this tells the nested SmartFigures to not create
            their own legends. However, if nested SmartFigures have ``general_legend=True``, they will create their own
            legends regardless of this parameter.
            Defaults to ``True``.

        Returns
        -------
        dict[str, dict[str, list[str | Any]]]
            A dictionary containing the legend information for the figure. The keys are "default" and "custom", and
            the values are dictionaries with the "labels" and "handles" keys, which give the list of each type of
            element. The "default" elements are the ones created by the Plottable elements' labels, while the "custom"
            elements are the ones created by the user using the
            :meth:`~graphinglib.SmartFigure.set_custom_legend` method. This is used to create a general legend for the
            entire SmartFigure and keeping trach of the default and custom elements to use the
            :attr:`~graphinglib.SmartFigure.hide_default_legend_elements` and
            :attr:`~graphinglib.SmartFigure.hide_custom_legend_elements` properties.
        """
        # Verify that all legend properties are single values when a general legend is requested
        if self._general_legend:
            legend_params = {
                "show_legend": self._show_legend,
                "legend_cols": self._legend_cols,
                "legend_loc": self._legend_loc,
                "hide_default_legend_elements": self._hide_default_legend_elements,
            }
            for param_name, param_value in legend_params.items():
                if isinstance(param_value, list):
                    raise IncompatibleArgumentsError(
                        f"When using a general legend, the '{param_name}' property must be a single value, not a list."
                    )

        # Get the normalized list or item parameter dict
        self._fill_per_subplot_params()

        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        num_cycle_colors = len(cycle_colors)
        subtitles_pad = self._subplot_p["subtitles_pad"]

        figure = self._figure
        assert figure is not None
        gridspec = figure.add_gridspec(
            self._num_rows,
            self._num_cols,
            wspace=self._width_padding,
            hspace=self._height_padding,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
        )
        self._gridspec = gridspec

        if self._global_reference_label:
            self._create_reference_label(figure)
            figure.suptitle(" ")  # Create a blank title to reserve space

        ax = None  # keep track of the last plt.Axes object, needed for sharing axes
        default_labels, default_handles = [], []
        custom_labels, custom_handles = [], []

        session = _DRAW_SESSION.get()
        if session is not None:
            session.bind_property(self, "figure.facecolor", figure.patch, "facecolor")
            for name in (*_LIVE_VISUAL_RC[1:], "_hidden_spines"):
                # Layout-only figures may have no direct axes for these settings.
                session.bind_update(self, name, lambda _: _combine_edits([]))
            for child in self._children.values():
                session.mark_present(child)

        limit_axes: list[tuple[int, Axes]] = []
        legend_targets: dict[int, Legend | None] = {}
        visibility: dict[str, dict[int, Callable[[bool], _ArtistEdit]]] = {
            name: {} for name in ("remove_axes", "remove_x_ticks", "remove_y_ticks")
        }
        grid_defaults = self._grid_defaults()
        grid_options = self._grid_options(grid_defaults)
        # Plottable and subfigure plotting
        for subplot_i, ((rows, cols), element) in enumerate(
            self._ordered_elements.items()
        ):
            if isinstance(element, SmartFigure):
                element._default_params = deepcopy(self._default_params)
                subfig_params_to_reset = []
                parent_rc_params = None
                local_grid_defaults = (
                    {key: plt.rcParams[key] for key in _GRID_RC}
                    if session is not None
                    else None
                )
                if is_matplotlib_style:
                    parent_rc_params = plt.rcParams.copy()
                    plt.rcParams.update(element._user_rc_dict)
                else:
                    element._default_params["rc_params"].update(element._user_rc_dict)
                    plt.rcParams.update(element._default_params["rc_params"])
                    subfig_params_to_reset = element._fill_in_missing_params(
                        element
                    )  # Fill "default" parameters

                # Check whether sub_x_labels/sub_y_labels/sub_titles are set and can be given as the main
                # x_label/y_label/title of the nested SmartFigure
                sub_params = [
                    self._subplot_p[sub_param][subplot_i]
                    for sub_param in ["sub_x_labels", "sub_y_labels", "subtitles"]
                ]  # list containing the sub_x_label, sub_y_label and subtitle for the current subplot
                # subfig_none_params contains True if the corresponding parameter is None in the nested SmartFigure
                subfig_none_params = [
                    getattr(element, param) is None
                    for param in ["x_label", "y_label", "title"]
                ]
                for attr, param_is_none, sub_param in zip(
                    ["x_label", "y_label", "title"], subfig_none_params, sub_params
                ):
                    if param_is_none and sub_param is not None:
                        setattr(element, attr, sub_param)

                try:
                    subfig = figure.add_subfigure(gridspec[rows, cols])
                    if session is not None:

                        def release_child(child=element, owned=subfig) -> None:
                            if child._figure is owned:
                                child._figure = None
                                child._gridspec = None

                        session.on_finish(release_child)
                    element._figure = subfig  # associates the current subfigure with the nested SmartFigure
                    element._reference_label_i = self._reference_label_i
                    legend_info = element._prepare_figure(
                        is_matplotlib_style=is_matplotlib_style,
                        make_legend=(not self._general_legend and make_legend),
                        **(
                            {
                                "animation_parent": (self, subplot_i),
                                "animation_grid_defaults": local_grid_defaults,
                            }
                            if session is not None
                            else {}
                        ),
                    )

                    self._reference_label_i = element._reference_label_i
                    default_labels += legend_info["labels"]["default"]
                    default_handles += legend_info["handles"]["default"]
                    custom_labels += legend_info["labels"]["custom"]
                    custom_handles += legend_info["handles"]["custom"]
                finally:
                    if is_matplotlib_style:
                        plt.rcParams.update(cast(Any, parent_rc_params))
                    else:
                        plt.rcParams.update(
                            self._default_params["rc_params"]
                        )  # Return to the parent SmartFigure's rc params
                        element._reset_params_to_default(
                            element, subfig_params_to_reset
                        )
                    element._default_params = {}
                    for param, param_was_none in zip(
                        ["x_label", "y_label", "title"], subfig_none_params
                    ):
                        if param_was_none:
                            setattr(element, param, None)

            elif isinstance(element, (Plottable, list)):
                current_elements = element if isinstance(element, list) else [element]
                subfig = figure.add_subfigure(gridspec[rows, cols])
                ax = subfig.add_subplot(
                    sharex=ax
                    if self._share_x
                    else None,  # This enables the coherent zoom and pan of the axes
                    sharey=ax
                    if self._share_y
                    else None,  # but it does not remove the ticklabels
                    projection=self._subplot_p["projection"][subplot_i],
                )

                if session is not None:
                    limit_axes.append((subplot_i, ax))

                # Plotting loop
                z_order = 2
                for index, current_element in enumerate(current_elements):
                    if current_element is not None:
                        params_to_reset = []
                        if not is_matplotlib_style:
                            params_to_reset = self._fill_in_missing_params(
                                current_element
                            )
                        try:
                            current_element._plot_element(
                                ax,
                                z_order,
                                cycle_color=cycle_colors[index % num_cycle_colors],
                            )
                        finally:
                            self._reset_params_to_default(
                                current_element, params_to_reset
                            )
                        try:
                            if current_element.label is not None:
                                default_handles.append(current_element.handle)
                                default_labels.append(current_element.label)
                        except AttributeError:
                            continue
                        z_order += 5

                # Add reference label
                if self._subplot_p["reference_labels"][subplot_i] and (
                    (1 if self._mode == "leaf" else len(self)) > 1
                    or isinstance(self._figure, SubFigure)
                ):
                    self._create_reference_label(ax, subplot_i)

                # Axes limits
                if self._subplot_p["x_lim"][subplot_i]:
                    ax.set_xlim(*self._subplot_p["x_lim"][subplot_i])
                if self._subplot_p["y_lim"][subplot_i]:
                    ax.set_ylim(*self._subplot_p["y_lim"][subplot_i])

                # Logarithmic scale
                if self._subplot_p["log_scale_x"][subplot_i]:
                    ax.set_xscale("log")
                if self._subplot_p["log_scale_y"][subplot_i]:
                    ax.set_yscale("log")

                if self._subplot_p["remove_axes"][subplot_i]:
                    ax.set_axis_off()
                if session is not None:
                    visibility["remove_axes"][subplot_i] = partial(_axis_removal, ax)

                ax.set_aspect(self._subplot_p["aspect_ratio"][subplot_i])
                ax.set_box_aspect(self._subplot_p["box_aspect_ratio"][subplot_i])
                ax.set_axisbelow(
                    False
                )  # ensure grid and ticks are above other elements

                # Invert axes
                # When axes are shared, check if already inverted to avoid double-toggling
                if (
                    self._subplot_p["invert_x_axis"][subplot_i]
                    and not ax.xaxis_inverted()
                ):
                    ax.invert_xaxis()
                if (
                    self._subplot_p["invert_y_axis"][subplot_i]
                    and not ax.yaxis_inverted()
                ):
                    ax.invert_yaxis()

                self._customize_ticks(ax, subplot_i)

                # If axes are shared, manually remove ticklabels from unnecessary plots as it is not done automatically
                # when adding subplots
                if self._share_x:
                    if rows.start != 0:
                        ax.tick_params(axis="x", labeltop=False)
                    if rows.stop != self._num_rows:
                        ax.tick_params(axis="x", labelbottom=False)
                if self._share_y:
                    if cols.start != 0:
                        ax.tick_params(axis="y", labelleft=False)
                    if cols.stop != self._num_cols:
                        ax.tick_params(axis="y", labelright=False)

                for dimension in ("x", "y"):
                    name = f"remove_{dimension}_ticks"
                    hidden = self._subplot_p[name][subplot_i]
                    if session is not None or hidden:
                        prepare = self._prepare_tick_removal(ax, dimension)
                        if hidden:
                            prepare(True).apply()
                        if session is not None:
                            visibility[name][subplot_i] = prepare

                show_grid = self._subplot_p["show_grid"][subplot_i]
                if show_grid is not None:
                    self._prepare_grid(ax, show_grid, grid_options).apply()

                # Axes subtitles
                if self._subtitles is not None:
                    pad = (
                        subtitles_pad[subplot_i] if subtitles_pad is not None else None
                    )
                    ax.set_title(self._subplot_p["subtitles"][subplot_i], pad=pad)

                # Axes sub_labels
                self._customize_ax_label(ax, subplot_i)
                if session is not None and not self.is_single_subplot:
                    self._bind_subplot_animation_text(session, ax, subplot_i)

                # Hidden spines
                if session is not None:
                    self._bind_animation_spines(session, ax)
                if self._hidden_spines is not None:
                    for name in self._hidden_spines:
                        if name in ax.spines:
                            ax.spines[name].set_visible(False)

                # Twin axes
                for i, twin_axis in enumerate(
                    [self._twin_x_axis, self._twin_y_axis], start=1
                ):
                    if twin_axis is not None:
                        twin_axis._default_params = deepcopy(self._default_params)
                        twin_axis_params_to_reset = []
                        parent_rc_params = None
                        if is_matplotlib_style:
                            parent_rc_params = plt.rcParams.copy()
                            plt.rcParams.update(cast(Any, twin_axis._user_rc_dict))
                        else:
                            twin_axis._default_params["rc_params"].update(
                                twin_axis._user_rc_dict
                            )
                            plt.rcParams.update(twin_axis._default_params["rc_params"])
                            twin_axis_params_to_reset = (
                                twin_axis._fill_in_missing_params(
                                    twin_axis, self._figure_style
                                )
                            )

                        twin_labels, twin_handles = twin_axis._prepare_twin_axis(
                            fig_axes=ax,
                            is_matplotlib_style=is_matplotlib_style,
                            cycle_colors=cycle_colors,
                            is_y=(i == 2),
                            z_order=200
                            * i,  # increment z_order to avoid overlap with the main axes
                            figure_style=self._figure_style,
                        )
                        default_labels.extend(twin_labels)
                        default_handles.extend(twin_handles)

                        if is_matplotlib_style:
                            plt.rcParams.update(cast(Any, parent_rc_params))
                        else:
                            plt.rcParams.update(
                                self._default_params["rc_params"]
                            )  # Return to the original rc params
                            twin_axis._reset_params_to_default(
                                twin_axis, twin_axis_params_to_reset
                            )
                        twin_axis._default_params = {}

                # Axes legend
                if self._subplot_p["hide_default_legend_elements"][subplot_i]:
                    default_labels = []
                    default_handles = []
                if not self._general_legend and make_legend:
                    if self.is_single_subplot:
                        custom_labels += self._custom_legend_labels
                        custom_handles += self._custom_legend_handles
                    if self._hide_custom_legend_elements or not self.is_single_subplot:
                        custom_labels = []
                        custom_handles = []
                    labels = default_labels + custom_labels
                    handles = default_handles + custom_handles

                    show_legend = self._subplot_p["show_legend"][subplot_i] is not False
                    if session is not None:
                        legend_targets[subplot_i] = None
                    if labels and (show_legend or session is not None):
                        legend_params = self._get_legend_params(
                            labels, handles, -0.1, subplot_i
                        )
                        # Set legend_ax to the uppermost drawn axis to avoid overlapping with any elements
                        if self._twin_y_axis is not None:
                            legend_ax = self._twin_y_axis._axes
                        elif self._twin_x_axis is not None:
                            legend_ax = self._twin_x_axis._axes
                        else:
                            legend_ax = ax
                        assert legend_ax is not None
                        try:
                            _legend = legend_ax.legend(
                                draggable=True,
                                **legend_params,
                            )
                        except Exception:
                            _legend = legend_ax.legend(
                                **legend_params,
                            )
                        _legend.set_zorder(10000)
                        if session is not None:
                            legend_targets[subplot_i] = _legend
                            session.bind_legend(
                                _legend, handles, initially_visible=show_legend
                            )
                    default_labels, default_handles = [], []
                    custom_labels, custom_handles = [], []

            elif element is not None:
                raise InvalidParameterTypeError(
                    f"Unsupported element type in list: {type(element).__name__}."
                )

        if session is not None:
            subplot_count = len(self._ordered_elements)
            if legend_targets:
                session.bind_legend_visibility(self, legend_targets, subplot_count)
            for name, targets in visibility.items():
                session.bind_axis_visibility(self, f"_{name}", targets, subplot_count)
            if limit_axes:
                self._bind_animation_grid(
                    session,
                    limit_axes,
                    subplot_count,
                    grid_defaults,
                    animation_grid_defaults,
                )
            for dimension in ("x", "y"):
                session.bind_axis_view(
                    self,
                    f"_{dimension}_lim",
                    f"_invert_{dimension}_axis",
                    f"_log_scale_{dimension}",
                    dimension,
                    [
                        (
                            index,
                            axes,
                            bool(self._subplot_p[f"invert_{dimension}_axis"][index]),
                        )
                        for index, axes in limit_axes
                    ],
                    subplot_count,
                )

        # Set a general axis for adding general labels/title and controlling padding
        general_ax = figure.add_subplot(gridspec[:, :], frameon=False)
        general_ax.grid(False)
        general_ax.set_facecolor((0, 0, 0, 0))
        general_ax.set_zorder(-1)
        general_ax.set_navigate(False)
        general_ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            bottom=False,
            top=False,
            left=False,
            right=False,
        )

        # General labels and title
        label_axes = ax if self.is_single_subplot else general_ax
        if label_axes is not None:
            self._customize_ax_label(label_axes)
        title_axes = label_axes if label_axes is not None else general_ax
        title_pad = self._pad_params.get("title_pad")
        if session is not None:
            self._bind_subplot_animation_text(
                session,
                title_axes,
                limit_axes[-1][0] if self.is_single_subplot and limit_axes else None,
                general=True,
                parent=animation_parent,
            )
        elif self._title:
            title_axes.set_title(self._title, pad=title_pad)

        # Annotations
        if self._annotations is not None:
            z_order = 5000
            for annotation in self._annotations:
                annotation._plot_element(cast(Any, figure), z_order)
                z_order += 5

        # Legend parameters
        if self._hide_custom_legend_elements:
            custom_labels = []
            custom_handles = []
        else:
            custom_labels += self._custom_legend_labels
            custom_handles += self._custom_legend_handles
        if (
            self._general_legend
        ):  # making a general legend is priorized over make_legend=False
            labels = default_labels + custom_labels
            handles = default_handles + custom_handles
            general_legend: Legend | None = None
            if labels and (self._show_legend or session is not None):
                legend_params = self._get_legend_params(labels, handles, 0)
                if session is not None:
                    location = self._legend_loc
                    if isinstance(location, str) and location.startswith("outside "):
                        # Native outside placement participates in constrained layout.
                        legend_params.pop("bbox_to_anchor", None)
                        legend_params["loc"] = location
                try:
                    _legend = figure.legend(
                        **legend_params,
                        draggable=True,
                    )
                except Exception:
                    _legend = figure.legend(
                        **legend_params,
                    )
                _legend.set_zorder(10000)
                general_legend = _legend
                if session is not None:
                    session.bind_legend(
                        _legend, handles, initially_visible=bool(self._show_legend)
                    )
            if session is not None:
                session.bind_legend_visibility(self, {0: general_legend}, 1)
            legend_info = {
                "labels": {"default": [], "custom": []},
                "handles": {"default": [], "custom": []},
            }
        else:
            legend_info = {
                "labels": {"default": default_labels, "custom": custom_labels},
                "handles": {"default": default_handles, "custom": custom_handles},
            }
        self._subplot_p = {}  # clear the ListOrItem subplot parameters to free memory
        return legend_info

    def _fill_per_subplot_params(self) -> None:
        """
        Fills the _subplot_p dictionary with parameters that can be broadcasted to all subplots in the
        :class:`~graphinglib.SmartFigure`. If a parameter is given as a single value, it is broadcasted to all
        subplots drawn by the SmartFigure. If it is given as a list, its length must not exceed the number of
        subfigures the SmartFigure draws. Shorter lists are padded using the default value for that parameter.
        """
        self_length = 1 if self._mode == "leaf" else len(self)
        blank_figure = (
            SmartFigure()
        )  # create a blank SmartFigure to get the default parameter values
        defaults: dict[str, Any] = {
            "x_lim": blank_figure._x_lim,
            "y_lim": blank_figure._y_lim,
            "log_scale_x": blank_figure._log_scale_x,
            "log_scale_y": blank_figure._log_scale_y,
            "remove_axes": blank_figure._remove_axes,
            "aspect_ratio": blank_figure._aspect_ratio,
            "box_aspect_ratio": blank_figure._box_aspect_ratio,
            "remove_x_ticks": blank_figure._remove_x_ticks,
            "remove_y_ticks": blank_figure._remove_y_ticks,
            "invert_x_axis": blank_figure._invert_x_axis,
            "invert_y_axis": blank_figure._invert_y_axis,
            "reference_labels": blank_figure._reference_labels,
            "reference_labels_loc": blank_figure._reference_labels_loc,
            "projection": blank_figure._projection,
            "legend_loc": blank_figure._legend_loc,
            "legend_cols": blank_figure._legend_cols,
            "show_legend": blank_figure._show_legend,
            "show_grid": blank_figure._show_grid,
            "hide_default_legend_elements": blank_figure._hide_default_legend_elements,
            "sub_x_labels": blank_figure._sub_x_labels,
            "sub_y_labels": blank_figure._sub_y_labels,
            "subtitles": blank_figure._subtitles,
            "sub_x_labels_pad": None,
            "sub_y_labels_pad": None,
            "subtitles_pad": None,
        }

        subplot_p: dict[str, list[Any]] = {}
        for param, default_value in defaults.items():
            if param[-3:] == "pad":
                value = self._pad_params.get(param)
            else:
                value = getattr(self, f"_{param}")
            if isinstance(value, list):
                if len(value) > self_length:
                    raise InvalidParameterError(
                        f"Number of {param} values ({len(value)}) must not exceed the number of subfigures "
                        f"({self_length})."
                    )
                elif len(value) < self_length:
                    # Pad the list with default values to reach the number of subfigures
                    subplot_p[param] = value + [default_value] * (
                        self_length - len(value)
                    )
                else:
                    subplot_p[param] = value
            else:
                subplot_p[param] = [value] * self_length
        self._subplot_p = subplot_p

    def _get_all_axes_recursive(
        self, figure_or_subfigure: Figure | SubFigure
    ) -> list[Axes]:
        """
        Recursively collect all axes from a figure and its subfigures, avoiding duplicates. This is used for aligning
        the spines when sharing axes.
        """
        all_axes = []
        previous_axes = set()

        # Get direct axes from this figure/subfigure
        direct_axes = figure_or_subfigure.get_axes()
        for ax in direct_axes:
            ax_id = id(ax)
            if (
                ax_id not in previous_axes and ax.get_navigate()
            ):  # skip dummy axes which are not navigable
                all_axes.append(ax)
                previous_axes.add(ax_id)

        # Recursively get axes from subfigures if they exist
        if hasattr(figure_or_subfigure, "subfigs"):
            for subfig in figure_or_subfigure.subfigs:
                subfig_axes = self._get_all_axes_recursive(
                    subfig
                )  # only axes in subfigures are extracted
                for ax in subfig_axes:
                    ax_id = id(ax)
                    if ax_id not in previous_axes:
                        all_axes.append(ax)
                        previous_axes.add(ax_id)

        return all_axes

    def _align_shared_x_spines(self) -> None:
        """
        Aligns subplot spines when sharing x axes. This method solves the constrained_layout behavior of misaligning the
        edge of subplots to fill the entire grid space, which leads to misaligned spines even when sharing the x axes.
        """
        for child in self._children.values():
            if not child._flatten_in_parent:
                child._align_shared_x_spines()

        tolerance = 0.3  # allowed difference between axes to consider them to be in the same column
        if self._share_x and self._num_rows > 1:
            try:
                figure = self._figure
                assert figure is not None
                plot_axes = self._get_all_axes_recursive(figure)
                if len(plot_axes) <= 1:
                    return

                # Group axes by column by looking at their center positions with get_position()
                groups = []
                for ax in plot_axes:
                    pos = ax.get_position()
                    center = pos.x0 + pos.width / 2

                    ax_placed = False  # whether the axis has been found close enough to an existing group
                    for group in groups:
                        ref_pos = group[0].get_position()
                        ref_center = ref_pos.x0 + ref_pos.width / 2

                        if abs(center - ref_center) < tolerance:
                            group.append(ax)
                            ax_placed = True
                            break

                    if not ax_placed:
                        groups.append([ax])

                for group in groups:
                    if len(group) <= 1:
                        continue

                    positions = [ax.get_position() for ax in group]
                    rightmost_left_edge = max(pos.x0 for pos in positions)
                    leftmost_right_edge = min(pos.x0 + pos.width for pos in positions)
                    aligned_size = leftmost_right_edge - rightmost_left_edge
                    for ax in group:
                        current_pos = ax.get_position()
                        ax.set_position(
                            (
                                rightmost_left_edge,
                                current_pos.y0,
                                aligned_size,
                                current_pos.height,
                            )
                        )

            except Exception:
                return

    def _customize_ticks(
        self,
        ax: Axes,
        subplot_i: int,
    ) -> None:
        """
        Customizes the ticks of the specified Axes according to the SmartFigure's tick parameters. This method is useful
        for inheritance to allow each SmartFigure class to customize the ticks their way.
        """
        # Handle x-axis ticks
        x_ticks = self._ticks.get("x_ticks")
        if x_ticks is not None:
            x_labels = self._ticks.get("x_tick_labels")
            if callable(x_labels):
                # Apply the callable to each tick
                x_labels = [x_labels(tick) for tick in x_ticks]
            ax.set_xticks(x_ticks, x_labels)

        ax.tick_params(axis="x", which="major", **self._tick_params["x major"])

        x_tick_spacing = self._ticks.get("x_tick_spacing")
        if x_tick_spacing is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
            # If a callable is provided for x_tick_labels, apply it with a FuncFormatter
            x_labels = self._ticks.get("x_tick_labels")
            if callable(x_labels):
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: x_labels(x))
                )

        # Handle y-axis ticks
        y_ticks = self._ticks.get("y_ticks")
        if y_ticks is not None:
            y_labels = self._ticks.get("y_tick_labels")
            if callable(y_labels):
                # Apply the callable to each tick
                y_labels = [y_labels(tick) for tick in y_ticks]
            ax.set_yticks(y_ticks, y_labels)

        ax.tick_params(axis="y", which="major", **self._tick_params["y major"])

        y_tick_spacing = self._ticks.get("y_tick_spacing")
        if y_tick_spacing is not None:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
            # If a callable is provided for y_tick_labels, apply it with a FuncFormatter
            y_labels = self._ticks.get("y_tick_labels")
            if callable(y_labels):
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda y, pos: y_labels(y))
                )

        minor_x_ticks = self._ticks.get("minor_x_ticks")
        if minor_x_ticks is not None:
            ax.set_xticks(minor_x_ticks, minor=True)
        ax.tick_params(axis="x", which="minor", **self._tick_params["x minor"])
        minor_x_tick_spacing = self._ticks.get("minor_x_tick_spacing")
        if minor_x_tick_spacing is not None:
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_x_tick_spacing))

        minor_y_ticks = self._ticks.get("minor_y_ticks")
        if minor_y_ticks is not None:
            ax.set_yticks(minor_y_ticks, minor=True)
        ax.tick_params(axis="y", which="minor", **self._tick_params["y minor"])
        minor_y_tick_spacing = self._ticks.get("minor_y_tick_spacing")
        if minor_y_tick_spacing is not None:
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_y_tick_spacing))

    def _bind_animation_spines(self, session: _AnimationSession, axes: Axes) -> None:
        session.bind_property(self, "axes.facecolor", axes.patch, "facecolor")
        for spine in axes.spines.values():
            session.bind_property(self, "axes.edgecolor", spine, "edgecolor")
            session.bind_property(self, "axes.linewidth", spine, "linewidth")
        named_spines = [
            (spine, name, spine.get_visible())
            for name, spine in axes.spines.items()
            if name in _SPINE_NAMES
        ]

        def prepare(hidden: tuple[str, ...]) -> _ArtistEdit:
            return _combine_edits(
                [
                    _edit(
                        spine.set_visible,
                        spine.get_visible(),
                        initial and name not in hidden,
                    )
                    for spine, name, initial in named_spines
                ]
            )

        session.bind_update(self, "_hidden_spines", prepare)

    @staticmethod
    def _grid_defaults() -> dict[str, tuple[str, bool]]:
        return {
            dimension: (
                plt.rcParams["axes.grid.which"],
                plt.rcParams["axes.grid.axis"] in ("both", dimension),
            )
            for dimension in ("x", "y")
        }

    def _grid_options(
        self, defaults: dict[str, tuple[str, bool]]
    ) -> dict[str, tuple[str, bool]]:
        return {
            dimension: (
                self._grid.get(f"which_{dimension}", which),
                self._grid.get(f"visible_{dimension}", visible),
            )
            for dimension, (which, visible) in defaults.items()
        }

    def _bind_animation_grid(
        self,
        session: _AnimationSession,
        axes: list[tuple[int, Axes]],
        subplot_count: int,
        defaults: dict[str, tuple[str, bool]],
        base_style: dict[str, Any] | None,
    ) -> None:
        targets = dict(axes)
        initial_linestyle = plt.rcParams["grid.linestyle"]
        inherited_style = (
            base_style
            if base_style is not None
            else {key: plt.rcParams[key] for key in _GRID_RC}
        )

        def prepare() -> _ArtistEdit:
            assert self._show_grid is not None
            entries = _subplot_visibility_entries(
                self._show_grid, targets, subplot_count
            )
            options = self._grid_options(defaults)
            style = {
                key: self._user_rc_dict.get(key, inherited_style[key])
                for key in _GRID_RC
            }
            return _combine_edits(
                [
                    self._prepare_grid(
                        targets[index], show, options, style, initial_linestyle
                    )
                    for index, show in entries.items()
                ]
            )

        session.bind_derived(self, ("_show_grid", "_grid"), prepare)

    def _prepare_grid(
        self,
        ax: Axes,
        show: bool,
        options: dict[str, tuple[str, bool]],
        style: dict[str, Any] | None = None,
        initial_linestyle: Any = None,
    ) -> _ArtistEdit:
        edits = []
        for dimension, (selected, enabled) in options.items():
            axis = ax.xaxis if dimension == "x" else ax.yaxis
            for which in ("major", "minor"):
                edits.append(
                    self._prepare_native_grid_group(
                        axis,
                        which,
                        show and enabled and selected in (which, "both"),
                        style,
                        initial_linestyle,
                    )
                )
        return _combine_edits(edits)

    @staticmethod
    def _prepare_native_grid_group(
        axis: Axis,
        which: Literal["major", "minor"],
        visible: bool,
        style: dict[str, Any] | None,
        initial_linestyle: Any,
    ) -> _ArtistEdit:
        params = axis.get_tick_params(which=which)
        old = {"gridOn": params.get("gridOn", False)}
        new: dict[str, Any] = {"gridOn": visible}
        if style is not None:
            get_ticks = (
                axis.get_major_ticks if which == "major" else axis.get_minor_ticks
            )
            gridline = get_ticks(1)[0].gridline
            old.update(
                grid_color=gridline.get_color(),
                grid_alpha=gridline.get_alpha(),
                grid_linestyle=params.get("grid_linestyle", initial_linestyle),
                grid_linewidth=gridline.get_linewidth(),
            )
            new.update({key.replace(".", "_"): value for key, value in style.items()})

        def write(params: dict[str, Any]) -> None:
            axis.set_tick_params(which=which, **params)

        return _edit(write, old, new)

    def _prepare_tick_removal(
        self, ax: Axes, dimension: str
    ) -> Callable[[bool], _ArtistEdit]:
        return _tick_removal(ax.xaxis if dimension == "x" else ax.yaxis, dimension)

    def _customize_ax_label(
        self,
        ax: Axes,
        subplot_i: int | None = None,
    ) -> None:
        """
        Customizes the x and y labels of the specified Axes according to the SmartFigure's label parameters. This method
        is useful for inheritance to allow each SmartFigure class to customize the labels their way.
        """
        if subplot_i is None:
            x_label, x_pad = self._x_label, self._pad_params.get("x_label_pad")
            y_label, y_pad = self._y_label, self._pad_params.get("y_label_pad")
        else:
            x_label, x_pad = [
                val[subplot_i]
                for val in [
                    self._subplot_p["sub_x_labels"],
                    self._subplot_p["sub_x_labels_pad"],
                ]
            ]
            y_label, y_pad = [
                val[subplot_i]
                for val in [
                    self._subplot_p["sub_y_labels"],
                    self._subplot_p["sub_y_labels_pad"],
                ]
            ]

        if x_label is not None:
            ax.set_xlabel(x_label, labelpad=x_pad, color=self._x_axis_label_color)
        if y_label is not None:
            ax.set_ylabel(y_label, labelpad=y_pad, color=self._y_axis_label_color)

    def _bind_subplot_animation_text(
        self,
        session: _AnimationSession,
        ax: Axes,
        index: int | None,
        *,
        general: bool = False,
        parent: tuple[SmartFigure, int] | None = None,
    ) -> None:
        def bind(
            name: str,
            read: Callable[[], str | None],
            write: Callable[[str], Any],
        ) -> None:
            sub_name = _SUBPLOT_TEXT[name]

            def subplot_text(owner: SmartFigure, slot: int) -> str | None:
                values = getattr(owner, sub_name)
                if isinstance(values, list):
                    return values[slot] if slot < len(values) else None
                return values

            def resolve() -> str | None:
                if general:
                    value = getattr(self, name)
                    if value is None and parent is not None:
                        value = subplot_text(*parent)
                    if value is not None and not (
                        name == "_title" and isinstance(value, str) and not value
                    ):
                        return value
                return subplot_text(self, index) if index is not None else None

            sources: list[tuple[object, str]] = [(self, name)] if general else []
            if parent is not None:
                sources.append((parent[0], sub_name))
            if index is not None:
                sources.append((self, sub_name))
            if index is None and parent is None:
                session.bind_axis_text(self, name, read, write)
            else:
                session.bind_resolved_axis_text(sources, resolve, read, write)

        use_wcs = isinstance(self, SmartFigureWCS) and index is not None
        label_axes = (
            ("x", ax.xaxis, self._x_axis_label_color),
            ("y", ax.yaxis, self._y_axis_label_color),
        )
        if general and self.is_single_subplot and index is None:
            label_axes = ()
        for name, axis, color in label_axes:
            artist = axis.label
            read_label, set_label = (
                (ax.get_xlabel, ax.set_xlabel)
                if name == "x"
                else (ax.get_ylabel, ax.set_ylabel)
            )
            pad = self._pad_params.get(f"{name}_label_pad") if general else None
            if index is not None and (
                not general or (pad is None and getattr(self, f"_{name}_label") is None)
            ):
                pad = self._subplot_p[f"sub_{name}_labels_pad"][index]
            if use_wcs:
                bind(
                    f"_{name}_label",
                    read_label,
                    partial(set_label, labelpad=pad, color=color),
                )
            else:
                if not artist.get_text():
                    set_label("", labelpad=pad, color=color)
                bind(f"_{name}_label", artist.get_text, artist.set_text)

        title_pad = self._pad_params.get("title_pad") if general else None
        if index is not None and (
            not general
            or (not self._title and (self._subtitles is not None or title_pad is None))
        ):
            title_pad = self._subplot_p["subtitles_pad"][index]
        title_location = plt.rcParams["axes.titlelocation"]
        title = self._title if general else None
        artist = ax.set_title(
            title or ax.get_title(loc=title_location), pad=title_pad, loc=title_location
        )
        bind("_title", artist.get_text, artist.set_text)

    def _create_reference_label(
        self,
        target: Axes | Figure | SubFigure,
        subplot_i: int | None = None,
    ) -> None:
        """
        Creates a reference label for the specified target (either an Axes, Figure or SubFigure). The label is
        positioned according to the specified location and is incremented for each reference label created.
        """
        if isinstance(target, Axes):
            trans = target.transAxes
        elif isinstance(target, (Figure, SubFigure)):
            trans = target.transSubfigure
        else:
            raise ValueError("Target must be either Axes, Figure or SubFigure.")

        assert self._reference_label_i is not None
        letter = ascii_lowercase[self._reference_label_i]
        formatted_letter = self._reference_labels_params.get(
            "format", lambda le: f"{le})"
        )(letter)
        reflabel_params = strip_inherit(self._reference_labels_params)
        target.text(
            x=0,
            y=1,
            s=formatted_letter,
            transform=trans + self._get_reference_label_translation(target, subplot_i),
            color=reflabel_params.get("color"),
            fontsize=reflabel_params.get("font_size"),
            fontweight=reflabel_params.get("font_weight"),
        )
        self._reference_label_i += 1

    def _get_reference_label_translation(
        self,
        target: Axes | Figure | SubFigure,
        subplot_i: int | None = None,
    ) -> ScaledTranslation:
        """
        Gives the translation to apply to the reference label to position it correctly relative to an Axes, Figure or
        SubFigure. The translation varies depending on the location of the reference label.
        """
        figure = self._figure
        assert figure is not None
        if isinstance(target, Axes):
            assert subplot_i is not None
            reflabel_loc = self._subplot_p["reference_labels_loc"][subplot_i]
            if isinstance(reflabel_loc, tuple):
                x_offset, y_offset = reflabel_loc
                return ScaledTranslation(x_offset, y_offset, figure.dpi_scale_trans)
            elif reflabel_loc == "outside":
                return ScaledTranslation(-5 / 72, 10 / 72, figure.dpi_scale_trans)
            elif reflabel_loc == "inside":
                return ScaledTranslation(10 / 72, -15 / 72, figure.dpi_scale_trans)
            else:
                raise InvalidParameterError(
                    "Invalid reference label location. Please specify either 'inside' or 'outside'."
                )

        elif isinstance(target, (Figure, SubFigure)):
            return ScaledTranslation(7 / 72, -10 / 72, figure.dpi_scale_trans)
        else:
            raise ValueError(
                "Target must be either an Axes, Figure or SubFigure instance."
            )

    def _get_legend_params(
        self,
        labels: list[str],
        handles: list[Any],
        outside_lower_center_y_offset: float,
        subplot_i: int = 0,
    ) -> dict[str, Any]:
        """
        Gives the parameters to use for the legend. The parameters are set according to the specified ``figure_style``
        and the location of the legend.

        Parameters
        ----------
        labels : list[str]
            The labels to use for the legend.
        handles : list[Any]
            The handles to use for the legend.
        outside_lower_center_y_offset : float
            The y offset to use for the legend when the location is set to an "outside" position. This is used to
            position the legend outside of the figure and to make it not overlap the ``x_label`` at the bottom. This
            parameter is useful as the vertical offset for the "outside lower center" location is not the same
            depending on if the legend is created for an Axes or a SubFigure.
        subplot_i : int, optional
            The index of the subplot for which the legend is being created. This is used to access the per-subplot
            legend parameters. If no value is given (e.g. when creating a general legend), the _subplot_p dict should
            contain only the same value and the first subplot's parameters are used for simplicity.
            Defaults to ``0``.

        Returns
        -------
        dict[str, Any]
            The parameters to use for the legend, that may be passed to the
            :meth:`matplotlib.axes.Axes.legend` or :meth:`matplotlib.figure.Figure.legend` methods as keyword arguments.
        """
        legend_params = {
            "handles": handles,
            "labels": labels,
            "handleheight": 1.3,
            "handler_map": {
                Polygon: HandlerPatch(patch_func=histogram_legend_artist),
                LineCollection: HandlerMultipleLines(),
                VerticalLineCollection: HandlerMultipleVerticalLines(),
            },
            "ncols": self._subplot_p["legend_cols"][subplot_i],
        }
        legend_loc = self._subplot_p["legend_loc"][subplot_i]
        if legend_loc is None:
            if self._general_legend:
                legend_params.update({"loc": "lower center"})
            else:
                legend_params.update({"loc": "best"})
        else:
            if "outside" in legend_loc:
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
                legend_params.update(
                    {
                        "loc": outside_keyword[legend_loc],
                        "bbox_to_anchor": outside_coords[legend_loc],
                    }
                )
            else:
                legend_params.update({"loc": legend_loc})
        return legend_params

    def _fill_in_missing_params(self, element: SmartFigure | Plottable) -> list[str]:
        """
        Fills in the missing parameters for a :class:`~graphinglib.SmartFigure` or a :class:`~graphinglib.Plottable`
        from the specified ``figure_style``.
        """
        params_to_reset = []
        # The following logic enables figures that inherit from SmartFigure to use the same default parameters
        object_type = (
            "SmartFigure"
            if isinstance(element, SmartFigure)
            else type(element).__name__
        )
        for try_i in range(2):
            try:
                for property_, value in vars(element).items():
                    if is_inherit(value) and not (property_ == "_figure_style"):
                        params_to_reset.append(property_)
                        default_value = self._default_params[object_type][property_]
                        setattr(element, property_, default_value)
                break
            except KeyError as e:
                if try_i == 1:
                    raise StyleFileError(
                        f"There was an error auto updating your {self._figure_style} style file following the recent "
                        "GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub"
                        " page. In the meantime, you can manually add the following parameter to your "
                        f"{self._figure_style} style file:\n {e.args[0]}."
                    ) from e
                file_updater = FileUpdater(resolved(self._figure_style))
                file_updater.update()
                file_loader = FileLoader(resolved(self._figure_style))
                new_defaults = file_loader.load()
                self._default_params.update(
                    (k, v)
                    for k, v in new_defaults.items()
                    if k not in self._default_params
                )
        return params_to_reset

    def _reset_params_to_default(
        self, element: Plottable | SmartFigure, params_to_reset: list[str]
    ) -> None:
        """
        Resets the parameters that were set to default in the :meth:`~graphinglib.SmartFigure._fill_in_missing_params`
        method.
        """
        for param in params_to_reset:
            setattr(element, param, INHERIT)

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
                plt.style.use(resolved(self._figure_style))
            plt.rcParams.update(cast(Any, self._user_rc_dict))
        else:
            params = self._default_params["rc_params"]
            try:
                if params["text.usetex"] and which("latex") is None:
                    params["text.usetex"] = False
            except KeyError:
                pass
            plt.rcParams.update(params)

    def set_rc_params(
        self,
        rc_params_dict: dict[str, Any] = {},
        reset: bool = False,
    ) -> Self:
        """
        Customize the visual style of the :class:`~graphinglib.SmartFigure`.

        Any rc parameter that is not specified in the dictionary will be set to the default value for the specified
        ``figure_style``.

        Parameters
        ----------
        rc_params_dict : dict[str, Any], optional
            Dictionary of rc parameters to update.
            Defaults to empty dictionary.
        reset : bool, optional
            If ``True``, resets all previously set rc parameters to their default values for the specified
            ``figure_style`` before applying the new parameters.
            Defaults to ``False``.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated rc parameters.
        """
        _reject_animated_layout(self)
        if reset:
            self._user_rc_dict.clear()
        for property_, value in rc_params_dict.items():
            self._user_rc_dict[property_] = value
        return self

    def set_visual_params(
        self,
        reset: bool = False,
        figure_face_color: str | None = None,
        axes_face_color: str | None = None,
        axes_edge_color: str | None = None,
        x_axis_label_color: str | None = None,
        y_axis_label_color: str | None = None,
        axes_label_pad: float | None = None,
        axes_line_width: float | None = None,
        color_cycle: list[str] | None = None,
        legend_face_color: str | None = None,
        legend_edge_color: str | None = None,
        legend_font_size: float | None = None,
        legend_handle_length: float | None = None,
        legend_handle_text_pad: float | None = None,
        font_family: str | None = None,
        font_size: float | None = None,
        font_weight: str | None = None,
        title_font_size: float | None = None,
        title_font_weight: str | None = None,
        text_color: str | None = None,
        use_latex: bool | None = None,
        hidden_spines: Iterable[Literal["right", "left", "top", "bottom"]]
        | None = None,
    ) -> Self:
        """
        Customize the visual style of the :class:`~graphinglib.SmartFigure`.

        Any parameter that is not specified (None) will be set to the default value for the specified ``figure_style``.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set visual parameters to their default values for the specified
            ``figure_style`` before applying the new parameters.
            Defaults to ``False``.
        figure_face_color : str, optional
            The color of the figure face.
        axes_face_color : str, optional
            The color of the axes face.
        axes_edge_color : str, optional
            The color of the axes edge.
        x_axis_label_color : str, optional
            The color of the x axis label.
        y_axis_label_color : str, optional
            The color of the y axis label.
        axes_label_pad : float, optional
            The padding between the axes labels and the axes.
            Typical range is ``2`` to ``12``.
        axes_line_width : float, optional
            The width of the axes lines.
            Typical range is ``0.5`` to ``3``.
        color_cycle : list[str], optional
            A list of colors to use for the color cycle.
        legend_face_color : str, optional
            The color of the legend face.
        legend_edge_color : str, optional
            The color of the legend edge.
        legend_font_size : float, optional
            The font size of the legend.
            Typical range is ``8`` to ``20``.
        legend_handle_length : float, optional
            The length of the legend handles.
            Typical range is ``1`` to ``4``.
        legend_handle_text_pad : float, optional
            The padding between the legend handles and the legend text.
            Typical range is ``0.2`` to ``1.5``.
        font_family : str, optional
            The font family to use.
        font_size : float, optional
            The font size to use.
            Typical range is ``8`` to ``20``.
        font_weight : str, optional
            The font weight to use.
            Values include ``"normal"``, ``"bold"``, ``"light"``, ``"ultralight"``, ``"heavy"``, and
            ``"black"``.
        title_font_size : float, optional
            The font size of the title.
            Typical range is ``10`` to ``24``.
        title_font_weight : str, optional
            The font weight of the title.
            Values include ``"normal"``, ``"bold"``, ``"light"``, ``"ultralight"``, ``"heavy"``, and
            ``"black"``.
        text_color : str, optional
            The color of the text.
        use_latex : bool, optional
            Whether or not to use latex.
        hidden_spines : Iterable[Literal["right", "left", "top", "bottom"]], optional
            The spines to hide. If specified, the corresponding spines will be hidden in the figure. This corresponds to
            the lines that form the borders of the plot.
            Values are ``"right"``, ``"left"``, ``"top"``, and ``"bottom"``.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated visual parameters.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        spines = None
        if hidden_spines is not None:
            if not isinstance(hidden_spines, Iterable):
                raise InvalidParameterTypeError(
                    "hidden_spines must be an iterable of spine names."
                )
            spines = tuple(hidden_spines)
            for spine in spines:
                if spine not in _SPINE_NAMES:
                    raise InvalidParameterError(
                        f"Invalid spine name: {spine}. Must be one of 'right', 'left', 'top' or 'bottom'."
                    )

        prop_cycle = plt.cycler(color=color_cycle) if color_cycle is not None else None

        rc_params_dict = {
            "figure.facecolor": figure_face_color,
            "axes.facecolor": axes_face_color,
            "axes.edgecolor": axes_edge_color,
            "axes.labelpad": axes_label_pad,
            "axes.linewidth": axes_line_width,
            "axes.prop_cycle": prop_cycle,
            "legend.facecolor": legend_face_color,
            "legend.edgecolor": legend_edge_color,
            "legend.fontsize": legend_font_size,
            "legend.handlelength": legend_handle_length,
            "legend.handletextpad": legend_handle_text_pad,
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
        session = _bound_session(self)
        if session is not None:
            params = {
                key: value for key, value in rc_params_dict.items() if value is not None
            }
            if (
                reset
                or x_axis_label_color is not None
                or y_axis_label_color is not None
                or params.keys() - set(_LIVE_VISUAL_RC)
            ):
                raise UnsupportedFeatureError(
                    "During animation set_visual_params supports figure and axes face colors, "
                    "axes edge color and line width, and hidden_spines."
                )
            validated = {
                key: plt.rcParams.validate[key](value) for key, value in params.items()
            }
            original = self._user_rc_dict
            for key, value in validated.items():
                session.queue_property(self, key, value, lambda: None)
            if validated:
                session.remember(
                    self,
                    "_user_rc_dict",
                    lambda: setattr(self, "_user_rc_dict", original),
                )
                self._user_rc_dict = {**original, **validated}
            if spines is not None:
                self._hidden_spines = _stage_live_value(self, "_hidden_spines", spines)
            return self

        if reset:
            for key in rc_params_dict.keys():
                self._user_rc_dict.pop(key, None)

        rc_params_dict = {
            key: value for key, value in rc_params_dict.items() if value is not None
        }
        self.set_rc_params(rc_params_dict)

        if spines is not None:
            self._hidden_spines = spines

        if x_axis_label_color is not None:
            self._x_axis_label_color = x_axis_label_color
        if y_axis_label_color is not None:
            self._y_axis_label_color = y_axis_label_color

        return self

    def set_ticks(
        self,
        reset: bool = False,
        x_ticks: Iterable[float] | None = None,
        y_ticks: Iterable[float] | None = None,
        x_tick_labels: Iterable[str] | Callable | None = None,
        y_tick_labels: Iterable[str] | Callable | None = None,
        x_tick_spacing: float | None = None,
        y_tick_spacing: float | None = None,
        minor_x_ticks: Iterable[float] | None = None,
        minor_y_ticks: Iterable[float] | None = None,
        minor_x_tick_spacing: float | None = None,
        minor_y_tick_spacing: float | None = None,
    ) -> Self:
        """
        Sets custom ticks and tick labels.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set ticks to their default values before applying the new parameters.
            Defaults to ``False``.
        x_ticks, y_ticks : Iterable[float], optional
            Tick positions for the x or y axis. If a value is specified, the corresponding ``x_tick_spacing`` or
            ``y_tick_spacing`` parameter must be ``None``.
        x_tick_labels, y_tick_labels : Iterable[str] | Callable, optional
            Tick labels for the x or y axis. Can be either:

            - An iterable of strings: If a value is specified, the corresponding ``x_ticks`` or ``y_ticks``
              parameter must also be given. The number of tick labels must match the number of ticks.
            - A callable that takes a float (the tick position) and returns a string: Can be used with ``x_ticks`` or
              ``y_ticks`` to apply the function to each tick position, or with ``x_tick_spacing`` or ``y_tick_spacing``
              to apply the function to any tick position using a custom formatter.
        x_tick_spacing, y_tick_spacing : float, optional
            Spacing between major ticks on the x or y axis. If a value is specified, the corresponding ``x_ticks`` or
            ``y_ticks`` parameter must be ``None``. When a callable ``x_tick_labels`` or ``y_tick_labels`` is provided
            with a spacing parameter, the callable will be used to format all tick labels automatically.
        minor_x_ticks, minor_y_ticks : Iterable[float], optional
            Minor tick positions for the x or y axis. If a value is specified, the corresponding
            ``minor_x_tick_spacing`` or ``minor_y_tick_spacing`` parameter must be ``None``.
        minor_x_tick_spacing, minor_y_tick_spacing : float, optional
            Spacing between minor ticks on the x or y axis. If a value is specified, the corresponding ``minor_x_ticks``
            or ``minor_y_ticks`` parameter must be ``None``.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated ticks.
        """
        # Normalize iterable tick/label inputs to lists so their lengths can be compared
        # and the values reused (a bare iterator would be exhausted after the first pass).
        _reject_animated_layout(self)
        if x_ticks is not None:
            x_ticks = list(x_ticks)
        if y_ticks is not None:
            y_ticks = list(y_ticks)
        if minor_x_ticks is not None:
            minor_x_ticks = list(minor_x_ticks)
        if minor_y_ticks is not None:
            minor_y_ticks = list(minor_y_ticks)
        if x_tick_labels is not None and not callable(x_tick_labels):
            x_tick_labels = list(x_tick_labels)
        if y_tick_labels is not None and not callable(y_tick_labels):
            y_tick_labels = list(y_tick_labels)

        # Check if tick labels are provided without ticks or spacing
        x_has_spacing = x_tick_spacing is not None
        y_has_spacing = y_tick_spacing is not None
        x_callable = callable(x_tick_labels)
        y_callable = callable(y_tick_labels)

        if any(
            [
                (x_tick_labels is not None)
                and x_ticks is None
                and not (x_has_spacing and x_callable),
                (y_tick_labels is not None)
                and y_ticks is None
                and not (y_has_spacing and y_callable),
            ]
        ):
            raise IncompatibleArgumentsError(
                "Ticks position must be specified when ticks labels are specified, "
                "unless a callable is provided with tick spacing."
            )

        if any(
            [
                (x_ticks is not None) and (x_tick_spacing is not None),
                (y_ticks is not None) and (y_tick_spacing is not None),
                (minor_x_ticks is not None) and (minor_x_tick_spacing is not None),
                (minor_y_ticks is not None) and (minor_y_tick_spacing is not None),
            ]
        ):
            raise IncompatibleArgumentsError(
                "Tick spacing and tick positions cannot be set simultaneously."
            )

        if (
            x_ticks is not None
            and x_tick_labels is not None
            and not callable(x_tick_labels)
        ):
            if len(x_ticks) != len(x_tick_labels):
                raise IncompatibleArgumentsError(
                    f"Number of x ticks ({len(x_ticks)}) and number of x tick labels "
                    f"({len(x_tick_labels)}) must be the same."
                )
        if (
            y_ticks is not None
            and y_tick_labels is not None
            and not callable(y_tick_labels)
        ):
            if len(y_ticks) != len(y_tick_labels):
                raise IncompatibleArgumentsError(
                    f"Number of y ticks ({len(y_ticks)}) and number of y tick labels "
                    f"({len(y_tick_labels)}) must be the same."
                )

        if reset:
            self._ticks.clear()

        params = [
            "x_ticks",
            "y_ticks",
            "x_tick_labels",
            "y_tick_labels",
            "x_tick_spacing",
            "y_tick_spacing",
            "minor_x_ticks",
            "minor_y_ticks",
            "minor_x_tick_spacing",
            "minor_y_tick_spacing",
        ]
        for param in params:
            value = locals()[param]
            if value is not None:
                self._ticks[param] = value

        return self

    def set_tick_params(
        self,
        axis: Literal["x", "y", "both"] | None = "both",
        which: Literal["major", "minor", "both"] = "major",
        reset: bool = False,
        direction: Literal["in", "out", "inout"] | None = None,
        length: float | None = None,
        width: float | None = None,
        color: str | None = None,
        pad: float | None = None,
        label_size: float | str | None = None,
        label_color: str | None = None,
        label_rotation: float | None = None,
        draw_bottom_ticks: bool | None = None,
        draw_top_ticks: bool | None = None,
        draw_left_ticks: bool | None = None,
        draw_right_ticks: bool | None = None,
        draw_bottom_labels: bool | None = None,
        draw_top_labels: bool | None = None,
        draw_left_labels: bool | None = None,
        draw_right_labels: bool | None = None,
    ) -> Self:
        """
        Sets the tick parameters for the figure. These parameters are given to the
        :meth:`matplotlib.axes.Axes.tick_params` method.

        Parameters
        ----------
        axis : {"x", "y", "both"}, optional
            The axis to set the tick parameters for. This method can be called multiple times to set the tick
            parameters specifically for each axes.
            Defaults to ``"both"``.
        which : {"major", "minor", "both"}, optional
            The ticks to set the parameters for. This method can be called multiple times to set the tick parameters
            specifically for each ticks type.
            Defaults to ``"major"``.
        reset : bool, optional
            If ``True``, all previously given tick parameters for this axis and tick type are reset to their default
            values before applying the new parameters.
            Defaults to ``False``.
        direction : {"in", "out", "inout"}, optional
            The direction of the ticks.
        length : float, optional
            The length of the ticks.
            Typical range is ``2`` to ``10``.
        width : float, optional
            The width of the ticks.
            Typical range is ``0.5`` to ``3``.
        color : str, optional
            The color of the ticks.
        pad : float, optional
            The padding to add between the tick labels and the ticks themselves.
            Typical range is ``2`` to ``10``.
        label_size : float | str, optional
            The font size of the tick labels. This can be a float or a string (e.g. "large").
            Typical range is ``8`` to ``20`` when a float is used.
        label_color : str, optional
            The color of the tick labels.
        label_rotation : float, optional
            The rotation of the tick labels, in degrees.
        draw_bottom_ticks, draw_top_ticks, draw_left_ticks, draw_right_ticks : bool, optional
            Whether to draw the ticks on the bottom, top, left or right side of the axes respectively.
        draw_bottom_labels, draw_top_labels, draw_left_labels, draw_right_labels : bool, optional
            Whether to draw the tick labels on the bottom, top, left or right side of the axes respectively.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated tick parameters.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        _reject_animated_layout(self)
        new_tick_params = {
            "direction": direction,
            "length": length,
            "width": width,
            "color": color,
            "pad": pad,
            "labelsize": label_size,
            "labelcolor": label_color,
            "labelrotation": label_rotation,
            "bottom": draw_bottom_ticks,
            "top": draw_top_ticks,
            "left": draw_left_ticks,
            "right": draw_right_ticks,
            "labelbottom": draw_bottom_labels,
            "labeltop": draw_top_labels,
            "labelleft": draw_left_labels,
            "labelright": draw_right_labels,
        }
        for axis_i in [axis] if axis != "both" else ["x", "y"]:
            for which_i in [which] if which != "both" else ["major", "minor"]:
                if reset:
                    self._tick_params[f"{axis_i} {which_i}"].clear()
                for param, value in new_tick_params.items():
                    if value is not None:
                        self._tick_params[f"{axis_i} {which_i}"][param] = value
        return self

    def set_grid(
        self,
        reset: bool = False,
        visible_x: bool = True,
        visible_y: bool = True,
        which_x: Literal["major", "minor", "both"] = "both",
        which_y: Literal["major", "minor", "both"] = "both",
        color: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
    ) -> Self:
        """
        Sets the grid parameters for the figure.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set grid parameters to their default values before applying the new
            parameters.
            Defaults to ``False``.
        visible_x, visible_y : bool, optional
            If ``True``, sets the x-axis or y-axis grid visible. If ``False``, the grid is not shown for the respective
            axis.
            Defaults to ``True`` for both axes.
        which_x, which_y : Literal["major", "minor", "both"], optional
            Sets whether major, minor or both grid lines are shown for the x-axis and y-axis respectively.
            Defaults to ``"both"`` for both axes.
        color : str, optional
            Sets the color of the grid lines.
            Default depends on the ``figure_style`` configuration.
        alpha : float, optional
            Sets the alpha value for the grid lines.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        line_style : str, optional
            Sets the line style of the grid lines.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        line_width : float, optional
            Sets the line width of the grid lines.
            Typical range is ``0.5`` to ``3``.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated grid parameters.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        if any(
            value not in (None, "major", "minor", "both")
            for value in (which_x, which_y)
        ):
            raise InvalidParameterError(
                "Grid tick groups must be 'major', 'minor' or 'both'."
            )

        grid = {} if reset else self._grid.copy()
        new_options = {
            "visible_x": visible_x,
            "visible_y": visible_y,
            "which_x": which_x,
            "which_y": which_y,
        }
        grid.update(
            (name, value) for name, value in new_options.items() if value is not None
        )
        explicit_style = strip_inherit(
            dict(zip(_GRID_RC, (color, alpha, line_style, line_width), strict=True))
        )
        rc = self._user_rc_dict.copy()
        if reset:
            for key in _GRID_RC:
                rc.pop(key, None)
        rc.update(explicit_style)

        session = _bound_session(self)
        if session is not None:
            for key, value in explicit_style.items():
                rc[key] = plt.rcParams.validate[key](value)
            previous_rc = self._user_rc_dict
            self._grid = _stage_live_value(self, "_grid", grid)
            session.remember(
                self,
                "_user_rc_dict",
                lambda: setattr(self, "_user_rc_dict", previous_rc),
            )
            self._user_rc_dict = rc
            self.show_grid = True
        else:
            self._grid = grid
            self._user_rc_dict = rc
            self._show_grid = True
        return self

    def set_custom_legend(
        self,
        elements: Iterable[LegendElement] | None = None,
        reset: bool = False,
    ) -> Self:
        """
        Sets a custom legend for the figure. If the SmartFigure contains multiple subplots, **custom legends only**
        **work if the ``general_legend`` parameter is set to ``True``**. Otherwise, custom legends can be added for
        non-general legends if the SmartFigure is currently used as a single plot (see the
        :attr:`~graphinglib.SmartFigure.is_single_subplot` property).

        .. note::
            The visibility of default or custom legend elements can be controlled individually with the
            :attr:`~graphinglib.SmartFigure.hide_default_legend_elements` and
            :attr:`~graphinglib.SmartFigure.hide_custom_legend_elements` properties.

        Parameters
        ----------
        elements : Iterable[LegendElement], optional
            Iterable of :class:`~graphinglib.LegendElement` objects to add to the legend.
        reset : bool, optional
            If ``True``, resets all previously set custom handles and labels before adding the new ones.
            Defaults to ``False``.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated custom legend.
        """
        _reject_animated_layout(self)
        if reset:
            self._custom_legend_handles = []
            self._custom_legend_labels = []

        if isinstance(elements, Iterable) and all(
            isinstance(el, LegendElement) for el in elements
        ):
            self._custom_legend_handles += [el.handle for el in elements]
            self._custom_legend_labels += [el.label for el in elements]
        elif elements is not None:
            raise InvalidParameterTypeError(
                "Elements must be an iterable of LegendElement objects."
            )

        return self

    def set_text_padding_params(
        self,
        reset: bool = False,
        x_label_pad: float | None = None,
        y_label_pad: float | None = None,
        title_pad: float | None = None,
        sub_x_labels_pad: Iterable[float] | None = None,
        sub_y_labels_pad: Iterable[float] | None = None,
        subtitles_pad: Iterable[float] | None = None,
    ) -> Self:
        """
        Sets the padding parameters for the figure's text elements. These parameters are used to set the padding between
        the axes and the labels and titles.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set text padding parameters to their default values before applying the
            new parameters.
            Defaults to ``False``.
        x_label_pad, y_label_pad : float, optional
            Padding between the main x-axis or y-axis label and the respective axis.
        title_pad : float, optional
            Padding between the main title and the top of the axes.
        sub_x_labels_pad, sub_y_labels_pad : Iterable[float], optional
            Padding for each subfigure's x-axis and y-axis labels.
        subtitles_pad : Iterable[float], optional
            Padding for each subfigure's subtitle.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated text padding parameters.
        """
        _reject_animated_layout(self)
        for pad_param in [x_label_pad, y_label_pad, title_pad]:
            if pad_param is not None and not isinstance(pad_param, (int, float)):
                raise InvalidParameterTypeError(
                    f"Padding parameters must be of type int or float, got {type(pad_param).__name__}."
                )
        for sub_pad_param in [sub_x_labels_pad, sub_y_labels_pad, subtitles_pad]:
            if sub_pad_param is not None and (
                not isinstance(sub_pad_param, Iterable)
                or not all(
                    isinstance(p, (int, float, type(None))) for p in sub_pad_param
                )
            ):
                raise InvalidParameterTypeError(
                    "Subfigure padding parameters must be an iterable of ints or floats."
                )

        if reset:
            self._pad_params.clear()

        params = [
            "x_label_pad",
            "y_label_pad",
            "title_pad",
            "sub_x_labels_pad",
            "sub_y_labels_pad",
            "subtitles_pad",
        ]
        for param in params:
            value = locals()[param]
            if value is not None:
                self._pad_params[param] = value

        return self

    def set_reference_labels_params(
        self,
        reset: bool = False,
        color: str | Inherit | None = None,
        start_index: int | None = None,
        font_size: float | Inherit | None = None,
        font_weight: str | Inherit | None = None,
        format: Callable | None = None,
    ) -> Self:
        """
        Sets advanced parameters for the reference labels that can be added to the subplots.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set reference label parameters to their default values before applying
            the new parameters.
            Defaults to ``False``.
        color : str | Inherit, optional
            The color of the reference labels. If ``"default"``, the color is set according to the text color of other
            text in the figure.
        start_index : int, optional
            Starting index for the reference labels. This allows to customize the starting label, for example, to start
            labeling from "b)" instead of "a)" by giving ``start_index = 1``.
        font_size : float | Inherit, optional
            The font size of the reference labels.
            Typical range is ``8`` to ``20``.
        font_weight : str | Inherit, optional
            The font weight of the reference labels.
            Values include ``"normal"``, ``"bold"``, ``"light"``, ``"ultralight"``, ``"heavy"``, and
            ``"black"``.
        format : Callable, optional
            A callable function to format the reference labels. By default, the reference labels are formatted as a),
            b), etc. The function must take a single str argument (the letter) and return a formatted str. For example,
            to have the reference labels in uppercase and in parentheses, the format could be defined as follows::

                format = lambda label: f"({label.upper()})"

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated reference labels parameters.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        _reject_animated_layout(self)
        if start_index is not None:
            if not isinstance(start_index, int):
                raise InvalidParameterTypeError("start_index must be an integer.")
            if start_index < 0:
                raise InvalidParameterError(
                    "start_index must be greater than or equal to 0."
                )
        if format is not None:
            try:
                format("a")
            except Exception as e:
                raise InvalidParameterTypeError(
                    "format must be a callable that takes a single str argument and returns a str."
                ) from e

        if reset:
            self._reference_labels_params.clear()

        params = ["color", "start_index", "font_size", "font_weight", "format"]
        for param in params:
            value = locals()[param]
            if value is not None:
                self._reference_labels_params[param] = value

        return self

    def create_twin_axis(
        self,
        is_y: bool = True,
        label: str | None = None,
        axis_lim: tuple[float, float] | None = None,
        log_scale: bool = False,
        remove_axes: bool = False,
        remove_ticks: bool = False,
        elements: Plottable | Iterable[Plottable | None] = [],
    ) -> SmartTwinAxis:
        """
        Creates a twin axis for the SmartFigure. This method creates a :class:`~graphinglib.SmartTwinAxis` object that
        can be used to plot elements on a secondary axis in the same subplot.

        Twin axes can only be created for SmartFigures that currently render as a single plot.

        Parameters
        ----------
        is_y : bool, optional
            If ``True``, the twin axis will be a y-axis, otherwise it will be an x-axis.
            Defaults to ``True``.
        label : str, optional
            Label for the twin axis.
        axis_lim : tuple[float, float], optional
            Limits for the twin axis.
        log_scale : bool, optional
            Whether to use a logarithmic scale for the twin axis.
            Defaults to ``False``.
        remove_axes : bool, optional
            Whether to remove the axes from the twin axis.
            Defaults to ``False``.
        remove_ticks : bool, optional
            Whether to remove the ticks from the twin axis.
            Defaults to ``False``.
        elements : Plottable | Iterable[Plottable | None], optional
            Elements to plot in the twin axis. A single :class:`~graphinglib.Plottable` is accepted and is treated as
            a one-element list. If an iterable is provided, any ``None`` elements are ignored.
            Defaults to empty list.

        Returns
        -------
        SmartTwinAxis
            The created twin axis object. The twin axis can also be accessed through the
            :attr:`~graphinglib.SmartFigure.twin_x_axis` or :attr:`~graphinglib.SmartFigure.twin_y_axis` properties.
        """
        if is_y and self._twin_y_axis is not None:
            raise InvalidOperationError(
                "A twin y-axis already exists for this SmartFigure."
            )
        elif not is_y and self._twin_x_axis is not None:
            raise InvalidOperationError(
                "A twin x-axis already exists for this SmartFigure."
            )

        twin_axis = SmartTwinAxis(
            label=label,
            axis_lim=axis_lim,
            log_scale=log_scale,
            remove_axes=remove_axes,
            remove_ticks=remove_ticks,
            elements=elements,
        )

        if is_y:
            self.twin_y_axis = twin_axis
        else:
            self.twin_x_axis = twin_axis

        return twin_axis


class SmartFigureWCS(SmartFigure):
    """
    This class implements a figure object for plotting :class:`~graphinglib.Plottable` elements with a
    `astropy.wcs.WCS <https://docs.astropy.org/en/stable/wcs/index.html>`_ projection.

    .. note::
        This class is available when the optional ``graphinglib[astro]`` extra is installed
        (it pulls in Astropy). Install with ``pip install graphinglib[astro]``.

    It allows for the creation of complex figures recursively, where each :class:`~graphinglib.SmartFigure` can contain
    other :class:`~graphinglib.SmartFigure` objects. The class supports a variety of customization options as well as
    the ability to use styles and themes for consistent visual appearance across different figures. A
    :class:`~graphinglib.SmartFigureWCS` follows the same general behavior as :class:`~graphinglib.SmartFigure`, but
    uses WCS projections for the plots it draws. The idea behind this class is that every SmartFigure contains a single
    x_label, y_label, title, projection, etc. and that nested SmartFigures can be inserted into the main SmartFigure
    to create complex figures with more parameters.

    Parameters
    ----------
    projection : WCS | list[WCS]
        The `World Coordinate System (WCS) <https://docs.astropy.org/en/stable/wcs/index.html>`_ object to use for the
        figure. This is used to plot data in a coordinate system that is not Cartesian, such as celestial coordinates.
        This can be given as a single WCS object or a list of WCS objects to apply to each subfigure drawn by the
        SmartFigure.
    num_rows, num_cols : int, optional
        Number of rows and columns for the base grid. These parameters determine the number of "squares" on which the
        plots can be placed.
        Defaults to ``1``.
    x_label, y_label : str, optional
        Labels for the x and y axes of the figure.
    size : tuple[float, float], optional
        Overall size of the figure. Note that this option is useless if the SmartFigure is nested inside another
        SmartFigure, as the size is then determined by the parent SmartFigure and the available space.
        Figure size is in inches; typical width is ``4`` to ``12`` and typical height is ``3`` to ``8``.
        Default depends on the ``figure_style`` configuration.
    title : str, optional
        General title of the figure.
    x_lim, y_lim : tuple[float, float] | list[tuple[float, float]], optional
        Limits for the x and y axes of the figure. This can be given as a single value or a list of values to apply to
        each subplot.
    sub_x_labels, sub_y_labels : Iterable[str | None], optional
        Labels for the x and y axes of each subfigure, respectively. This is only useful for figures that are not a
        single subplot and when each subfigure needs its own x and y labels. This prevents the creation of nested
        :class:`~graphinglib.SmartFigure` objects for each subfigure only to set the x and y labels. This list cannot
        be longer than the number of subfigures drawn by the SmartFigure and None values can be used to skip specific
        subfigures.
    subtitles : Iterable[str | None], optional
        Labels for the subtitles of each subfigure, respectively. Similarly to `sub_x_labels` and `sub_y_labels`, this
        allows to set subtitles for each subfigure without needing to create nested
        :class:`~graphinglib.SmartFigure` objects. It is only useful for figures that are not a single subplot and when
        each subfigure needs its own subtitle. This list cannot be longer than the number of subfigures drawn by the
        SmartFigure and None values can be used to skip specific subfigures.
    log_scale_x, log_scale_y : bool | list[bool | None], optional
        Whether to use a logarithmic scale for the x and y axes, respectively. This can be given as a single value or
        a list of values to apply to each subplot.
        Defaults to ``False``.
    remove_axes : bool | list[bool | None], optional
        Whether to remove the axes from the figure. This can be given as a single boolean or a list of booleans to apply
        to each subplot.
        Defaults to ``False``.
    aspect_ratio : float | Literal["auto", "equal"] | list[float | Literal["auto", "equal"]], optional
        Aspect ratio of the figure. If set to "auto", the aspect ratio is determined automatically to fill the available
        space. If set to "equal", the aspect ratio is set to 1:1. If set to a float, the aspect ratio represents the
        ratio of the height to the width of the data. This can be given as a single value or a list of values to apply
        to each subplot.
        Values are ``"equal"``, ``"auto"``, or a positive float. Defaults to ``"auto"``.

        .. warning::
            This parameter must not be confused with the `box_aspect_ratio` parameter, which is the aspect ratio of the
            box containing the elements. The `aspect_ratio` parameter is the aspect ratio of the data itself, which
            does not change the size of the plot but rather how the data is displayed within the plot.

    box_aspect_ratio : float | list[float], optional
        Aspect ratio of the box containing the elements, i.e. the ratio of the height to the width of the plot. This can
        be given as a single value or a list of values to apply to each subplot.

        .. warning::
            This parameter must not be confused with the `aspect_ratio` parameter, which is the aspect ratio of the
            data itself. The `box_aspect_ratio` parameter changes the size of the plot, which does not affect the
            figure's axes.

    remove_x_ticks, remove_y_ticks : bool | list[bool | None], optional
        Whether to remove the x and y ticks from the figure, respectively. This can be given as a single value or a list
        of values to apply to each subplot.
        Defaults to ``False``.
    invert_x_axis, invert_y_axis : bool | list[bool | None], optional
        Whether to invert the x and y axes, respectively. Use a single value or a list aligned with the subplots.
        During animation, a list entry of ``None`` leaves that subplot alone.
        Defaults to ``False``.
    reference_labels : bool | list[bool], optional
        Whether or not to add reference labels to the subfigures. If set to ``True``, each subfigure will be labeled
        alphabetically in the form of "a)", "b)", etc. This can be given as a single value or a list of values to apply
        to each subplot.
        Defaults to ``True``.

        .. note::
            For nested figures, each subfigure controls its own reference labels. This means that if a nested
            SmartFigure turns off reference labels, the plots in it will not be labeled, even if the parent SmartFigure
            has reference labels turned on.

    global_reference_label : bool, optional
        Whether to use a single reference label for the entire figure instead of individual labels for each subfigure.
        If set to ``True``, the reference label will be placed in the top left corner of the global SmartFigure. This is
        useful for labeling the entire figure rather than individual subfigures.
        Defaults to ``False``.

        .. warning::
            As the global reference label is placed more left than the reference label, this forces the horizontal shift
            of the axes, which may lead to overlapping between axes. Consider modifying the `size` or `width_padding`
            parameters to avoid this issue.
    reference_labels_loc : Literal["inside", "outside"] | tuple[float, float] | list, optional
        Location of the reference labels of the SubFigures, which can be either ``"inside"``, ``"outside"``, or a tuple of
        (x, y) relative coordinates to the top-left corner of each subfigure. This can be given as a single value or a
        list of values to apply to each subplot.
        Defaults to ``"outside"``.
    width_padding, height_padding : float, optional
        Padding between the subfigures in the x and y directions, respectively. The default value of ``None`` results in
        a default small amount of padding. This may be set to 0 to completely remove the space between subfigures, but
        note that axes labels may need to be removed to delete additional space.
    width_ratios, height_ratios : ArrayLike, optional
        Ratios of the widths and heights of the subfigures, respectively. These ratios determine how much space each
        column and row of subfigures will take up in the overall figure. The length of these arrays must match the
        number of columns and rows, respectively. By default, all subfigures are given equal space.
    share_x, share_y : bool, optional
        Whether to share the x and y axes between subfigures, respectively. This means that all subfigures will have
        the same x and y limits, and the ticks will be shared as well. This is useful for comparing data across
        subfigures.

        .. note::
            Sharing axes only works for plots drawn directly in the SmartFigure. If you insert an existing nested
            SmartFigure, that nested figure keeps its own axes sharing settings.

    general_legend : bool, optional
        Whether to create a general legend for the entire figure. If set to ``True``, a single legend will be created
        to regroup all the legends from the subplots. If set to ``False``, all subplots will have their own legend. If
        nested SmartFigures set this parameter to ``False``, their legend is added to the parent's general legend.
        However, if a nested SmartFigure sets its general legend to ``True``, it will be created separately and will not
        be added to the parent's general legend.
        Defaults to ``False``.
    legend_loc : str | tuple | list[str | tuple], optional
        Location of the legend. This can be a string (e.g., "upper right") or a tuple of (x, y) relative coordinates.
        The supported string locations are: {"upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center", "center", "outside upper center",
        "outside center right", "outside lower center", "outside center left"}. Additionally, only if ``general_legend``
        is set to ``False``, the legend location can also be set to "best". This option can be given as a single value
        or a list of values to apply to each subplot.
        Defaults to ``"best"`` if ``general_legend`` is set to ``False``, otherwise it defaults to ``"lower center"``.

        .. warning::
            If ``general_legend`` is set to ``True`` and the legend location is set to a position containing "outside",
            the legend may not be displayed correctly in some matplotlib backends. In such cases, it is recommended to
            use inline figures in a Jupyter notebook or save the figure to a file to ensure proper display of the
            legend outside the figure.

    legend_cols : int | list[int], optional
        Number of columns to display the labels in the legend. This is only used if the legend is displayed. This can be
        given as a single value or a list of values to apply to each subplot.
        Defaults to ``1``.
    show_legend : bool | list[bool | None], optional
        Whether to show the legend for the figure. This allows to easily toggle the visibility of the legend. This can
        be given as a single value or a list of values to apply to each subplot.
        Defaults to ``True``.
    twin_x_axis, twin_y_axis : SmartTwinAxis, optional
        Twin axes for the x and y axes, respectively. This allows to attach additional axes to the main axes of the
        SmartFigure, which can be useful for displaying additional information or data on the same plot without
        cluttering the main axes. The twin axes can be used to plot additional data with different scales or units. See
        the :class:`~graphinglib.SmartTwinAxis` class for more details on how to use twin axes and the
        :meth:`~graphinglib.SmartFigure.create_twin_axis` method for wrapping the creation of twin axes.
    figure_style : str, optional
        The figure style to use for the figure. The default style can be set using ``gl.set_default_style()``.
        Defaults to ``"default"``.
    elements : Plottable | Iterable[Plottable | SmartFigure] | Iterable[Iterable[Plottable | SmartFigure]], optional
        The elements to plot in the figure.
        If a single :class:`~graphinglib.Plottable` is provided, it is added as the only element in the SmartFigure's
        single plot. If an iterable of :class:`~graphinglib.Plottable` objects is provided, all the elements are added
        to the SmartFigure's single plot.
        If a flat iterable is provided for a figure with multiple cells, the items are assigned to the cells from
        left-to-right, top-to-bottom. Each item can be a child :class:`~graphinglib.SmartFigure`, a
        :class:`~graphinglib.Plottable`, an iterable of :class:`~graphinglib.Plottable` objects, or ``None``.
        Plottables and iterables of Plottables create a child plot in that cell. The iterable can be shorter than the
        total number of cells, in which case the remaining cells are left empty.
        If a :class:`~graphinglib.SmartFigure` is provided, it occupies exactly one cell in the parent figure. Its
        ``num_rows`` and ``num_cols`` only describe its internal layout.

        .. note::
            The ``elements`` parameter does not use a child :class:`~graphinglib.SmartFigure`'s shape to determine how
            much space it occupies in the parent. If you want a bare :class:`~graphinglib.Plottable`, an iterable of
            Plottables, or a child :class:`~graphinglib.SmartFigure` to span multiple subplots in the parent, you
            should use the __setitem__ method instead.
            For example, to add an element spanning the complete first row , use ``fig[0,:] = element``.

    annotations : Iterable[Text], optional
        Text annotations to add on the figure. The provided Text elements must have figure-relative coordinates, i.e.
        in the range [0, 1] for both x and y. This allows to add annotations that are not tied to a specific subplot,
        for example, to add general notes or labels on the figure.
    """

    def __init__(
        self,
        projection: ListOrItem[WCS],
        num_rows: int = 1,
        num_cols: int = 1,
        x_label: str | None = None,
        y_label: str | None = None,
        size: tuple[float, float] | Inherit = INHERIT,
        title: str | None = None,
        x_lim: ListOrItem[tuple[float | None, float | None] | None] = None,
        y_lim: ListOrItem[tuple[float | None, float | None] | None] = None,
        sub_x_labels: Iterable[str | None] | None = None,
        sub_y_labels: Iterable[str | None] | None = None,
        subtitles: Iterable[str | None] | None = None,
        log_scale_x: bool | list[bool | None] = False,
        log_scale_y: bool | list[bool | None] = False,
        remove_axes: bool | list[bool | None] = False,
        aspect_ratio: ListOrItem[float | Literal["auto", "equal"]] = "auto",
        box_aspect_ratio: ListOrItem[float | None] = None,
        remove_x_ticks: bool | list[bool | None] = False,
        remove_y_ticks: bool | list[bool | None] = False,
        invert_x_axis: bool | list[bool | None] = False,
        invert_y_axis: bool | list[bool | None] = False,
        reference_labels: ListOrItem[bool] = True,
        global_reference_label: bool = False,
        reference_labels_loc: ListOrItem[
            Literal["inside", "outside"] | tuple[float, float]
        ] = "outside",
        width_padding: float | None = None,
        height_padding: float | None = None,
        width_ratios: ArrayLike | None = None,
        height_ratios: ArrayLike | None = None,
        share_x: bool = False,
        share_y: bool = False,
        general_legend: bool = False,
        legend_loc: ListOrItem[str | tuple | None] = None,
        legend_cols: ListOrItem[int] = 1,
        show_legend: bool | list[bool | None] = True,
        twin_x_axis: SmartTwinAxis | None = None,
        twin_y_axis: SmartTwinAxis | None = None,
        figure_style: str | Inherit = INHERIT,
        elements: Plottable
        | Iterable[Plottable | SmartFigure | None | Iterable[Plottable | None]] = [],
        annotations: Iterable[Text] | None = None,
    ) -> None:
        _require_astropy("SmartFigureWCS")

        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            x_label=x_label,
            y_label=y_label,
            size=size,
            title=title,
            x_lim=x_lim,
            y_lim=y_lim,
            sub_x_labels=sub_x_labels,
            sub_y_labels=sub_y_labels,
            subtitles=subtitles,
            log_scale_x=log_scale_x,
            log_scale_y=log_scale_y,
            remove_axes=remove_axes,
            aspect_ratio=aspect_ratio,
            box_aspect_ratio=box_aspect_ratio,
            remove_x_ticks=remove_x_ticks,
            remove_y_ticks=remove_y_ticks,
            invert_x_axis=invert_x_axis,
            invert_y_axis=invert_y_axis,
            reference_labels=reference_labels,
            global_reference_label=global_reference_label,
            reference_labels_loc=reference_labels_loc,
            width_padding=width_padding,
            height_padding=height_padding,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            share_x=share_x,
            share_y=share_y,
            projection=projection,
            general_legend=general_legend,
            legend_loc=legend_loc,
            legend_cols=legend_cols,
            show_legend=show_legend,
            twin_x_axis=twin_x_axis,
            twin_y_axis=twin_y_axis,
            figure_style=figure_style,
            elements=elements,
            annotations=annotations,
        )

        self._default_tick_params = {  # The following are the default parameters of WCSAxes objects
            "x major": {"bottom": True, "top": True, "labelbottom": True},
            "y major": {"left": True, "right": True, "labelleft": True},
            "x minor": {},
            "y minor": {},
        }
        self._tick_params = deepcopy(self._default_tick_params)

    @property
    def projection(self) -> ListOrItem[WCS]:
        return self._projection

    @projection.setter
    def projection(self, value: ListOrItem[WCS]) -> None:
        _reject_animated_layout(self)
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, WCS):
                raise InvalidParameterTypeError(
                    "The projection of a SmartFigureWCS must be a WCS object."
                )
        self._projection = value
        if hasattr(self, "_children"):
            self._sync_auto_child_projections()

    def _prepare_figure(
        self,
        is_matplotlib_style: bool = False,
        make_legend: bool = True,
        *,
        animation_parent: tuple[SmartFigure, int] | None = None,
        animation_grid_defaults: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, list[str | Any]]]:
        """
        Wraps the parent method to check if the number of projections matches the number of subfigures drawn by the
        SmartFigure.
        """
        if isinstance(self._projection, list) and len(self._projection) != len(self):
            raise InvalidParameterError(
                f"Number of WCS projections ({len(self._projection)}) must be equal to the number of subfigures "
                f"({len(self)})."
            )
        return super()._prepare_figure(
            is_matplotlib_style,
            make_legend,
            animation_parent=animation_parent,
            animation_grid_defaults=animation_grid_defaults,
        )

    def _customize_ticks(
        self,
        ax: Axes,
        subplot_i: int,
    ) -> None:
        """
        Customizes the ticks of the specified Axes according to the SmartFigure's tick parameters. This method is useful
        for inheritance to allow each SmartFigure class to customize the ticks their way.
        """
        # ax is a WCSAxes here; its `.coords` accessor is not in matplotlib's Axes stub.
        x_axis, y_axis = cast(Any, ax).coords
        x_axis.set_auto_axislabel(False)
        y_axis.set_auto_axislabel(False)

        if self._ticks.get("x_tick_formatter") is not None:
            x_axis.set_major_formatter(self._ticks.get("x_tick_formatter"))
        if self._ticks.get("y_tick_formatter") is not None:
            y_axis.set_major_formatter(self._ticks.get("y_tick_formatter"))

        x_axis.set_ticks(
            values=self._ticks.get("x_ticks"),
            spacing=self._ticks.get("x_tick_spacing"),
            number=self._ticks.get("number_of_x_ticks"),
        )
        y_axis.set_ticks(
            values=self._ticks.get("y_ticks"),
            spacing=self._ticks.get("y_tick_spacing"),
            number=self._ticks.get("number_of_y_ticks"),
        )

        if self._ticks.get("minor_x_tick_frequency") is not None:
            x_axis.display_minor_ticks(True)
            x_axis.set_minor_frequency(self._ticks.get("minor_x_tick_frequency"))
        if self._ticks.get("minor_y_tick_frequency") is not None:
            y_axis.display_minor_ticks(True)
            y_axis.set_minor_frequency(self._ticks.get("minor_y_tick_frequency"))

        # Manually set the tick_params using the recommended API
        for i, axis, ax_params in zip(
            ["x", "y"],
            [x_axis, y_axis],
            [self._tick_params["x major"], self._tick_params["y major"]],
        ):
            axis.set_ticks(
                size=ax_params.get("length"),
                width=ax_params.get("width"),
                color=ax_params.get("color"),
                direction=ax_params.get("direction"),
            )
            axis.set_ticklabel(
                size=ax_params.get("labelsize"),
                color=ax_params.get("labelcolor"),
                pad=ax_params.get("pad"),
                rotation=ax_params.get("labelrotation"),
            )

            # Only allow valid positions for each axis
            if i == "x":
                valid_tick_pos = ["bottom", "top"]
                valid_label_pos = ["labelbottom", "labeltop"]
                tick_pos_str = "".join(
                    pos[0] for pos in valid_tick_pos if ax_params.get(pos)
                )
                label_pos_str = "".join(
                    pos[5] for pos in valid_label_pos if ax_params.get(pos)
                )
            else:  # i == "y"
                valid_tick_pos = ["left", "right"]
                valid_label_pos = ["labelleft", "labelright"]
                tick_pos_str = "".join(
                    pos[0] for pos in valid_tick_pos if ax_params.get(pos)
                )
                label_pos_str = "".join(
                    pos[5] for pos in valid_label_pos if ax_params.get(pos)
                )

            if tick_pos_str:
                axis.set_ticks_position(tick_pos_str)
            else:
                axis.set_ticks_visible(False)

            if label_pos_str:
                axis.set_ticklabel_position(label_pos_str)
            else:
                axis.set_ticklabel_visible(False)

        if self._tick_params["x minor"].get("length") is not None:
            ax.tick_params(
                axis="x",
                which="minor",
                length=self._tick_params["x minor"].get("length"),
            )
        if self._tick_params["y minor"].get("length") is not None:
            ax.tick_params(
                axis="y",
                which="minor",
                length=self._tick_params["y minor"].get("length"),
            )

    def _grid_options(
        self, defaults: dict[str, tuple[str, bool]]
    ) -> dict[str, tuple[str, bool]]:
        return {
            dimension: ("major", enabled)
            for dimension, (_, enabled) in super()._grid_options(defaults).items()
        }

    def _prepare_grid(
        self,
        ax: Axes,
        show: bool,
        options: dict[str, tuple[str, bool]],
        style: dict[str, Any] | None = None,
        initial_linestyle: Any = None,
    ) -> _ArtistEdit:
        edits = []
        for index, dimension in enumerate(("x", "y")):
            coordinate = cast(Any, ax).coords[index]
            _, enabled = options[dimension]
            # CoordinateHelper exposes grid visibility through its drawing options.
            previous = coordinate._grid_lines_kwargs
            old = {"draw_grid": previous["visible"]}
            new = {"draw_grid": show and enabled}
            if style is not None:
                new.update(
                    {
                        "edgecolor": style["grid.color"],
                        "alpha": style["grid.alpha"],
                        "linestyle": style["grid.linestyle"],
                        "linewidth": style["grid.linewidth"],
                    }
                )
                old.update(
                    {name: previous[name] for name in new if name != "draw_grid"}
                )

            def write(params, coord=coordinate) -> None:
                coord.grid(grid_type=getattr(coord, "_grid_type", None), **params)

            edits.append(_edit(write, old, new))
        return _combine_edits(edits)

    def _prepare_tick_removal(
        self, ax: Axes, dimension: str
    ) -> Callable[[bool], _ArtistEdit]:
        axis = cast(Any, ax).coords[0 if dimension == "x" else 1]

        def read() -> tuple[bool, bool]:
            return axis.get_ticks_visible(), axis.get_ticklabel_visible()

        def write(flags: tuple[bool, bool]) -> None:
            axis.set_ticks_visible(flags[0])
            axis.set_ticklabel_visible(flags[1])

        initial = read()

        def prepare(remove: bool) -> _ArtistEdit:
            return _edit(write, read(), (False, False) if remove else initial)

        return prepare

    # WCSAxes uses Astropy Quantity ticks and WCS-specific tick controls, so this
    # intentionally is not a behavioral substitute for SmartFigure.set_ticks.
    def set_ticks(  # ty: ignore[invalid-method-override]
        self,
        reset: bool = False,
        x_ticks: list[Quantity] | None = None,
        y_ticks: list[Quantity] | None = None,
        x_tick_spacing: Quantity | None = None,
        y_tick_spacing: Quantity | None = None,
        number_of_x_ticks: int | None = None,
        number_of_y_ticks: int | None = None,
        x_tick_formatter: str | Callable | None = None,
        y_tick_formatter: str | Callable | None = None,
        minor_x_tick_frequency: int | None = None,
        minor_y_tick_frequency: int | None = None,
    ) -> Self:
        """
        Sets custom ticks and tick labels.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set ticks to their default values before applying the new parameters.
            Defaults to ``False``.
        x_ticks, y_ticks : list[Quantity], optional
            Tick positions for the x or y axis. If a value is specified, the corresponding ``x_tick_spacing`` and
            ``number_of_x_ticks`` or ``y_tick_spacing`` and ``number_of_y_ticks`` parameters must be ``None``.
        x_tick_spacing, y_tick_spacing : Quantity, optional
            Spacing between major ticks on the x or y axis. If a value is specified, the corresponding ``x_ticks`` and
            ``number_of_x_ticks`` or ``y_ticks`` and ``number_of_y_ticks`` parameters must be ``None``.
        number_of_x_ticks, number_of_y_ticks : int, optional
            Number of ticks to display on the x or y axis. If specified, the ``x_ticks`` and ``x_tick_spacing`` or
            ``y_ticks`` and ``y_tick_spacing`` parameters must be ``None``.

            .. note::
                This value is not absolute, but rather a suggestion to the WCSAxes. The actual number of ticks
                displayed may vary depending on the data and limits of the axes.

        x_tick_formatter, y_tick_formatter : str | Callable, optional
            A function or a string format to apply to the x or y tick labels. If a function is provided, it should take
            a single argument (the tick value) and return a formatted string. If a string is provided, it should be a
            format string that will be applied to each tick value. See the `astropy documentation
            <https://docs.astropy.org/en/latest/visualization/wcsaxes/ticks_labels_grid.html>`_ for more details.

            .. warning::
                Callable formatters must only be given if the coordinate axis is not in angular units.

            Examples::

                >>> x_tick_formatter = "hh:mm:ss.s"

                ``1h01m34.1s``

                >>> x_tick_formatter = lambda x: f"{x:.2f} s"  # only for non-angular coordinate axes

                ``1.23 s``

        minor_x_tick_frequency, minor_y_tick_frequency : float, optional
            Frequency of minor ticks on the x or y axis. This gives the number of minor ticks between each major tick.

            .. note::
                The frequency includes the major tick, so a frequency of 2 means that there is one minor tick between
                each major tick.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated ticks.
        """
        _reject_animated_layout(self)
        if any(
            [
                (x_ticks is not None) and (x_tick_spacing is not None),
                (y_ticks is not None) and (y_tick_spacing is not None),
            ]
        ):
            raise IncompatibleArgumentsError(
                "Tick spacing and tick positions cannot be set simultaneously."
            )

        if any(
            [
                (x_ticks is not None) and (number_of_x_ticks is not None),
                (y_ticks is not None) and (number_of_y_ticks is not None),
            ]
        ):
            raise IncompatibleArgumentsError(
                "Number of ticks and tick positions cannot be set simultaneously."
            )

        if any(
            [
                (x_tick_spacing is not None) and (number_of_x_ticks is not None),
                (y_tick_spacing is not None) and (number_of_y_ticks is not None),
            ]
        ):
            raise IncompatibleArgumentsError(
                "Number of ticks and tick spacing cannot be set simultaneously."
            )

        if reset:
            self._ticks.clear()

        params = [
            "x_ticks",
            "y_ticks",
            "x_tick_spacing",
            "y_tick_spacing",
            "number_of_x_ticks",
            "number_of_y_ticks",
            "x_tick_formatter",
            "y_tick_formatter",
            "minor_x_tick_frequency",
            "minor_y_tick_frequency",
        ]
        for param in params:
            value = locals()[param]
            if value is not None:
                self._ticks[param] = value

        return self

    # WCSAxes cannot express SmartFigure's major/minor split; minor tick length
    # is the only independent minor-tick control exposed by Astropy.
    def set_tick_params(  # ty: ignore[invalid-method-override]
        self,
        axis: Literal["x", "y", "both"] | None = "both",
        reset: bool = False,
        direction: Literal["in", "out"] | None = None,
        length: float | None = None,
        minor_length: float | None = None,
        width: float | None = None,
        color: str | None = None,
        pad: float | None = None,
        label_size: float | str | None = None,
        label_color: str | None = None,
        label_rotation: float | None = None,
        draw_bottom_ticks: bool | None = None,
        draw_top_ticks: bool | None = None,
        draw_left_ticks: bool | None = None,
        draw_right_ticks: bool | None = None,
        draw_bottom_labels: bool | None = None,
        draw_top_labels: bool | None = None,
        draw_left_labels: bool | None = None,
        draw_right_labels: bool | None = None,
    ) -> Self:
        """
        Sets the tick parameters for the figure. These parameters are given to the ``tick_params`` method of the
        `astropy.visualization.wcsaxes.WCSAxes
        <https://docs.astropy.org/en/stable/api/astropy.visualization.wcsaxes.WCSAxes.html>`_.

        .. warning::
            Due to how the :class:`~astropy.visualization.wcsaxes.WCSAxes` are implemented, only the length of the minor
            ticks can be controlled independently from the major ticks. The other parameters are applied to both major
            and minor ticks for a specified axis.

        Parameters
        ----------
        axis : {"x", "y", "both"}, optional
            The axis to set the tick parameters for. This method can be called multiple times to set the tick
            parameters specifically for each axes.
            Defaults to ``"both"``.
        reset : bool, optional
            If ``True``, resets all previously given tick parameters to their default values before applying the new
            parameters.
            Defaults to ``False``.
        direction : {"in", "out"}, optional
            The direction of the ticks.

            .. warning::
                Contrary to the :meth:`~graphinglib.SmartFigure.set_tick_params` method, the ``direction`` parameter
                cannot be set to ``"inout"`` since they are not supported by `astropy.visualization.wcsaxes.WCSAxes
                <https://docs.astropy.org/en/stable/api/astropy.visualization.wcsaxes.WCSAxes.html>`_.
        length : float, optional
            The length of the ticks.
            Typical range is ``2`` to ``10``.
        minor_length : float, optional
            The length of the minor ticks. This is the only parameter that can be set independently from the major ticks
            due to the way the :class:`astropy.visualization.wcsaxes.WCSAxes` are implemented.
            Typical range is ``1`` to ``6``.
        width : float, optional
            The width of the ticks.
            Typical range is ``0.5`` to ``3``.
        color : str, optional
            The color of the ticks.
        pad : float, optional
            The padding to add between the tick labels and the ticks themselves.
            Typical range is ``2`` to ``10``.
        label_size : float | str, optional
            The font size of the tick labels. This can be a float or a string (e.g. "large").
            Typical range is ``8`` to ``20`` when a float is used.
        label_color : str, optional
            The color of the tick labels.
        label_rotation : float, optional
            The rotation of the tick labels, in degrees.
        draw_bottom_ticks, draw_top_ticks, draw_left_ticks, draw_right_ticks : bool, optional
            Whether to draw the ticks on the bottom, top, left or right side of the axes respectively.
        draw_bottom_labels, draw_top_labels, draw_left_labels, draw_right_labels : bool, optional
            Whether to draw the tick labels on the bottom, top, left or right side of the axes respectively.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated tick parameters.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        _reject_animated_layout(self)
        new_tick_params = {
            "direction": direction,
            "length": length,
            "width": width,
            "color": color,
            "pad": pad,
            "labelsize": label_size,
            "labelcolor": label_color,
            "labelrotation": label_rotation,
            "bottom": draw_bottom_ticks,
            "top": draw_top_ticks,
            "left": draw_left_ticks,
            "right": draw_right_ticks,
            "labelbottom": draw_bottom_labels,
            "labeltop": draw_top_labels,
            "labelleft": draw_left_labels,
            "labelright": draw_right_labels,
        }
        for axis_i in [axis] if axis != "both" else ["x", "y"]:
            if reset:
                self._tick_params[f"{axis_i} major"] = deepcopy(
                    self._default_tick_params[f"{axis_i} major"]
                )
                self._tick_params[f"{axis_i} minor"] = deepcopy(
                    self._default_tick_params[f"{axis_i} minor"]
                )
            for param, value in new_tick_params.items():
                if value is not None:
                    self._tick_params[f"{axis_i} major"][param] = value
            if minor_length is not None:
                self._tick_params[f"{axis_i} minor"]["length"] = minor_length
        return self

    # WCSAxes grids only support major grid lines, so SmartFigure's which_x /
    # which_y controls are intentionally absent from this API.
    def set_grid(  # ty: ignore[invalid-method-override]
        self,
        visible_x: bool = True,
        visible_y: bool = True,
        color: str | Inherit = INHERIT,
        alpha: float | Inherit = INHERIT,
        line_style: str | Inherit = INHERIT,
        line_width: float | Inherit = INHERIT,
    ) -> Self:
        """
        Sets the grid parameters for the figure.

        .. note::
            Contrary to the :class:`~graphinglib.SmartFigure` class, this method does not support plotting grid lines
            for minor ticks. This is because the :class:`astropy.visualization.wcsaxes.WCSAxes` do not support minor
            ticks for the grid lines.

        Parameters
        ----------
        visible_x, visible_y : bool, optional
            If ``True``, sets the x-axis or y-axis grid visible. If ``False``, the grid is not shown for the respective
            axis.
            Defaults to ``True`` for both axes.
        color : str, optional
            Sets the color of the grid lines.
            Default depends on the ``figure_style`` configuration.
        alpha : float, optional
            Sets the alpha value for the grid lines.
            Range is ``0`` (transparent) to ``1`` (opaque).
            Default depends on the ``figure_style`` configuration.
        line_style : str, optional
            Sets the line style of the grid lines.
            Values include ``"-"``, ``"--"``, ``"-."``, ``":"``, ``"solid"``, ``"dashed"``, ``"dashdot"``, and
            ``"dotted"``.
            Default depends on the ``figure_style`` configuration.
        line_width : float, optional
            Sets the line width of the grid lines.
            Typical range is ``0.5`` to ``3``.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated grid parameters.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        return super().set_grid(
            visible_x=visible_x,
            visible_y=visible_y,
            which_x="major",
            which_y="major",
            color=color,
            alpha=alpha,
            line_style=line_style,
            line_width=line_width,
        )


class SmartTwinAxis:
    """
    This class implements a twin axis for the :class:`~graphinglib.SmartFigure` and :class:`~graphinglib.SmartFigureWCS`
    classes.

    Behaves like a :class:`~graphinglib.SmartFigure` object, but is not meant to be used on its own. Elements can be
    added to the twin axis using the :meth:`~graphinglib.SmartTwinAxis.add_elements` method and the same methods as for
    the :class:`~graphinglib.SmartFigure` class can be used to customize the twin axis.

    Parameters
    ----------
    label : str, optional
        Label for the twin axis.
    axis_lim : tuple[float, float], optional
        Limits for the twin axis.
    log_scale : bool, optional
        Whether to use a logarithmic scale for the twin axis.
        Defaults to ``False``.
    remove_axes : bool, optional
        Whether to remove the axes from the twin axis.
        Defaults to ``False``.
    remove_ticks : bool, optional
        Whether to remove the ticks from the twin axis.
        Defaults to ``False``.
    invert_axis : bool, optional
        Whether to invert the twin axis.
        Defaults to ``False``.
    elements : Plottable | Iterable[Plottable | None], optional
        Elements to plot in the twin axis. A single :class:`~graphinglib.Plottable` is accepted and is treated as a
        one-element list. If an iterable is provided, any ``None`` values are ignored. See the ``elements`` parameter of the
        :class:`~graphinglib.SmartFigure` class for more details.
    """

    def __init__(
        self,
        label: str | None = None,
        axis_lim: tuple[float, float] | None = None,
        log_scale: bool = False,
        remove_axes: bool = False,
        remove_ticks: bool = False,
        invert_axis: bool = False,
        elements: Plottable | Iterable[Plottable | None] = [],
    ) -> None:
        self.label = label
        self.axis_lim = axis_lim
        self.log_scale = log_scale
        self.remove_axes = remove_axes
        self.remove_ticks = remove_ticks
        self.invert_axis = invert_axis
        self.elements = elements

        self._ticks = {}
        self._tick_params: dict[str, dict[str, Any]] = {"major": {}, "minor": {}}

        self._edge_color = None
        self._line_width = None
        self._hide_spine = None
        self._user_rc_dict: dict[str, Any] = {}
        self._default_params = {}
        self._axes: Axes | None = (
            None  # used for keeping a reference to the Axes which enables drawing the legend on top
        )

    @property
    def label(self) -> str | None:
        return self._label

    @label.setter
    def label(self, value: str | None) -> None:
        self._label = _stage_live_value(self, "_label", value)

    @property
    def axis_lim(self) -> tuple[float, float] | None:
        return self._axis_lim

    @axis_lim.setter
    def axis_lim(self, value: tuple[float, float] | None) -> None:
        if value is not None:
            if not isinstance(value, tuple):
                raise InvalidParameterTypeError("axis_lim must be a tuple.")
            if len(value) != 2:
                raise InvalidParameterError("axis_lim must be a tuple of length 2.")
        self._axis_lim = _stage_live_value(self, "_axis_lim", value)

    @property
    def log_scale(self) -> bool:
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterTypeError("log_scale must be a boolean.")
        self._log_scale = _stage_live_value(self, "_log_scale", value)

    @property
    def remove_axes(self) -> bool:
        return self._remove_axes

    @remove_axes.setter
    def remove_axes(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterTypeError("remove_axes must be a bool.")
        self._remove_axes = _stage_live_value(self, "_remove_axes", value)

    @property
    def remove_ticks(self) -> bool:
        return self._remove_ticks

    @remove_ticks.setter
    def remove_ticks(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterTypeError("remove_ticks must be a bool.")
        self._remove_ticks = _stage_live_value(self, "_remove_ticks", value)

    @property
    def invert_axis(self) -> bool:
        return self._invert_axis

    @invert_axis.setter
    def invert_axis(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise InvalidParameterTypeError("invert_axis must be a bool.")
        self._invert_axis = _stage_live_value(self, "_invert_axis", value)

    @property
    def elements(self) -> list[Plottable | None]:
        return self._elements

    @elements.setter
    def elements(self, value: Plottable | Iterable[Plottable | None]) -> None:
        _reject_animated_layout(self)
        self._elements = []  # systematically reset the elements when setting them with the property
        if isinstance(value, Plottable):
            self.add_elements(value)
            return
        self.add_elements(*value)

    def __len__(self) -> int:
        """
        Gives the number of elements in the :class:`~graphinglib.SmartTwinAxis`.
        """
        return len(self._elements)

    def __getitem__(self, key: int) -> Plottable:
        """
        Gives the element(s) at the specified key in the SmartTwinAxis's list of elements.

        Parameters
        ----------
        key : int
            The key specifying the location in the SmartTwinAxis to access.

        Returns
        -------
        Plottable
            The element at the specified key. If there is no element at the given key, an empty list is returned.
        """
        if not isinstance(key, int):
            raise InvalidParameterTypeError(
                f"Key must be an integer, not {type(key).__name__}."
            )
        key_ = key + len(self._elements) if key < 0 else key
        if key_ >= len(self._elements) or key_ < 0:
            raise IndexError(
                f"Key {key} is out of bounds for the SmartTwinAxis with {len(self._elements)} elements."
            )
        return self._elements[key_]

    def __iter__(self) -> Iterator[Plottable]:
        """
        Iterates over the elements in the SmartTwinAxis in order of their position in the grid, from top-left to
        bottom-right.
        """
        yield from self._elements

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.SmartTwinAxis` object.
        """
        return deepcopy(self)

    def copy_with(self, **kwargs) -> Self:
        """
        Returns a deep copy of the SmartTwinAxis with specified attributes overridden.

        Parameters
        ----------
        **kwargs
            Public writable properties to override in the copied SmartTwinAxis. The keys should be property names to
            modify and the values are the new values for those properties.

        Returns
        -------
        SmartTwinAxis
            A new SmartTwinAxis instance with the specified attributes overridden.

        Examples
        --------
        Copy an existing SmartTwinAxis to remove change its label::

            twin_ax_2 = twin_ax_1.copy_with(label="New Label")
        """
        return _copy_with_overrides(self, **kwargs)

    def add_elements(self, *elements: Plottable | None) -> Self:
        """
        Adds one or more :class:`~graphinglib.Plottable` elements to the twin axis.

        Parameters
        ----------
        elements : :class:`~graphinglib.Plottable`
            Elements to plot in the :class:`~graphinglib.SmartTwinAxis`. If ``None`` values are present, they are
            ignored and not added to the twin axis.

        Returns
        -------
        Self
            For convenience, the same :class:`~graphinglib.SmartTwinAxis` with the added elements.
        """
        _reject_animated_layout(self)
        if not SmartFigure._is_iterable_of_plottables(elements):
            raise InvalidParameterTypeError(
                "Elements must be an iterable of Plottable objects."
            )
        self._elements += [el for el in elements if el is not None]
        return self

    def _prepare_twin_axis(
        self,
        fig_axes: Axes,
        is_matplotlib_style: bool,
        cycle_colors: list[str],
        is_y: bool,
        z_order: int,
        figure_style: str | Inherit,
    ) -> tuple[list[str], list[Any]]:
        """
        Prepares the twin axis to be displayed.

        Parameters
        ----------
        fig_axes : matplotlib.axes.Axes
            The parent axes to create the twin axis from.
        is_matplotlib_style : bool
            Whether the figure style is a matplotlib style.
        cycle_colors : list[str]
            Color cycle for plotting elements.
        is_y : bool
            Whether the twin axis is a y-axis (``True``) or an x-axis (``False``).
        z_order : int
            The z-order for the elements plotted on the twin axis. This is used to ensure that the elements on the twin
            axis are drawn above the elements of the original axis.
        figure_style : str | Inherit
            The figure style to use for the twin axis. This is used for the
            :meth:`~graphinglib.SmartTwinAxis._fill_in_missing_params` method.

        Returns
        -------
        tuple[list[str], list[Any]]
            A tuple containing the labels and handles for legend creation.
        """
        # Create the twin axis
        if is_y:
            ax = fig_axes.twinx()
            ax_set_label, ax_set_lim, ax_set_scale, spine_str = (
                ax.set_ylabel,
                ax.set_ylim,
                ax.set_yscale,
                "right",
            )
        else:
            ax = fig_axes.twiny()
            ax_set_label, ax_set_lim, ax_set_scale, spine_str = (
                ax.set_xlabel,
                ax.set_xlim,
                ax.set_xscale,
                "top",
            )

        self._axes = ax
        session = _DRAW_SESSION.get()
        if session is not None:
            session.mark_present(self)

            def release_twin() -> None:
                if self._axes is ax:
                    self._axes = None

            session.on_finish(release_twin)

        if self._label:
            ax_set_label(self._label)
        if session is not None:
            session.bind_axis_text(
                self, "_label", ax.get_ylabel if is_y else ax.get_xlabel, ax_set_label
            )
        if self._axis_lim:
            ax_set_lim(*self._axis_lim)

        # Artificially modify the axes edge color and line width to modify only a single spine
        if self._edge_color:
            ax.spines[spine_str].set_color(self._edge_color)
        if self._line_width:
            ax.spines[spine_str].set_linewidth(self._line_width)
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        spine = ax.spines[spine_str]
        spine.set_visible(not self._hide_spine)
        if session is not None:
            session.bind_property(self, "_edge_color", spine, "edgecolor")
            session.bind_property(self, "_line_width", spine, "linewidth")

            def prepare_hide(hidden: bool) -> _ArtistEdit:
                return _edit(spine.set_visible, spine.get_visible(), not hidden)

            session.bind_update(self, "_hide_spine", prepare_hide)

        self._customize_ticks(is_y)

        # Logarithmic scale
        if self._log_scale:
            ax_set_scale("log")

        if self._remove_axes:
            ax.set_axis_off()
        if self._remove_ticks or session is not None:
            prepare_ticks = _tick_removal(
                ax.yaxis if is_y else ax.xaxis, "y" if is_y else "x"
            )
            if self._remove_ticks:
                prepare_ticks(True).apply()
            if session is not None:
                session.bind_update(self, "_remove_ticks", prepare_ticks)

        # Invert axis
        if self.invert_axis:
            if is_y and not ax.yaxis_inverted():
                ax.invert_yaxis()
            elif not is_y and not ax.xaxis_inverted():
                ax.invert_xaxis()

        if session is not None:
            session.bind_update(self, "_remove_axes", partial(_axis_removal, ax))
            session.bind_axis_view(
                self,
                "_axis_lim",
                "_invert_axis",
                "_log_scale",
                "y" if is_y else "x",
                [(0, ax, self.invert_axis)],
                1,
            )

        # Plotting loop
        num_cycle_colors = len(cycle_colors)
        labels, handles = [], []
        for index, element in enumerate(self._elements):
            if isinstance(element, Plottable):
                params_to_reset = []
                if not is_matplotlib_style:
                    params_to_reset = self._fill_in_missing_params(
                        element, figure_style
                    )

                element._plot_element(
                    ax,
                    z_order,
                    cycle_color=cycle_colors[index % num_cycle_colors],
                )
                if not is_matplotlib_style:
                    self._reset_params_to_default(element, params_to_reset)
                try:
                    if element.label is not None:
                        handles.append(element.handle)
                        labels.append(element.label)
                except AttributeError:
                    continue
                z_order += 5
            elif element is not None:
                raise InvalidParameterTypeError(
                    f"Unsupported element type: {type(element).__name__}."
                )

        return labels, handles

    def _customize_ticks(
        self,
        is_y: bool,
    ) -> None:
        """
        Customizes the ticks of the specified Axes according to the SmartTwinAxis's tick parameters.
        """
        axes = self._axes
        assert axes is not None
        if is_y:
            ax_set_ticks, axis_str, ax_axis = (
                axes.set_yticks,
                "y",
                axes.yaxis,
            )
        else:
            ax_set_ticks, axis_str, ax_axis = (
                axes.set_xticks,
                "x",
                axes.xaxis,
            )

        ticks = self._ticks.get("ticks")
        if ticks is not None:
            tick_labels = self._ticks.get("tick_labels")
            if callable(tick_labels):
                # Apply the callable to each tick
                tick_labels = [tick_labels(tick) for tick in ticks]
            ax_set_ticks(ticks, tick_labels)

        axes.tick_params(axis=axis_str, which="major", **self._tick_params["major"])

        tick_spacing = self._ticks.get("tick_spacing")
        if tick_spacing is not None:
            ax_axis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            # If a callable is provided for tick_labels, apply it with a FuncFormatter
            tick_labels = self._ticks.get("tick_labels")
            if callable(tick_labels):
                ax_axis.set_major_formatter(
                    ticker.FuncFormatter(lambda pos, x: tick_labels(pos))
                )

        minor_ticks = self._ticks.get("minor_ticks")
        if minor_ticks is not None:
            ax_set_ticks(minor_ticks, minor=True)
        axes.tick_params(axis=axis_str, which="minor", **self._tick_params["minor"])
        minor_tick_spacing = self._ticks.get("minor_tick_spacing")
        if minor_tick_spacing is not None:
            ax_axis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))

    def _fill_in_missing_params(
        self,
        element: SmartFigure | SmartTwinAxis | Plottable,
        figure_style: str | Inherit,
    ) -> list[str]:
        """
        Fills in the missing parameters for a :class:`~graphinglib.Plottable` from the parent's ``figure_style``.
        """
        params_to_reset = []
        object_type = type(element).__name__
        for try_i in range(2):
            try:
                for property_, value in vars(element).items():
                    if is_inherit(value) and not (property_ == "_figure_style"):
                        params_to_reset.append(property_)
                        default_value = self._default_params[object_type][property_]
                        setattr(element, property_, default_value)
                break
            except KeyError as e:
                if try_i == 1:
                    raise StyleFileError(
                        f"There was an error auto updating your {figure_style} style file following the recent "
                        "GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub"
                        " page. In the meantime, you can manually add the following parameter to your "
                        f"{figure_style} style file:\n {e.args[0]}."
                    ) from e
                file_updater = FileUpdater(resolved(figure_style))
                file_updater.update()
                file_loader = FileLoader(resolved(figure_style))
                new_defaults = file_loader.load()
                self._default_params.update(
                    (k, v)
                    for k, v in new_defaults.items()
                    if k not in self._default_params
                )
        return params_to_reset

    def _reset_params_to_default(
        self, element: Plottable | SmartTwinAxis, params_to_reset: list[str]
    ) -> None:
        """
        Resets the parameters that were set to default in the :meth:`~graphinglib.SmartTwinAxis._fill_in_missing_params`
        method.
        """
        for param in params_to_reset:
            setattr(element, param, INHERIT)

    def set_rc_params(
        self,
        rc_params_dict: dict[str, Any] = {},
        reset: bool = False,
    ) -> Self:
        """
        Customize the visual style of the :class:`~graphinglib.SmartTwinAxis`.

        Any rc parameter that is not specified in the dictionary will be set to the default value for the specified
        ``figure_style`` from the parent :class:`~graphinglib.SmartFigure`.

        Parameters
        ----------
        rc_params_dict : dict[str, Any], optional
            Dictionary of rc parameters to update.
            Defaults to empty dictionary.
        reset : bool, optional
            If ``True``, resets all previously set rc parameters to their default values for the specified
            ``figure_style`` before applying the new parameters.
            Defaults to ``False``.

        Returns
        -------
        Self
            For convenience, the same SmartTwinAxis with the updated rc parameters.
        """
        _reject_animated_layout(self)
        if reset:
            self._user_rc_dict.clear()
            self._edge_color = None
            self._line_width = None
        for property_, value in rc_params_dict.items():
            self._user_rc_dict[property_] = value
        return self

    def set_visual_params(
        self,
        reset: bool = False,
        edge_color: str | None = None,
        label_color: str | None = None,
        label_pad: float | None = None,
        line_width: float | None = None,
        font_family: str | None = None,
        font_size: float | None = None,
        font_weight: str | None = None,
        use_latex: bool | None = None,
        hide_spine: bool | None = None,
    ) -> Self:
        """
        Customize the visual style of the twin axis.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set visual parameters to their default values for the specified
            ``figure_style`` before applying the new parameters.
            Defaults to ``False``.
        edge_color : str, optional
            The color of the spine.
        label_color : str, optional
            The color of the label.
        label_pad : float, optional
            The padding between the axis and the label.
            Typical range is ``2`` to ``12``.
        line_width : float, optional
            The width of the spine.
            Typical range is ``0.5`` to ``3``.
        font_family : str, optional
            The font family to use.
        font_size : float, optional
            The font size to use.
            Typical range is ``8`` to ``20``.
        font_weight : str, optional
            The font weight to use.
            Values include ``"normal"``, ``"bold"``, ``"light"``, ``"ultralight"``, ``"heavy"``, and
            ``"black"``.
        use_latex : bool, optional
            Whether or not to use latex.
        hide_spine : bool, optional
            Whether to hide the spine of the axes. This corresponds to the line that forms the border of the plot.

        Returns
        -------
        Self
            For convenience, the same SmartTwinAxis with the updated visual parameters.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        if hide_spine is not None and not isinstance(hide_spine, bool):
            raise InvalidParameterTypeError("hide_spine must be a boolean.")
        rc_params_dict = {
            "axes.labelcolor": label_color,
            "axes.labelpad": label_pad,
            "font.family": font_family,
            "font.size": font_size,
            "font.weight": font_weight,
            "text.usetex": use_latex,
        }

        if _bound_session(self) is not None:
            if reset or any(value is not None for value in rc_params_dict.values()):
                raise UnsupportedFeatureError(
                    "During animation twin set_visual_params supports edge_color, line_width, and hide_spine."
                )
            if edge_color is not None:
                edge_color = plt.rcParams.validate["axes.edgecolor"](edge_color)
            if line_width is not None:
                line_width = plt.rcParams.validate["axes.linewidth"](line_width)
            if edge_color is not None:
                self._edge_color = _stage_live_value(self, "_edge_color", edge_color)
            if line_width is not None:
                self._line_width = _stage_live_value(self, "_line_width", line_width)
            if hide_spine is not None:
                self._hide_spine = _stage_live_value(self, "_hide_spine", hide_spine)
            return self

        if reset:
            self._edge_color = edge_color
            self._line_width = line_width
            for key in rc_params_dict.keys():
                self._user_rc_dict.pop(key, None)
        else:
            if edge_color is not None:
                self._edge_color = edge_color
            if line_width is not None:
                self._line_width = line_width

        rc_params_dict = {
            key: value for key, value in rc_params_dict.items() if value is not None
        }
        self.set_rc_params(rc_params_dict)

        if hide_spine is not None:
            self._hide_spine = hide_spine

        return self

    def set_ticks(
        self,
        reset: bool = False,
        ticks: Iterable[float] | None = None,
        tick_labels: Iterable[str] | Callable[[float], str] | None = None,
        tick_spacing: float | None = None,
        minor_ticks: Iterable[float] | None = None,
        minor_tick_spacing: float | None = None,
    ) -> Self:
        """
        Sets custom ticks and tick labels.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set ticks to their default values before applying the new parameters.
            Defaults to ``False``.
        ticks : Iterable[float], optional
            Tick positions for the axis. If a value is specified, the ``tick_spacing`` parameter must be ``None``.
        tick_labels : Iterable[str] | Callable[[float], str], optional
            Tick labels for the axis. Can be either:

            - An iterable of strings: If a value is specified, the ``ticks`` parameter must also be given. The number of
              tick labels must match the number of ticks.
            - A callable that takes a float (the tick position) and returns a string: Can be used with ``ticks`` to apply
              the function to each tick position, or with ``tick_spacing`` to apply the function to any tick position
              using a custom formatter.
        tick_spacing : float, optional
            Spacing between major ticks on the axis. When a callable ``tick_labels`` is provided with spacing,
            the callable will be used to format all tick labels automatically.
        minor_ticks : Iterable[float], optional
            Minor tick positions for the axis.
        minor_tick_spacing : float, optional
            Spacing between minor ticks on the axis.

        Returns
        -------
        Self
            For convenience, the same SmartTwinAxis with the updated ticks.
        """
        # Normalize iterable tick/label inputs to lists so their lengths can be compared
        # and the values reused (a bare iterator would be exhausted after the first pass).
        _reject_animated_layout(self)
        if ticks is not None:
            ticks = list(ticks)
        if minor_ticks is not None:
            minor_ticks = list(minor_ticks)
        if tick_labels is not None and not callable(tick_labels):
            tick_labels = list(tick_labels)

        # Check if tick labels are provided without ticks or spacing
        has_spacing = tick_spacing is not None
        is_callable = callable(tick_labels)

        if (
            (tick_labels is not None)
            and ticks is None
            and not (has_spacing and is_callable)
        ):
            raise IncompatibleArgumentsError(
                "Ticks position must be specified when ticks labels are specified, "
                "unless a callable is provided with tick spacing."
            )

        if any(
            [
                (ticks is not None) and (tick_spacing is not None),
                (minor_ticks is not None) and (minor_tick_spacing is not None),
            ]
        ):
            raise IncompatibleArgumentsError(
                "Tick spacing and tick positions cannot be set simultaneously."
            )

        if ticks is not None and tick_labels is not None and not callable(tick_labels):
            if len(ticks) != len(tick_labels):
                raise IncompatibleArgumentsError(
                    f"Number of ticks ({len(ticks)}) and number of tick labels ({len(tick_labels)}) must be the same."
                )

        if reset:
            self._ticks.clear()

        params = [
            "ticks",
            "tick_labels",
            "tick_spacing",
            "minor_ticks",
            "minor_tick_spacing",
        ]
        for param in params:
            value = locals()[param]
            if value is not None:
                self._ticks[param] = value

        return self

    def set_tick_params(
        self,
        which: Literal["major", "minor", "both"] = "major",
        reset: bool = False,
        direction: Literal["in", "out", "inout"] | None = None,
        length: float | None = None,
        width: float | None = None,
        color: str | None = None,
        pad: float | None = None,
        label_size: float | str | None = None,
        label_color: str | None = None,
        label_rotation: float | None = None,
        draw_ticks: bool | None = None,
        draw_labels: bool | None = None,
    ) -> Self:
        """
        Sets the tick parameters. These parameters are given to the :meth:`matplotlib.axes.Axes.tick_params` method.

        Parameters
        ----------
        which : {"major", "minor", "both"}, optional
            The ticks to set the parameters for. This method can be called multiple times to set the tick parameters
            specifically for each ticks type.
            Defaults to ``"major"``.
        reset : bool, optional
            If ``True``, resets all previously given tick parameters to their default values before applying the new
            parameters.
            Defaults to ``False``.
        direction : {"in", "out", "inout"}, optional
            The direction of the ticks.
        length : float, optional
            The length of the ticks.
            Typical range is ``2`` to ``10``.
        width : float, optional
            The width of the ticks.
            Typical range is ``0.5`` to ``3``.
        color : str, optional
            The color of the ticks.
        pad : float, optional
            The padding to add between the tick labels and the ticks themselves.
            Typical range is ``2`` to ``10``.
        label_size : float | str, optional
            The font size of the tick labels. This can be a float or a string (e.g. "large").
            Typical range is ``8`` to ``20`` when a float is used.
        label_color : str, optional
            The color of the tick labels.
        label_rotation : float, optional
            The rotation of the tick labels, in degrees.
        draw_ticks : bool, optional
            Whether to draw the ticks of the axis.
        draw_labels : bool, optional
            Whether to draw the tick labels of the axis.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated tick parameters.

        Notes
        -----
        Color parameters accept Matplotlib color formats: named colors (``"blue"``), short color strings
        (``"b"``), hex strings (``"#0000ff"``), grayscale strings (``"0.5"``), and RGB/RGBA tuples with
        values between ``0`` and ``1`` (``(0, 0, 1)`` or ``(0, 0, 1, 0.5)``).
        """
        _reject_animated_layout(self)
        new_tick_params = {
            "direction": direction,
            "length": length,
            "width": width,
            "color": color,
            "pad": pad,
            "labelsize": label_size,
            "labelcolor": label_color,
            "labelrotation": label_rotation,
            "bottom": draw_ticks,
            "top": draw_ticks,
            "left": draw_ticks,
            "right": draw_ticks,
            "labelbottom": draw_labels,
            "labeltop": draw_labels,
            "labelleft": draw_labels,
            "labelright": draw_labels,
        }
        for which_i in [which] if which != "both" else ["major", "minor"]:
            if reset:
                self._tick_params[which_i].clear()
            for param, value in new_tick_params.items():
                if value is not None:
                    self._tick_params[which_i][param] = value
        return self
