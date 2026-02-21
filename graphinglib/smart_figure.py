from __future__ import annotations as _annotations_

from collections import OrderedDict
from copy import deepcopy
from logging import warning
from shutil import which
from string import ascii_lowercase
from typing import Any, Callable, Iterable, Iterator, Literal, Self, TypeVar, Union

try:  # Optional dependency: astropy
    from astropy.units import Quantity
    from astropy.wcs import WCS

    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False
    WCS = type("WCSPlaceholder", (), {})  # type: ignore[assignment]
    Quantity = type("QuantityPlaceholder", (), {})  # type: ignore[assignment]

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure, SubFigure
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.projections import get_projection_names
from matplotlib.transforms import ScaledTranslation
from numpy.typing import ArrayLike

from .file_manager import FileLoader, FileUpdater, get_default_style, get_styles
from .graph_elements import GraphingException, Plottable, Text
from .legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    LegendElement,
    VerticalLineCollection,
    histogram_legend_artist,
)
from .tools import _copy_with_overrides

T = TypeVar("T")
ListOrItem = Union[T, list[T]]


def _require_astropy(feature: str = "this feature") -> None:
    """Raise a clear error when an astro-extra feature is used without the optional dependency installed."""
    if not _ASTROPY_AVAILABLE:
        raise GraphingException(
            f"{feature} requires the optional `graphinglib[astro]` extra (installs Astropy). "
            "Install it with `pip install graphinglib[astro]`."
        )


HAS_ASTROPY = _ASTROPY_AVAILABLE


class SmartFigure:
    """
    This class implements a figure object for plotting :class:`~graphinglib.Plottable` elements.

    It allows for the creation of complex figures recursively, where each :class:`~graphinglib.SmartFigure` can contain
    other :class:`~graphinglib.SmartFigure` objects. The class supports a variety of customization options as well as
    the ability to use styles and themes for consistent visual appearance across different figures. The idea behind this
    class is that every SmartFigure contains a single x_label, y_label, title, projection, etc. and that nested
    SmartFigures can be inserted into the main SmartFigure to create complex figures with more parameters.

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
        Default depends on the ``figure_style`` configuration.
    title : str, optional
        General title of the figure.
    x_lim, y_lim : tuple[float, float] | list[tuple[float, float]], optional
        Limits for the x and y axes of the figure. This can be given as a single value or a list of values to apply to
        each subplot.
    sub_x_labels, sub_y_labels : Iterable[str], optional
        Labels for the x and y axes of each subfigure, respectively. This is only useful for figures that are not a
        single subplot and when each subfigure needs its own x and y labels. This prevents the creation of nested
        :class:`~graphinglib.SmartFigure` objects for each subfigure only to set the x and y labels. This list cannot
        be longer than the number of non-empty subplots and None values can be used to skip specific subplots.
    subtitles : Iterable[str], optional
        Labels for the subtitles of each subfigure, respectively. Similarly to `sub_x_labels` and `sub_y_labels`, this
        allows to set subtitles for each subfigure without needing to create nested
        :class:`~graphinglib.SmartFigure` objects. It is only useful for figures that are not a single subplot and when
        each subfigure needs its own subtitle. This list cannot be longer than the number of non-empty subplots and None
        values can be used to skip specific subplots.
    log_scale_x, log_scale_y : bool | list[bool], optional
        Whether to use a logarithmic scale for the x and y axes, respectively. This can be given as a single value or
        a list of values to apply to each subplot.
        Defaults to ``False``.
    remove_axes : bool | list[bool], optional
        Whether to remove the axes from the figure. This can be given as a single boolean or a list of booleans to apply
        to each subplot.
        Defaults to ``False``.
    aspect_ratio : float | Literal["auto", "equal"] | list[float | Literal["auto", "equal"]], optional
        Aspect ratio of the figure. If set to "auto", the aspect ratio is determined automatically to fill the available
        space. If set to "equal", the aspect ratio is set to 1:1. If set to a float, the aspect ratio represents the
        ratio of the height to the width of the data. This can be given as a single value or a list of values to apply
        to each subplot.
        Defaults to "auto".

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

    remove_x_ticks, remove_y_ticks : bool | list[bool], optional
        Whether to remove the x and y ticks from the figure, respectively. This can be given as a single value or a list
        of values to apply to each subplot.
        Defaults to ``False``.
    invert_x_axis, invert_y_axis : bool | list[bool], optional
        Whether to invert the x and y axes, respectively. This can be given as a single value or a list of values to
        apply to each subplot.
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
        Location of the reference labels of the SubFigures, which can be either "inside", "outside" or a tuple of
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
            Sharing axes only works for plots directly inside the SmartFigure. If a nested SmartFigure is used, the
            axes sharing will not be applied to the nested SmartFigure. Instead, the nested SmartFigure will have its
            own axes sharing settings.

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
    show_legend : bool | list[bool], optional
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
    elements : Iterable[Plottable | SmartFigure] | Iterable[Iterable[Plottable | SmartFigure]], optional
        The elements to plot in the figure.
        If an iterable of depth 1 is provided and the figure is 1x1, all the elements are added to the unique plot. For
        other geometries, the elements are added one by one in the order they are provided to each subplot, and the
        iterable should not be longer than the number of subplots.
        If an iterable of depth 2 is provided, each sub-iterable is added to the corresponding subplot, in the order
        they are provided. The number of sub-iterables should be equal to the number of subplots.
        If ``None`` elements are present in the iterable, the corresponding subplots are not drawn and a blank space is
        left in the figure. If iterables containing only ``None`` are given in the main iterable, the corresponding
        subplots are drawn but empty.

        .. note::
            This method for adding elements only allows to add elements to single subplots. If you want to add elements
            that span multiple subplots, you should use the __setitem__ method instead.
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
        size: tuple[float, float] | Literal["default"] = "default",
        title: str | None = None,
        x_lim: ListOrItem[tuple[float, float] | None] = None,
        y_lim: ListOrItem[tuple[float, float] | None] = None,
        sub_x_labels: Iterable[str] | None = None,
        sub_y_labels: Iterable[str] | None = None,
        subtitles: Iterable[str] | None = None,
        log_scale_x: ListOrItem[bool] = False,
        log_scale_y: ListOrItem[bool] = False,
        remove_axes: ListOrItem[bool] = False,
        aspect_ratio: ListOrItem[float | Literal["auto", "equal"]] = "auto",
        box_aspect_ratio: ListOrItem[float | None] = None,
        remove_x_ticks: ListOrItem[bool] = False,
        remove_y_ticks: ListOrItem[bool] = False,
        invert_x_axis: ListOrItem[bool] = False,
        invert_y_axis: ListOrItem[bool] = False,
        reference_labels: ListOrItem[bool] = True,
        global_reference_label: bool = False,
        reference_labels_loc: ListOrItem[
            Literal["inside", "outside"] | tuple[float, float]
        ] = "outside",
        width_padding: float = None,
        height_padding: float = None,
        width_ratios: ArrayLike = None,
        height_ratios: ArrayLike = None,
        share_x: bool = False,
        share_y: bool = False,
        projection: ListOrItem[Any | None] = None,
        general_legend: bool = False,
        legend_loc: ListOrItem[str | tuple | None] = None,
        legend_cols: ListOrItem[int] = 1,
        show_legend: ListOrItem[bool] = True,
        twin_x_axis: SmartTwinAxis | None = None,
        twin_y_axis: SmartTwinAxis | None = None,
        figure_style: str = "default",
        elements: Iterable[Plottable | SmartFigure | None]
        | Iterable[Iterable[Plottable | None]] = [],
        annotations: Iterable[Text] | None = None,
    ) -> None:
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
        self.elements = elements
        self.annotations = annotations

        self._figure = None
        self._gridspec = None
        self._reference_label_i = None

        self._ticks = {}
        self._tick_params = {"x major": {}, "y major": {}, "x minor": {}, "y minor": {}}
        self._pad_params = {}
        self._reference_labels_params = {}

        self.show_grid = False
        self._grid = {}

        self.hide_custom_legend_elements = False
        self.hide_default_legend_elements = False
        self._custom_legend_handles = []
        self._custom_legend_labels = []

        self._hidden_spines = None
        self._user_rc_dict = {}
        self._default_params = {}
        self._subplot_p = {}  # used to store the ListOrItem parameters that can be different for each subplot

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @num_rows.setter
    def num_rows(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("num_rows must be an integer.")
        if value < 1:
            raise ValueError("num_rows must be greater than 0.")
        # Check if the number of rows is being reduced and conflicts with existing elements
        try:
            if self._num_rows > value:
                removed_rows = list(range(value, self._num_rows))
                for pos, element in self._elements.items():
                    if (pos[0].stop - 1) in removed_rows and element:
                        raise GraphingException(
                            "Cannot remove rows from the SmartFigure when there are elements in "
                            "them. Please remove the elements first."
                        )
        except AttributeError:
            # The figure is being created, so the _num_rows attribute is not yet set
            pass
        self._num_rows = value

    @property
    def num_cols(self) -> int:
        return self._num_cols

    @num_cols.setter
    def num_cols(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("num_cols must be an integer.")
        if value < 1:
            raise ValueError("num_cols must be greater than 0.")
        # Check if the number of rows is being reduced and conflicts with existing elements
        try:
            if self._num_cols > value:
                removed_cols = list(range(value, self._num_cols))
                for pos, element in self._elements.items():
                    if (pos[1].stop - 1) in removed_cols and element:
                        raise GraphingException(
                            "Cannot remove cols from the SmartFigure when there are elements in "
                            "them. Please remove the elements first."
                        )
        except AttributeError:
            # The figure is being created, so the _num_cols attribute is not yet set
            pass
        self._num_cols = value

    @property
    def shape(self) -> tuple[int, int]:
        return self._num_rows, self._num_cols

    @shape.setter
    def shape(self, value: tuple[int, int]) -> None:
        self.num_rows, self.num_cols = value

    @property
    def x_label(self) -> str | None:
        return self._x_label

    @x_label.setter
    def x_label(self, value: str) -> None:
        self._x_label = value

    @property
    def y_label(self) -> str | None:
        return self._y_label

    @y_label.setter
    def y_label(self, value: str) -> None:
        self._y_label = value

    @property
    def size(self) -> tuple[float, float] | Literal["default"]:
        return self._size

    @size.setter
    def size(self, value: tuple[float, float] | Literal["default"]):
        if not isinstance(value, tuple) and value != "default":
            raise TypeError("size must be a tuple or 'default'.")
        if isinstance(value, tuple) and len(value) != 2:
            raise ValueError("size must be a tuple of length 2.")
        if isinstance(value, tuple) and (value[0] <= 0 or value[1] <= 0):
            raise ValueError("size values must be greater than 0.")
        self._size = value

    @property
    def title(self) -> str | None:
        return self._title

    @title.setter
    def title(self, value: str | None) -> None:
        self._title = value

    @property
    def x_lim(self) -> ListOrItem[tuple[float, float] | None]:
        return self._x_lim

    @x_lim.setter
    def x_lim(self, value: ListOrItem[tuple[float, float] | None]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if not isinstance(v, tuple):
                    raise TypeError("x_lim must be a tuple.")
                if len(v) != 2:
                    raise ValueError("x_lim must be a tuple of length 2.")
        self._x_lim = value

    @property
    def y_lim(self) -> ListOrItem[tuple[float, float] | None]:
        return self._y_lim

    @y_lim.setter
    def y_lim(self, value: ListOrItem[tuple[float, float] | None]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if not isinstance(v, tuple):
                    raise TypeError("y_lim must be a tuple.")
                if len(v) != 2:
                    raise ValueError("y_lim must be a tuple of length 2.")
        self._y_lim = value

    @property
    def sub_x_labels(self) -> Iterable[str] | None:
        return self._sub_x_labels

    @sub_x_labels.setter
    def sub_x_labels(self, value: Iterable[str] | None) -> None:
        if value is not None:
            if not isinstance(value, Iterable):
                raise TypeError("sub_x_labels must be an iterable of strings.")
        self._sub_x_labels = value

    @property
    def sub_y_labels(self) -> Iterable[str] | None:
        return self._sub_y_labels

    @sub_y_labels.setter
    def sub_y_labels(self, value: Iterable[str] | None) -> None:
        if value is not None:
            if not isinstance(value, Iterable):
                raise TypeError("sub_y_labels must be an iterable of strings.")
        self._sub_y_labels = value

    @property
    def subtitles(self) -> Iterable[str] | None:
        return self._subtitles

    @subtitles.setter
    def subtitles(self, value: Iterable[str] | None) -> None:
        if value is not None:
            if not isinstance(value, Iterable):
                raise TypeError("subtitles must be an iterable of strings.")
        self._subtitles = value

    @property
    def log_scale_x(self) -> ListOrItem[bool]:
        return self._log_scale_x

    @log_scale_x.setter
    def log_scale_x(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("log_scale_x must be a bool.")
        self._log_scale_x = value

    @property
    def log_scale_y(self) -> ListOrItem[bool]:
        return self._log_scale_y

    @log_scale_y.setter
    def log_scale_y(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("log_scale_y must be a bool.")
        self._log_scale_y = value

    @property
    def remove_axes(self) -> ListOrItem[bool]:
        return self._remove_axes

    @remove_axes.setter
    def remove_axes(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("remove_axes must be a bool.")
        self._remove_axes = value

    @property
    def aspect_ratio(self) -> ListOrItem[float | Literal["auto", "equal"]]:
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, value: ListOrItem[float | Literal["auto", "equal"]]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, (float, int)) and v != "auto" and v != "equal":
                raise TypeError("aspect_ratio must be a float, 'auto' or 'equal'.")
            if isinstance(v, (float, int)) and v <= 0:
                raise ValueError("aspect_ratio must be greater than 0.")
        self._aspect_ratio = value

    @property
    def box_aspect_ratio(self) -> ListOrItem[float | None]:
        return self._box_aspect_ratio

    @box_aspect_ratio.setter
    def box_aspect_ratio(self, value: ListOrItem[float | None]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if not isinstance(v, (float, int)):
                    raise TypeError("box_aspect_ratio must be a number.")
                if v <= 0:
                    raise ValueError("box_aspect_ratio must be greater than 0.")
        self._box_aspect_ratio = value

    @property
    def remove_x_ticks(self) -> ListOrItem[bool]:
        return self._remove_x_ticks

    @remove_x_ticks.setter
    def remove_x_ticks(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("remove_x_ticks must be a bool.")
        self._remove_x_ticks = value

    @property
    def remove_y_ticks(self) -> ListOrItem[bool]:
        return self._remove_y_ticks

    @remove_y_ticks.setter
    def remove_y_ticks(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("remove_y_ticks must be a bool.")
        self._remove_y_ticks = value

    @property
    def invert_x_axis(self) -> ListOrItem[bool]:
        return self._invert_x_axis

    @invert_x_axis.setter
    def invert_x_axis(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("invert_x_axis must be a bool.")
        self._invert_x_axis = value

    @property
    def invert_y_axis(self) -> ListOrItem[bool]:
        return self._invert_y_axis

    @invert_y_axis.setter
    def invert_y_axis(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("invert_y_axis must be a bool.")
        self._invert_y_axis = value

    @property
    def reference_labels(self) -> ListOrItem[bool]:
        return self._reference_labels

    @reference_labels.setter
    def reference_labels(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("reference_labels must be a bool.")
        self._reference_labels = value

    @property
    def global_reference_label(self) -> bool:
        return self._global_reference_label

    @global_reference_label.setter
    def global_reference_label(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("global_reference_label must be a bool.")
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
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, tuple):
                if len(v) != 2:
                    raise ValueError(
                        "If reference_labels_loc is a tuple, it must be of length 2."
                    )
            elif v not in ["inside", "outside"]:
                raise ValueError(
                    "reference_labels_loc must be either 'inside' or 'outside'."
                )
        self._reference_labels_loc = value

    @property
    def width_padding(self) -> float | None:
        return self._width_padding

    @width_padding.setter
    def width_padding(self, value: float | None) -> None:
        if value is not None:
            if not isinstance(value, (float, int)):
                raise TypeError("width_padding must be a number.")
            if value < 0:
                raise ValueError("width_padding must be greater than or equal to 0.")
        self._width_padding = value

    @property
    def height_padding(self) -> float | None:
        return self._height_padding

    @height_padding.setter
    def height_padding(self, value: float | None) -> None:
        if value is not None:
            if not isinstance(value, (float, int)):
                raise TypeError("height_padding must be a number.")
            if value < 0:
                raise ValueError("height_padding must be greater than or equal to 0.")
        self._height_padding = value

    @property
    def width_ratios(self) -> ArrayLike | None:
        return self._width_ratios

    @width_ratios.setter
    def width_ratios(self, value: ArrayLike | None) -> None:
        if value is not None:
            if not hasattr(value, "__len__"):
                raise TypeError("width_ratios must be an ArrayLike.")
            if not all(isinstance(x, (float, int)) for x in value):
                raise TypeError("width_ratios must contain only numbers.")
            if len(value) != self._num_cols:
                raise ValueError("width_ratios must have the same length as num_cols.")
        self._width_ratios = value

    @property
    def height_ratios(self) -> ArrayLike | None:
        return self._height_ratios

    @height_ratios.setter
    def height_ratios(self, value: ArrayLike | None) -> None:
        if value is not None:
            if not hasattr(value, "__len__"):
                raise TypeError("height_ratios must be an ArrayLike.")
            if not all(isinstance(x, (float, int)) for x in value):
                raise TypeError("height_ratios must contain only numbers.")
            if len(value) != self._num_rows:
                raise ValueError("height_ratios must have the same length as num_rows.")
        self._height_ratios = value

    @property
    def share_x(self) -> bool:
        return self._share_x

    @share_x.setter
    def share_x(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("share_x must be a bool.")
        self._share_x = value

    @property
    def share_y(self) -> bool:
        return self._share_y

    @share_y.setter
    def share_y(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("share_y must be a bool.")
        self._share_y = value

    @property
    def projection(self) -> ListOrItem[Any | None]:
        return self._projection

    @projection.setter
    def projection(self, value: ListOrItem[Any | None]) -> None:
        valid_projections = get_projection_names()
        if "3d" in valid_projections:
            valid_projections.remove("3d")
        for v in value if isinstance(value, list) else [value]:
            if v is not None:
                if isinstance(v, str):
                    if v == "3d":
                        raise GraphingException("3D projection is not supported.")
                    if v not in valid_projections:
                        raise ValueError(
                            f"projection must be one of {valid_projections} or a valid object."
                        )
                elif isinstance(v, WCS):
                    raise GraphingException(
                        "WCS projection should be used with the SmartFigureWCS object."
                    )
        self._projection = value

    @property
    def general_legend(self) -> bool:
        return self._general_legend

    @general_legend.setter
    def general_legend(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("general_legend must be a bool.")
        self._general_legend = value

    @property
    def legend_loc(self) -> ListOrItem[str | tuple | None]:
        return self._legend_loc

    @legend_loc.setter
    def legend_loc(self, value: ListOrItem[str | tuple | None]) -> None:
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
                        raise ValueError(f"legend_loc must be one of {choices}.")
                    if self._general_legend and v == "best":
                        raise ValueError(
                            "legend_loc cannot be 'best' when general_legend is True."
                        )
                elif isinstance(v, tuple):
                    if len(v) != 2:
                        raise ValueError(
                            "legend_loc must be a string or a tuple of length 2."
                        )
                else:
                    raise TypeError("legend_loc must be a string or tuple.")
        self._legend_loc = value

    @property
    def legend_cols(self) -> ListOrItem[int]:
        return self._legend_cols

    @legend_cols.setter
    def legend_cols(self, value: ListOrItem[int]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, int):
                raise TypeError("legend_cols must be an integer.")
            if v < 1:
                raise ValueError("legend_cols must be greater than 0.")
        self._legend_cols = value

    @property
    def show_legend(self) -> ListOrItem[bool]:
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("show_legend must be a bool.")
        self._show_legend = value

    @property
    def twin_x_axis(self) -> SmartTwinAxis | None:
        return self._twin_x_axis

    @twin_x_axis.setter
    def twin_x_axis(self, value: SmartTwinAxis | None) -> None:
        if value is not None:
            if not self.is_single_subplot:
                raise GraphingException(
                    "Twin axes can only be created for single subplot SmartFigures."
                )
            if not isinstance(value, SmartTwinAxis):
                raise TypeError("twin_x_axis must be a SmartTwinAxis instance.")
        self._twin_x_axis = value

    @property
    def twin_y_axis(self) -> SmartTwinAxis | None:
        return self._twin_y_axis

    @twin_y_axis.setter
    def twin_y_axis(self, value: SmartTwinAxis | None) -> None:
        if value is not None:
            if not self.is_single_subplot:
                raise GraphingException(
                    "Twin axes can only be created for single subplot SmartFigures."
                )
            if not isinstance(value, SmartTwinAxis):
                raise TypeError("twin_y_axis must be a SmartTwinAxis instance.")
        self._twin_y_axis = value

    @property
    def figure_style(self) -> str:
        return self._figure_style

    @figure_style.setter
    def figure_style(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("figure_style must be a string.")
        available_styles = ["default", "matplotlib"] + get_styles(matplotlib=True)
        if value not in available_styles:
            raise ValueError(f"figure_style must be one of {available_styles}.")
        self._figure_style = value

    @property
    def elements(self) -> dict[tuple[slice, slice], Plottable | SmartFigure]:
        return self._elements

    @elements.setter
    def elements(
        self,
        value: Iterable[Plottable | SmartFigure | None]
        | Iterable[Iterable[Plottable | None]],
    ) -> None:
        """
        Sets the elements of the SmartFigure with the same rules as the constructor. For adding elements instead of
        replacing them, use the :meth:`~graphinglib.SmartFigure.add_elements` or the
        :meth:`~graphinglib.SmartFigure.__setitem__` methods.
        """
        self._elements = {}  # systematically reset the elements when setting them with the property
        self.add_elements(*value)

    @property
    def annotations(self) -> Iterable[Text] | None:
        return self._annotations

    @annotations.setter
    def annotations(self, value: Iterable[Text] | None) -> None:
        if value is not None:
            if not isinstance(value, Iterable) or not all(
                isinstance(t, Text) for t in value
            ):
                raise TypeError("annotations must be an iterable of Text elements.")
        self._annotations = value

    @property
    def show_grid(self) -> ListOrItem[bool]:
        """
        Whether to show the grid lines on the figure. A grid first needs to be created using the
        :meth: `~graphinglib.SmartFigure.set_grid` method. This can be used to easily toggle the visibility of a
        previously created grid.
        """
        return self._show_grid

    @show_grid.setter
    def show_grid(self, value: ListOrItem[bool]) -> None:
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("show_grid must be a bool.")
        self._show_grid = value

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
        if not isinstance(value, bool):
            raise TypeError("hide_custom_legend_elements must be a bool.")
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
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, bool):
                raise TypeError("hide_default_legend_elements must be a bool.")
        self._hide_default_legend_elements = value

    @property
    def is_single_subplot(self) -> bool:
        """
        Whether the SmartFigure is a single subplot (1x1). This is useful to determine if the SmartFigure can be used
        as a single plot or if it contains multiple subplots.

        .. note::
            This property is used to verify if custom legend elements can be added to the SmartFigure even if the
            :attr:`~graphinglib.SmartFigure.general_legend` is set to ``False``.
        """
        return self.num_rows == 1 and self.num_cols == 1

    def __len__(self) -> int:
        """
        Gives the number of non-empty subplots in the :class:`~graphinglib.SmartFigure`.
        """
        return len(self._elements)

    def __setitem__(
        self,
        key: int | slice | tuple[int | slice],
        element: Plottable | Iterable[Plottable | None] | SmartFigure | None,
    ) -> None:
        """
        Assigns a Plottable, a list of Plottable objects, or a SmartFigure to a specified position in the SmartFigure.
        The indexing follows classical 2D numpy-like indexing, where the first element corresponds to the row and the
        second element corresponds to the column.

        Parameters
        ----------
        key : int | slice | tuple[int | slice]
            The key specifying the location(s) in the SmartFigure to assign the element(s). If a tuple of ints is
            provided, the element is placed in the corresponding square of the grid, following classical 2D numpy-like
            indexing. If slices are provided, the element can span multiple squares in the grid. If ``num_rows`` or
            ``num_cols`` is set to 1, the key can be a single int or slice. Otherwise, the key must be a two-tuple.
        element : Plottable | Iterable[Plottable | None] | SmartFigure | None
            The element(s) to assign. Must be a Plottable, an iterable of Plottable objects, or a SmartFigure. If None,
            the element at the specified key will be removed. Note that the exact slice used for inserting Plottables or
            a SmartFigure must be provided to remove it.

            .. note::
                It is also possible to add a list of Plottables to a subplot already containing Plottables using the
                ``+=`` operator.

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

        We can add elements using the ``+=`` operator and remove them using ``None``::

            fig[0, 0] += [gl.Curve(x2, y2)]
            fig[0, 1] = None

        Which will result in the following layout::

            +------------+------------+
            | 0,0        | 0,1        |
            | Curve      |            |
            | Curve      |            |
            +------------+------------+
            | 1,0          1,1        |
            | Histogram               |
            +-------------------------+

        We can also insert a nested SmartFigure into a specific region of the SmartFigure and remove the bottom plot::

            subfigure = SmartFigure(num_rows=2, num_cols=1)
            subfigure.add_elements(gl.Heatmap(data1), gl.Heatmap(data2))
            fig[0, 1] = subfigure
            fig[1, :] = None

        Which will lead to the following layout::

            +------------+------------+
            | 0,0        | Heatmap    |
            | Curve      +------------+
            |            | Heatmap    |
            +------------+------------+
            | 1,0        | 1,1        |
            +------------+------------+
        """
        if not any(
            [
                element is None,
                isinstance(element, (Plottable, SmartFigure)),
                SmartFigure._is_iterable_of_plottables(element),
            ]
        ):
            raise TypeError(
                "Element must be a Plottable, an iterable of Plottables, or a SmartFigure."
            )
        key_ = self._validate_and_normalize_key(key)
        if element is None:
            self._elements.pop(key_, None)
        else:
            # Normalize all iterables to lists for consistency
            if isinstance(element, Plottable):
                el = [element]
            elif isinstance(element, Iterable) and not isinstance(element, SmartFigure):
                el = list(element)
            else:
                el = element
            self._elements[key_] = el

    def __getitem__(
        self, key: int | slice | tuple[int | slice]
    ) -> list[Plottable] | SmartFigure:
        """
        Gives the element(s) at the specified key in the SmartFigure. This can be used to modify or extract directly an
        element in the SmartFigure. The indexing follows classical 2D numpy-like indexing, where the first element
        corresponds to the row and the second element corresponds to the column.

        Parameters
        ----------
        key : int | slice | tuple[int | slice]
            The key specifying the location(s) in the SmartFigure to access. If a tuple of ints is provided, the element
            is accessed in the corresponding square of the grid, following classical 2D numpy-like indexing. If slices
            are provided, an element spanning multiple squares in the grid can be retrieved. If ``num_rows`` or
            ``num_cols`` is set to 1, the key can be a single int or slice to index into the single row or column.
            Otherwise, the key must be a two-tuple.

            .. note::
                The exact slice of the element must be provided to access it. This means that if an element spans
                multiple subplots, the given slice also needs to span these subplots.

        Returns
        -------
        list[Plottable] | SmartFigure
            The element(s) at the specified key, which can be a list of Plottables or a SmartFigure. If there is no
            elements at the given key, an empty list is returned.
        """
        key_ = self._validate_and_normalize_key(key)
        return self._elements.get(key_, [])

    def __iter__(self) -> Iterator[list[Plottable] | SmartFigure]:
        """
        Iterates over the elements in the SmartFigure in order of their position in the grid, from top-left to
        bottom-right.
        """
        for element in self._ordered_elements.values():
            yield element

    def __deepcopy__(self, memo: dict) -> Self:
        """
        Creates a deep copy of the SmartFigure instance, intentionally excluding the '_figure' and '_gridspec'
        attributes from the copy. These attributes are matplotlib objects and are not duplicated to avoid issues with
        copying live figure state.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        excluded_attrs = ["_figure", "_gridspec"]
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
        return _copy_with_overrides(self, **kwargs)

    @property
    def _ordered_elements(self) -> OrderedDict:
        """
        Gives the _elements dict sorted by the starting position of the slices. This is used to ensure that the
        elements are plotted in the correct order when creating the figure.
        """
        return OrderedDict(
            sorted(
                self._elements.items(),
                key=lambda item: (item[0][0].start, item[0][1].start),
            )
        )

    def _validate_and_normalize_key(
        self, key: int | slice | tuple[int | slice]
    ) -> tuple[slice]:
        """
        Validates and normalizes the key for indexing into the SmartFigure. This method ensures that the key is
        either a single integer, a slice, or a tuple of integers/slices. It also checks for out-of-bounds indices and
        raises appropriate exceptions if the key is invalid. The returned key is always a tuple of two slices without
        None or negative indices..

        Parameters
        ----------
        key : int | slice | tuple[int | slice]
            The key to validate and normalize.

        Returns
        -------
        tuple[slice]
            The normalized key as a two-tuple of slices.
        """
        if not isinstance(key, tuple):
            key = (key,)

        # 1D SmartFigures
        if self._num_rows == 1 or self._num_cols == 1:
            if len(key) == 1:
                key = (0, key[0]) if self._num_rows == 1 else (key[0], 0)
            elif len(key) != 2:
                raise ValueError(
                    "Key must be 1D (int or slice) or 2D with one zero index for 1D SmartFigure."
                )

        # 2D SmartFigures
        else:
            if len(key) != 2:
                raise ValueError("2D indexing must use a tuple of length 2.")

        # Bounds check
        new_keys = []
        for i, (k, axis_size) in enumerate(zip(key, (self._num_rows, self._num_cols))):
            if isinstance(k, int):
                new_k = k + axis_size if k < 0 else k
                if not (0 <= new_k < axis_size):
                    raise IndexError(
                        f"Index {k} out of bounds for axis {i} with size {axis_size}."
                    )
                new_keys.append(slice(new_k, new_k + 1, None))
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                start = start + axis_size if start < 0 else start
                stop = k.stop if k.stop is not None else axis_size
                stop = stop + axis_size if stop < 0 else stop
                if start < 0 or stop > axis_size:
                    raise IndexError(
                        f"{k} out of bounds for axis {i} with size {axis_size}."
                    )
                if start >= stop:
                    raise IndexError(
                        f"{k} for axis {i} must have stop larger than start."
                    )
                if k.step is not None:
                    raise ValueError(f"{k} step for axis {i} must be None.")
                new_keys.append(slice(start, stop, None))
            else:
                raise TypeError(
                    f"Key element {k} for axis {i} must be an int or a slice."
                )
        return tuple(new_keys)

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
        return isinstance(item, Iterable) and all(
            isinstance(el, (Plottable, type(None))) for el in item
        )

    def add_elements(
        self,
        *elements: Plottable | SmartFigure | None | Iterable[Plottable | None],
    ) -> Self:
        """
        Adds one or more :class:`~graphinglib.Plottable` or :class:`~graphinglib.SmartFigure` to the current
        SmartFigure. This method is equivalent to using the :meth:`~graphinglib.SmartFigure.__setitem__` method, but can
        only add elements spanning single subplots.

        Parameters
        ----------
        elements : Plottable | SmartFigure | Iterable[Plottable | SmartFigure]
            Elements to plot in the :class:`~graphinglib.SmartFigure`. Each given element is added in turn to each
            subplot in the order they are provided. Iterables of :class:`~graphinglib.Plottable` objects can be provided
            to add multiple elements in the same subplot. The number of provided elements must be at most the number of
            subplots unless the :class:`~graphinglib.SmartFigure` is a single subplot, in which case all elements are
            added to the unique plot. If ``None`` elements are present, the corresponding subplot is skipped and not
            drawn. If iterables containing only ``None`` are given, the corresponding subplots are drawn but will appear
            empty.

            .. note::
                This method for adding elements only allows to add elements to single subplots. If you want to add
                elements that span multiple subplots, you should use the :meth:`~graphinglib.SmartFigure.__setitem__`
                method instead.

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
        if len(elements) > 0:
            if (
                len(elements) > self._num_cols * self._num_rows
                and not self.is_single_subplot
            ):
                raise ValueError(
                    "Too many elements provided for the number of subplots."
                )
            # Check the type of each element
            for i, element in enumerate(elements):
                index = (0, 0) if self.is_single_subplot else divmod(i, self._num_cols)
                if isinstance(element, Plottable):
                    self[index] += [element]
                elif SmartFigure._is_iterable_of_plottables(element):
                    self[index] += list(element)
                elif isinstance(element, SmartFigure):
                    self[index] = element
                elif element is not None:
                    raise TypeError(
                        f"Element at index {i} must be a Plottable, an iterable of Plottables, or a SmartFigure, "
                        f"not {type(element).__name__}."
                    )
        return self

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
            plt.get_current_fig_manager().full_screen_toggle()

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
                    subfig = self.copy_with(elements=[element], num_rows=1, num_cols=1)
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

    def _initialize_parent_smart_figure(
        self,
    ) -> None:
        """
        Initializes the parent :class:`~graphinglib.SmartFigure` for plotting. This method initializes the appropriate
        figure style, parameters and matplotlib figure and calls the :meth:`~graphinglib.SmartFigure._prepare_figure`
        method.
        """
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

        parent_figure_params_to_reset = self._fill_in_missing_params(
            self
        )  # Fill "default" parameters
        self._default_params["rc_params"].update(
            self._user_rc_dict
        )  # Custom rc parameters supersede the defaults
        self._fill_in_rc_params(is_matplotlib_style)

        # The following try/except removes lingering figures when errors occur during the plotting process
        try:
            self._figure = plt.figure(constrained_layout=True, figsize=self._size)
            self._figure.get_layout_engine().set(w_pad=0, h_pad=0)
            self._reference_label_i = self._reference_labels_params.get(
                "start_index", 0
            )
            self._prepare_figure(is_matplotlib_style)
            self._figure.canvas.draw()
            self._align_shared_x_spines()
        except Exception as e:
            plt.close()
            raise e

        self._reset_params_to_default(self, parent_figure_params_to_reset)
        self._default_params = {}

    def _prepare_figure(
        self,
        is_matplotlib_style: bool = False,
        make_legend: bool = True,
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
                    raise GraphingException(
                        f"When using a general legend, the '{param_name}' property must be a single value, not a list."
                    )

        # Get the normalized list or item parameter dict
        self._fill_per_subplot_params()

        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        num_cycle_colors = len(cycle_colors)
        subtitles_pad = self._subplot_p["subtitles_pad"]

        self._gridspec = self._figure.add_gridspec(
            self._num_rows,
            self._num_cols,
            wspace=self._width_padding,
            hspace=self._height_padding,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
        )

        if self._global_reference_label:
            self._create_reference_label(self._figure)
            self._figure.suptitle(" ")  # Create a blank title to reserve space

        ax = None  # keep track of the last plt.Axes object, needed for sharing axes
        default_labels, default_handles = [], []
        custom_labels, custom_handles = [], []

        # Plottable and subfigure plotting
        for subplot_i, ((rows, cols), element) in enumerate(
            self._ordered_elements.items()
        ):
            if isinstance(element, SmartFigure):
                element._default_params = deepcopy(self._default_params)
                subfig_params_to_reset = []
                parent_rc_params = None
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

                subfig = self._figure.add_subfigure(self._gridspec[rows, cols])
                element._figure = subfig  # associates the current subfigure with the nested SmartFigure
                element._reference_label_i = self._reference_label_i
                legend_info = element._prepare_figure(
                    is_matplotlib_style=is_matplotlib_style,
                    make_legend=(not self._general_legend and make_legend),
                )

                self._reference_label_i = element._reference_label_i
                default_labels += legend_info["labels"]["default"]
                default_handles += legend_info["handles"]["default"]
                custom_labels += legend_info["labels"]["custom"]
                custom_handles += legend_info["handles"]["custom"]

                if is_matplotlib_style:
                    plt.rcParams.update(parent_rc_params)
                else:
                    plt.rcParams.update(
                        self._default_params["rc_params"]
                    )  # Return to the parent SmartFigure's rc params
                    element._reset_params_to_default(element, subfig_params_to_reset)
                element._default_params = {}
                for param, param_was_none in zip(
                    ["x_label", "y_label", "title"], subfig_none_params
                ):
                    if param_was_none:
                        setattr(element, param, None)

            elif isinstance(element, (Plottable, list)):
                current_elements = element if isinstance(element, list) else [element]
                subfig = self._figure.add_subfigure(self._gridspec[rows, cols])
                ax = subfig.add_subplot(
                    sharex=ax
                    if self._share_x
                    else None,  # This enables the coherent zoom and pan of the axes
                    sharey=ax
                    if self._share_y
                    else None,  # but it does not remove the ticklabels
                    projection=self._subplot_p["projection"][subplot_i],
                )

                # Plotting loop
                z_order = 2
                for index, current_element in enumerate(current_elements):
                    if current_element is not None:
                        params_to_reset = []
                        if not is_matplotlib_style:
                            params_to_reset = self._fill_in_missing_params(
                                current_element
                            )
                        current_element._plot_element(
                            ax,
                            z_order,
                            cycle_color=cycle_colors[index % num_cycle_colors],
                        )
                        if not is_matplotlib_style:
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
                    len(self) > 1 or isinstance(self._figure, SubFigure)
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

                # Remove axes
                if self._subplot_p["remove_axes"][subplot_i]:
                    ax.axis("off")

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

                # Customize grid
                if self._subplot_p["show_grid"][subplot_i]:
                    ax.grid(
                        self._grid.get("visible_x"),
                        which=self._grid.get("which_x"),
                        axis="x",
                    )
                    ax.grid(
                        self._grid.get("visible_y"),
                        which=self._grid.get("which_y"),
                        axis="y",
                    )

                # Axes subtitles
                if self._subtitles is not None:
                    pad = (
                        subtitles_pad[subplot_i] if subtitles_pad is not None else None
                    )
                    ax.set_title(self._subplot_p["subtitles"][subplot_i], pad=pad)

                # Axes sub_labels
                self._customize_ax_label(ax, subplot_i)

                # Hidden spines
                if self._hidden_spines is not None:
                    for spine in set(self._hidden_spines):
                        ax.spines[spine].set_visible(False)

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
                            plt.rcParams.update(twin_axis._user_rc_dict)
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
                            plt.rcParams.update(parent_rc_params)
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

                    if self._subplot_p["show_legend"][subplot_i] and labels:
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
                    default_labels, default_handles = [], []
                    custom_labels, custom_handles = [], []

            elif element is not None:
                raise GraphingException(
                    f"Unsupported element type in list: {type(element).__name__}."
                )

        # Set a general axis for adding general labels/title and controlling padding
        general_ax = self._figure.add_subplot(self._gridspec[:, :], frameon=False)
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

        # General labels
        if self.is_single_subplot:
            if (
                ax is not None
            ):  # makes sure an element was plotted and that an axis was created
                self._customize_ax_label(ax)
        else:
            self._customize_ax_label(general_ax)

        # Title (if the SmartFigure is not a single subplot)
        if self._title:
            if self.is_single_subplot and ax is not None:
                ax.set_title(self._title, pad=self._pad_params.get("title_pad"))
            else:
                general_ax.set_title(self._title, pad=self._pad_params.get("title_pad"))

        # Annotations
        if self._annotations is not None:
            z_order = 5000
            for annotation in self._annotations:
                annotation._plot_element(self._figure, z_order)
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
            if labels and self._show_legend:
                legend_params = self._get_legend_params(labels, handles, 0)
                try:
                    _legend = self._figure.legend(
                        **legend_params,
                        draggable=True,
                    )
                except Exception:
                    _legend = self._figure.legend(
                        **legend_params,
                    )
                _legend.set_zorder(10000)
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

    def _fill_per_subplot_params(self) -> dict[str, Any]:
        """
        Fills the _subplot_p dictionary with parameters that can be broadcasted to all subplots in the
        :class:`~graphinglib.SmartFigure`. If a parameter is given as a single value, it is broadcasted to all subplots.
        If it is given as a list, its length must not exceed the number of non-empty subplots. Shorter lists are padded
        using the default value for that parameter.
        """
        self_length = len(self)
        blank_figure = (
            SmartFigure()
        )  # create a blank SmartFigure to get the default parameter values
        subplot_p = {
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

        for param, default_value in subplot_p.items():
            if param[-3:] == "pad":
                value = self._pad_params.get(param)
            else:
                value = getattr(self, f"_{param}")
            if isinstance(value, list):
                if len(value) > self_length:
                    raise GraphingException(
                        f"Number of {param} values ({len(value)}) must not exceed the number of non-empty subplots "
                        f"({self_length})."
                    )
                elif len(value) < self_length:
                    # Pad the list with default values to reach the number of non-empty subplots
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
        for element in self._ordered_elements.values():
            if isinstance(element, SmartFigure):
                element._align_shared_x_spines()

        tolerance = 0.3  # allowed difference between axes to consider them to be in the same column
        if self._share_x and self._num_rows > 1:
            try:
                plot_axes = self._get_all_axes_recursive(
                    self._figure
                )  # gives all the axes in the figure
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
                            [
                                rightmost_left_edge,
                                current_pos.y0,
                                aligned_size,
                                current_pos.height,
                            ]
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
        if self._ticks.get("x_ticks") is not None:
            x_labels = self._ticks.get("x_tick_labels")
            if callable(x_labels):
                # Apply the callable to each tick
                x_labels = [x_labels(tick) for tick in self._ticks.get("x_ticks")]
            ax.set_xticks(self._ticks.get("x_ticks"), x_labels)

        ax.tick_params(axis="x", which="major", **self._tick_params["x major"])

        if self._ticks.get("x_tick_spacing") is not None:
            ax.xaxis.set_major_locator(
                ticker.MultipleLocator(self._ticks.get("x_tick_spacing"))
            )
            # If a callable is provided for x_tick_labels, apply it with a FuncFormatter
            x_labels = self._ticks.get("x_tick_labels")
            if callable(x_labels):
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, pos: x_labels(x))
                )

        # Handle y-axis ticks
        if self._ticks.get("y_ticks") is not None:
            y_labels = self._ticks.get("y_tick_labels")
            if callable(y_labels):
                # Apply the callable to each tick
                y_labels = [y_labels(tick) for tick in self._ticks.get("y_ticks")]
            ax.set_yticks(self._ticks.get("y_ticks"), y_labels)

        ax.tick_params(axis="y", which="major", **self._tick_params["y major"])

        if self._ticks.get("y_tick_spacing") is not None:
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(self._ticks.get("y_tick_spacing"))
            )
            # If a callable is provided for y_tick_labels, apply it with a FuncFormatter
            y_labels = self._ticks.get("y_tick_labels")
            if callable(y_labels):
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda y, pos: y_labels(y))
                )

        if self._ticks.get("minor_x_ticks") is not None:
            ax.set_xticks(self._ticks.get("minor_x_ticks"), minor=True)
        ax.tick_params(axis="x", which="minor", **self._tick_params["x minor"])
        if self._ticks.get("minor_x_tick_spacing") is not None:
            ax.xaxis.set_minor_locator(
                ticker.MultipleLocator(self._ticks.get("minor_x_tick_spacing"))
            )

        if self._ticks.get("minor_y_ticks") is not None:
            ax.set_yticks(self._ticks.get("minor_y_ticks"), minor=True)
        ax.tick_params(axis="y", which="minor", **self._tick_params["y minor"])
        if self._ticks.get("minor_y_tick_spacing") is not None:
            ax.yaxis.set_minor_locator(
                ticker.MultipleLocator(self._ticks.get("minor_y_tick_spacing"))
            )

        # Remove ticks
        if self._subplot_p["remove_x_ticks"][subplot_i]:
            ax.tick_params(
                "x",
                which="both",
                labelbottom=False,
                labeltop=False,
                bottom=False,
                top=False,
            )
        if self._subplot_p["remove_y_ticks"][subplot_i]:
            ax.tick_params(
                "y",
                which="both",
                labelleft=False,
                labelright=False,
                left=False,
                right=False,
            )

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
            ax.set_xlabel(x_label, labelpad=x_pad)
        if y_label is not None:
            ax.set_ylabel(y_label, labelpad=y_pad)

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

        letter = ascii_lowercase[self._reference_label_i]
        formatted_letter = self._reference_labels_params.get(
            "format", lambda le: f"{le})"
        )(letter)
        reflabel_params = {
            k: v for k, v in self._reference_labels_params.items() if v != "default"
        }
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
        if isinstance(target, Axes):
            reflabel_loc = self._subplot_p["reference_labels_loc"][subplot_i]
            if isinstance(reflabel_loc, tuple):
                return ScaledTranslation(*reflabel_loc, self._figure.dpi_scale_trans)
            elif reflabel_loc == "outside":
                return ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
            elif reflabel_loc == "inside":
                return ScaledTranslation(
                    10 / 72, -15 / 72, self._figure.dpi_scale_trans
                )
            else:
                raise ValueError(
                    "Invalid reference label location. Please specify either 'inside' or 'outside'."
                )

        elif isinstance(target, (Figure, SubFigure)):
            return ScaledTranslation(7 / 72, -10 / 72, self._figure.dpi_scale_trans)
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
                    if (
                        (type(value) is str)
                        and (value == "default")
                        and not (property_ == "_figure_style")
                    ):
                        params_to_reset.append(property_)
                        default_value = self._default_params[object_type][property_]
                        setattr(element, property_, default_value)
                break
            except KeyError as e:
                if try_i == 1:
                    raise GraphingException(
                        f"There was an error auto updating your {self._figure_style} style file following the recent "
                        "GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub"
                        " page. In the meantime, you can manually add the following parameter to your "
                        f"{self._figure_style} style file:\n {e.args[0]}."
                    )
                file_updater = FileUpdater(self._figure_style)
                file_updater.update()
                file_loader = FileLoader(self._figure_style)
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
            try:
                if params["text.usetex"] and which("latex") is None:
                    params["text.usetex"] = False
            except KeyError:
                pass
            plt.rcParams.update(params)

    def set_rc_params(
        self,
        rc_params_dict: dict[str, str | float] = {},
        reset: bool = False,
    ) -> Self:
        """
        Customize the visual style of the :class:`~graphinglib.SmartFigure`.

        Any rc parameter that is not specified in the dictionary will be set to the default value for the specified
        ``figure_style``.

        Parameters
        ----------
        rc_params_dict : dict[str, str | float], optional
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
        axes_label_color: str | None = None,
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
        axes_label_color : str, optional
            The color of the axes labels.
        axes_label_pad : float, optional
            The padding between the axes labels and the axes.
        axes_line_width : float, optional
            The width of the axes lines.
        color_cycle : list[str], optional
            A list of colors to use for the color cycle.
        legend_face_color : str, optional
            The color of the legend face.
        legend_edge_color : str, optional
            The color of the legend edge.
        legend_font_size : float, optional
            The font size of the legend.
        legend_handle_length : float, optional
            The length of the legend handles.
        legend_handle_text_pad : float, optional
            The padding between the legend handles and the legend text.
        font_family : str, optional
            The font family to use.
        font_size : float, optional
            The font size to use.
        font_weight : str, optional
            The font weight to use.
        title_font_size : float, optional
            The font size of the title.
        title_font_weight : str, optional
            The font weight of the title.
        text_color : str, optional
            The color of the text.
        use_latex : bool, optional
            Whether or not to use latex.
        hidden_spines : Iterable[Literal["right", "left", "top", "bottom"]], optional
            The spines to hide. If specified, the corresponding spines will be hidden in the figure. This corresponds to
            the lines that form the borders of the plot.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated visual parameters.
        """
        if color_cycle is not None:
            color_cycle = plt.cycler(color=color_cycle)

        rc_params_dict = {
            "figure.facecolor": figure_face_color,
            "axes.facecolor": axes_face_color,
            "axes.edgecolor": axes_edge_color,
            "axes.labelcolor": axes_label_color,
            "axes.labelpad": axes_label_pad,
            "axes.linewidth": axes_line_width,
            "axes.prop_cycle": color_cycle,
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
        if reset:
            for key in rc_params_dict.keys():
                self._user_rc_dict.pop(key, None)

        rc_params_dict = {
            key: value for key, value in rc_params_dict.items() if value is not None
        }
        self.set_rc_params(rc_params_dict)

        if hidden_spines is not None:
            if not isinstance(hidden_spines, Iterable):
                raise TypeError("hidden_spines must be an iterable of spine names.")
            for spine in hidden_spines:
                if spine not in ["right", "left", "top", "bottom"]:
                    raise ValueError(
                        f"Invalid spine name: {spine}. Must be one of 'right', 'left', 'top' or 'bottom'."
                    )
            self._hidden_spines = hidden_spines

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
            raise GraphingException(
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
            raise GraphingException(
                "Tick spacing and tick positions cannot be set simultaneously."
            )

        if (
            x_ticks is not None
            and x_tick_labels is not None
            and not callable(x_tick_labels)
        ):
            if len(x_ticks) != len(x_tick_labels):
                raise GraphingException(
                    f"Number of x ticks ({len(x_ticks)}) and number of x tick labels "
                    f"({len(x_tick_labels)}) must be the same."
                )
        if (
            y_ticks is not None
            and y_tick_labels is not None
            and not callable(y_tick_labels)
        ):
            if len(y_ticks) != len(y_tick_labels):
                raise GraphingException(
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
        which: Literal["major", "minor", "both"] | None = "major",
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
        width : float, optional
            The width of the ticks.
        color : str, optional
            The color of the ticks.
        pad : float, optional
            The padding to add between the tick labels and the ticks themselves.
        label_size : float | str, optional
            The font size of the tick labels. This can be a float or a string (e.g. "large").
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
        """
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
        color: str | Literal["default"] = "default",
        alpha: float | Literal["default"] = "default",
        line_style: str | Literal["default"] = "default",
        line_width: float | Literal["default"] = "default",
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
            Default depends on the ``figure_style`` configuration.
        line_style : str, optional
            Sets the line style of the grid lines.
            Default depends on the ``figure_style`` configuration.
        line_width : float, optional
            Sets the line width of the grid lines.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated grid parameters.
        """
        if reset:
            self._grid.clear()
            self._user_rc_dict.update(
                {
                    "grid.color": "default",
                    "grid.alpha": "default",
                    "grid.linestyle": "default",
                    "grid.linewidth": "default",
                }
            )

        self._show_grid = True
        params = ["visible_x", "visible_y", "which_x", "which_y"]
        for param in params:
            value = locals()[param]
            if value is not None:
                self._grid[param] = value

        rc_params_dict = {
            "grid.color": color,
            "grid.alpha": alpha,
            "grid.linestyle": line_style,
            "grid.linewidth": line_width,
        }
        rc_params_dict = {k: v for k, v in rc_params_dict.items() if v != "default"}
        self.set_rc_params(rc_params_dict)
        return self

    def set_custom_legend(
        self,
        elements: Iterable[LegendElement] | None = None,
        reset: bool = False,
    ) -> Self:
        """
        Sets a custom legend for the figure. If the SmartFigure contains multiple subplots, **custom legends only**
        **work if the ``general_legend`` parameter is set to ``True``**. Otherwise, custom legends can be added for
        non-general legends if the SmartFigure is a single subplot (see the
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
        if reset:
            self._custom_legend_handles = []
            self._custom_legend_labels = []

        if isinstance(elements, Iterable) and all(
            isinstance(el, LegendElement) for el in elements
        ):
            self._custom_legend_handles += [el.handle for el in elements]
            self._custom_legend_labels += [el.label for el in elements]
        elif elements is not None:
            raise TypeError("Elements must be an iterable of LegendElement objects.")

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
        for pad_param in [x_label_pad, y_label_pad, title_pad]:
            if pad_param is not None and not isinstance(pad_param, (int, float)):
                raise TypeError(
                    f"Padding parameters must be of type int or float, got {type(pad_param).__name__}."
                )
        for sub_pad_param in [sub_x_labels_pad, sub_y_labels_pad, subtitles_pad]:
            if sub_pad_param is not None and (
                not isinstance(sub_pad_param, Iterable)
                or not all(
                    isinstance(p, (int, float, type(None))) for p in sub_pad_param
                )
            ):
                raise TypeError(
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
        color: str | Literal["default"] | None = None,
        start_index: int | None = None,
        font_size: float | Literal["default"] | None = None,
        font_weight: str | Literal["default"] | None = None,
        format: Callable = None,
    ) -> Self:
        """
        Sets advanced parameters for the reference labels that can be added to the subplots.

        Parameters
        ----------
        reset : bool, optional
            If ``True``, resets all previously set reference label parameters to their default values before applying
            the new parameters.
            Defaults to ``False``.
        color : str | Literal["default"], optional
            The color of the reference labels. If ``"default"``, the color is set according to the text color of other
            text in the figure.
        start_index : int, optional
            Starting index for the reference labels. This allows to customize the starting label, for example, to start
            labeling from "b)" instead of "a)" by giving ``start_index = 1``.
        font_size : float | Literal["default"], optional
            The font size of the reference labels.
        font_weight : str | Literal["default"], optional
            The font weight of the reference labels.
        format : Callable, optional
            A callable function to format the reference labels. By default, the reference labels are formatted as a),
            b), etc. The function must take a single str argument (the letter) and return a formatted str. For example,
            to have the reference labels in uppercase and in parentheses, the format could be defined as follows::

                format = lambda label: f"({label.upper()})"

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated reference labels parameters.
        """
        if start_index is not None:
            if not isinstance(start_index, int):
                raise TypeError("start_index must be an integer.")
            if start_index < 0:
                raise ValueError("start_index must be greater than or equal to 0.")
        if format is not None:
            try:
                format("a")
            except Exception as e:
                raise TypeError(
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
        elements: Iterable[Plottable | None] = [],
    ) -> SmartTwinAxis:
        """
        Creates a twin axis for the SmartFigure. This method creates a :class:`~graphinglib.SmartTwinAxis` object that
        can be used to plot elements on a secondary axis in the same subplot.

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
        elements : Iterable[Plottable | None], optional
            Elements to plot in the twin axis. This must be an iterable of
            :class:`~graphinglib.Plottable` objects. If ``None`` elements are present, they are ignored.
            Defaults to empty list.

        Returns
        -------
        SmartTwinAxis
            The created twin axis object. The twin axis can also be accessed through the
            :attr:`~graphinglib.SmartFigure.twin_x_axis` or :attr:`~graphinglib.SmartFigure.twin_y_axis` properties.
        """
        if is_y and self._twin_y_axis is not None:
            raise GraphingException(
                "A twin y-axis already exists for this SmartFigure."
            )
        elif not is_y and self._twin_x_axis is not None:
            raise GraphingException(
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
    the ability to use styles and themes for consistent visual appearance across different figures. The idea behind this
    class is that every SmartFigure contains a single x_label, y_label, title, projection, etc. and that nested
    SmartFigures can be inserted into the main SmartFigure to create complex figures with more parameters.

    Parameters
    ----------
    projection : WCS | list[WCS]
        The `World Coordinate System (WCS) <https://docs.astropy.org/en/stable/wcs/index.html>`_ object to use for the
        figure. This is used to plot data in a coordinate system that is not Cartesian, such as celestial coordinates.
        This can be given as a single WCS object or a list of WCS objects to apply to each subplot.
    num_rows, num_cols : int, optional
        Number of rows and columns for the base grid. These parameters determine the number of "squares" on which the
        plots can be placed.
        Defaults to ``1``.
    x_label, y_label : str, optional
        Labels for the x and y axes of the figure.
    size : tuple[float, float], optional
        Overall size of the figure. Note that this option is useless if the SmartFigure is nested inside another
        SmartFigure, as the size is then determined by the parent SmartFigure and the available space.
        Default depends on the ``figure_style`` configuration.
    title : str, optional
        General title of the figure.
    x_lim, y_lim : tuple[float, float] | list[tuple[float, float]], optional
        Limits for the x and y axes of the figure. This can be given as a single value or a list of values to apply to
        each subplot.
    sub_x_labels, sub_y_labels : Iterable[str], optional
        Labels for the x and y axes of each subfigure, respectively. This is only useful for figures that are not a
        single subplot and when each subfigure needs its own x and y labels. This prevents the creation of nested
        :class:`~graphinglib.SmartFigure` objects for each subfigure only to set the x and y labels. This list cannot
        be longer than the number of non-empty subplots and None values can be used to skip specific subplots.
    subtitles : Iterable[str], optional
        Labels for the subtitles of each subfigure, respectively. Similarly to `sub_x_labels` and `sub_y_labels`, this
        allows to set subtitles for each subfigure without needing to create nested
        :class:`~graphinglib.SmartFigure` objects. It is only useful for figures that are not a single subplot and when
        each subfigure needs its own subtitle. This list cannot be longer than the number of non-empty subplots and None
        values can be used to skip specific subplots.
    log_scale_x, log_scale_y : bool | list[bool], optional
        Whether to use a logarithmic scale for the x and y axes, respectively. This can be given as a single value or
        a list of values to apply to each subplot.
        Defaults to ``False``.
    remove_axes : bool | list[bool], optional
        Whether to remove the axes from the figure. This can be given as a single boolean or a list of booleans to apply
        to each subplot.
        Defaults to ``False``.
    aspect_ratio : float | Literal["auto", "equal"] | list[float | Literal["auto", "equal"]], optional
        Aspect ratio of the figure. If set to "auto", the aspect ratio is determined automatically to fill the available
        space. If set to "equal", the aspect ratio is set to 1:1. If set to a float, the aspect ratio represents the
        ratio of the height to the width of the data. This can be given as a single value or a list of values to apply
        to each subplot.
        Defaults to "auto".

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

    remove_x_ticks, remove_y_ticks : bool | list[bool], optional
        Whether to remove the x and y ticks from the figure, respectively. This can be given as a single value or a list
        of values to apply to each subplot.
        Defaults to ``False``.
    invert_x_axis, invert_y_axis : bool | list[bool], optional
        Whether to invert the x and y axes, respectively. This can be given as a single value or a list of values to
        apply to each subplot.
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
        Location of the reference labels of the SubFigures, which can be either "inside", "outside" or a tuple of
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
            Sharing axes only works for plots directly inside the SmartFigure. If a nested SmartFigure is used, the
            axes sharing will not be applied to the nested SmartFigure. Instead, the nested SmartFigure will have its
            own axes sharing settings.

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
    show_legend : bool | list[bool], optional
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
    elements : Iterable[Plottable | SmartFigure] | Iterable[Iterable[Plottable | SmartFigure]], optional
        The elements to plot in the figure.
        If an iterable of depth 1 is provided and the figure is 1x1, all the elements are added to the unique plot. For
        other geometries, the elements are added one by one in the order they are provided to each subplot, and the
        iterable should not be longer than the number of subplots.
        If an iterable of depth 2 is provided, each sub-iterable is added to the corresponding subplot, in the order
        they are provided. The number of sub-iterables should be equal to the number of subplots.
        If ``None`` elements are present in the iterable, the corresponding subplots are not drawn and a blank space is
        left in the figure. If iterables containing only ``None`` are given in the main iterable, the corresponding
        subplots are drawn but empty.

        .. note::
            This method for adding elements only allows to add elements to single subplots. If you want to add elements
            that span multiple subplots, you should use the __setitem__ method instead.
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
        size: tuple[float, float] | Literal["default"] = "default",
        title: str | None = None,
        x_lim: ListOrItem[tuple[float, float] | None] = None,
        y_lim: ListOrItem[tuple[float, float] | None] = None,
        sub_x_labels: Iterable[str] | None = None,
        sub_y_labels: Iterable[str] | None = None,
        subtitles: Iterable[str] | None = None,
        log_scale_x: ListOrItem[bool] = False,
        log_scale_y: ListOrItem[bool] = False,
        remove_axes: ListOrItem[bool] = False,
        aspect_ratio: ListOrItem[float | Literal["auto", "equal"]] = "auto",
        box_aspect_ratio: ListOrItem[float | None] = None,
        remove_x_ticks: ListOrItem[bool] = False,
        remove_y_ticks: ListOrItem[bool] = False,
        invert_x_axis: ListOrItem[bool] = False,
        invert_y_axis: ListOrItem[bool] = False,
        reference_labels: ListOrItem[bool] = True,
        global_reference_label: bool = False,
        reference_labels_loc: ListOrItem[
            Literal["inside", "outside"] | tuple[float, float]
        ] = "outside",
        width_padding: float = None,
        height_padding: float = None,
        width_ratios: ArrayLike = None,
        height_ratios: ArrayLike = None,
        share_x: bool = False,
        share_y: bool = False,
        general_legend: bool = False,
        legend_loc: ListOrItem[str | tuple | None] = None,
        legend_cols: ListOrItem[int] = 1,
        show_legend: ListOrItem[bool] = True,
        twin_x_axis: SmartTwinAxis | None = None,
        twin_y_axis: SmartTwinAxis | None = None,
        figure_style: str = "default",
        elements: Iterable[Plottable | SmartFigure | None]
        | Iterable[Iterable[Plottable | None]] = [],
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
        for v in value if isinstance(value, list) else [value]:
            if not isinstance(v, WCS):
                raise GraphingException(
                    "The projection of a SmartFigureWCS must be a WCS object."
                )
        self._projection = value

    def _prepare_figure(
        self,
        is_matplotlib_style: bool = False,
        make_legend: bool = True,
    ) -> dict[str, dict[str, list[str | Any]]]:
        """
        Wraps the parent method to check if the number of projections matches the number of non-empty subplots.
        """
        if isinstance(self._projection, list) and len(self._projection) != len(self):
            raise GraphingException(
                f"Number of WCS projections ({len(self._projection)}) must be equal to the number of non-empty "
                f"subplots ({len(self)})."
            )
        return super()._prepare_figure(is_matplotlib_style, make_legend)

    def _customize_ticks(
        self,
        ax: Axes,
        subplot_i: int,
    ) -> None:
        """
        Customizes the ticks of the specified Axes according to the SmartFigure's tick parameters. This method is useful
        for inheritance to allow each SmartFigure class to customize the ticks their way.
        """
        x_axis, y_axis = ax.coords
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

        # Remove ticks
        if self._subplot_p["remove_x_ticks"][subplot_i]:
            x_axis.set_ticks_visible(False)
            x_axis.set_ticklabel_visible(False)
        if self._subplot_p["remove_y_ticks"][subplot_i]:
            y_axis.set_ticks_visible(False)
            y_axis.set_ticklabel_visible(False)

    def set_ticks(
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
        super().set_ticks(
            reset=reset,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_spacing=x_tick_spacing,
            y_tick_spacing=y_tick_spacing,
        )
        if any(
            [
                (x_ticks is not None) and (number_of_x_ticks is not None),
                (y_ticks is not None) and (number_of_y_ticks is not None),
            ]
        ):
            raise GraphingException(
                "Number of ticks and tick positions cannot be set simultaneously."
            )

        if any(
            [
                (x_tick_spacing is not None) and (number_of_x_ticks is not None),
                (y_tick_spacing is not None) and (number_of_y_ticks is not None),
            ]
        ):
            raise GraphingException(
                "Number of ticks and tick spacing cannot be set simultaneously."
            )

        params = [
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

    def set_tick_params(
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
        minor_length : float, optional
            The length of the minor ticks. This is the only parameter that can be set independently from the major ticks
            due to the way the :class:`astropy.visualization.wcsaxes.WCSAxes` are implemented.
        width : float, optional
            The width of the ticks.
        color : str, optional
            The color of the ticks.
        pad : float, optional
            The padding to add between the tick labels and the ticks themselves.
        label_size : float | str, optional
            The font size of the tick labels. This can be a float or a string (e.g. "large").
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
        """
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

    def set_grid(
        self,
        visible_x: bool = True,
        visible_y: bool = True,
        color: str | Literal["default"] = "default",
        alpha: float | Literal["default"] = "default",
        line_style: str | Literal["default"] = "default",
        line_width: float | Literal["default"] = "default",
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
            Default depends on the ``figure_style`` configuration.
        line_style : str, optional
            Sets the line style of the grid lines.
            Default depends on the ``figure_style`` configuration.
        line_width : float, optional
            Sets the line width of the grid lines.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        Self
            For convenience, the same SmartFigure with the updated grid parameters.
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
    elements : Iterable[Plottable | None], optional
        Elements to plot in the twin axis. This must be an iterable of :class:`~graphinglib.Plottable` objects. If
        ``None`` values are present, they are ignored.
    """

    def __init__(
        self,
        label: str | None = None,
        axis_lim: tuple[float, float] | None = None,
        log_scale: bool = False,
        remove_axes: bool = False,
        remove_ticks: bool = False,
        invert_axis: bool = False,
        elements: Iterable[Plottable | None] = [],
    ) -> None:
        self.label = label
        self.axis_lim = axis_lim
        self.log_scale = log_scale
        self.remove_axes = remove_axes
        self.remove_ticks = remove_ticks
        self.invert_axis = invert_axis
        self.elements = elements

        self._ticks = {}
        self._tick_params = {"major": {}, "minor": {}}

        self._edge_color = None
        self._line_width = None
        self._hide_spine = None
        self._user_rc_dict = {}
        self._default_params = {}
        self._axes = None  # used for keeping a reference to the Axes which enables drawing the legend on top

    @property
    def label(self) -> str | None:
        return self._label

    @label.setter
    def label(self, value: str | None) -> None:
        self._label = value

    @property
    def axis_lim(self) -> tuple[float, float] | None:
        return self._axis_lim

    @axis_lim.setter
    def axis_lim(self, value: tuple[float, float] | None) -> None:
        if value is not None:
            if not isinstance(value, tuple):
                raise TypeError("axis_lim must be a tuple.")
            if len(value) != 2:
                raise ValueError("axis_lim must be a tuple of length 2.")
        self._axis_lim = value

    @property
    def log_scale(self) -> bool:
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("log_scale must be a boolean.")
        self._log_scale = value

    @property
    def remove_axes(self) -> bool:
        return self._remove_axes

    @remove_axes.setter
    def remove_axes(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("remove_axes must be a bool.")
        self._remove_axes = value

    @property
    def remove_ticks(self) -> bool:
        return self._remove_ticks

    @remove_ticks.setter
    def remove_ticks(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("remove_ticks must be a bool.")
        self._remove_ticks = value

    @property
    def invert_axis(self) -> bool:
        return self._invert_axis

    @invert_axis.setter
    def invert_axis(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("invert_axis must be a bool.")
        self._invert_axis = value

    @property
    def elements(self) -> list[Plottable | None]:
        return self._elements

    @elements.setter
    def elements(self, value: Iterable[Plottable | None]) -> None:
        """
        Sets the elements of the SmartTwinAxis with the same rules as the constructor. For adding elements instead of
        replacing them, use the :meth:`~graphinglib.SmartTwinAxis.add_elements` method.
        """
        self._elements = []  # systematically reset the elements when setting them with the property
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
            raise TypeError(f"Key must be an integer, not {type(key).__name__}.")
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
        if not SmartFigure._is_iterable_of_plottables(elements):
            raise TypeError("Elements must be an iterable of Plottable objects.")
        self._elements += [el for el in elements if el is not None]
        return self

    def _prepare_twin_axis(
        self,
        fig_axes: Axes,
        is_matplotlib_style: bool,
        cycle_colors: list[str],
        is_y: bool,
        z_order: int,
        figure_style: str,
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
        figure_style : str
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

        if self._label:
            ax_set_label(self._label)
        if self._axis_lim:
            ax_set_lim(*self._axis_lim)

        # Artificially modify the axes edge color and line width to modify only a single spine
        if self._edge_color:
            ax.spines[spine_str].set_color(self._edge_color)
        if self._line_width:
            ax.spines[spine_str].set_linewidth(self._line_width)
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.spines[spine_str].set_visible(not self._hide_spine)

        self._customize_ticks(is_y)

        # Logarithmic scale
        if self._log_scale:
            ax_set_scale("log")

        # Remove axes
        if self._remove_axes:
            ax.axis("off")

        # Invert axis
        if self.invert_axis:
            if is_y:
                ax.invert_yaxis()
            else:
                ax.invert_xaxis()

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
                raise GraphingException(
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
        if is_y:
            ax_set_ticks, axis_str, ax_axis = (
                self._axes.set_yticks,
                "y",
                self._axes.yaxis,
            )
        else:
            ax_set_ticks, axis_str, ax_axis = (
                self._axes.set_xticks,
                "x",
                self._axes.xaxis,
            )

        if self._ticks.get("ticks") is not None:
            tick_labels = self._ticks.get("tick_labels")
            if callable(tick_labels):
                # Apply the callable to each tick
                tick_labels = [tick_labels(tick) for tick in self._ticks.get("ticks")]
            ax_set_ticks(self._ticks.get("ticks"), tick_labels)

        self._axes.tick_params(
            axis=axis_str, which="major", **self._tick_params["major"]
        )

        if self._ticks.get("tick_spacing") is not None:
            ax_axis.set_major_locator(
                ticker.MultipleLocator(self._ticks.get("tick_spacing"))
            )
            # If a callable is provided for tick_labels, apply it with a FuncFormatter
            tick_labels = self._ticks.get("tick_labels")
            if callable(tick_labels):
                ax_axis.set_major_formatter(
                    ticker.FuncFormatter(lambda pos, x: tick_labels(pos))
                )

        if self._ticks.get("minor_ticks") is not None:
            ax_set_ticks(self._ticks.get("minor_ticks"), minor=True)
        self._axes.tick_params(
            axis=axis_str, which="minor", **self._tick_params["minor"]
        )
        if self._ticks.get("minor_tick_spacing") is not None:
            ax_axis.set_minor_locator(
                ticker.MultipleLocator(self._ticks.get("minor_tick_spacing"))
            )

        # Remove ticks
        if self._remove_ticks:
            self._axes.tick_params(
                axis_str,
                which="both",
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False,
                bottom=False,
                left=False,
                right=False,
                top=False,
            )

    def _fill_in_missing_params(
        self, element: SmartFigure | Plottable, figure_style: str
    ) -> list[str]:
        """
        Fills in the missing parameters for a :class:`~graphinglib.Plottable` from the parent's ``figure_style``.
        """
        params_to_reset = []
        object_type = type(element).__name__
        for try_i in range(2):
            try:
                for property_, value in vars(element).items():
                    if (
                        (type(value) is str)
                        and (value == "default")
                        and not (property_ == "_figure_style")
                    ):
                        params_to_reset.append(property_)
                        default_value = self._default_params[object_type][property_]
                        setattr(element, property_, default_value)
                break
            except KeyError as e:
                if try_i == 1:
                    raise GraphingException(
                        f"There was an error auto updating your {figure_style} style file following the recent "
                        "GraphingLib update. Please notify the developers by creating an issue on GraphingLib's GitHub"
                        " page. In the meantime, you can manually add the following parameter to your "
                        f"{figure_style} style file:\n {e.args[0]}."
                    )
                file_updater = FileUpdater(figure_style)
                file_updater.update()
                file_loader = FileLoader(figure_style)
                new_defaults = file_loader.load()
                self._default_params.update(
                    (k, v)
                    for k, v in new_defaults.items()
                    if k not in self._default_params
                )
        return params_to_reset

    def _reset_params_to_default(
        self, element: Plottable, params_to_reset: list[str]
    ) -> None:
        """
        Resets the parameters that were set to default in the :meth:`~graphinglib.SmartTwinAxis._fill_in_missing_params`
        method.
        """
        for param in params_to_reset:
            setattr(element, param, "default")

    def set_rc_params(
        self,
        rc_params_dict: dict[str, str | float] = {},
        reset: bool = False,
    ) -> Self:
        """
        Customize the visual style of the :class:`~graphinglib.SmartTwinAxis`.

        Any rc parameter that is not specified in the dictionary will be set to the default value for the specified
        ``figure_style`` from the parent :class:`~graphinglib.SmartFigure`.

        Parameters
        ----------
        rc_params_dict : dict[str, str | float], optional
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
        line_width : float, optional
            The width of the spine.
        font_family : str, optional
            The font family to use.
        font_size : float, optional
            The font size to use.
        font_weight : str, optional
            The font weight to use.
        use_latex : bool, optional
            Whether or not to use latex.
        hide_spine : bool, optional
            Whether to hide the spine of the axes. This corresponds to the line that forms the border of the plot.

        Returns
        -------
        Self
            For convenience, the same SmartTwinAxis with the updated visual parameters.
        """
        rc_params_dict = {
            "axes.labelcolor": label_color,
            "axes.labelpad": label_pad,
            "font.family": font_family,
            "font.size": font_size,
            "font.weight": font_weight,
            "text.usetex": use_latex,
        }

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
            if not isinstance(hide_spine, bool):
                raise TypeError(
                    "hide_spine must be a boolean or an iterable of spine names."
                )
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
        # Check if tick labels are provided without ticks or spacing
        has_spacing = tick_spacing is not None
        is_callable = callable(tick_labels)

        if (
            (tick_labels is not None)
            and ticks is None
            and not (has_spacing and is_callable)
        ):
            raise GraphingException(
                "Ticks position must be specified when ticks labels are specified, "
                "unless a callable is provided with tick spacing."
            )

        if any(
            [
                (ticks is not None) and (tick_spacing is not None),
                (minor_ticks is not None) and (minor_tick_spacing is not None),
            ]
        ):
            raise GraphingException(
                "Tick spacing and tick positions cannot be set simultaneously."
            )

        if ticks is not None and tick_labels is not None and not callable(tick_labels):
            if len(ticks) != len(tick_labels):
                raise GraphingException(
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
        which: Literal["major", "minor", "both"] | None = "major",
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
        width : float, optional
            The width of the ticks.
        color : str, optional
            The color of the ticks.
        pad : float, optional
            The padding to add between the tick labels and the ticks themselves.
        label_size : float | str, optional
            The font size of the tick labels. This can be a float or a string (e.g. "large").
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
        """
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
