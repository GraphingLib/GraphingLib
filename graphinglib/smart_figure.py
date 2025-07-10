from __future__ import annotations
from shutil import which
from typing import Literal, Optional, Any, Self, Callable, Iterable
from logging import warning
from string import ascii_lowercase
from collections import OrderedDict
from copy import deepcopy
from difflib import get_close_matches
from astropy.wcs import WCS
from astropy.units import Quantity

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import is_interactive
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from matplotlib.transforms import ScaledTranslation
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from matplotlib.projections import get_projection_names
from matplotlib.artist import Artist

from .file_manager import (
    FileLoader,
    FileUpdater,
    get_default_style,
    get_styles,
)
from .graph_elements import GraphingException, Plottable, Text
from .legend_artists import (
    HandlerMultipleLines,
    HandlerMultipleVerticalLines,
    VerticalLineCollection,
    histogram_legend_artist,
)

from numpy.typing import ArrayLike


class SmartFigure:
    """
    This class implements a figure object for plotting :class:`~graphinglib.graph_elements.Plottable` elements. It
    allows for the creation of complex figures recursively, where each :class:`~graphinglib.smart_figure.SmartFigure`
    can contain other :class:`~graphinglib.smart_figure.SmartFigure` objects. The class supports a variety of
    customization options as well as the ability to use styles and themes for consistent visual appearance across
    different figures. The idea behind this class is that every SmartFigure contains a single x_label, y_label, title,
    projection, etc. and that nested SmartFigures can be inserted into the main SmartFigure to create complex figures
    with more parameters.

    Parameters
    ----------
    num_rows, num_cols : int
        Number of rows and columns for the base grid. These parameters determine the number of "squares" on which the
        plots can be placed.
    x_label, y_label : str, optional
        Labels for the x and y axes of the figure.
    size : tuple[float, float]
        Overall size of the multifigure.
        Default depends on the ``figure_style`` configuration.
    title : str, optional
        General title of the figure.
    x_lim, y_lim : tuple[float, float], optional
        Limits for the x and y axes of the figure.
    log_scale_x, log_scale_y : bool
        Whether to use a logarithmic scale for the x and y axes, respectively.
        Defaults to ``False``.
    remove_axes : bool
        Whether to remove the axes from the figure.
        Defaults to ``False``.
    aspect_ratio : float | Literal["auto", "equal"]
        Aspect ratio of the figure. If set to "auto", the aspect ratio is determined automatically to fill the available
        space. If set to "equal", the aspect ratio is set to 1:1. If set to a float, the aspect ratio represents the
        ratio of the height to the width of the figure.
        Defaults to "auto".
    remove_x_ticks, remove_y_ticks : bool
        Whether to remove the x and y ticks from the figure, respectively.
        Defaults to ``False``.
    reference_labels : bool
        Whether or not to add reference labels to the subfigures. If set to ``True``, each subfigure will be labeled
        alphabetically in the form of "a)", "b)", etc.
        Defaults to ``True``.

        .. note::
            For nested figures, each subfigure controls its own reference labels. This means that if a nested
            SmartFigure turns off reference labels, the plots in it will not be labeled, even if the parent SmartFigure
            has reference labels turned on.

    global_reference_label : bool
        Whether to use a single reference label for the entire figure instead of individual labels for each subfigure.
        If set to ``True``, the reference label will be placed in the top left corner of the global SmartFigure. This is
        useful for labeling the entire figure rather than individual subfigures.
        Defaults to ``False``.
    reflabel_loc : Literal["inside", "outside"]
        Location of the reference labels of the SubFigures, either "inside" or "outside".
        Defaults to ``"outside"``.
    width_padding, height_padding : float
        Padding between the subfigures in the x and y directions, respectively. The default value of ``None`` results in
        a default small amount of padding. This may be set to 0 to completely remove the space between subfigures, but
        note that axes labels may need to be removed to delete additional space.
    width_ratios, height_ratios : ArrayLike
        Ratios of the widths and heights of the subfigures, respectively. These ratios determine how much space each
        column and row of subfigures will take up in the overall figure. The length of these arrays must match the
        number of columns and rows, respectively. By default, all subfigures are given equal space.
    share_x, share_y : bool
        Whether to share the x and y axes between subfigures, respectively. This means that all subfigures will have
        the same x and y limits, and the ticks will be shared as well. This is useful for comparing data across
        subfigures.

        .. note::
            Sharing axes only works for plots directly inside the SmartFigure. If a nested SmartFigure is used, the
            axes sharing will not be applied to the nested SmartFigure. Instead, the nested SmartFigure will have its
            own axes sharing settings.

    projection : Any, optional
        Projection type for the subfigures. This can be a string of a matplotlib projection (e.g., "polar") or an object
        capable of creating a projection (e.g. astropy.wcs.WCS).

        .. note::
            3D projections are not supported at the moment.

    general_legend : bool
        Whether to create a general legend for the entire figure. If set to ``True``, a single legend will be created
        to regroup all the legends from the subplots. If set to ``False``, all subplots will have their own legend. If
        nested SmartFigures set this parameter to ``False``, their legend is added to the parent's general legend.
        However, if a nested SmartFigure sets its general legend to ``True``, it will be created separately and will not
        be added to the parent's general legend.
        Defaults to ``False``.
    legend_loc : str | tuple, optional
        Location of the legend. This can be a string (e.g., "upper right") or a tuple of (x, y) relative coordinates.
        The supported string locations are: {"upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center", "center", "outside upper center",
        "outside center right", "outside lower center", "outside center left"}. Additionally, only if ``general_legend``
        is set to ``False``, the legend location can also be set to "best".
        Defaults to ``"best"`` if ``general_legend`` is set to ``False``, otherwise it defaults to ``"lower center"``.

        .. warning::
            If ``general_legend`` is set to ``True`` and the legend location is set to a position containing "outside",
            the legend may not be displayed correctly in some matplotlib backends. In such cases, it is recommended to
            use inline figures in a Jupyter notebook or save the figure to a file to ensure proper display of the
            legend outside the figure.

    legend_cols : int
        Number of columns to display the labels in the legend. This is only used if the legend is displayed.
        Defaults to ``1``.
    show_legend : bool
        Whether to show the legend for the figure. This allows to easily toggle the visibility of the legend.
        Defaults to ``True``.
    figure_style : str
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
        left in the figure. If empty lists or iterables containing only ``None`` are given in the list, the
        corresponding subplots are drawn but empty.

        .. note::
            This method for adding elements only allows to add elements to single subplots. If you want to add elements
            that span multiple subplots, you should use the __setitem__ method instead.
            For example, to add an element spanning the complete first row , use ``fig[0,:] = element``.
    """
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
        aspect_ratio: float | Literal["auto", "equal"] = "auto",
        remove_x_ticks: bool = False,
        remove_y_ticks: bool = False,
        reference_labels: bool = True,
        global_reference_label: bool = False,
        reflabel_loc: Literal["inside", "outside"] = "outside",
        width_padding: float = None,
        height_padding: float = None,
        width_ratios: ArrayLike = None,
        height_ratios: ArrayLike = None,
        share_x: bool = False,
        share_y: bool = False,
        projection: Optional[Any] = None,
        general_legend: bool = False,
        legend_loc: Optional[str | tuple] = None,
        legend_cols: int = 1,
        show_legend: bool = True,
        figure_style: str = "default",
        elements: Optional[Iterable[Plottable | SmartFigure] | Iterable[Iterable[Plottable | SmartFigure]]] = [],
    ) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.x_label = x_label
        self.y_label = y_label
        self.size = size
        self.title = title
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.log_scale_x = log_scale_x
        self.log_scale_y = log_scale_y
        self.remove_axes = remove_axes
        self.aspect_ratio = aspect_ratio
        self.remove_x_ticks = remove_x_ticks
        self.remove_y_ticks = remove_y_ticks
        self.reference_labels = reference_labels
        self.global_reference_label = global_reference_label
        self.reflabel_loc = reflabel_loc
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
        self.figure_style = figure_style

        self._elements = {}
        if any(elements):
            if not (isinstance(elements, Iterable)
                    and all(isinstance(el, (Plottable, SmartFigure, Iterable, type(None))) for el in elements)):
                raise TypeError(
                    f"Elements must be an iterable of Plottable objects or SmartFigures, not {type(elements).__name__}."
                )
            if self.is_single_subplot and not SmartFigure._is_iterable_of_elements(elements[0]):
                elements = [elements]
            for i, element in enumerate(elements):
                if not any([
                    isinstance(element, (Plottable, SmartFigure, type(None))),
                    SmartFigure._is_iterable_of_elements(element),
                ]):
                    raise TypeError(
                        f"Element at index {i} must be a Plottable, an iterable of Plottables, or a SmartFigure."
                    )
            if len(elements) > num_cols * num_rows:
                raise ValueError("Too many elements provided for the number of subplots.")
            for i, element in enumerate(elements):
                self[divmod(i, self._num_cols)] = element

        self._figure = None
        self._gridspec = None
        self._reference_label_i = None

        self._x_ticks = None
        self._y_ticks = None
        self._x_tick_labels = None
        self._y_tick_labels = None
        self._x_tick_spacing = None
        self._y_tick_spacing = None
        self._minor_x_ticks = None
        self._minor_y_ticks = None
        self._minor_x_tick_spacing = None
        self._minor_y_tick_spacing = None
        self._tick_params = {"x major": {}, "y major": {}, "x minor": {}, "y minor": {}}

        self.show_grid = False
        self._grid_visible_x = None
        self._grid_visible_y = None
        self._grid_show_on_top = None
        self._grid_which_x = None
        self._grid_which_y = None

        self.hide_custom_legend_elements = False
        self.hide_default_legend_elements = False
        self._custom_legend_handles = []
        self._custom_legend_labels = []

        self._user_rc_dict = {}
        self._default_params = {}

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
                        raise GraphingException("Cannot remove rows from the SmartFigure when there are elements in "
                                                "them. Please remove the elements first.")
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
                        raise GraphingException("Cannot remove cols from the SmartFigure when there are elements in "
                                                "them. Please remove the elements first.")
        except AttributeError:
            # The figure is being created, so the _num_cols attribute is not yet set
            pass
        self._num_cols = value

    @property
    def x_label(self) -> str:
        return self._x_label

    @x_label.setter
    def x_label(self, value: str) -> None:
        self._x_label = value

    @property
    def y_label(self) -> str:
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
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value

    @property
    def x_lim(self) -> tuple[float, float]:
        return self._x_lim

    @x_lim.setter
    def x_lim(self, value: tuple[float, float]) -> None:
        if value is not None:
            if not isinstance(value, tuple):
                raise TypeError("x_lim must be a tuple.")
            if len(value) != 2:
                raise ValueError("x_lim must be a tuple of length 2.")
        self._x_lim = value

    @property
    def y_lim(self) -> tuple[float, float]:
        return self._y_lim

    @y_lim.setter
    def y_lim(self, value: tuple[float, float]) -> None:
        if value is not None:
            if not isinstance(value, tuple):
                raise TypeError("y_lim must be a tuple.")
            if len(value) != 2:
                raise ValueError("y_lim must be a tuple of length 2.")
        self._y_lim = value

    @property
    def log_scale_x(self) -> bool:
        return self._log_scale_x

    @log_scale_x.setter
    def log_scale_x(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("log_scale_x must be a bool.")
        self._log_scale_x = value

    @property
    def log_scale_y(self) -> bool:
        return self._log_scale_y

    @log_scale_y.setter
    def log_scale_y(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("log_scale_y must be a bool.")
        self._log_scale_y = value

    @property
    def remove_axes(self) -> bool:
        return self._remove_axes

    @remove_axes.setter
    def remove_axes(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("remove_axes must be a bool.")
        self._remove_axes = value

    @property
    def aspect_ratio(self) -> float | Literal["auto", "equal"]:
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, value: float | Literal["auto", "equal"]) -> None:
        if not isinstance(value, (float, int)) and value != "auto" and value != "equal":
            raise TypeError("aspect_ratio must be a float, 'auto' or 'equal'.")
        if isinstance(value, (float, int)) and value <= 0:
            raise ValueError("aspect_ratio must be greater than 0.")
        self._aspect_ratio = value

    @property
    def remove_x_ticks(self) -> bool:
        return self._remove_x_ticks

    @remove_x_ticks.setter
    def remove_x_ticks(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("remove_x_ticks must be a bool.")
        self._remove_x_ticks = value

    @property
    def remove_y_ticks(self) -> bool:
        return self._remove_y_ticks

    @remove_y_ticks.setter
    def remove_y_ticks(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("remove_y_ticks must be a bool.")
        self._remove_y_ticks = value

    @property
    def reference_labels(self) -> bool:
        return self._reference_labels

    @reference_labels.setter
    def reference_labels(self, value: bool) -> None:
        if not isinstance(value, bool):
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
    def reflabel_loc(self) -> Literal["inside", "outside"]:
        return self._reflabel_loc

    @reflabel_loc.setter
    def reflabel_loc(self, value: Literal["inside", "outside"]) -> None:
        if value not in ["inside", "outside"]:
            raise ValueError("reflabel_loc must be either 'inside' or 'outside'.")
        self._reflabel_loc = value

    @property
    def width_padding(self) -> float:
        return self._width_padding

    @width_padding.setter
    def width_padding(self, value: float) -> None:
        if value is not None:
            if not isinstance(value, (float, int)):
                raise TypeError("width_padding must be a number.")
            if value < 0:
                raise ValueError("width_padding must be greater than or equal to 0.")
        self._width_padding = value

    @property
    def height_padding(self) -> float:
        return self._height_padding

    @height_padding.setter
    def height_padding(self, value: float) -> None:
        if value is not None:
            if not isinstance(value, (float, int)):
                raise TypeError("height_padding must be a number.")
            if value < 0:
                raise ValueError("height_padding must be greater than or equal to 0.")
        self._height_padding = value

    @property
    def width_ratios(self) -> ArrayLike:
        return self._width_ratios

    @width_ratios.setter
    def width_ratios(self, value: ArrayLike) -> None:
        if value is not None:
            if not hasattr(value, "__len__"):
                raise TypeError("width_ratios must be an ArrayLike.")
            if not all(isinstance(x, (float, int)) for x in value):
                raise TypeError("width_ratios must contain only numbers.")
            if len(value) != self._num_cols:
                raise ValueError("width_ratios must have the same length as num_cols.")
        self._width_ratios = value

    @property
    def height_ratios(self) -> ArrayLike:
        return self._height_ratios

    @height_ratios.setter
    def height_ratios(self, value: ArrayLike) -> None:
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
    def projection(self) -> Any:
        return self._projection

    @projection.setter
    def projection(self, value: Any) -> None:
        if value is not None:
            valid_projections = get_projection_names()
            if "3d" in valid_projections:
                valid_projections.remove("3d")
            if isinstance(value, str):
                if value == "3d":
                    raise GraphingException("3D projection is not supported.")
                if value not in valid_projections:
                    raise ValueError(f"projection must be one of {valid_projections} or a valid object.")
            elif isinstance(value, WCS):
                warning("WCS projection should be used with the SmartFigureWCS object. This class may not offer all "
                        "needed functionalities.")
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
    def legend_loc(self) -> str | tuple:
        return self._legend_loc

    @legend_loc.setter
    def legend_loc(self, value: str | tuple) -> None:
        if value is not None:
            if isinstance(value, str):
                choices = ["best", "upper right", "upper left", "lower left", "lower right", "right", "center left",
                           "center right", "lower center", "upper center", "center", "outside upper center",
                           "outside center right", "outside lower center", "outside center left"]
                if value not in choices:
                    raise ValueError(f"legend_loc must be one of {choices}.")
                if self._general_legend and value == "best":
                    raise ValueError("legend_loc cannot be 'best' when general_legend is True.")
            elif isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError("legend_loc must be a string or a tuple of length 2.")
            else:
                raise TypeError("legend_loc must be a string or tuple.")
        self._legend_loc = value

    @property
    def legend_cols(self) -> int:
        return self._legend_cols

    @legend_cols.setter
    def legend_cols(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("legend_cols must be an integer.")
        if value < 1:
            raise ValueError("legend_cols must be greater than 0.")
        self._legend_cols = value

    @property
    def show_legend(self) -> bool:
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("show_legend must be a bool.")
        self._show_legend = value

    @property
    def figure_style(self) -> str:
        return self._figure_style

    @figure_style.setter
    def figure_style(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("figure_style must be a string.")
        available_styles = ["default"] + get_styles(matplotlib=True)
        if value not in available_styles:
            raise ValueError(f"figure_style must be one of {available_styles}.")
        self._figure_style = value

    @property
    def show_grid(self) -> bool:
        """
        Whether to show the grid lines on the figure. A grid first needs to be created using the
        :meth: `~graphinglib.smart_figure.SmartFigure.set_grid` method. This can be used to easily toggle the visibility
        of a previously created grid.
        """
        return self._show_grid

    @show_grid.setter
    def show_grid(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("show_grid must be a bool.")
        self._show_grid = value

    @property
    def hide_custom_legend_elements(self) -> bool:
        """
        Whether to hide custom legend elements. This is useful if a custom legend was previously created using the
        :meth:`~graphinglib.smart_figure.SmartFigure.set_custom_legend` method and you want to hide these elements. Each
        SmartFigure controls its own custom legend elements, so if this property is set to True in a nested SmartFigure,
        the custom legend elements will be hidden even if the parent SmartFigure attempts to create a general legend.
        However, both the nested and parent SmartFigures need to set this property to False to display the custom
        elements of a nested SmartFigure in a global general legend.
        """
        return self._hide_custom_legend_elements

    @hide_custom_legend_elements.setter
    def hide_custom_legend_elements(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("hide_custom_legend_elements must be a bool.")
        self._hide_custom_legend_elements = value

    @property
    def hide_default_legend_elements(self) -> bool:
        """
        Whether to hide default legend elements. This is useful if a custom legend was previously created using the
        :meth:`~graphinglib.smart_figure.SmartFigure.set_custom_legend` method and you want to hide the default labels
        created with each :class:`~graphinglib.graph_elements.Plottable` element's label. Each SmartFigure controls its
        own default legend elements, so if this property is set to True in a nested SmartFigure, the default
        elements will be hidden even if the parent SmartFigure attempts to create a general legend. However, both the
        nested and parent SmartFigures need to set this property to False to display the default elements of a nested
        SmartFigure in a global general legend.

        .. warning::
            The use of this property for simply toggling the visibility of the legend is discouraged. Instead, use the
            :meth:`~graphinglib.smart_figure.SmartFigure.show_legend` property to show or hide all the legend elements.
            This should only be used if a custom legend was created.
        """
        return self._hide_default_legend_elements

    @hide_default_legend_elements.setter
    def hide_default_legend_elements(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("hide_default_legend_elements must be a bool.")
        self._hide_default_legend_elements = value

    @property
    def is_single_subplot(self) -> bool:
        """
        Whether the SmartFigure is a single subplot (1x1). This is useful to determine if the SmartFigure can be used
        as a single plot or if it contains multiple subplots.

        .. note::
            This property is used to verify if custom legend elements can be added to the SmartFigure even if the
            :attr:`~graphinglib.smart_figure.SmartFigure.general_legend` is set to ``False``.
        """
        return self.num_rows == 1 and self.num_cols == 1

    def __len__(self) -> int:
        """
        Gives the number of elements in the SmartFigure.
        """
        return len(self._elements)

    def __setitem__(
        self,
        key: int | slice | tuple[int | slice],
        element: Plottable | Iterable[Plottable] | SmartFigure
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
        element : Plottable | Iterable[Plottable] | SmartFigure
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

            subfigure = SmartFigure(num_rows=2, num_cols=1, elements=[gl.Heatmap(data1), gl.Heatmap(data2)])
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
        if not any([
            element is None,
            isinstance(element, (Plottable, SmartFigure)),
            SmartFigure._is_iterable_of_elements(element),
        ]):
            raise TypeError("Element must be a Plottable, an iterable of Plottables, or a SmartFigure.")
        key_ = self._keys_to_slices(self._validate_and_normalize_key(key))
        if element is None:
            self._elements.pop(key_, None)
        else:
            # Normalize all iterables to lists for consistency
            if isinstance(element, Plottable):
                el = [element]
            elif isinstance(element, Iterable):
                el = list(element)
            else:
                el = element
            self._elements[key_] = el

    def __getitem__(self, key: int | slice | tuple[int | slice]) -> list[Plottable] | SmartFigure:
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
        key_ = self._keys_to_slices(self._validate_and_normalize_key(key))
        return self._elements.get(key_, [])

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
        Returns a deep copy of the :class:`~graphinglib.smart_figure.SmartFigure` object.
        """
        return deepcopy(self)

    def copy_with(self, **kwargs) -> Self:
        """
        Returns a deep copy of the SmartFigure with specified attributes overridden. This is useful when including
        SmartFigures in other SmartFigures, as it allows to modify the attributes in a single call.

        Parameters
        ----------
        kwargs
            Properties to override in the copied SmartFigure. The keys should be property names to modify and the values
            are the new values for those properties.

        Returns
        -------
        SmartFigure
            A new SmartFigure instance with the specified attributes overridden.

        Examples
        --------
        Copy an existing SmartFigure to remove the x and y labels::

            fig2 = fig1.copy_with(x_label=None, y_label=None)
        """
        properties = [attr for attr in dir(self.__class__) if isinstance(getattr(self.__class__, attr, None), property)]
        properties = list(filter(lambda x: x[0] != "_", properties))      # filter out hidden properties
        new_copy = deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_copy, key):
                setattr(new_copy, key, value)
            else:
                close_match = get_close_matches(key, properties, n=1, cutoff=0.6)
                if close_match:
                    raise AttributeError(f"SmartFigure has no attribute '{key}'. Did you mean '{close_match[0]}'?")
                else:
                    raise AttributeError(f"SmartFigure has no attribute '{key}'.")
        return new_copy

    @property
    def _ordered_elements(self) -> OrderedDict:
        """
        Gives the _elements dict sorted by the starting position of the slices. This is used to ensure that the
        elements are plotted in the correct order when creating the figure.
        """
        return OrderedDict(sorted(self._elements.items(), key=lambda item: (item[0][0].start, item[0][1].start)))

    def _keys_to_slices(self, keys: tuple[slice | int]) -> tuple[slice]:
        """
        Converts a given two-tuple of integers or slices into a tuple of slices for normalization. The starting or
        ending ``None`` values of slices, if present, are replaced with 0 or the size of the axis respectively.
        """
        new_slices = [k if isinstance(k, slice) else slice(k, k+1, None) for k in keys]   # convert int -> slice
        new_slices = [s if s.start is not None else slice(0, s.stop) for s in new_slices] # convert starting None -> 0
        new_slices = [s if s.stop is not None else slice(s.start, stop)
                      for s, stop in zip(new_slices, (self._num_rows, self._num_cols))]   # convert stop None -> size
        return tuple(new_slices)

    def _validate_and_normalize_key(self, key: int | slice | tuple[int | slice]) -> tuple[int | slice]:
        """
        Validates and normalizes the key for indexing into the SmartFigure. This method ensures that the key is
        either a single integer, a slice, or a tuple of integers/slices. It also checks for out-of-bounds indices and
        raises appropriate exceptions if the key is invalid. The returned key is always a tuple of integers or slices.

        Parameters
        ----------
        key : int | slice | tuple[int | slice]
            The key to validate and normalize.

        Returns
        -------
        tuple[int | slice]
            The normalized key as a tuple of integers or slices.
        """
        if not isinstance(key, tuple):
            key = (key,)

        # 1D SmartFigures
        if self._num_rows == 1 or self._num_cols == 1:
            if len(key) == 1:
                key = (0, key[0]) if self._num_rows == 1 else (key[0], 0)
            elif len(key) != 2:
                raise ValueError("Key must be 1D (int or slice) or 2D with one zero index for 1D SmartFigure.")

        # 2D SmartFigures
        else:
            if len(key) != 2:
                raise ValueError("2D indexing must use a tuple of length 2.")

        # Bounds check
        for i, (k, axis_size) in enumerate(zip(key, (self._num_rows, self._num_cols))):
            if isinstance(k, int):
                if not (0 <= k < axis_size):
                    raise IndexError(f"Index {k} out of bounds for axis {i} with size {axis_size}.")
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else axis_size
                if start < 0 or stop > axis_size:
                    raise IndexError(f"{k} out of bounds for axis {i} with size {axis_size}.")
                if start >= stop:
                    raise IndexError(f"{k} for axis {i} must have stop larger than start.")
                if k.step is not None:
                    raise ValueError(f"{k} step for axis {i} must be None.")
            else:
                raise TypeError(f"Key element {k} for axis {i} must be an int or a slice.")

        return key

    @staticmethod
    def _is_iterable_of_elements(item: Any) -> bool:
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
        return isinstance(item, Iterable) and all(isinstance(el, (Plottable, type(None))) for el in item)

    def add_elements(self, *elements: Plottable) -> None:
        """
        Adds one or more :class:`~graphinglib.graph_elements.Plottable` elements to the
        :class:`~graphinglib.smart_figure.SmartFigure`. This convenience method is equivalent to using __setitem__, but
        only works if the SmartFigure contains a single plot (1x1). Otherwise, the __setitem__ method should be used.

        Parameters
        ----------
        elements : :class:`~graphinglib.graph_elements.Plottable`
            Elements to plot in the :class:`~graphinglib.smart_figure.SmartFigure`.

        See Also
        --------
        :meth:`~graphinglib.smart_figure.SmartFigure.__setitem__`
            For more information on how to use the __setitem__ method to add elements to the SmartFigure.
        """
        if self._num_rows != 1 or self._num_cols != 1:
            raise GraphingException("The add_elements() method only works for 1x1 SmartFigures.")
        self[0,0] += elements

    def show(
        self,
        fullscreen: bool = False,
    ) -> None:
        """
        Plots and displays the :class:`~graphinglib.smart_figure.SmartFigure`. The
        :meth:`~graphinglib.smart_figure.SmartFigure.save` method is recommended to see properly what the figure looks
        like, as the display may not show the full figure or the appropriate spacings in some cases.

        .. warning::
            If the SmartFigure contains a general legend and the legend location is set to an "outside" position, it may
            not be displayed correctly in matplotlib windows. Inline figures in a Jupyter notebook or saving the figure
            to a file using the :meth:`~graphinglib.smart_figure.SmartFigure.save` method are recommended to get the
            figure properly displayed.

        Parameters
        ----------
        fullscreen : bool, optional
            If True, the figure will be displayed in fullscreen mode.
            Defaults to ``False``.
        """
        self._initialize_parent_smart_figure()

        # Create an artificial axis to add padding around the figure
        # This is needed because the figure is created with h_pad=0 and w_pad=0 creating 0 padding
        ax_dummy = self._figure.add_subplot(self._gridspec[:, :])
        ax_dummy.xaxis.grid(False)
        ax_dummy.yaxis.grid(False)
        ax_dummy.set_facecolor((0, 0, 0, 0))
        ax_dummy.set_zorder(-1)
        ax_dummy.set_navigate(False)
        ax_dummy.tick_params(colors=(0,0,0,0), axis="both", direction="in",
                             labelright=True, labeltop=True, labelsize=0.01)
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
            warning("The general legend location is set to an 'outside' position and matplotlib windows may not be "
                    "able to show it properly. Consider using inline figures in a jupyter notebook or saving the "
                    "figure to a file instead to get the full figure.")

        if fullscreen:
            plt.get_current_fig_manager().full_screen_toggle()

        plt.show()
        if not any(plt.fignum_exists(num) for num in plt.get_fignums()):    # check if the parameters can be reset
            plt.rcParams.update(plt.rcParamsDefault)
            self._figure.clear()
            self._figure = None
            self._gridspec = None

    def save(
        self,
        file_name: str,
        dpi: Optional[int] = None,
        transparent: bool = False,
    ) -> None:
        """
        Saves the :class:`~graphinglib.smart_figure.SmartFigure` to a file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the figure to. The file extension determines the format (e.g., .png, .pdf).
        dpi : int, optional
            The resolution in dots per inch. If None, the figure's DPI is used.
        transparent : bool, optional
            Whether to save the figure with a transparent background.
            Defaults to ``False``.
        """
        self._initialize_parent_smart_figure()
        plt.savefig(
            file_name,
            bbox_inches="tight",
            dpi=dpi if dpi is not None else "figure",
            transparent=transparent,
        )
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)
        self._figure.clear()
        self._figure = None
        self._gridspec = None

    def _initialize_parent_smart_figure(
        self,
    ) -> None:
        """
        Initializes the parent :class:`~graphinglib.smart_figure.SmartFigure` for plotting. This method initializes the
        appropriate figure style, parameters and matplotlib figure and calls the
        :meth:`~graphinglib.smart_figure.SmartFigure._prepare_figure` method.
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

        parent_figure_params_to_reset = self._fill_in_missing_params(self)  # Fill "default" parameters
        self._default_params["rc_params"].update(self._user_rc_dict)  # Custom rc parameters supersede the defaults
        self._fill_in_rc_params(is_matplotlib_style)

        # The following try/except removes lingering figures when errors occur during the plotting process
        try:
            self._figure = plt.figure(constrained_layout=True, figsize=self._size)
            self._figure.get_layout_engine().set(w_pad=0, h_pad=0)
            self._reference_label_i = 0
            self._prepare_figure(is_matplotlib_style)
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
            argument is passed to the :meth:`~graphinglib.smart_figure.SmartFigure._fill_in_rc_params` method, and
            determines if missing plottable parameters should be filled in.
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
            :meth:`~graphinglib.smart_figure.SmartFigure.set_custom_legend` method. This is used to create a general
            legend for the entire SmartFigure and keeping trach of the default and custom elements to use the
            :attr:`~graphinglib.smart_figure.SmartFigure.hide_default_legend_elements` and
            :attr:`~graphinglib.smart_figure.SmartFigure.hide_custom_legend_elements` properties.
        """
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
        default_labels, default_handles = [], []
        custom_labels, custom_handles = [], []
        for (rows, cols), element in self._ordered_elements.items():
            if isinstance(element, SmartFigure):
                element._default_params = deepcopy(self._default_params)
                element._default_params["rc_params"].update(element._user_rc_dict)
                plt.rcParams.update(element._default_params["rc_params"])
                subfig_params_to_reset = element._fill_in_missing_params(element)  # Fill "default" parameters

                subfig = self._figure.add_subfigure(self._gridspec[rows, cols])
                element._figure = subfig        # associates the current subfigure with the nested SmartFigure
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

                plt.rcParams.update(self._default_params["rc_params"])  # Return to the parent SmartFigure's rc params
                element._reset_params_to_default(element, subfig_params_to_reset)
                element._default_params = {}

            elif isinstance(element, (Plottable, list)):
                current_elements = element if isinstance(element, list) else [element]
                subfig = self._figure.add_subfigure(self._gridspec[rows, cols])
                ax = subfig.add_subplot(
                    sharex=ax if self._share_x else None,  # This enables the coherent zoom and pan of the axes
                    sharey=ax if self._share_y else None,  # but it does not remove the ticklabels
                    projection=self._projection,
                )

                # Plotting loop
                z_order = 2
                for index, current_element in enumerate(current_elements):
                    if isinstance(current_element, Plottable):
                        params_to_reset = []
                        if not is_matplotlib_style:
                            params_to_reset = self._fill_in_missing_params(current_element)
                        if isinstance(current_element, Text) and current_element.relative_to == "figure":
                            target = subfig
                        else:
                            target = ax
                        current_element._plot_element(
                            target,
                            z_order,
                            cycle_color=cycle_colors[index % num_cycle_colors],
                        )
                        if not is_matplotlib_style:
                            self._reset_params_to_default(current_element, params_to_reset)
                        try:
                            if current_element.label is not None:
                                default_handles.append(current_element.handle)
                                default_labels.append(current_element.label)
                        except AttributeError:
                            continue
                        z_order += 5
                    elif current_element is not None:
                        raise GraphingException(f"Unsupported element type: {type(current_element).__name__}")

                # If only text objects were plotted, the axes is hidden and the other properties are not set
                if all(isinstance(element_i, Text) for element_i in element):
                    ax.axis("off")
                else:
                    # Add reference label
                    if self._reference_labels and (len(self) > 1 or isinstance(self._figure, SubFigure)):
                        self._create_ref_label(ax)

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

                    self._customize_ticks(ax)

                    # If axes are shared, manually remove ticklabels from unnecessary plots as it is not done
                    # automatically when adding subplots
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
                    if self._show_grid:
                        ax.grid(self._grid_visible_x, which=self._grid_which_x, axis="x")
                        ax.grid(self._grid_visible_y, which=self._grid_which_y, axis="y")
                        if self._grid_show_on_top:
                            ax.set_axisbelow(False)

                    # Axes legend
                    if not self._general_legend and make_legend:
                        if self._hide_default_legend_elements:
                            default_labels = []
                            default_handles = []
                        if self.is_single_subplot:
                            custom_labels += self._custom_legend_labels
                            custom_handles += self._custom_legend_handles
                        if self._hide_custom_legend_elements or not self.is_single_subplot:
                            custom_labels = []
                            custom_handles = []
                        labels = default_labels + custom_labels
                        handles = default_handles + custom_handles

                        if self._show_legend and labels:
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
                        default_labels, default_handles = [], []
                        custom_labels, custom_handles = [], []

                    # Axes title (if the SmartFigure is a single subplot)
                    if self._title and self.is_single_subplot:
                        ax.set_title(self._title)

            elif element is not None:
                raise GraphingException(f"Unsupported element type in list: {type(element).__name__}")

        # Axes labels
        if self.is_single_subplot:
            if ax is not None:  # makes sure an element was plotted and that an axis was created
                self._customize_ax_label(ax)
        else:
            suplabel_params = {
                "fontsize" : plt.rcParams["font.size"],
                "color" : plt.rcParams["axes.labelcolor"],
                "fontweight" : plt.rcParams["font.weight"],
            }
            self._figure.supxlabel(self._x_label, **suplabel_params)
            self._figure.supylabel(self._y_label, **suplabel_params)

        # Title (if the SmartFigure is not a single subplot)
        if self._title and not self.is_single_subplot:
            self._figure.suptitle(self._title)

        # General legend
        custom_labels += self._custom_legend_labels
        custom_handles += self._custom_legend_handles
        if self._hide_default_legend_elements:
            default_labels = []
            default_handles = []
        if self._hide_custom_legend_elements:
            custom_labels = []
            custom_handles = []
        labels = default_labels + custom_labels
        handles = default_handles + custom_handles
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
            legend_info = {
                "labels": {"default": [], "custom": []},
                "handles": {"default": [], "custom": []},
            }
        else:
            legend_info = {
                "labels": {"default": default_labels, "custom": custom_labels},
                "handles": {"default": default_handles, "custom": custom_handles},
            }
        return legend_info

    def _customize_ticks(
        self,
        ax: Axes,
    ) -> None:
        """
        Customizes the ticks of the specified Axes according to the SmartFigure's tick parameters. This method is useful
        for inheritance to allow each SmartFigure class to customize the ticks their way.
        """
        if self._x_ticks is not None:
            ax.set_xticks(self._x_ticks, self._x_tick_labels)
        ax.tick_params(axis="x", which="major", **self._tick_params["x major"])
        if self._x_tick_spacing is not None:
            ax.xaxis.set_major_locator(
                ticker.MultipleLocator(self._x_tick_spacing)
            )

        if self._y_ticks is not None:
            ax.set_yticks(self._y_ticks, self._y_tick_labels)
        ax.tick_params(axis="y", which="major", **self._tick_params["y major"])
        if self._y_tick_spacing is not None:
            ax.yaxis.set_major_locator(
                ticker.MultipleLocator(self._y_tick_spacing)
            )

        if self._minor_x_ticks is not None:
            ax.set_xticks(self._minor_x_ticks, minor=True)
        ax.tick_params(axis="x", which="minor", **self._tick_params["x minor"])
        if self._minor_x_tick_spacing is not None:
            ax.xaxis.set_minor_locator(
                ticker.MultipleLocator(self._minor_x_tick_spacing)
            )

        if self._minor_y_ticks is not None:
            ax.set_yticks(self._minor_y_ticks, minor=True)
        ax.tick_params(axis="y", which="minor", **self._tick_params["y minor"])
        if self._minor_y_tick_spacing is not None:
            ax.yaxis.set_minor_locator(
                ticker.MultipleLocator(self._minor_y_tick_spacing)
            )

        # Remove ticks
        if self._remove_x_ticks:
            ax.tick_params("x", which="both", labelbottom=False, labeltop=False, bottom=False, top=False)
        if self._remove_y_ticks:
            ax.tick_params("y", which="both", labelleft=False, labelright=False, left=False, right=False)

    def _customize_ax_label(
        self,
        ax: Axes,
    ) -> None:
        """
        Customizes the x and y labels of the specified Axes according to the SmartFigure's label parameters. This method
        is useful for inheritance to allow each SmartFigure class to customize the labels their way.
        """
        ax.set_xlabel(self._x_label)
        ax.set_ylabel(self._y_label)

    def _create_ref_label(
        self,
        target: Axes | Figure | SubFigure
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
        target.text(
            0,
            1,
            ascii_lowercase[self._reference_label_i] + ")",
            transform=trans + self._get_reflabel_translation(target)
        )
        self._reference_label_i += 1

    def _get_reflabel_translation(
        self,
        target: Axes | Figure | SubFigure,
    ) -> ScaledTranslation:
        """
        Gives the translation to apply to the reference label to position it correctly relative to an Axes, Figure or
        SubFigure. The translation varies depending on the location of the reference label.
        """
        if isinstance(target, Axes):
            if self._reflabel_loc == "outside":
                return ScaledTranslation(-5 / 72, 10 / 72, self._figure.dpi_scale_trans)
            elif self._reflabel_loc == "inside":
                return ScaledTranslation(10 / 72, -15 / 72, self._figure.dpi_scale_trans)
            else:
                raise ValueError("Invalid reference label location. Please specify either 'inside' or 'outside'.")

        elif isinstance(target, (Figure, SubFigure)):
            return ScaledTranslation(7 / 72, -10 / 72, self._figure.dpi_scale_trans)
        else:
            raise ValueError("Target must be either an Axes, Figure or SubFigure instance.")

    def _get_legend_params(
        self,
        labels: list[str],
        handles: list[Any],
        outside_lower_center_y_offset: float,
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

        Returns
        -------
        dict[str, Any]
            The parameters to use for the legend, that may be passed to the
            :meth:`matplotlib.axes.Axes.legend` or :meth:`matplotlib.figure.Figure.legend` methods as keyword arguments.
        """
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
        if self._legend_loc is None:
            if self._general_legend:
                legend_params.update({"loc": "lower center"})
            else:
                legend_params.update({"loc": "best"})
        else:
            if "outside" in self._legend_loc:
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

    def _fill_in_missing_params(self, element: SmartFigure | Plottable) -> list[str]:
        """
        Fills in the missing parameters for a ``SmartFigure``or a ``Plottable`` from the specified ``figure_style``.
        """
        params_to_reset = []
        # The following logic enables figures that inherit from SmartFigure to use the same default parameters
        object_type = "SmartFigure" if isinstance(element, SmartFigure) else type(element).__name__
        for try_i in range(2):
            try:
                for property_, value in vars(element).items():
                    if (type(value) == str) and (value == "default") and not (property_ == "_figure_style"):
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
                        f"{self._figure_style} style file:\n {e.args[0]}"
                    )
                file_updater = FileUpdater(self._figure_style)
                file_updater.update()
                file_loader = FileLoader(self._figure_style)
                new_defaults = file_loader.load()
                self._default_params.update((k, v) for k, v in new_defaults.items() if k not in self._default_params)
        return params_to_reset

    def _reset_params_to_default(
        self, element: Plottable | SmartFigure, params_to_reset: list[str]
    ) -> None:
        """
        Resets the parameters that were set to default in the
        :meth:`~graphinglib.smart_figure.SmartFigure._fill_in_missing_params` method.
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
        for property_, value in rc_params_dict.items():
            self._user_rc_dict[property_] = value

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
        y_ticks: Optional[list[float]] = None,
        x_tick_labels: Optional[list[str]] = None,
        y_tick_labels: Optional[list[str]] = None,
        x_tick_spacing: Optional[float] = None,
        y_tick_spacing: Optional[float] = None,
        minor_x_ticks: Optional[list[float]] = None,
        minor_y_ticks: Optional[list[float]] = None,
        minor_x_tick_spacing: Optional[float] = None,
        minor_y_tick_spacing: Optional[float] = None,
    ) -> None:
        """
        Sets custom ticks and tick labels.

        Parameters
        ----------
        x_ticks, y_ticks : list[float], optional
            Tick positions for the x or y axis. If a value is specified, the corresponding ``x_tick_spacing`` or
            ``y_tick_spacing`` parameter must be ``None``.
        x_tick_labels, y_tick_labels : list[str], optional
            Tick labels for the x or y axis. If a value is specified, the corresponding ``x_ticks`` or ``y_ticks``
            parameter must also be given.
        x_tick_spacing, y_tick_spacing : float, optional
            Spacing between ticks on the x or y axis. If a value is specified, the corresponding ``x_ticks`` or
            ``y_ticks`` parameter must be ``None``.
        minor_x_ticks, minor_y_ticks : list[float], optional
            Minor tick positions for the x or y axis. If a value is specified, the corresponding
            ``minor_x_tick_spacing`` or ``minor_y_tick_spacing`` parameter must be ``None``.
        minor_x_tick_spacing, minor_y_tick_spacing : float, optional
            Spacing between minor ticks on the x or y axis. If a value is specified, the corresponding ``minor_x_ticks``
            or ``minor_y_ticks`` parameter must be ``None``.
        """
        if any([
            (x_tick_labels is not None) and x_ticks is None,
            (y_tick_labels is not None) and y_ticks is None,
        ]):
            raise GraphingException("Ticks position must be specified when ticks labels are specified")

        if any([
            (x_ticks is not None) and (x_tick_spacing is not None),
            (y_ticks is not None) and (y_tick_spacing is not None),
            (minor_x_ticks is not None) and (minor_x_tick_spacing is not None),
            (minor_y_ticks is not None) and (minor_y_tick_spacing is not None),
        ]):
            raise GraphingException("Tick spacing and tick positions cannot be set simultaneously")

        self._x_ticks = x_ticks
        self._y_ticks = y_ticks
        self._x_tick_labels = x_tick_labels
        self._y_tick_labels = y_tick_labels
        self._x_tick_spacing = x_tick_spacing
        self._y_tick_spacing = y_tick_spacing
        self._minor_x_ticks = minor_x_ticks
        self._minor_y_ticks = minor_y_ticks
        self._minor_x_tick_spacing = minor_x_tick_spacing
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
            If ``True``, all previously given tick parameters are reset to their default values before applying the new
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
        draw_bottom_tick, draw_top_tick, draw_left_tick, draw_right_tick : bool, optional
            Whether to draw the ticks on the bottom, top, left or right side of the axes respectively.
        draw_bottom_label, draw_top_label, draw_left_label, draw_right_label : bool, optional
            Whether to draw the tick labels on the bottom, top, left or right side of the axes respectively.
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
        which_x: Literal["major", "minor", "both"] = "both",
        which_y: Literal["major", "minor", "both"] = "both",
        color: str | Literal["default"] = "default",
        alpha: float | Literal["default"] = "default",
        line_style: str | Literal["default"] = "default",
        line_width: float | Literal["default"] = "default",
    ) -> None:
        """
        Sets the grid parameters for the figure.

        Parameters
        ----------
        visible_x, visible_y : bool, optional
            If ``True``, sets the x-axis or y-axis grid visible. If ``False``, the grid is not shown for the respective
            axis.
            Defaults to ``True`` for both axes.
        show_on_top : bool, optional
            If ``True``, sets the grid lines to be shown on top of the plot elements. This can be useful to see the grid
            lines above a plotted :class:`~graphinglib.data_plotting_2d.Heatmap` for example.
            Defaults to ``False``.
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

    def set_custom_legend(
        self,
        handles: Optional[list[Artist]] = [],
        labels: Optional[list[str]] = [],
        reset: bool = False,
    ) -> None:
        """
        Sets a custom legend for the figure. If the SmartFigure contains multiple subfigures, custom legends only work
        if the ``general_legend`` parameter is set to ``True``. Otherwise, custom legends can be added for non-general
        legends if the SmartFigure is a single subplot (see the
        :attr:`~graphinglib.smart_figure.SmartFigure.is_single_subplot` property).

        .. note::
            The visibility of default or custom legend elements can be controlled individually with the
            :attr:`~graphinglib.smart_figure.SmartFigure.hide_default_legend_elements` and
            :attr:`~graphinglib.smart_figure.SmartFigure.hide_custom_legend_elements` properties.

        Parameters
        ----------
        handles : list[Artist], optional
            Handles to add to the legend. Any object accepted by the :function:`~matplotlib.pyplot.legend` function can
            be used. These can be for example :class:`~matplotlib.lines.Line2D` or :class:`~matplotlib.patches.Patch`
            objects. See the matplotlib documentation for details:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend
        labels : list[str], optional
            List of labels for the legend. If the handles directly contain the labels, this parameter can be set to
            ``None``. However, if given, the length of the labels list must match the length of the handles list.

            .. note::
                If labels are given, the handles must also be given.
        reset : bool, optional
            Whether or not to reset the custom handles and labels previously added with this method before adding the
            new ones.
            Defaults to ``False``.
        """
        if labels and not handles:
            raise GraphingException("Handles must be specified if labels are given.")

        if not labels:
            new_labels = [handle.get_label() for handle in handles]
        else:
            new_labels = labels.copy()
            if len(handles) != len(labels):
                raise GraphingException("If labels are given, their number must match the number of handles.")

        if reset:
            self._custom_legend_handles = []
            self._custom_legend_labels = []

        self._custom_legend_handles += handles
        self._custom_legend_labels += new_labels


class SmartFigureWCS(SmartFigure):
    """
    This class implements a figure object for plotting :class:`~graphinglib.graph_elements.Plottable` elements. It
    allows for the creation of complex figures recursively, where each :class:`~graphinglib.smart_figure.SmartFigure`
    can contain other :class:`~graphinglib.smart_figure.SmartFigure` objects. The class supports a variety of
    customization options as well as the ability to use styles and themes for consistent visual appearance across
    different figures. The idea behind this class is that every SmartFigure contains a single x_label, y_label, title,
    projection, etc. and that nested SmartFigures can be inserted into the main SmartFigure to create complex figures
    with more parameters.

    Parameters
    ----------
    projection : WCS
        The World Coordinate System (WCS) object to use for the figure. This is used to plot data in a coordinate system
        that is not Cartesian, such as celestial coordinates.
    num_rows, num_cols : int
        Number of rows and columns for the base grid. These parameters determine the number of "squares" on which the
        plots can be placed.
    x_label, y_label : str, optional
        Labels for the x and y axes of the figure.
    size : tuple[float, float]
        Overall size of the multifigure.
        Default depends on the ``figure_style`` configuration.
    title : str, optional
        General title of the figure.
    x_lim, y_lim : tuple[float, float], optional
        Limits for the x and y axes of the figure.
    log_scale_x, log_scale_y : bool
        Whether to use a logarithmic scale for the x and y axes, respectively.
        Defaults to ``False``.
    remove_axes : bool
        Whether to remove the axes from the figure.
        Defaults to ``False``.
    aspect_ratio : float | Literal["auto", "equal"]
        Aspect ratio of the figure. If set to "auto", the aspect ratio is determined automatically to fill the available
        space. If set to "equal", the aspect ratio is set to 1:1. If set to a float, the aspect ratio represents the
        ratio of the height to the width of the figure.
        Defaults to "auto".
    remove_x_ticks, remove_y_ticks : bool
        Whether to remove the x and y ticks from the figure, respectively.
        Defaults to ``False``.
    reference_labels : bool
        Whether or not to add reference labels to the subfigures. If set to ``True``, each subfigure will be labeled
        alphabetically in the form of "a)", "b)", etc.
        Defaults to ``True``.

        .. note::
            For nested figures, each subfigure controls its own reference labels. This means that if a nested
            SmartFigure turns off reference labels, the plots in it will not be labeled, even if the parent SmartFigure
            has reference labels turned on.

    global_reference_label : bool
        Whether to use a single reference label for the entire figure instead of individual labels for each subfigure.
        If set to ``True``, the reference label will be placed in the top left corner of the global SmartFigure. This is
        useful for labeling the entire figure rather than individual subfigures.
        Defaults to ``False``.
    reflabel_loc : Literal["inside", "outside"]
        Location of the reference labels of the SubFigures, either "inside" or "outside".
        Defaults to ``"outside"``.
    width_padding, height_padding : float
        Padding between the subfigures in the x and y directions, respectively. The default value of ``None`` results in
        a default small amount of padding. This may be set to 0 to completely remove the space between subfigures, but
        note that axes labels may need to be removed to delete additional space.
    width_ratios, height_ratios : ArrayLike
        Ratios of the widths and heights of the subfigures, respectively. These ratios determine how much space each
        column and row of subfigures will take up in the overall figure. The length of these arrays must match the
        number of columns and rows, respectively. By default, all subfigures are given equal space.
    share_x, share_y : bool
        Whether to share the x and y axes between subfigures, respectively. This means that all subfigures will have
        the same x and y limits, and the ticks will be shared as well. This is useful for comparing data across
        subfigures.

        .. note::
            Sharing axes only works for plots directly inside the SmartFigure. If a nested SmartFigure is used, the
            axes sharing will not be applied to the nested SmartFigure. Instead, the nested SmartFigure will have its
            own axes sharing settings.

    general_legend : bool
        Whether to create a general legend for the entire figure. If set to ``True``, a single legend will be created
        to regroup all the legends from the subplots. If set to ``False``, all subplots will have their own legend. If
        nested SmartFigures set this parameter to ``False``, their legend is added to the parent's general legend.
        However, if a nested SmartFigure sets its general legend to ``True``, it will be created separately and will not
        be added to the parent's general legend.
        Defaults to ``False``.
    legend_loc : str | tuple, optional
        Location of the legend. This can be a string (e.g., "upper right") or a tuple of (x, y) relative coordinates.
        The supported string locations are: {"upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center", "center", "outside upper center",
        "outside center right", "outside lower center", "outside center left"}. Additionally, only if ``general_legend``
        is set to ``False``, the legend location can also be set to "best".
        Defaults to ``"best"`` if ``general_legend`` is set to ``False``, otherwise it defaults to ``"lower center"``.

        .. warning::
            If ``general_legend`` is set to ``True`` and the legend location is set to a position containing "outside",
            the legend may not be displayed correctly in some matplotlib backends. In such cases, it is recommended to
            use inline figures in a Jupyter notebook or save the figure to a file to ensure proper display of the
            legend outside the figure.

    legend_cols : int
        Number of columns to display the labels in the legend. This is only used if the legend is displayed.
        Defaults to ``1``.
    show_legend : bool
        Whether to show the legend for the figure. This allows to easily toggle the visibility of the legend.
        Defaults to ``True``.
    figure_style : str
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
        left in the figure. If empty lists or iterables containing only ``None`` are given in the list, the
        corresponding subplots are drawn but empty.

        .. note::
            This method for adding elements only allows to add elements to single subplots. If you want to add elements
            that span multiple subplots, you should use the __setitem__ method instead.
            For example, to add an element spanning the complete first row , use ``fig[0,:] = element``.
    """
    def __init__(
        self,
        projection: WCS,
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
        aspect_ratio: float | Literal["auto", "equal"] = "auto",
        remove_x_ticks: bool = False,
        remove_y_ticks: bool = False,
        reference_labels: bool = True,
        global_reference_label: bool = False,
        reflabel_loc: Literal["inside", "outside"] = "outside",
        width_padding: float = None,
        height_padding: float = None,
        width_ratios: ArrayLike = None,
        height_ratios: ArrayLike = None,
        share_x: bool = False,
        share_y: bool = False,
        general_legend: bool = False,
        legend_loc: Optional[str | tuple] = None,
        legend_cols: int = 1,
        show_legend: bool = True,
        figure_style: str = "default",
        elements: Optional[Iterable[Plottable | SmartFigure] | Iterable[Iterable[Plottable | SmartFigure]]] = [],
    ) -> None:
        super().__init__(
            num_rows=num_rows,
            num_cols=num_cols,
            x_label=x_label,
            y_label=y_label,
            size=size,
            title=title,
            x_lim=x_lim,
            y_lim=y_lim,
            log_scale_x=log_scale_x,
            log_scale_y=log_scale_y,
            remove_axes=remove_axes,
            aspect_ratio=aspect_ratio,
            remove_x_ticks=remove_x_ticks,
            remove_y_ticks=remove_y_ticks,
            reference_labels=reference_labels,
            global_reference_label=global_reference_label,
            reflabel_loc=reflabel_loc,
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
            figure_style=figure_style,
            elements=elements,
        )

        self._number_of_x_ticks = None
        self._number_of_y_ticks = None
        self._x_tick_formatter = None
        self._y_tick_formatter = None
        self._minor_x_tick_frequency = None
        self._minor_y_tick_frequency = None
        self._default_tick_params = {       # The following are the default parameters of WCSAxes objects
            "x major": {"bottom" : True, "top" : True, "labelbottom" : True},
            "y major": {"left" : True, "right" : True, "labelleft" : True},
            "x minor": {},
            "y minor": {},
        }
        self._tick_params = deepcopy(self._default_tick_params)

    @property
    def projection(self) -> Any:
        return self._projection

    @projection.setter
    def projection(self, value: WCS) -> None:
        if not isinstance(value, WCS):
            raise GraphingException("The projection of a SmartFigureWCS must be a WCS object.")
        self._projection = value

    def _customize_ticks(
        self,
        ax: Axes,
    ) -> None:
        """
        Customizes the ticks of the specified Axes according to the SmartFigure's tick parameters. This method is useful
        for inheritance to allow each SmartFigure class to customize the ticks their way.
        """
        x_axis, y_axis = ax.coords
        x_axis.set_auto_axislabel(False)
        y_axis.set_auto_axislabel(False)

        if self._x_tick_formatter is not None:
            x_axis.set_major_formatter(self._x_tick_formatter)
        if self._y_tick_formatter is not None:
            y_axis.set_major_formatter(self._y_tick_formatter)

        x_axis.set_ticks(values=self._x_ticks, spacing=self._x_tick_spacing, number=self._number_of_x_ticks)
        y_axis.set_ticks(values=self._y_ticks, spacing=self._y_tick_spacing, number=self._number_of_y_ticks)

        if self._minor_x_tick_frequency is not None:
            x_axis.display_minor_ticks(True)
            x_axis.set_minor_frequency(self._minor_x_tick_frequency)
        if self._minor_y_tick_frequency is not None:
            y_axis.display_minor_ticks(True)
            y_axis.set_minor_frequency(self._minor_y_tick_frequency)

        # Manually set the tick_params using the recommended API
        for i, axis, ax_params in zip(
            ["x", "y"], [x_axis, y_axis], [self._tick_params["x major"], self._tick_params["y major"]]
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
                tick_pos_str = "".join(pos[0] for pos in valid_tick_pos if ax_params.get(pos))
                label_pos_str = "".join(pos[5] for pos in valid_label_pos if ax_params.get(pos))
            else:  # i == "y"
                valid_tick_pos = ["left", "right"]
                valid_label_pos = ["labelleft", "labelright"]
                tick_pos_str = "".join(pos[0] for pos in valid_tick_pos if ax_params.get(pos))
                label_pos_str = "".join(pos[5] for pos in valid_label_pos if ax_params.get(pos))

            if tick_pos_str:
                axis.set_ticks_position(tick_pos_str)
            else:
                axis.set_ticks_visible(False)

            if label_pos_str:
                axis.set_ticklabel_position(label_pos_str)
            else:
                axis.set_ticklabel_visible(False)

        if self._tick_params[f"x minor"].get("length") is not None:
            ax.tick_params(axis="x", which="minor", length=self._tick_params[f"x minor"].get("length"))
        if self._tick_params[f"y minor"].get("length") is not None:
            ax.tick_params(axis="y", which="minor", length=self._tick_params[f"y minor"].get("length"))

        # Remove ticks
        if self._remove_x_ticks:
            x_axis.set_ticks_visible(False)
            x_axis.set_ticklabel_visible(False)
        if self._remove_y_ticks:
            y_axis.set_ticks_visible(False)
            y_axis.set_ticklabel_visible(False)

    def _customize_ax_label(
        self,
        ax: Axes,
    ) -> None:
        """
        Customizes the x and y labels of the specified Axes according to the SmartFigure's label parameters. This method
        is useful for inheritance to allow each SmartFigure class to customize the labels their way.

        .. note::
            This method converts the given ``axes.labelpad`` value so the default value of matplotlib (``4.0``) is
            converted to the default value of class:`astropy.visualization.wcsaxes.WCSAxes` (``1.0``). The given
            ``axes.labelpad`` value is divided by ``4`` to achieve this conversion.
        """
        x_axis, y_axis = ax.coords
        if self._x_label is not None:
            x_axis.set_axislabel(self._x_label, minpad=plt.rcParams["axes.labelpad"] / 4)
        if self._y_label is not None:
            y_axis.set_axislabel(self._y_label, minpad=plt.rcParams["axes.labelpad"] / 4)

    def set_ticks(
        self,
        x_ticks: Optional[list[Quantity]] = None,
        y_ticks: Optional[list[Quantity]] = None,
        x_tick_spacing: Optional[Quantity] = None,
        y_tick_spacing: Optional[Quantity] = None,
        number_of_x_ticks: Optional[int] = None,
        number_of_y_ticks: Optional[int] = None,
        x_tick_formatter: Optional[Callable | str] = None,
        y_tick_formatter: Optional[Callable | str] = None,
        minor_x_tick_frequency: Optional[int] = None,
        minor_y_tick_frequency: Optional[int] = None,
    ) -> None:
        """
        Sets custom ticks and tick labels.

        x_ticks, y_ticks : list[Quantity], optional
            Tick positions for the x or y axis. If a value is specified, the corresponding ``x_tick_spacing`` and
            ``number_of_x_ticks`` or ``y_tick_spacing`` and ``number_of_y_ticks`` parameters must be ``None``.
        x_tick_spacing, y_tick_spacing : Quantity, optional
            Spacing between ticks on the x or y axis. If a value is specified, the corresponding ``x_ticks`` and
            ``number_of_x_ticks`` or ``y_ticks`` and ``number_of_y_ticks`` parameters must be ``None``.
        number_of_x_ticks, number_of_y_ticks : int, optional
            Number of ticks to display on the x or y axis. If specified, the ``x_ticks`` and ``x_tick_spacing`` or
            ``y_ticks`` and ``y_tick_spacing`` parameters must be ``None``.

            .. note::
                This value is not absolute, but rather a suggestion to the WCSAxes. The actual number of ticks
                displayed may vary depending on the data and limits of the axes.

        x_tick_formatter, y_tick_formatter : Callable | str, optional
            A function or a string format to apply to the x or y tick labels. If a function is provided, it should take
            a single argument (the tick value) and return a formatted string. If a string is provided, it should be a
            format string that will be applied to each tick value. See the astropy documentation for more details:
            https://docs.astropy.org/en/latest/visualization/wcsaxes/ticks_labels_grid.html

            .. example::
                >>> x_tick_formatter = "hh:mm:ss.s"

                ``1h01m34.1s``

                >>> x_tick_formatter = lambda x: f"{x:.2f} s"

                ``1.23 s``

        minor_x_tick_frequency, minor_y_tick_frequency : float, optional
            Frequency of minor ticks on the x or y axis. This gives the number of minor ticks between each major tick.

            .. note::
                The frequency includes the major tick, so a frequency of 2 means that there is one minor tick between
                each major tick.
        """
        super().set_ticks(
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_spacing=x_tick_spacing,
            y_tick_spacing=y_tick_spacing,
        )
        if any([
            (x_ticks is not None) and (number_of_x_ticks is not None),
            (y_ticks is not None) and (number_of_y_ticks is not None),
        ]):
            raise GraphingException("Number of ticks and tick positions cannot be set simultaneously")

        if any([
            (x_tick_spacing is not None) and (number_of_x_ticks is not None),
            (y_tick_spacing is not None) and (number_of_y_ticks is not None),
        ]):
            raise GraphingException("Number of ticks and tick spacing cannot be set simultaneously")

        self._number_of_x_ticks = number_of_x_ticks
        self._number_of_y_ticks = number_of_y_ticks
        self._x_tick_formatter = x_tick_formatter
        self._y_tick_formatter = y_tick_formatter
        self._minor_x_tick_frequency = minor_x_tick_frequency
        self._minor_y_tick_frequency = minor_y_tick_frequency

    def set_tick_params(
        self,
        axis: Optional[Literal["x", "y", "both"]] = "both",
        reset: Optional[bool] = False,
        direction: Optional[Literal["in", "out"]] = None,
        length: Optional[float] = None,
        minor_length: Optional[float] = None,
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
        """
        Sets the tick parameters for the figure. These parameters are given to the
        :meth:`astropy.visualization.wcsaxes.coordinate_helpers.tick_params` method.

        .. warning::
            Due to how the :class:`astropy.visualization.wcsaxes.WCSAxes` are implemented, only the length of the minor
            ticks can be controlled independently from the major ticks. The other parameters are applied to both major
            and minor ticks for a specified axis.

        Parameters
        ----------
        axis : {"x", "y", "both"}, optional
            The axis to set the tick parameters for. This method can be called multiple times to set the tick
            parameters specifically for each axes.
            Defaults to ``"both"``.
        reset : bool, optional
            If ``True``, all previously given tick parameters are reset to their default values before applying the new
            parameters.
            Defaults to ``False``.
        direction : {"in", "out"}, optional
            The direction of the ticks.

            .. warning::
                Contrary to the :meth:`~graphinglib.smart_figure.SmartFigure.set_tick_params` method, the ``direction``
                parameter cannot be set to ``"inout"`` due to how :class:`astropy.visualization.wcsaxes.WCSAxes` work.
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
        draw_bottom_tick, draw_top_tick, draw_left_tick, draw_right_tick : bool, optional
            Whether to draw the ticks on the bottom, top, left or right side of the axes respectively.
        draw_bottom_label, draw_top_label, draw_left_label, draw_right_label : bool, optional
            Whether to draw the tick labels on the bottom, top, left or right side of the axes respectively.
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
            "bottom": draw_bottom_tick,
            "top": draw_top_tick,
            "left": draw_left_tick,
            "right": draw_right_tick,
            "labelbottom": draw_bottom_label,
            "labeltop": draw_top_label,
            "labelleft": draw_left_label,
            "labelright": draw_right_label,
        }
        for axis_i in [axis] if axis != "both" else ["x", "y"]:
            if reset:
                self._tick_params[f"{axis_i} major"] = deepcopy(self._default_tick_params[f"{axis_i} major"])
                self._tick_params[f"{axis_i} minor"] = deepcopy(self._default_tick_params[f"{axis_i} minor"])
            for param, value in new_tick_params.items():
                if value is not None:
                    self._tick_params[f"{axis_i} major"][param] = value
            if minor_length is not None:
                self._tick_params[f"{axis_i} minor"]["length"] = minor_length

    def set_grid(
        self,
        visible_x: bool = True,
        visible_y: bool = True,
        show_on_top: bool = False,
        color: str | Literal["default"] = "default",
        alpha: float | Literal["default"] = "default",
        line_style: str | Literal["default"] = "default",
        line_width: float | Literal["default"] = "default",
    ) -> None:
        """
        Sets the grid parameters for the figure.

        .. note::
            Contrary to the :class:`~graphinglib.smart_figure.SmartFigure` class, this method does not support plotting
            grid lines for minor ticks. This is because the :class:`astropy.visualization.wcsaxes.WCSAxes` do not
            support minor ticks for the grid lines.

        Parameters
        ----------
        visible_x, visible_y : bool, optional
            If ``True``, sets the x-axis or y-axis grid visible. If ``False``, the grid is not shown for the respective
            axis.
            Defaults to ``True`` for both axes.
        show_on_top : bool, optional
            If ``True``, sets the grid lines to be shown on top of the plot elements. This can be useful to see the grid
            lines above a plotted :class:`~graphinglib.data_plotting_2d.Heatmap` for example.
            Defaults to ``False``.
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
        """
        super().set_grid(
            visible_x=visible_x,
            visible_y=visible_y,
            show_on_top=show_on_top,
            which_x="major",
            which_y="major",
            color=color,
            alpha=alpha,
            line_style=line_style,
            line_width=line_width,
        )
