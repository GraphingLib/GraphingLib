from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.image import imread
from numpy.typing import ArrayLike
from scipy.interpolate import griddata

from .graph_elements import Plottable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@runtime_checkable
class Plottable2D(Plottable, Protocol):
    """
    Dummy class to allow type hinting of Plottable2D objects.
    """

    pass


@dataclass
class Heatmap(Plottable2D):
    """
    The class implements heatmaps.

    Parameters
    ----------
    image : ArrayLike | str
        Image to display. If an array of values is given, the 2D array will be interpreted as the values of the image.
        If a str if given, the corresponding file will be read as an image.
    x_axis_range, y_axis_range : tuple[float, float], optional
        The range of x and y values used for the axes as tuples containing the start and end of the range. These values
        are ignored when ``x_mesh`` and ``y_mesh`` are provided.
    x_mesh, y_mesh : ArrayLike, optional
        Mesh grids defining the coordinates of the heatmap values. When provided, the heatmap is plotted using
        ``pcolormesh`` instead of ``imshow``.
    color_map : str, Colormap
        The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
        string (named colormap from Matplotlib) or a Colormap object.
        Default depends on the ``figure_style`` configuration.
    color_map_range: tuple[float, float], optional
        The data range that the color map will cover.
    show_color_bar : bool
        Whether or not to display the color bar next to the plot.
        Defaults to ``True``.
    alpha : float
        Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
        Defaults to 1.0.
    aspect_ratio : str or float
        Aspect ratio of the axes. This value is ignored when ``x_mesh`` and ``y_mesh`` are provided.
        Default depends on the ``figure_style`` configuration.
    origin_position : str
        Position of the origin of the axes (upper left or lower left corner). This value is ignored when ``x_mesh`` and
        ``y_mesh`` are provided.
        Default depends on the ``figure_style`` configuration.
    interpolation : str
        Interpolation method to be applied to the image. This value is ignored when ``x_mesh`` and ``y_mesh`` are
        provided.
        Defaults to ``"none"``.

        .. seealso::

            For other interpolation methods, refer to
            `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.
    norm : str or Normalize, optional
        Normalization of the colormap. Default is ``None``.
    """

    def __init__(
        self,
        image: ArrayLike | str,
        x_axis_range: Optional[tuple[float, float]] = None,
        y_axis_range: Optional[tuple[float, float]] = None,
        x_mesh: Optional[ArrayLike] = None,
        y_mesh: Optional[ArrayLike] = None,
        color_map: str | Colormap = "default",
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Literal["default"] = "default",
        alpha: float = 1.0,
        aspect_ratio: str | float = "default",
        origin_position: str = "default",
        interpolation: str = "none",
        norm: Optional[str | Normalize] = None,
    ) -> None:
        """
        The class implements heatmaps.

        Parameters
        ----------
        image : ArrayLike | str
            Image to display. If an array of values is given, the 2D array will be interpreted as the values of the
            image. If a str if given, the corresponding file will be read as an image.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range. These
            values are ignored when ``x_mesh`` and ``y_mesh`` are provided.
        x_mesh, y_mesh : ArrayLike, optional
            Mesh grids defining the coordinates of the heatmap values. When provided, the heatmap is plotted using
            ``pcolormesh`` instead of ``imshow``.
        color_map : str, Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range that the color map will cover.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Defaults to ``True``.
        alpha : float
            Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
            Defaults to 1.0.
        aspect_ratio : str or float
            Aspect ratio of the axes. This value is ignored when ``x_mesh`` and ``y_mesh`` are provided.
            Default depends on the ``figure_style`` configuration.
        origin_position : str
            Position of the origin of the axes (upper left or lower left corner). This value is ignored when ``x_mesh``
            and ``y_mesh`` are provided.
            Default depends on the ``figure_style`` configuration.
        interpolation : str
            Interpolation method to be applied to the image. This value is ignored when ``x_mesh`` and ``y_mesh`` are
            provided.
            Defaults to ``"none"``.

            .. seealso::

                For other interpolation methods, refer to
                `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.
        norm : str or Normalize, optional
            Normalization of the colormap. Default is ``None``.
        """
        self.show_color_bar = show_color_bar
        self.image = image
        self.x_axis_range = x_axis_range
        self.y_axis_range = y_axis_range
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self.color_map = color_map
        self.color_map_range = color_map_range
        self.alpha = alpha
        self.aspect_ratio = aspect_ratio
        self.origin_position = origin_position
        self.interpolation = interpolation
        self._norm = norm

        self._color_bar_params: dict = {}

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        color_map: str | Colormap = "default",
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool = True,
        alpha: float = 1.0,
        aspect_ratio: str | float = "default",
        origin_position: str = "default",
        interpolation: str = "none",
        number_of_points: tuple[int, int] = (50, 50),
        norm: Optional[str | Normalize] = None,
    ) -> Self:
        """
        Creates a heatmap from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], ArrayLike]
            Function to be plotted. Works with regular functions and lambda functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        color_map : str, Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range that the color map will cover.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Defaults to ``True``.
        alpha : float
            Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
            Defaults to 1.0.
        aspect_ratio : str or float
            Aspect ratio of the axes.
            Default depends on the ``figure_style`` configuration.
        origin_position : str
            Position of the origin of the axes (upper left or lower left corner).
            Default depends on the ``figure_style`` configuration.
        interpolation : str
            Interpolation method to be applied to the image.
            Defaults to ``"none"``.

            .. seealso::
                For other interpolation methods, refer to
                `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.

        number_of_points : tuple[int, int]
            Number of points in the x and y coordinates.
            Defaults to ``(50, 50)``.
        norm : str or Normalize, optional
            Normalization of the colormap. Default is ``None``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Heatmap` object created from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points[0])
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points[1])
        x_grid, y_grid = np.meshgrid(x, y)
        z = func(x_grid, y_grid)
        return cls(
            image=z,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
            color_map=color_map,
            color_map_range=color_map_range,
            show_color_bar=show_color_bar,
            alpha=alpha,
            aspect_ratio=aspect_ratio,
            origin_position=origin_position,
            interpolation=interpolation,
            norm=norm,
        )

    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        values: ArrayLike,
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        grid_interpolation: str = "nearest",
        fill_value: float = np.nan,
        color_map: str | Colormap = "default",
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool = True,
        alpha: float = 1.0,
        aspect_ratio: str | float = "default",
        origin_position: str = "default",
        interpolation: str = "none",
        number_of_points: tuple[int, int] = (50, 50),
        norm: Optional[str | Normalize] = None,
    ) -> Self:
        """
        Creates a heatmap by interpolating unevenly distributed data points on a grid.

        Parameters
        ----------
        points : ArrayLike
            The list or array of points at which values are known.
        values : ArrayLike
            The list or array of values at given points.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        grid_interpolation : str
            Interpolation method to be used when interpolating the uneavenly distributed data on a grid.
            Must be one of {"nearest", "linear", "cubic"}.
        color_map : str, Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range that the color map will cover.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Defaults to ``True``.
        alpha : float
            Opacity value of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
            Defaults to 1.0.
        aspect_ratio : str or float
            Aspect ratio of the axes.
            Default depends on the ``figure_style`` configuration.
        origin_position : str
            Position of the origin of the axes (upper left or lower left corner).
            Default depends on the ``figure_style`` configuration.
        interpolation : str
            Interpolation method to be applied to the image.
            Defaults to ``"none"``.

            .. seealso::
                For other interpolation methods, refer to
                `Interpolations for imshow <https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html>`_.

        number_of_points : tuple[int, int]
            Number of points in the x and y coordinates.
            Defaults to ``(50, 50)``.
        norm : str or Normalize, optional
            Normalization of the colormap. Default is ``None``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Heatmap` object created from data points.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points[0])
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points[1])
        x_grid, y_grid = np.meshgrid(x, y)
        grid = griddata(
            points,
            values,
            (x_grid, y_grid),
            method=grid_interpolation,
            fill_value=fill_value,
        )
        return cls(
            image=grid,
            x_axis_range=x_axis_range,
            y_axis_range=y_axis_range,
            color_map=color_map,
            color_map_range=color_map_range,
            show_color_bar=show_color_bar,
            alpha=alpha,
            aspect_ratio=aspect_ratio,
            origin_position=origin_position,
            interpolation=interpolation,
            norm=norm,
        )

    @property
    def image(self) -> ArrayLike | str:
        return self._image

    @image.setter
    def image(self, image: ArrayLike | str) -> None:
        if isinstance(image, str):
            self._image = imread(image)
            self._show_color_bar = False
        else:
            self._image = np.asarray(image)

    @property
    def x_axis_range(self) -> Optional[tuple[float, float]]:
        return self._x_axis_range

    @x_axis_range.setter
    def x_axis_range(self, x_axis_range: Optional[tuple[float, float]]) -> None:
        self._x_axis_range = x_axis_range

    @property
    def y_axis_range(self) -> Optional[tuple[float, float]]:
        return self._y_axis_range

    @y_axis_range.setter
    def y_axis_range(self, y_axis_range: Optional[tuple[float, float]]) -> None:
        self._y_axis_range = y_axis_range

    @property
    def x_mesh(self) -> Optional[ArrayLike]:
        return self._x_mesh

    @x_mesh.setter
    def x_mesh(self, x_mesh: Optional[ArrayLike]) -> None:
        self._x_mesh = None if x_mesh is None else np.asarray(x_mesh)

    @property
    def y_mesh(self) -> Optional[ArrayLike]:
        return self._y_mesh

    @y_mesh.setter
    def y_mesh(self, y_mesh: Optional[ArrayLike]) -> None:
        self._y_mesh = None if y_mesh is None else np.asarray(y_mesh)

    @property
    def color_map(self) -> str | Colormap:
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: str | Colormap) -> None:
        self._color_map = color_map

    @property
    def color_map_range(self) -> tuple[float, float]:
        return self._color_map_range

    @color_map_range.setter
    def color_map_range(self, color_map_range: tuple[float, float]) -> None:
        self._color_map_range = color_map_range

    @property
    def show_color_bar(self) -> bool:
        return self._show_color_bar

    @show_color_bar.setter
    def show_color_bar(self, show_color_bar: bool) -> None:
        self._show_color_bar = show_color_bar

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def aspect_ratio(self) -> str | float:
        return self._aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, aspect_ratio: str | float) -> None:
        self._aspect_ratio = aspect_ratio

    @property
    def origin_position(self) -> str:
        return self._origin_position

    @origin_position.setter
    def origin_position(self, origin_position: str) -> None:
        self._origin_position = origin_position

    @property
    def interpolation(self) -> str:
        return self._interpolation

    @interpolation.setter
    def interpolation(self, interpolation: str) -> None:
        self._interpolation = interpolation

    @property
    def color_bar_params(self) -> dict:
        return self._color_bar_params

    @property
    def _xy_range(self) -> Optional[tuple[float, float, float, float]]:
        if self._x_axis_range is not None and self._y_axis_range is not None:
            return self._x_axis_range + self._y_axis_range
        return None

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_2d.Heatmap`.
        """
        return deepcopy(self)

    def set_color_bar_params(
        self,
        label: Optional[str] = None,
        position: Optional[str] = None,
        **color_bar_params,
    ) -> None:
        """
        Sets the color bar parameters.

        Parameters
        ----------
        label : str, optional
            Label of the color bar.
        position : str, optional
            Position of the color bar relative to the ``Figure``. It can be "left",
            "right", "top" or "bottom". This also determines the orientation of the
            color bar (vertical if the color bar is plotted on the "left" or "right",
            horizontal otherwise). If None, the color bar is plotted on the right
            side of the ``Figure``.
        **color_bar_params:
            Additional keyword arguments are passed to ``plt.colorbar`` call.
        """
        self._color_bar_params = color_bar_params
        if label is not None:
            self._color_bar_params["label"] = label
        if position is not None:
            self._color_bar_params["location"] = position

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        params = {
            "cmap": self._color_map,
            "alpha": self._alpha,
            "norm": self._norm,
        }
        if self._color_map_range:
            params["vmin"] = min(self._color_map_range)
            params["vmax"] = max(self._color_map_range)
        use_pcolormesh = self._x_mesh is not None and self._y_mesh is not None
        if use_pcolormesh:
            params = {k: v for k, v in params.items() if v != "default"}
            image = axes.pcolormesh(
                self._x_mesh,
                self._y_mesh,
                self._image,
                zorder=z_order,
                **params,
            )
        else:
            params.update(
                {
                    "aspect": self._aspect_ratio,
                    "origin": self._origin_position,
                    "interpolation": self._interpolation,
                    "extent": self._xy_range,
                }
            )

            params = {k: v for k, v in params.items() if v != "default"}
            image = axes.imshow(
                self._image,
                zorder=z_order,
                **params,
            )
        fig = axes.get_figure()
        if self._show_color_bar:
            fig.colorbar(image, ax=axes, **self._color_bar_params)


@dataclass
class VectorField(Plottable2D):
    """
    This class implements vector fields.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        x and y coordinates of the vectors.
    u_data, v_data : ArrayLike
        Magnitudes in the x and y coordinates.
    arrow_width : float
        Width of the arrow shaft. Acts as a multiplier for the standard arrow width.
        Default depends on the ``figure_style`` configuration.
    arrow_head_size : float
        Size of the arrow head. Acts as a multiplier for the standard arrow head size.
        Default depends on the ``figure_style`` configuration.
    scale : float
        Scaling of the arrow lengths. If ``None``, the arrows will be automatically scaled to look nice. Use 1 for no scaling.
    angle_in_data_coords : bool
        Whether to use the screen coordinates or the data coordinates to determine the vector directions.
        Defaults to ``True``.
    color : str
        Color of the vector arrows.
        Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        u_data: ArrayLike,
        v_data: ArrayLike,
        arrow_width: float | Literal["default"] = "default",
        arrow_head_size: float | Literal["default"] = "default",
        scale: Optional[float] = None,
        make_angles_axes_independent: bool = False,
        color: str | Literal["default"] = "default",
    ) -> None:
        """
        This class implements vector fields.

        Parameters
        ----------
        x_data, y_data : ArrayLike
            x and y coordinates of the vectors.
        u_data, v_data : ArrayLike
            Magnitudes in the x and y coordinates.
        arrow_width : float
            Width of the arrow shaft. Acts as a multiplier for the standard arrow width.
            Default depends on the ``figure_style`` configuration.
        arrow_head_size : float
            Size of the arrow head. Acts as a multiplier for the standard arrow head size.
            Default depends on the ``figure_style`` configuration.
        scale : float
            Scaling of the arrow lengths. If ``None``, the arrows will be automatically scaled to look nice. Use 1 for no scaling.
            Default is ``None``.
        make_angles_axes_independent : bool
            Whether to use the screen coordinates or the data coordinates to
            determine the vector directions. If ``True``, vectors with u = v will
            always appear as 45 degree angles with respect to the screen, regardless
            of the axes limits. If ``False``, the vectors will scale with the
            aspect ratio of the axes.
            Defaults to ``False``.
        color : str
            Color of the vector arrows.
            Default depends on the ``figure_style`` configuration.
        """
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        self._u_data = np.asarray(u_data)
        self._v_data = np.asarray(v_data)
        self._arrow_width = arrow_width
        self._arrow_head_size = arrow_head_size
        self._scale = scale
        self._make_angles_axes_independent = make_angles_axes_independent

        self._color = color

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        number_of_arrows_x: int = 10,
        number_of_arrows_y: int = 10,
        arrow_width: float | Literal["default"] = "default",
        arrow_head_size: float | Literal["default"] = "default",
        scale: Optional[float] = None,
        make_angles_axes_independent: bool = False,
        color: str | Literal["default"] = "default",
    ) -> Self:
        """
        Creates a :class:`~graphinglib.data_plotting_2d.VectorField` from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]]
            Function to be plotted. Works with regular functions and lambda functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        number_of_arrows_x, number_of_arrows_y : int
            Number of arrows to plot in the x and y direction. Defaults to 10.
        arrow_width : float
            Width of the arrow shaft. Acts as a multiplier for the standard arrow width.
            Default depends on the ``figure_style`` configuration.
        arrow_head_size : float
            Size of the arrow head. Acts as a multiplier for the standard arrow head size.
            Default depends on the ``figure_style`` configuration.
        scale : float
            Scaling of the arrow lengths. If ``None``, the arrows will be automatically scaled to look nice. Use 1 for no scaling.
            Default is ``None``.
        make_angles_axes_independent : bool
            Whether to use the screen coordinates or the data coordinates to
            determine the vector directions. If ``True``, vectors with u = v will
            always appear as 45 degree angles with respect to the screen, regardless
            of the axes limits. If ``False``, the vectors will scale with the
            aspect ratio of the axes.
            Defaults to ``False``.
        color : str
            Color of the vector arrows.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.VectorField` object from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[-1], number_of_arrows_x)
        y = np.linspace(y_axis_range[0], y_axis_range[-1], number_of_arrows_y)
        x_grid, y_grid = np.meshgrid(x, y)
        u, v = func(x_grid, y_grid)
        return cls(
            x_grid,
            y_grid,
            u,
            v,
            arrow_width,
            arrow_head_size,
            scale,
            make_angles_axes_independent,
            color,
        )

    @property
    def x_data(self) -> ArrayLike:
        return self._x_data

    @x_data.setter
    def x_data(self, x_data: ArrayLike) -> None:
        self._x_data = np.asarray(x_data)

    @property
    def y_data(self) -> ArrayLike:
        return self._y_data

    @y_data.setter
    def y_data(self, y_data: ArrayLike) -> None:
        self._y_data = np.asarray(y_data)

    @property
    def u_data(self) -> ArrayLike:
        return self._u_data

    @u_data.setter
    def u_data(self, u_data: ArrayLike) -> None:
        self._u_data = np.asarray(u_data)

    @property
    def v_data(self) -> ArrayLike:
        return self._v_data

    @v_data.setter
    def v_data(self, v_data: ArrayLike) -> None:
        self._v_data = np.asarray(v_data)

    @property
    def arrow_width(self) -> float:
        return self._arrow_width

    @arrow_width.setter
    def arrow_width(self, arrow_width: float) -> None:
        self._arrow_width = arrow_width

    @property
    def arrow_head_size(self) -> float:
        return self._arrow_head_size

    @arrow_head_size.setter
    def arrow_head_size(self, arrow_head_size: float) -> None:
        self._arrow_head_size = arrow_head_size

    @property
    def scale(self) -> Optional[float]:
        return self._scale

    @scale.setter
    def scale(self, scale: Optional[float]) -> None:
        self._scale = scale

    @property
    def make_angles_axes_independent(self) -> bool:
        return self._make_angles_axes_independent

    @make_angles_axes_independent.setter
    def make_angles_axes_independent(self, value: bool) -> None:
        self._make_angles_axes_independent = value

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, color: str) -> None:
        self._color = color

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_2d.VectorField`.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self._make_angles_axes_independent:
            angle = "uv"
        else:
            angle = "xy"
        params = {
            "angles": angle,
            "width": 0.005 * self._arrow_width,
            "headwidth": 4 * self._arrow_head_size / self._arrow_width,
            "headlength": 4 * self._arrow_head_size / self._arrow_width,
            "headaxislength": 4 * self._arrow_head_size / self._arrow_width,
            "color": self._color,
            "scale": 1 / self._scale if self._scale is not None else None,
            "scale_units": "xy",
        }
        params = {k: v for k, v in params.items() if v != "default"}
        axes.quiver(
            self._x_data,
            self._y_data,
            self._u_data,
            self._v_data,
            zorder=z_order,
            **params,
        )


@dataclass
class Contour(Plottable2D):
    """
    This class implements contour plots.

    Parameters
    ----------
    z_data : ArrayLike
        Data for each point of the mesh.
    x_mesh, y_mesh : ArrayLike, optional
        Mesh grids defining the coordinates of the contour values. If not provided, a mesh grid will be created based on
        the shape of ``z_data``.
    levels : int | ArrayLike
        If `levels` is an integer, it defines the number of levels to use in the contour.
        If `levels` is an array, it defines the value of each contour level.
        Default depends on the ``figure_style`` configuration.
    color_map : str or Colormap
        The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
        string (named colormap from Matplotlib) or a Colormap object.
        Default depends on the ``figure_style`` configuration.
    color_map_range: tuple[float, float], optional
        The data range that the color map will cover.
    show_color_bar : bool
        Whether or not to display the color bar next to the plot.
        Default depends on the ``figure_style`` configuration.
    filled : bool
        Wheter or not to fill the contour with color.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the filled contour.
        Default depends on the ``figure_style`` configuration.
    line_widths : float | ArrayLike
        If the contour is not filled, the width of the contour lines. If an array is provided, it defines the line width
        for each contour level.
        Default depends on the ``figure_style`` configuration.
    """

    _z_data: ArrayLike
    _x_mesh: ArrayLike
    _y_mesh: ArrayLike
    _levels: int | Literal["default"] = "default"
    _color_map: str | Colormap | Literal["default"] = "default"
    _show_color_bar: bool | Literal["default"] = "default"
    _filled: bool | Literal["default"] = "default"
    _alpha: float | Literal["default"] = "default"
    _line_widths: float | ArrayLike | Literal["default"] = "default"

    def __init__(
        self,
        z_data: ArrayLike,
        x_mesh: Optional[ArrayLike] = None,
        y_mesh: Optional[ArrayLike] = None,
        levels: int | ArrayLike | Literal["default"] = "default",
        color_map: str | Colormap | Literal["default"] = "default",
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Literal["default"] = "default",
        filled: bool | Literal["default"] = "default",
        alpha: float | Literal["default"] = "default",
        line_widths: float | ArrayLike | Literal["default"] = "default",
    ) -> None:
        """
        This class implements contour plots.

        Parameters
        ----------
        z_data : ArrayLike
            Data for each point of the mesh.
        x_mesh, y_mesh : ArrayLike, optional
            Mesh grids defining the coordinates of the contour values. If not provided, a mesh grid will be created
            based on the shape of ``z_data``.
        levels : int | ArrayLike
            If `levels` is an integer, it defines the number of levels to use in the contour.
            If `levels` is an array, it defines the value of each contour level.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range that the color map will cover.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        filled : bool
            Wheter or not to fill the contour with color.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the filled contour.
            Default depends on the ``figure_style`` configuration.
        line_widths : float | ArrayLike
            If the contour is not filled, the width of the contour lines. If an array is provided, it defines the line
            width for each contour level.
            Default depends on the ``figure_style`` configuration.
        """
        self.z_data = z_data
        self.x_mesh = x_mesh
        self.y_mesh = y_mesh
        self._levels = levels
        self._color_map = color_map
        self._color_map_range = color_map_range
        self._show_color_bar = show_color_bar
        self._filled = filled
        self._alpha = alpha
        self._line_widths = line_widths

        self._color_bar_params: dict = {}

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        levels: int | ArrayLike | Literal["default"] = "default",
        color_map: str | Colormap | Literal["default"] = "default",
        color_map_range: Optional[tuple[float, float]] = None,
        show_color_bar: bool | Literal["default"] = "default",
        filled: bool | Literal["default"] = "default",
        alpha: float | Literal["default"] = "default",
        line_widths: float | ArrayLike | Literal["default"] = "default",
        number_of_points: tuple[int, int] = (500, 500),
    ) -> Self:
        """
        Creates a Contour object from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], ArrayLike]
            Function to be plotted. Works with regular functions and lambda functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        levels : int | ArrayLike
            If `levels` is an integer, it defines the number of levels to use in the contour.
            If `levels` is an array, it defines the value of each contour level.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Default depends on the ``figure_style`` configuration.
        color_map_range: tuple[float, float], optional
            The data range that the color map will cover.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        filled : bool
            Wheter or not to fill the contour with color.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the filled contour.
            Default depends on the ``figure_style`` configuration.
        line_widths : float | ArrayLike
            If the contour is not filled, the width of the contour lines. If an array is provided, it defines the line
            width for each contour level.
            Default depends on the ``figure_style`` configuration.
        number_of_points : tuple[int, int]
            Number of points in the x and y coordinates.
            Defaults to ``(50, 50)``.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Contour` object from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points[0])
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points[1])
        x_mesh, y_mesh = np.meshgrid(x, y)
        z_data = func(x_mesh, y_mesh)
        return cls(
            z_data,
            x_mesh,
            y_mesh,
            levels,
            color_map,
            color_map_range,
            show_color_bar,
            filled,
            alpha,
            line_widths,
        )

    @property
    def x_mesh(self) -> ArrayLike:
        return self._x_mesh

    @x_mesh.setter
    def x_mesh(self, x_mesh: ArrayLike) -> None:
        self._x_mesh = None if x_mesh is None else np.asarray(x_mesh)

    @property
    def y_mesh(self) -> ArrayLike:
        return self._y_mesh

    @y_mesh.setter
    def y_mesh(self, y_mesh: ArrayLike) -> None:
        self._y_mesh = None if y_mesh is None else np.asarray(y_mesh)

    @property
    def z_data(self) -> ArrayLike:
        return self._z_data

    @z_data.setter
    def z_data(self, z_data: ArrayLike) -> None:
        self._z_data = np.asarray(z_data)

    @property
    def levels(self) -> int | ArrayLike | Literal["default"]:
        return self._levels

    @levels.setter
    def levels(self, levels: int | ArrayLike | Literal["default"]) -> None:
        self._levels = levels

    @property
    def color_map(self) -> str | Colormap:
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: str | Colormap) -> None:
        self._color_map = color_map

    @property
    def color_map_range(self) -> tuple[float, float]:
        return self._color_map_range

    @color_map_range.setter
    def color_map_range(self, color_map_range: tuple[float, float]) -> None:
        self._color_map_range = color_map_range

    @property
    def show_color_bar(self) -> bool:
        return self._show_color_bar

    @show_color_bar.setter
    def show_color_bar(self, show_color_bar: bool) -> None:
        self._show_color_bar = show_color_bar

    @property
    def filled(self) -> bool:
        return self._filled

    @filled.setter
    def filled(self, filled: bool) -> None:
        self._filled = filled

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha

    @property
    def line_widths(self) -> float | ArrayLike:
        return self._line_widths

    @line_widths.setter
    def line_widths(self, line_widths: float | ArrayLike) -> None:
        self._line_widths = line_widths

    @property
    def color_bar_params(self) -> dict:
        return self._color_bar_params

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_2d.Contour`.
        """
        return deepcopy(self)

    def set_color_bar_params(
        self,
        label: Optional[str] = None,
        position: Optional[str] = None,
        **color_bar_params,
    ) -> None:
        """
        Sets the color bar parameters.

        Parameters
        ----------
        label : str, optional
            Label of the color bar.
        position : str, optional
            Position of the color bar relative to the ``Figure``. It can be "left",
            "right", "top" or "bottom". This also determines the orientation of the
            color bar (vertical if the color bar is plotted on the "left" or "right",
            horizontal otherwise). If None, the color bar is plotted on the right
            side of the ``Figure``.
        **color_bar_params:
            Additional keyword arguments are passed to ``plt.colorbar`` call.
        """
        self._color_bar_params = color_bar_params
        if label is not None:
            self._color_bar_params["label"] = label
        if position is not None:
            self._color_bar_params["location"] = position

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self._x_mesh is None or self._y_mesh is None:
            x_mesh, y_mesh = np.meshgrid(
                np.arange(self._z_data.shape[1]),
                np.arange(self._z_data.shape[0]),
            )
        else:
            x_mesh = self._x_mesh
            y_mesh = self._y_mesh
        params = {
            "levels": self._levels,
            "cmap": self._color_map,
            "alpha": self._alpha,
            "linewidths": self._line_widths if not self._filled else None,
        }
        if self._color_map_range is not None:
            params["vmin"] = min(self._color_map_range)
            params["vmax"] = max(self._color_map_range)

        params = {
            k: v for k, v in params.items() if not isinstance(v, str) or v != "default"
        }
        if self._filled:
            cont = axes.contourf(
                x_mesh,
                y_mesh,
                self._z_data,
                zorder=z_order,
                **params,
            )
        else:
            cont = axes.contour(
                x_mesh,
                y_mesh,
                self._z_data,
                zorder=z_order,
                **params,
            )
        if self._show_color_bar:
            fig = axes.get_figure()
            fig.colorbar(cont, ax=axes, **self._color_bar_params)


@dataclass
class Stream(Plottable2D):
    """
    This class implements stream plots.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        x and y coordinates of the vectors as a mesh grid.
    u_data, v_data : ArrayLike
        Magnitudes of the vectors for each point of the mesh grid.
    density : float or tuple[float, float]
        Density of stream lines. Can be specified independently for the x and y coordinates
        by specifying a density tuple instead. Defaults to 1.
    line_width : float
        Width of the stream lines. Default depends on the ``figure_style`` configuration.
    color : str or ArrayLike
        Color of the stream lines. If an array of intensities is provided, the values are mapped to the specified color map.
        Default depends on the ``figure_style`` configuration.
    color_map : str or Colormap
        Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
        Default depends on the ``figure_style`` configuration.
    arrow_size : float
        Arrow size multiplier. Default depends on the ``figure_style`` configuration.
    """

    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        u_data: ArrayLike,
        v_data: ArrayLike,
        density: float | tuple[float, float] = 1,
        line_width: float | Literal["default"] = "default",
        color: str | ArrayLike | Literal["default"] = "default",
        color_map: str | Colormap | Literal["default"] = "default",
        arrow_size: float | Literal["default"] = "default",
    ) -> None:
        """
        This class implements stream plots.

        Parameters
        ----------
        x_data, y_data : ArrayLike
            x and y coordinates of the vectors as a mesh grid.
        u_data, v_data : ArrayLike
            Magnitudes of the vectors for each point of the mesh grid.
        density : float or tuple[float, float]
            Density of stream lines. Can be specified independently for the x and y coordinates
            by specifying a density tuple instead. Defaults to 1.
        line_width : float
            Width of the stream lines. Default depends on the ``figure_style`` configuration.
        color : str or ArrayLike
            Color of the stream lines. If an array of intensities is provided, the values are mapped to the specified color map.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map of the stream lines, to be used in combination with the color parameter to specify intensity.
            Default depends on the ``figure_style`` configuration.
        arrow_size : float
            Arrow size multiplier. Default depends on the ``figure_style`` configuration.
        """
        self._x_data = np.asarray(x_data)
        self._y_data = np.asarray(y_data)
        self._u_data = np.asarray(u_data)
        self._v_data = np.asarray(v_data)
        self._density = density
        self._line_width = line_width
        self._color = color
        self._color_map = color_map
        self._arrow_size = arrow_size

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], tuple[ArrayLike, ArrayLike]],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        number_of_points_x: int = 30,
        number_of_points_y: int = 30,
        density: float | tuple[float, float] = 1,
        line_width: float | Literal["default"] = "default",
        color: str | Literal["default"] = "default",
        color_map: str | Colormap | Literal["default"] = "default",
        arrow_size: float | Literal["default"] = "default",
    ) -> Self:
        """
        Creates a :class:`~graphinglib.data_plotting_2d.Stream` from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], [ArrayLike, ArrayLike]]
            Function to be plotted. Works with regular functions and lambda functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the start and end of the range.
        number_of_points_x, number_of_points_y : int
            Number of points to fill the x and y ranges.
            Defaults to 30.
        density : float or tuple[float, float]
            Density of stream lines. Can be specified independently for the x and y coordinates
            by specifying a density tuple instead. Defaults to 1.
        line_width : float
            Width of the stream lines. Default depends on the ``figure_style`` configuration.
        color : str
            Color of the stream lines. Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            Color map of the stream lines. Default depends on the ``figure_style`` configuration.
        arrow_size : float
            Arrow size multiplier. Default depends on the ``figure_style`` configuration.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Stream` object from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points_x)
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points_y)
        x_grid, y_grid = np.meshgrid(x, y)
        u, v = func(x_grid, y_grid)
        return cls(x, y, u, v, density, line_width, color, color_map, arrow_size)

    def copy(self) -> Self:
        """
        Returns a deep copy of the :class:`~graphinglib.data_plotting_2d.Stream`.
        """
        return deepcopy(self)

    def _plot_element(self, axes: plt.Axes, z_order: int, **kwargs) -> None:
        """
        Plots the element in the specified Axes.
        """
        params = {
            "density": self._density,
            "linewidth": self._line_width,
            "cmap": self._color_map,
            "arrowsize": self._arrow_size,
        }
        params = {k: v for k, v in params.items() if v != "default"}
        if isinstance(self._color, str) and self._color == "default":
            pass
        else:
            params["color"] = self._color

        axes.streamplot(
            x=self._x_data,
            y=self._y_data,
            u=self._u_data,
            v=self._v_data,
            zorder=z_order,
            **params,
        )
