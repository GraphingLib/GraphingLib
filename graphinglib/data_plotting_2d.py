from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.image import imread
from numpy.typing import ArrayLike


@dataclass
class Heatmap:
    """
    The class implements heatmaps.

    Parameters
    ----------
    image : ArrayLike or str
        Image to display as an array of values or from a file.
    x_axis_range, y_axis_range : tuple[float, float], optional
        The range of x and y values used for the axes as tuples containing the
        start and end of the range.
    color_map : str, Colormap
        The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
        string (named colormap from Matplotlib) or a Colormap object.
        Default depends on the ``figure_style`` configuration.
    show_color_bar : bool
        Whether or not to display the color bar next to the plot.
        Defaults to ``True``.
    alpha_value : float
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
    """

    image: ArrayLike | str
    x_axis_range: Optional[tuple[float, float]] = None
    y_axis_range: Optional[tuple[float, float]] = None
    color_map: str | Colormap = "default"
    show_color_bar: bool = True
    alpha_value: float = 1.0
    aspect_ratio: str | float = "default"
    origin_position: str = "default"
    interpolation: str = "none"

    def __post_init__(self) -> None:
        if isinstance(self.image, str):
            self.image = imread(self.image)
            self.show_color_bar = False
        else:
            self.image = np.array(self.image)
        if self.x_axis_range is not None and self.y_axis_range is not None:
            self._xy_range = self.x_axis_range + self.y_axis_range

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        color_map: str | Colormap = "default",
        show_color_bar: bool = True,
        alpha_value: float = 1.0,
        aspect_ratio: str | float = "default",
        origin_position: str = "default",
        interpolation: str = "none",
        number_of_points: tuple[int, int] = (50, 50),
    ) -> Self:
        """
        Creates a heatmap from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], ArrayLike]
            Function to be plotted. Works with regular functions and lambda
            functions.
        x_axis_range, y_axis_range : tuple[float, float], optional
            The range of x and y values used for the axes as tuples containing the
            start and end of the range.
        color_map : str, Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Default depends on the ``figure_style`` configuration.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Defaults to ``True``.
        alpha_value : float
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

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Heatmap` object created from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points[0])
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points[1])
        x_grid, y_grid = np.meshgrid(x, y)
        z = func(x_grid, y_grid)
        return cls(
            z,
            x_axis_range,
            y_axis_range,
            color_map,
            show_color_bar,
            alpha_value,
            aspect_ratio,
            origin_position,
            interpolation,
        )

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self.x_axis_range is not None and self.y_axis_range is not None:
            image = axes.imshow(
                self.image,
                cmap=self.color_map,
                alpha=self.alpha_value,
                aspect=self.aspect_ratio,
                origin=self.origin_position,
                interpolation=self.interpolation,
                extent=self._xy_range,
                zorder=z_order,
            )
        else:
            image = axes.imshow(
                self.image,
                cmap=self.color_map,
                alpha=self.alpha_value,
                aspect=self.aspect_ratio,
                origin=self.origin_position,
                interpolation=self.interpolation,
                zorder=z_order,
            )
        fig = axes.get_figure()
        if self.show_color_bar:
            fig.colorbar(image, ax=axes)


@dataclass
class VectorField:
    """
    This class implements vector fields.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        x and y coordinates of the vectors.
    u_data, v_data : ArrayLike
        Magnitudes in the x and y coordinates.
    arrow_length_multiplier : float, optional
        Arrow length scaling factor.
        Default depends on the ``figure_style`` configuration.
    arrow_width : float
        Arrow width.
        Default depends on the ``figure_style`` configuration.
    arrow_head_width : float
        Arrow head width.
        Default depends on the ``figure_style`` configuration.
    arrow_head_length : float
        Arrow head length.
        Default depends on the ``figure_style`` configuration.
    arrow_head_axis_length : float
        Arrow head axis length.
        Default depends on the ``figure_style`` configuration.
    angle_in_data_coords : bool
        Wheter to use the screen coordinates or the data coordinates to
        determine the vector directions.
        Defaults to ``True``.
    color : str
        Color of the vector arrows.
        Default depends on the ``figure_style`` configuration.
    """

    x_data: ArrayLike
    y_data: ArrayLike
    u_data: ArrayLike
    v_data: ArrayLike
    arrow_length_multiplier: Optional[float | Literal["default"]] = "default"
    arrow_width: float | Literal["default"] = "default"
    arrow_head_width: float | Literal["default"] = "default"
    arrow_head_length: float | Literal["default"] = "default"
    arrow_head_axis_length: float | Literal["default"] = "default"
    angle_in_data_coords: bool = True
    color: str | Literal["default"] = "default"

    def __post_init__(self) -> None:
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        self.u_data = np.array(self.u_data)
        self.v_data = np.array(self.v_data)

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], [ArrayLike, ArrayLike]],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        number_of_arrows_x: int = 10,
        number_of_arrows_y: int = 10,
        arrow_length_multiplier: Optional[float | Literal["default"]] = "default",
        arrow_width: float | Literal["default"] = "default",
        arrow_head_width: float | Literal["default"] = "default",
        arrow_head_length: float | Literal["default"] = "default",
        arrow_head_axis_length: float | Literal["default"] = "default",
        angle_in_data_coords: bool = True,
        color: str | Literal["default"] = "default",
    ) -> Self:
        """
        Creates a :class:`~graphinglib.data_plotting_2d.VectorField` from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], [ArrayLike, ArrayLike]]
            Function to be plotted. Works with regular functions and lambda
            functions.
        x_data, y_data : ArrayLike
            x and y coordinates of the vectors.
        u_data, v_data : ArrayLike
            Magnitudes in the x and y coordinates.
        number_of_arrows_x, number_of_arrows_y : int
            Number of arrows to plot in the x and y direction. Defaults to 10.
        arrow_length_multiplier : float, optional
            Arrow length scaling factor.
            Default depends on the ``figure_style`` configuration.
        arrow_width : float
            Arrow width.
            Default depends on the ``figure_style`` configuration.
        arrow_head_width : float
            Arrow head width.
            Default depends on the ``figure_style`` configuration.
        arrow_head_length : float
            Arrow head length.
            Default depends on the ``figure_style`` configuration.
        arrow_head_axis_length : float
            Arrow head axis length.
            Default depends on the ``figure_style`` configuration.
        angle_in_data_coords : bool
            Wheter to use the screen coordinates or the data coordinates to
            determine the vector directions.
            Defaults to ``True``.
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
            arrow_length_multiplier,
            arrow_width,
            arrow_head_width,
            arrow_head_length,
            arrow_head_axis_length,
            angle_in_data_coords,
            color,
        )

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self.angle_in_data_coords:
            angle = "xy"
        else:
            angle = "uv"
        axes.quiver(
            self.x_data,
            self.y_data,
            self.u_data,
            self.v_data,
            color=self.color,
            zorder=z_order,
            angles=angle,
            # scale=self.arrow_length_multiplier,
            width=self.arrow_width,
            headwidth=self.arrow_head_width,
            headlength=self.arrow_head_length,
            headaxislength=self.arrow_head_axis_length,
        )


@dataclass
class Contour:
    """
    This class implements contour plots.

    Parameters
    ----------
    x_mesh, y_mesh : ArrayLike
        x and y coordinates of the mesh grid.
    z_data : ArrayLike
        Data for each point of the mesh.
    number_of_levels : int
        Number of distinct levels of contour plot.
        Default depends on the ``figure_style`` configuration.
    color_map : str or Colormap
        The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
        string (named colormap from Matplotlib) or a Colormap object.
        Default depends on the ``figure_style`` configuration.
    show_color_bar : bool
        Whether or not to display the color bar next to the plot.
        Default depends on the ``figure_style`` configuration.
    filled : bool
        Wheter or not to fill the contour with color.
        Default depends on the ``figure_style`` configuration.
    alpha : float
        Opacity of the filled contour.
        Default depends on the ``figure_style`` configuration.
    """

    x_mesh: ArrayLike
    y_mesh: ArrayLike
    z_data: ArrayLike
    number_of_levels: int | Literal["default"] = "default"
    color_map: str | Colormap | Literal["default"] = "default"
    show_color_bar: bool | Literal["default"] = "default"
    filled: bool | Literal["default"] = "default"
    alpha: float | Literal["default"] = "default"

    def __post_init__(self) -> None:
        self.x_mesh = np.array(self.x_mesh)
        self.y_mesh = np.array(self.y_mesh)
        self.z_data = np.array(self.z_data)

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], ArrayLike],
        x_axis_range: tuple[float, float],
        y_axis_range: tuple[float, float],
        number_of_levels: int | Literal["default"] = "default",
        color_map: str | Colormap | Literal["default"] = "default",
        show_color_bar: bool | Literal["default"] = "default",
        filled: bool | Literal["default"] = "default",
        alpha: float | Literal["default"] = "default",
        number_of_points: tuple[int, int] = (500, 500),
    ) -> Self:
        """
        Creates a Contour object from a function.

        Parameters
        ----------
        func : Callable[[ArrayLike, ArrayLike], ArrayLike]
            Function to be plotted. Works with regular functions and lambda
            functions.
        x_mesh, y_mesh : ArrayLike
            x and y coordinates of the mesh grid.
        z_data : ArrayLike
            Data for each point of the mesh.
        number_of_levels : int
            Number of distinct levels of contour plot.
            Default depends on the ``figure_style`` configuration.
        color_map : str or Colormap
            The color map to use for the :class:`~graphinglib.data_plotting_2d.Heatmap`. Can either be specified as a
            string (named colormap from Matplotlib) or a Colormap object.
            Default depends on the ``figure_style`` configuration.
        show_color_bar : bool
            Whether or not to display the color bar next to the plot.
            Default depends on the ``figure_style`` configuration.
        filled : bool
            Wheter or not to fill the contour with color.
            Default depends on the ``figure_style`` configuration.
        alpha : float
            Opacity of the filled contour.
            Default depends on the ``figure_style`` configuration.

        Returns
        -------
        A :class:`~graphinglib.data_plotting_2d.Contour` object from a function.
        """
        x = np.linspace(x_axis_range[0], x_axis_range[1], number_of_points[0])
        y = np.linspace(y_axis_range[0], y_axis_range[1], number_of_points[1])
        x_mesh, y_mesh = np.meshgrid(x, y)
        z_data = func(x_mesh, y_mesh)
        return cls(
            x_mesh,
            y_mesh,
            z_data,
            number_of_levels,
            color_map,
            show_color_bar,
            filled,
            alpha,
        )

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified
        `Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html>`_.
        """
        if self.filled:
            cont = axes.contourf(
                self.x_mesh,
                self.y_mesh,
                self.z_data,
                levels=self.number_of_levels,
                cmap=self.color_map,
                alpha=self.alpha,
                zorder=z_order,
            )
        else:
            cont = axes.contour(
                self.x_mesh,
                self.y_mesh,
                self.z_data,
                levels=self.number_of_levels,
                cmap=self.color_map,
                alpha=self.alpha,
                zorder=z_order,
            )
        if self.show_color_bar:
            fig = axes.get_figure()
            fig.colorbar(cont, ax=axes)


@dataclass
class Stream:
    """
    This class implements stream plots.

    Parameters
    ----------
    x_data, y_data : ArrayLike
        x and y coordinates of the vectors.
    u_data, v_data : ArrayLike
        Magnitudes in the x and y coordinates.
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
    """

    x_data: ArrayLike
    y_data: ArrayLike
    u_data: ArrayLike
    v_data: ArrayLike
    density: float | tuple[float, float] = 1
    line_width: float | Literal["default"] = "default"
    color: str | Literal["default"] = "default"
    color_map: str | Colormap | Literal["default"] = "default"
    arrow_size: float | Literal["default"] = "default"

    def __post_init__(self) -> None:
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        self.u_data = np.array(self.u_data)
        self.v_data = np.array(self.v_data)

    @classmethod
    def from_function(
        cls,
        func: Callable[[ArrayLike, ArrayLike], [ArrayLike, ArrayLike]],
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
            Function to be plotted. Works with regular functions and lambda
            functions.
        x_axis_range : tuple[float, float]
            Range of x values.
        y_axis_range : tuple[float, float]
            Range of y values.
        number_of_points_x : int
            Number of points to fill the x range. Defaults to 30.
        number_of_points_y : int
            Number of points to fill the y range. Defaults to 30.
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

    def _plot_element(self, axes: plt.Axes, z_order: int) -> None:
        """
        Plots the element in the specified Axes.
        """
        axes.streamplot(
            self.x_data,
            self.y_data,
            self.u_data,
            self.v_data,
            density=self.density,
            linewidth=self.line_width,
            color=self.color,
            cmap=self.color_map,
            arrowsize=self.arrow_size,
            zorder=z_order,
        )
