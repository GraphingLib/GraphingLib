from dataclasses import dataclass
from typing import Callable, Optional, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from matplotlib.image import imread
from numpy.typing import ArrayLike


@dataclass
class Heatmap:
    """
    Basic heatmap class.
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
