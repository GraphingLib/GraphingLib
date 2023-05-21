from dataclasses import dataclass

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from matplotlib.colors import Colormap
from matplotlib.image import imread


@dataclass
class Heatmap:
    """
    Basic heatmap class.
    """

    image: ArrayLike | str
    x_axis_range: tuple[float, float] | None = None
    y_axis_range: tuple[float, float] | None = None
    color_map: str | Colormap = "default"
    show_color_bar: bool = True
    alpha_value: float = 1.0
    aspect_ratio: str | float = "default"
    origin_position: str = "default"
    interpolation: str = "none"

    def __post_init__(self):
        if isinstance(self.image, str):
            self.image = imread(self.image)
            self.show_color_bar = False
        if self.x_axis_range is not None and self.y_axis_range is not None:
            self.__xy_range__ = self.x_axis_range + self.y_axis_range

    def plot_element(self, axes: plt.Axes, z_order: int):
        if self.x_axis_range is not None and self.y_axis_range is not None:
            image = axes.imshow(
                self.image,
                cmap=self.color_map,
                alpha=self.alpha_value,
                aspect=self.aspect_ratio,
                origin=self.origin_position,
                interpolation=self.interpolation,
                extent=self.__xy_range__,
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
