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
    color_map: str | Colormap = "default"
    color_bar: bool = True
    alpha_value: float = 1.0
    aspect_ratio: str | float = "default"
    origin_position: str = "default"

    def __post_init__(self):
        if isinstance(self.image, str):
            self.image = imread(self.image)
            self.color_bar = False

    def plot_element(self, axes: plt.Axes):
        image = axes.imshow(
            self.image,
            cmap=self.color_map,
            alpha=self.alpha_value,
            aspect=self.aspect_ratio,
            origin=self.origin_position,
        )
        fig = axes.get_figure()
        if self.color_bar:
            fig.colorbar(image, ax=axes)
