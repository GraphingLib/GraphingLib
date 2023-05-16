from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.colors import Colormap


@dataclass
class Heatmap:
    """
    Basic heatmap class.
    """

    image: ArrayLike
    color_map: str | Colormap = "default"
    alpha_value: float = 1.0
    aspect_ratio: str | float = "default"
    origin_position: str = "default"

    def plot_element(self, axes: plt.Axes):
        axes.imshow(
            self.image,
            cmap=self.color_map,
            alpha=self.alpha_value,
            aspect=self.aspect_ratio,
            origin=self.origin_position,
        )
