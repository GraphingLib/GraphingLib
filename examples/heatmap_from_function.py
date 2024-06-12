"""
Heatmap From Function
=====================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl


def func(x, y):
    return np.sin(x) * np.cos(y)


heatmap = gl.Heatmap.from_function(
    func,
    x_axis_range=(0, 10),
    y_axis_range=(0, 10),
    color_map="viridis",
    show_color_bar=True,
    aspect_ratio="equal",
    number_of_points=(20, 20),
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(heatmap)
fig.show()
