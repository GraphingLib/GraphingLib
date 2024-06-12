"""
Heatmap From Data
=====================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

x_grid, y_grid = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
data = x_grid * np.sin(y_grid) + np.random.normal(0, 1, (20, 20))

heatmap = gl.Heatmap(
    data,
    color_map="plasma",
    show_color_bar=True,
    aspect_ratio="equal",
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(heatmap)
fig.show()
