"""
Contour From Data
=================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create fictitious data
x_grid, y_grid = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
data = (x_grid - 5) ** 3 + y_grid**2

contour = gl.Contour(
    x_mesh=x_grid,
    y_mesh=y_grid,
    z_data=data,
    color_map="viridis",
    show_color_bar=True,
    filled=True,
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(contour)
fig.show()
