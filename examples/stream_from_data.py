"""
Stream From Data
================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create fictitious data
x_grid, y_grid = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
u_data = np.sin(x_grid) + np.cos(y_grid) + np.random.normal(0, 0.5, (20, 20))
v_data = 2 * np.sin(y_grid) + np.random.normal(0, 0.5, (20, 20))

intensities = np.sqrt(u_data**2 + v_data**2)

# Create stream plot
stream = gl.Stream(
    x_data=x_grid,
    y_data=y_grid,
    u_data=u_data,
    v_data=v_data,
    color=intensities,
    color_map="viridis",
    density=2,
)

# Create figure and add stream plot
fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(stream)
fig.show()
