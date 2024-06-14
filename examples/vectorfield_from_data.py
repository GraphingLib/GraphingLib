"""
VectorField From Data
=====================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create fictitious data
x_grid, y_grid = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
u_data = x_grid**2 - y_grid**2 + np.random.normal(0, 5, (20, 20))
v_data = 2 * x_grid * y_grid + np.random.normal(0, 5, (20, 20))

vector_field = gl.VectorField(
    x_data=x_grid,
    y_data=y_grid,
    u_data=u_data,
    v_data=v_data,
    arrow_head_size=3 / 4,
    arrow_width=0.8,
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(vector_field)
fig.show()
