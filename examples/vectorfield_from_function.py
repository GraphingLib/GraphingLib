"""
VectorField From Function
=========================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl


# Create function to generate data
def func(x, y):
    return x**2 - y**2, np.sin(x) * x * y


vector_field = gl.VectorField.from_function(
    func,
    x_axis_range=(0, 10),
    y_axis_range=(0, 10),
    arrow_head_size=3 / 4,
    arrow_width=0.8,
    color="C2",
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(vector_field)
fig.show()
