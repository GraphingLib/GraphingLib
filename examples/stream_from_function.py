"""
Stream From Function
====================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl


# Create function to generate data
def func(x, y):
    return np.sin(y - x**2), 2 * x * y ** (1 / 2)


stream = gl.Stream.from_function(
    func,
    x_axis_range=(0, 5),
    y_axis_range=(0, 10),
    density=2,
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(stream)

fig.show()
