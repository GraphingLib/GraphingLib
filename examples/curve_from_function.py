"""
Curve From Function
===================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl


# Define a function
def func(x):
    return 100 * np.sin(x) / x


# Create a curve from the function
curve_a = gl.Curve.from_function(
    func,
    x_min=-10,
    x_max=10,
    color="C2",
    label="Curve A",
)

# Create another curve from a different function using lambda
curve_b = gl.Curve.from_function(
    lambda x: (x**3 - 2 * x**2 + 5) / 10,
    x_min=-10,
    x_max=10,
    line_width=3,
    line_style="--",
    label="Curve B",
)

# Create a figure and add the curves
fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(curve_a, curve_b)
fig.show()
