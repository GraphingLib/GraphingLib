"""
Curve Slices
============

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create a curve from a function
curve = gl.Curve.from_function(
    lambda x: np.sin(x) * np.exp(-x / 10),
    x_min=-10,
    x_max=10,
    label="Curve",
)

# Slice the curve from x=-5 to x=5 and change the color
sliced_curve = curve.create_slice_x(x1=-5, x2=5, color="red", label="Important section")

# Slicing in the y direction creates a clipping effect when the curve goes out of bounds
clipping = curve.create_slice_y(
    y1=0, y2=0.5, color="orange", line_style="--", label="Clipped section"
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
# Add the curves to the figure in the order they should be layered
fig.add_elements(curve, sliced_curve, clipping)
fig.show()
