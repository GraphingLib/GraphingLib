"""
Curve Intersections
===================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create two curves
curve_a = gl.Curve.from_function(
    lambda x: (x**4 - 3 * x + 3) / 100,
    x_min=-10,
    x_max=10,
    label="Trajectory A",
)

curve_b = gl.Curve.from_function(
    lambda x: 100 * np.sin(2 * x) / x + 4 * x,
    x_min=-10,
    x_max=10,
    label="Trajectory B",
)

# Create points at each intersection of the curves
intersection_points = curve_a.create_intersection_points(curve_b)

# Create the figure and add the curves and intersection points
fig = gl.Figure(size=(8, 6), x_label="x position", y_label="y position")
fig.add_elements(curve_a, curve_b, *intersection_points)
fig.show()
