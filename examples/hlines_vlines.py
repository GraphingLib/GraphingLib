"""
Hlines and Vlines
=================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

curve = gl.Curve.from_function(
    lambda x: 0.1 * x**2 + np.sin(3 * x) - np.cos(2 * x) + 1, 0, 5
)

# Identify two points on the curve and add their coordinates
point_1 = curve.create_point_at_x(0.9)
point_2 = curve.create_point_at_x(4.3)
point_1.add_coordinates()
point_2.add_coordinates()
point_2.h_align = "right"

# Create horizontal and vertical lines at the points
hlines = gl.Hlines(
    y=[point_1.y, point_2.y],
    x_min=[0, 0],
    x_max=[point_1.x, point_2.x],
    line_styles="--",
    colors="gray",
)
vlines = gl.Vlines(
    x=[point_1.x, point_2.x],
    y_min=[0, 0],
    y_max=[point_1.y, point_2.y],
    line_styles="--",
    colors="gray",
)

figure = gl.Figure(x_lim=(0, 5), y_lim=(0, 6))
figure.add_elements(curve, hlines, vlines, point_1, point_2)
figure.show()
