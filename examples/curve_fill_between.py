"""
Curve Fill Between
==================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create a curve from a function
curve_1 = gl.Curve.from_function(
    lambda x: 100 * np.sin(x) / x,
    x_min=-10,
    x_max=10,
    color="C2",
    label="Curve A",
)

curve_2 = gl.Curve.from_function(
    lambda x: (x**3 - 2 * x**2 + 500) / 10,
    x_min=-10,
    x_max=10,
    line_width=3,
    line_style="--",
    label="Curve B",
)

# This method returns the area under the curve and shades it if fill_between=True
# Left half of figure is shaded under curve_1 (between curve_1 and x-axis)
area_under_curve = curve_1.get_area_between(x1=-10, x2=0, fill_between=True)

# Right half of figure is shaded between curve_1 and curve_2
area_between_curves = curve_2.get_area_between(
    x1=0,
    x2=10,
    other_curve=curve_1,
    fill_between=True,
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.set_grid()
fig.add_elements(curve_1, curve_2)
fig.show()
