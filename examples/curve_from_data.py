"""
Curve From Data
===================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create fictional data
x = np.linspace(-10, 10, 100)
y1 = 3 * x**2 - 2 * x + 5 + np.random.normal(0, 10, 100)
y2 = -2 * x**2 + 3 * x + 200 + np.random.normal(0, 30, 100)

# Create curves from the data
curve_a = gl.Curve(x, y1, label="Curve A")
curve_b = gl.Curve(x, y2, line_style="-.", color="C4", label="Curve B")

# Create a figure and add the curves
fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(curve_a, curve_b)
fig.show()
