"""
Error Curves
============

_thumb: .4, .4
"""

import graphinglib as gl
import numpy as np

curve = gl.Curve.from_function(
    lambda x: x**2 - 30 * np.sin(0.7 * x) + 3, x_min=-10, x_max=10, label="Data"
)

curve.add_error_curves(y_error=10)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(curve)
fig.show()
