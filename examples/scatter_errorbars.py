"""
Scatter Errorbars
=================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create fictional data
x_data = np.linspace(0, 10, 10)
y_data = 3 * x_data**2 - 2 * x_data

# Errorbars can be constant or variable
x_error = 0.3
y_error = 8 * np.sin(x_data) + 10

# Add errorbars with float or array/list of floats
scatter = gl.Scatter(x_data, y_data, label="Data")
scatter.add_errorbars(x_error=x_error, y_error=y_error)

fig = gl.Figure(x_label="x values", y_label="y values", title="Scatter with Errorbars")
fig.add_elements(scatter)
fig.show()
