"""
Dark Style
==========

_thumb: .4, .4
"""

import graphinglib as gl
import numpy as np

# Generating data for a 2D plot of a Lissajous curve
A = 5
B = 4
a = 3
b = 2
delta = np.pi / 2

t = np.linspace(0, 2 * np.pi, 1000)
x = A * np.sin(a * t + delta)
y = B * np.sin(b * t)

curve = gl.Curve(x_data=x, y_data=y)

# Creating a figure with a dark style
fig = gl.Figure(
    x_label="x",
    y_label="y",
    figure_style="dark",
)
fig.add_elements(curve)
fig.show()
