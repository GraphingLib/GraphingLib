"""
Histogram
=========

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

values = np.random.normal(loc=2, scale=5, size=500)

histogram = gl.Histogram(
    values,
    bins=30,
    hist_type="bar",
    face_color="C2",
    edge_color="k",
    line_width=1,
    label="Distribution of values",
)

figure = gl.Figure(x_label="Values", y_label="Counts", size=(8, 6))
figure.add_elements(histogram)
figure.show()
