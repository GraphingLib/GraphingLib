"""
Histogram
=========

_thumb: .4, .4
"""

import numpy as np
import graphinglib as gl

values = np.random.normal(loc=2, scale=5, size=500)

histogram = gl.Histogram(values, number_of_bins=30, label="Distribution of values")

figure = gl.Figure(x_label="Values", y_label="Counts", size=(8, 6))
figure.add_elements(histogram)
figure.show()
