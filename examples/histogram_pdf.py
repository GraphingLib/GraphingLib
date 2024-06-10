"""
Histogram PDF
=============

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Generate random values
values = np.random.normal(loc=2, scale=5, size=500)

# Create histogram
histogram = gl.Histogram(values, number_of_bins=30, label="Distribution of values")

# Add a normal probability density function overlay
# The pdf is calculated using the mean and standard deviation of the data
histogram.add_pdf("normal")

# Access mean and standard deviation of the data
mean = histogram.mean
std = histogram.standard_deviation

figure = gl.Figure(x_label="Values", y_label="Probability Density", size=(8, 6))
figure.add_elements(histogram)
figure.show()
