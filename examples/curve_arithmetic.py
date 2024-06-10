"""
Curve Arithmetic
================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create fictional data
x = np.linspace(-10, 10, 100)
raw_data = 3 * x + 5 + np.random.normal(0, 1, 100)

# Create curve from the data
raw_data_curve = gl.Curve(x, raw_data, label="Raw data")

# Identify trend
trend = gl.FitFromPolynomial(raw_data_curve, degree=1)

trend_curve = gl.Curve.from_function(
    trend.function,
    x_min=-10,
    x_max=10,
    color="lightgrey",
    line_style="--",
    label="Trend",
)

# Arithmetic between curves: +, -, *, /
corrected_curve = raw_data_curve - trend_curve

# Arithmetic with a scalar: +, -, *, /, **, abs
corrected_curve += 5

corrected_curve.color = "C1"
corrected_curve.label = "Corrected Curve"


# Create a figure and add the curves
fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(trend_curve, raw_data_curve, corrected_curve)
fig.show()
