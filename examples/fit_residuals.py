"""
Fit Residuals
=============

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl

# Create fictional data
np.random.seed(0)
x_data = np.linspace(0, 10, 100)
y_data = 3 * x_data**2 - 2 * x_data + np.random.normal(0, 10, 100)

# Create scatter plot
scatter = gl.Scatter(x_data, y_data, label="Data")

# Create a polynomial fit
fit = gl.FitFromPolynomial(scatter, degree=2, label="Fit", color="red")

# Create histogram of residuals
residuals = gl.Histogram.from_fit_residuals(fit, 15, label="Residual")
residuals.add_pdf("normal")

# Show the fit and its residual distribution side by side
figure = gl.SmartFigure(
    1,
    2,
    elements=[[scatter, fit], residuals],
    size=(10, 5),
    y_lim=[None, (0, 0.055)],
    reference_labels=False,
)
figure.show()
