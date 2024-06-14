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

# Create figures and add the elements
fig1 = gl.Figure()
fig1.add_elements(scatter, fit)

fig2 = gl.Figure(y_lim=(0, 0.055))
fig2.add_elements(residuals)

# Combine figures into a single multifigure
multifigure = gl.MultiFigure.from_row(
    [fig1, fig2], size=(10, 5), reference_labels=False
)
multifigure.show()
