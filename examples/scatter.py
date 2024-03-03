"""
Scatter
=======

_thumb: .4, .4
"""

import numpy as np
import graphinglib as gl


# Create noisy data0
x = np.linspace(0, 10, 100)
y = x**2 - 3 * x + 3 + np.random.normal(0, 7, 100)

scatter = gl.Scatter(x, y, "Data")
fit = gl.FitFromPolynomial(scatter, 2, "Fit")

# Use the fit to predict value of y at x = 5
print(f"Value of fit at x = 5 is y = {fit.function(5)}")
predicted_point = fit.create_point_at_x(5, color="red")

fig = gl.Figure()
fig.add_elements(scatter, fit, predicted_point)
fig.show()
