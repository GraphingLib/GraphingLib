"""
Curve Calculus
==============

_thumb: .4, .4
"""

import graphinglib as gl

curve = gl.Curve.from_function(lambda x: x**2 - 5, x_min=-5, x_max=5)

# Create derivative, integral, normal, and tangent curves
derivative_curve = curve.create_derivative_curve(label="Derivative")
integral_curve = curve.create_integral_curve(label="Integral")
normal_curve = curve.create_normal_curve(2, label="Normal at x=2")
tangent_curve = curve.create_tangent_curve(2, label="Tangent at x=2")

# Create figures and add the curves
fig1 = gl.Figure(y_lim=(-6, 25))
fig1.add_elements(curve, derivative_curve, integral_curve)

fig2 = gl.Figure(x_lim=(-5, 5), y_lim=(-6, 4), aspect_ratio=1)
fig2.add_elements(curve, normal_curve, tangent_curve)

# Create a multifigure from the figures
multifig = gl.MultiFigure.from_row([fig1, fig2], size=(12, 6), reference_labels=False)
multifig.show()
