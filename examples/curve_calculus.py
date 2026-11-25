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

# Create a panel for each group of curves
figure = gl.SmartFigure(
    1,
    2,
    elements=[
        [curve, derivative_curve, integral_curve],
        [curve, normal_curve, tangent_curve],
    ],
    size=(12, 6),
    x_lim=[None, (-5, 5)],
    y_lim=[(-6, 25), (-6, 4)],
    aspect_ratio=["auto", 1],
    reference_labels=False,
)
figure.show()
