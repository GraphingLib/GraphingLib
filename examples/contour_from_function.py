"""
Contour From Function
=====================

_thumb: .4, .4
"""

import numpy as np

import graphinglib as gl


# Create function to generate data
def func(x, y):
    return (
        np.exp(-((x - 3) ** 2 + (y - 3) ** 2) / 2)
        + np.exp(-((x - 6) ** 2 + (y - 6) ** 2) / 2)
        + np.random.normal(0, 0.1)
    )


contour = gl.Contour.from_function(
    func,
    x_axis_range=(1, 8),
    y_axis_range=(1, 8),
    color_map="viridis",
    show_color_bar=True,
    filled=False,
    number_of_points=(20, 20),
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(contour)
fig.show()
