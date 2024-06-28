"""
Mandelbrot Set
==============

_thumb: .4, .4
"""

import numpy as np
import graphinglib as gl


# Generating data for a 2D plot of a Mandelbrot set
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


# Define the plotting region and resolution
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height = 400, 400
max_iter = 20

# Create the image
image = np.zeros((height, width))
for x in range(width):
    for y in range(height):
        re = xmin + (x / width) * (xmax - xmin)
        im = ymin + (y / height) * (ymax - ymin)
        c = complex(re, im)
        m = mandelbrot(c, max_iter)
        color = 255 - int(m * 255 / max_iter)
        image[y, x] = color

heatmap = gl.Heatmap(
    image=image,
    x_axis_range=(xmin, xmax),
    y_axis_range=(ymin, ymax),
)

fig = gl.Figure(x_lim=(xmin, xmax), y_lim=(ymin, ymax))
fig.add_elements(heatmap)
fig.show()
