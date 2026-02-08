"""
Heatmap Rotation
================

_thumb: .4, .4
"""

import graphinglib as gl
import numpy as np

# Load image
image_source = "../docs/_static/icons/GraphingLib-favicon_250x250.png"

# Create heatmap
heatmap = gl.Heatmap(
    image_source,
    aspect_ratio="equal",
)

# Rotate the image using x_mesh and y_mesh
angle = np.pi / 6

m, n = heatmap.image.shape[:2]  # Get image dimensions, excluding color channels
y_mesh, x_mesh = np.mgrid[m:0:-1, :n]
x_rot = x_mesh * np.cos(angle) - y_mesh * np.sin(angle)
y_rot = x_mesh * np.sin(angle) + y_mesh * np.cos(angle)

heatmap.x_mesh = x_rot
heatmap.y_mesh = y_rot

fig = gl.Figure(aspect_ratio=1, size=(6, 6), x_label="x values", y_label="y values")
fig.add_elements(heatmap)
fig.show()
