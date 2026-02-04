"""
Heatmap From Image
==================

_thumb: .4, .4
"""

import graphinglib as gl

# Load image
image_source = "../docs/_static/icons/GraphingLib-favicon_250x250.png"

# Create heatmap
heatmap = gl.Heatmap(
    image_source,
    color_map="plasma",
    show_color_bar=True,
    aspect_ratio="equal",
)

fig = gl.Figure(size=(8, 6), x_label="x values", y_label="y values")
fig.add_elements(heatmap)
fig.show()
