"""
Shape Unions
============

_thumb: .4, .4
"""

import graphinglib as gl


# Create G shape
main_circle = gl.Circle(x_center=1, y_center=1, radius=1, fill_color="C0")
triangle = gl.Polygon(vertices=[(1, 1), (5, 5), (5, 1)])
center_circle = gl.Circle(x_center=1, y_center=1, radius=0.7)
bar = gl.Rectangle(x_bottom_left=1, y_bottom_left=0.7, width=0.7, height=0.3)

g_shape = (
    main_circle.create_difference(triangle, copy_style=True)
    .create_difference(center_circle, copy_style=True)
    .create_union(bar, copy_style=True)
)

# Create L shape
main_rectangle = gl.Rectangle(
    x_bottom_left=0, y_bottom_left=0, width=1, height=2, fill_color="C1"
)
smaller_rectangle = gl.Rectangle(
    x_bottom_left=0.3, y_bottom_left=0.3, width=0.7, height=1.7
)
l_shape = main_rectangle.create_difference(smaller_rectangle, copy_style=True)
l_shape.translate(dx=2.5, dy=0)

fig = gl.Figure(aspect_ratio=1, x_lim=(-1, 4.5), y_lim=(-1, 3))
fig.add_elements(g_shape, l_shape)
fig.show()
