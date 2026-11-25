"""
Animating a Julia Set
=====================

The parameter ``c`` travels around an ellipse in the complex plane. Each frame
colors points by how quickly they escape under the iteration ``z = z*z + c``.
The inset samples the same fixed region at higher resolution while a marker
tracks ``c`` on the parameter plot.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import graphinglib as gl

FRAMES = 30
FULL_X = (-1.74, 1.74)
FULL_Y = (-1.31, 1.31)
ZOOM_X = (-0.92, -0.46)
ZOOM_Y = (-0.16, 0.30)
STEPS = 72


def plane(x_range, y_range, width, height):
    x = np.linspace(*x_range, width)
    y = np.linspace(*y_range, height)
    return x[np.newaxis, :] + 1j * y[:, np.newaxis]


def escape_field(grid, c):
    """Color escaping pixels by smoothed iteration count."""
    z = grid.ravel().copy()
    active = np.arange(z.size)
    count = np.zeros(z.size)
    for step in range(1, STEPS + 1):
        z = z * z + c
        radius = np.abs(z)
        escapes = radius > 4
        count[active[escapes]] = step + 1 - np.log2(np.log(radius[escapes]))
        active = active[~escapes]
        z = z[~escapes]
        if not active.size:
            break
    return np.log1p(count.reshape(grid.shape))


def save_animation(filename):
    angles = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
    orbit = -0.74 + 0.115j + 0.087 * np.cos(angles) + 0.082j * np.sin(angles)
    full_grid = plane(FULL_X, FULL_Y, 416, 312)
    zoom_grid = plane(ZOOM_X, ZOOM_Y, 270, 270)
    palette = LinearSegmentedColormap.from_list(
        "escape",
        [
            (0.00, "#141326"),
            (0.19, "#141326"),
            (0.30, "#153052"),
            (0.45, "#266f9d"),
            (0.56, "#23adbd"),
            (0.68, "#63d7c0"),
            (0.79, "#f0c86e"),
            (0.90, "#fa8e52"),
            (1.00, "#fff0be"),
        ],
    )
    c = orbit[0]

    def initial_heatmap(grid, x_range, y_range):
        return gl.Heatmap(
            escape_field(grid, c),
            x_axis_range=x_range,
            y_axis_range=y_range,
            color_map=palette,
            color_map_range=(0, 4.15),
            show_color_bar=False,
            origin_position="lower",
            aspect_ratio="equal",
            interpolation="bilinear",
        )

    overview = initial_heatmap(full_grid, FULL_X, FULL_Y)
    detail = initial_heatmap(zoom_grid, ZOOM_X, ZOOM_Y)
    xmin, xmax = ZOOM_X
    ymin, ymax = ZOOM_Y
    zoom_outline = gl.Curve(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        color="#a3b8cf",
        line_width=0.8,
    )
    view = gl.SmartFigure(
        elements=[overview, zoom_outline],
        title="Julia boundary",
        x_lim=FULL_X,
        y_lim=FULL_Y,
        aspect_ratio="equal",
        remove_x_ticks=True,
        remove_y_ticks=True,
        show_legend=False,
        reference_labels=False,
    )
    view.create_inset(
        (0.69, 0.65, 0.29, 0.30),
        elements=[detail],
        x_lim=ZOOM_X,
        y_lim=ZOOM_Y,
        indicate_zoom=True,
    )

    closed_orbit = np.r_[orbit, orbit[0]]
    path = gl.Curve(
        closed_orbit.real, closed_orbit.imag, color="#6d91bd", line_width=1.7
    )
    cursor = gl.Point(
        c.real, c.imag, face_color="#ffb666", edge_color="#fff0cc", marker_size=102
    )
    readout = gl.Text(
        -0.74,
        -0.005,
        f"c = {c.real:.3f} {c.imag:+.3f}i",
        color="#fff0cc",
        font_size=13,
        h_align="center",
    )
    parameters = gl.SmartFigure(
        elements=[path, cursor, readout],
        title="Parameter path",
        x_label="Re(c)",
        y_label="Im(c)",
        x_lim=(-0.873, -0.609),
        y_lim=(-0.025, 0.219),
        box_aspect_ratio=1,
        show_legend=False,
        reference_labels=False,
    )
    parameters.set_visual_params(
        x_axis_label_color="#c6d3e4", y_axis_label_color="#c6d3e4"
    )
    parameters.set_ticks(x_ticks=[-0.85, -0.75, -0.65], y_ticks=[0.0, 0.1, 0.2])
    figure = gl.SmartFigure(
        1,
        2,
        elements=[view, parameters],
        size=(9.2, 5.1),
        width_ratios=[1.8, 1],
        width_padding=0.08,
        figure_style="dark",
        show_legend=False,
        reference_labels=False,
    )
    figure.set_rc_params(
        rc_params_dict={
            "figure.facecolor": "#121b2b",
            "axes.facecolor": "#141326",
            "axes.edgecolor": "#607691",
            "axes.labelcolor": "#c6d3e4",
            "xtick.color": "#b8cada",
            "ytick.color": "#b8cada",
            "text.color": "#ecf6ff",
            "axes.titlesize": 14,
            "font.size": 10.0,
        },
    )

    def update(frame):
        c = orbit[frame]
        overview.image = escape_field(full_grid, c)
        detail.image = escape_field(zoom_grid, c)
        cursor.coordinates = (c.real, c.imag)
        readout.text = f"c = {c.real:.3f} {c.imag:+.3f}i"

    figure.save_animation(
        filename, update, number_of_frames=FRAMES, interval=110, dpi=85
    )


if __name__ == "__main__":
    save_animation("animated_julia_set.gif")
