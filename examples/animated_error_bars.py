"""
Animating Error Bars
====================

The samples follow a traveling sine wave. Their lower and upper errors
change independently. Each frame updates the measurements and both sets of
errors with one call to :meth:`~graphinglib.Curve.set_data`.
"""

import numpy as np

import graphinglib as gl

FRAMES = 24
INTERVAL = 90


def save_animation(filename):
    x = np.linspace(0, 2 * np.pi, 15)

    def measurements(frame):
        phase = 2 * np.pi * frame / FRAMES
        values = np.sin(x - phase)
        lower = 0.05 + 0.11 * (1 + np.cos(x - phase))
        upper = 0.07 + 0.13 * (1 + np.sin(x - phase))
        return values, np.vstack((lower, upper))

    values, errors = measurements(0)
    trace = gl.Curve(x, values, color="#007c83", line_width=2.7)
    trace.add_errorbars(
        y_error=errors, errorbars_color="#bd532f", cap_width=4, cap_thickness=1.2
    )
    figure = gl.SmartFigure(
        elements=trace,
        figure_style="plain",
        size=(8.1, 4.6),
        x_lim=(0, 2 * np.pi),
        y_lim=(-1.35, 1.35),
        x_label="Position (rad)",
        y_label="Signal",
        title="Traveling sine wave with asymmetric errors",
    )

    def update(frame):
        values, errors = measurements(frame)
        trace.set_data(x, values, y_error=errors)

    figure.save_animation(
        filename, update, number_of_frames=FRAMES, interval=INTERVAL, dpi=96
    )


if __name__ == "__main__":
    save_animation("animated_error_bars.gif")
