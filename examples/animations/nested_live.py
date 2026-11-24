"""Animate a nested, spanning SmartFigure with four kinds of live artist."""

import argparse
from pathlib import Path

import numpy as np

import graphinglib as gl

FRAME_COUNT = 16
INTERVAL = 120


def build_animation():
    grid = np.linspace(-2.0, 2.0, 48)
    xx, yy = np.meshgrid(grid, grid)
    samples = np.arange(1.0, 7.0)
    angles = np.arange(12) * (2 * np.pi / 12)

    def frame_data(frame):
        phase = frame * 2 * np.pi / FRAME_COUNT
        curve_x = samples + 0.16 * np.sin(phase)
        curve_y = np.sin(curve_x * 0.78 - phase)
        errors = np.vstack(
            (
                0.13 + 0.10 * (1 + np.sin(samples + phase)),
                0.17 + 0.10 * (1 + np.cos(samples + phase)),
            )
        )
        orbit_x = (1 + 0.25 * np.sin(3 * angles + phase)) * np.cos(angles + phase)
        orbit_y = (1 + 0.25 * np.sin(3 * angles + phase)) * np.sin(angles + phase)
        image = np.exp(
            -((xx - 1.05 * np.cos(phase)) ** 2 + (yy - 1.05 * np.sin(phase)) ** 2)
            / 0.39
        )
        return curve_x, curve_y, errors, orbit_x, orbit_y, image

    x, y, errors, orbit_x, orbit_y, image = frame_data(0)
    curve = gl.Curve(x, y, color="#244e86", line_width=2.7)
    curve.add_errorbars(
        y_error=errors, cap_width=4, cap_thickness=1.1, errorbars_color="#c13c24"
    )
    scatter = gl.Scatter(
        orbit_x, orbit_y, face_color="#087f6d", edge_color="none", marker_size=54
    )
    heatmap = gl.Heatmap(
        image,
        x_axis_range=(-2, 2),
        y_axis_range=(-2, 2),
        color_map="magma",
        color_map_range=(0, 1),
        show_color_bar=False,
    )
    counter = gl.Text(
        0.50,
        0.61,
        f"FRAME 01 / {FRAME_COUNT:02d}",
        color="#244e86",
        font_size=21,
        h_align="center",
        v_align="center",
        highlight_color="#e9f0f9",
        highlight_padding=0.42,
    )
    clock = gl.Text(
        0.50,
        0.30,
        "Elapsed  0.00 s",
        color="#344d69",
        font_size=12,
        h_align="center",
        v_align="center",
    )

    def panel(elements, title, **kwargs):
        return gl.SmartFigure(
            elements=elements, title=title, reference_labels=False, **kwargs
        )

    curve_panel = panel(
        [curve], "Curve and uncertainty", x_lim=(0.5, 6.6), y_lim=(-1.5, 1.55)
    )
    scatter_panel = panel(
        [scatter], "Scatter offsets", x_lim=(-1.6, 1.6), y_lim=(-1.55, 1.55)
    )
    heatmap_panel = panel(
        [heatmap],
        "Raster heatmap",
        x_lim=(-2, 2),
        y_lim=(-2, 2),
        remove_x_ticks=True,
        remove_y_ticks=True,
    )
    text_panel = panel([], "Live annotation", annotations=[counter, clock])
    left = gl.SmartFigure(
        2,
        1,
        elements=[curve_panel, scatter_panel],
        title="Data and geometry",
        reference_labels=False,
    )
    right = gl.SmartFigure(
        2,
        1,
        elements=[heatmap_panel, text_panel],
        title="Pixels and text",
        reference_labels=False,
    )
    root = gl.SmartFigure(
        2,
        2,
        figure_style="plain",
        size=(10, 6.2),
        width_padding=0.08,
        reference_labels=False,
    )
    root[:, 0] = left
    root[:, 1] = right

    def update(frame):
        x, y, errors, orbit_x, orbit_y, image = frame_data(frame)
        curve.set_data(x, y, y_error=errors)
        scatter.set_data(orbit_x, orbit_y)
        heatmap.image = image
        counter.text = f"FRAME {frame + 1:02d} / {FRAME_COUNT:02d}"
        counter.y = 0.61 + 0.08 * np.sin(frame * 2 * np.pi / FRAME_COUNT)
        clock.text = f"Elapsed  {INTERVAL / 1000 * frame:.2f} s"

    return root, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("nested-live.gif"))
    parser.add_argument(
        "--play", action="store_true", help="Open an interactive window."
    )
    args = parser.parse_args()
    figure, update = build_animation()
    if args.play:
        figure.show_animation(update, FRAME_COUNT, interval=INTERVAL)
    else:
        figure.save_animation(
            args.output, update, FRAME_COUNT, interval=INTERVAL, dpi=105
        )


if __name__ == "__main__":
    main()
