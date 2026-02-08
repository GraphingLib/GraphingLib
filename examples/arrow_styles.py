"""
Arrow Styles
============

_thumb: .4, .4
"""

import graphinglib as gl

arrow_styles = ["->", "-|>", "-[", "]->", "simple", "fancy", "wedge"]

one_sided_arrows = []
for i, style in enumerate(reversed(arrow_styles)):
    one_sided_arrows.extend(
        [
            gl.Arrow(
                (0, i),
                (1, i),
                width=2,
                head_size=3,
                color="black",
                style=style,
                two_sided=False,
            ),
            gl.Text(
                0.5,
                i + 0.3,
                f'style="{style}"',
                color="black",
                font_size=10,
                h_align="center",
                v_align="center",
            ),
        ]
    )

two_sided_arrows = []
for i, style in enumerate(reversed(arrow_styles[:3]), start=4):
    two_sided_arrows.extend(
        [
            gl.Arrow(
                (0, i),
                (1, i),
                width=2,
                head_size=3,
                color="black",
                style=style,
                two_sided=True,
            ),
            gl.Text(
                0.5,
                i + 0.3,
                f'style="{style}", two_sided=True',
                color="black",
                font_size=10,
                h_align="center",
                v_align="center",
            ),
        ]
    )

fig = gl.SmartFigure(
    num_cols=2,
    x_lim=(-0.25, 1.25),
    y_lim=(-0.5, len(arrow_styles) - 0.5),
    remove_x_ticks=True,
    remove_y_ticks=True,
    reference_labels=False,
    subtitles=["One-sided Arrows", "Two-sided Arrows"],
    elements=[one_sided_arrows, two_sided_arrows],
)
fig.show()
