"""
Infinite Square Well
====================

_thumb: .4, .4
"""

import numpy as np
import graphinglib as gl

# Define parameters
n_levels = 3
well_width = 1

# Calculate and create energy lines and labels
energies = [(n**2 * np.pi**2) / (2 * well_width**2) for n in range(1, n_levels + 1)]

energy_lines = gl.Hlines(
    y=energies,
    x_min=0,
    x_max=well_width,
    colors="grey",
    line_styles="--",
    line_widths=1,
)

energy_labels = [
    gl.Text(
        x=well_width + 0.05,
        y=energy,
        text=f"$E_{n + 1}$",
        color="gray",
        font_size=12,
        h_align="left",
        v_align="center",
    )
    for n, energy in enumerate(energies)
]

# Calculate and create wavefunctions
x = np.linspace(0, well_width, 1000)
wavefunctions = [
    np.sqrt(2 / well_width) * np.sin(n * np.pi * x / well_width)
    for n in range(1, n_levels + 1)
]
# Scale the wavefunctions for better visualization and add the energies to shift them up
wavefunctions = [wave * 3 + energies[i] for i, wave in enumerate(wavefunctions)]

wave_curves = []
for wave in wavefunctions:
    wave_curves.append(gl.Curve(x_data=x, y_data=wave))

# Create well
well = gl.Rectangle(
    x_bottom_left=0,
    y_bottom_left=0,
    width=well_width,
    height=max(energies) * 2,  # Out of the plot range
    fill=False,
    edge_color="black",
)

# Create figure and add elements
fig = gl.Figure(
    x_label="$x$",
    y_label="Energy and $\\psi_n(x)$",
    x_lim=(-0.2 * well_width, 1.2 * well_width),
    y_lim=(-2, max(energies) * 1.3),
    title="Infinite Square Well",
)
# Use * to unpack the lists
fig.add_elements(well, energy_lines, *energy_labels, *wave_curves)
fig.show()
