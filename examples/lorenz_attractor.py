"""
Lorenz Attractor
================

_thumb: .4, .4
"""

import graphinglib as gl


# Generating data for a 2D plot of the Lorenz attractor
def lorenz_attractor(sigma, beta, rho, initial_state, num_steps, dt):
    x, y, z = initial_state
    xs, ys, zs = [], [], []

    for _ in range(num_steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        x += dx * dt
        y += dy * dt
        z += dz * dt

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return xs, ys, zs


# Parameters for the Lorenz attractor
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0
initial_state = (0.0, 1.0, 1.05)
num_steps = 5000
dt = 0.01

# Generating the Lorenz attractor data
xs, ys, zs = lorenz_attractor(sigma, beta, rho, initial_state, num_steps, dt)

# 2D plot of the Lorenz attractor
curve = gl.Curve(x_data=xs, y_data=zs, line_width=1)

# Choosing a style from the matplotlib library
fig = gl.Figure(x_label="x", y_label="z", figure_style="Solarize_Light2")

fig.add_elements(curve)
fig.show()
