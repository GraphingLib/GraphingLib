
<div style="width: 100%;">
  <a href="https://www.graphinglib.org/" target="_blank">
    <img src="images/readme_banner.svg" style="width: 100%;" alt="Click to see the documentation website">
  </a>
</div>

[![PyPI version](https://badge.fury.io/py/graphinglib.svg)](https://badge.fury.io/py/graphinglib)
[![Documentation Status](https://readthedocs.org/projects/graphinglib/badge/?version=latest)](https://graphinglib.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi downloads](https://img.shields.io/pypi/dm/graphinglib)](https://pypi.org/project/graphinglib/)
![GitHub stars](https://img.shields.io/github/stars/GraphingLib/GraphingLib.svg?style=social&label=Star&maxAge=2592000)
[![DOI](https://zenodo.org/badge/613172394.svg)](https://doi.org/10.5281/zenodo.19936792)

# GraphingLib

<div style="text-align: center;">
  <img src="https://github.com/GraphingLib/GraphingLib/blob/main/images/GraphingLib-Logo-Bolder.svg?raw=true" alt="graphinglib logo">
</div>

GraphingLib is an open-source data visualization library in Python, designed as a wrapper for matplotlib. It integrates powerful data manipulation features from libraries such as scipy, shapely, and others.

GraphingLib has the following explicit goals:

1. **Simplify Plotting:** Provide the simplest, most intuitive, and user-friendly API to enable users to create plots in as few lines of code as possible.
2. **Data Analysis Functions**: Implement common data analysis functions and operations to streamline the visualization process.
3. **Custom Figure Styles:** Facilitate the customization and reuse of figure styles.

## How is GraphingLib different?

- **Object-Oriented Plotting:** Figures and plotted elements are all objects, making it easier to manage and manipulate plots and elements.
- **Curve Fitting:** Perform curve fitting with a single line of code.
- **Curve Operations:** Carry out differentiation, integration, arithmetic, intersections, and other standard operations on Curve objects.
- **GUI Style Editor:** Use the GraphingLib Style Editor to create and modify custom styles, and set them as your default style.
- **Polygon Manipulation:** Obtain useful information such as area, centroid, and perimeter of polygons, and manipulate them using transform and set operations methods.
- **SmartFigures:** Create modular figures with multiple sub-figures and an intuitive syntax.

## Getting started

To get started with GraphingLib, check out our comprehensive [documentation and examples](https://www.graphinglib.org/) available on our website. Whether you're a beginner or an experienced user, our documentation provides step-by-step guides to help you make the most out of GraphingLib. Here are a few ways to install GraphingLib:

From PyPI with

```bash
pip install graphinglib
```

From source with

```bash
pip install git+https://github.com/GraphingLib/GraphingLib.git
```

Using Poetry with

```bash
poetry add graphinglib
```

Using `uv` with

```bash
uv add graphinglib
```

Optional extras:

- Astronomical projections (SmartFigureWCS): install `pip install graphinglib[astro]`

## Contributing

We welcome contributions from the community. If you're interested in contributing to GraphingLib, please read our [contribution guidelines](https://graphinglib.org/latest/contributing/index.html) on our documentation website.


## Example
Here is a short example showing how to use GraphingLib to create a figure with a scatter plot, a fit, and a histogram of the residuals.
    
```python
import graphinglib as gl
import numpy as np

# Data creation
np.random.seed(2)
x_data = np.linspace(0, 10, 100)
y_data = 3 * x_data**2 - 2 * x_data + np.random.normal(0, 10, 100)

# Create elements
scatter = gl.Scatter(x_data, y_data, label="Position data")
fit = gl.FitFromPolynomial(scatter, degree=2, label="Fit", color="red")
residuals = gl.Histogram.from_fit_residuals(fit, bins=15)
residuals.add_pdf("normal")

# Create and show figure
fig = gl.SmartFigure(
    num_cols=2,
    num_rows=1,
    size=(10, 5),
    y_lim=[None, (0, 0.06)],
    sub_x_labels=["Time [s]", "Distance from fit [mm]"],
    sub_y_labels=["Position [mm]", "Frequency [-]"],
    subtitles=["Position as a function of time", "Histogram of fit residuals"],
)
fig[0] = [scatter, fit]
fig[1] = [residuals]
fig.show()
```

![image](images/example_fit.svg)
