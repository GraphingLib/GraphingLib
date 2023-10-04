# GraphingLib

![graphinglib logo](https://github.com/GraphingLib/GraphingLib/blob/master/images/GraphingLib-Logo-Bolder.svg?raw=true)

## 1. Description

GraphingLib is an object oriented library combining the functionalities of Matplotlib and Scipy. With this library it is possible to create scientific graphs to visualise data while fitting the data with simple, single-line commmands.

GraphingLib also provides the ability to create multiple predefined themes for different applications. Once those themes are specified, they can be applied to figures with a one-word parameter.

## 2. Installation

From PyPI with

```text
pip install graphinglib
```

From source with

```text
pip install git+https://github.com/GraphingLib/GraphingLib.git
```

With Poetry with

```text
poetry add graphinglib
```

## 3. Documentation

As of v1.1.0, a documentation page has been created and is accessible [here](https://graphinglib.readthedocs.io/). Note however that it is still a work in progress and that it will be improving in the future.

## 4. Quick usage

Here is a simple example of a curve fit using GraphingLib. For more examples, see the examples folder.

```python
import graphinglib as gl
import numpy as np

# Create some noisy random data :
x_data = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 5, len(x_data))
y_data = 0.05 * x_data ** 3 + x_data ** 2 + x_data + noise

# Create the Scatter object
scatter = gl.Scatter(x_data, y_data, label="Scatter plot")

# Create a curve fit
fit = gl.FitFromPolynomial(scatter, degree=3, label="Curve fit")

# Create the figure object and add the Scatter object to the figure
fig_1 = gl.Figure(x_label="x values", y_label="y values")
fig_1.add_element(scatter)

# Display the figure
fig_1.display()
```
![quick usage image](https://github.com/GraphingLib/GraphingLib/blob/master/images/Quick-Usage.png?raw=true)

## 5. Why GraphingLib

It is our belief that the best way to explain the simplicity of GraphingLib is by providing an example. This is why we have included the code below, creating the exact same graph as before, but using the commands defined in Matplotlib and Scipy.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Create function to be plotted
def my_func(x_data, a, b, c, d):
    return a * x_data**3 + b * x_data**2 + c * x_data + d


# Create some noisy random data :
x_data = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 5, len(x_data))
y_data = my_func(x_data, 0.05, 1, 1, 1) + noise

# Create the figure
fig, ax = plt.subplots()

# Create a scatter plot
ax.scatter(x_data, y_data, label="Scatter plot", color="k")

# Create a curve fit
popt, pcov = curve_fit(my_func, x_data, y_data)
ax.plot(
    x_data,
    my_func(x_data, popt[0], popt[1], popt[2], popt[3]),
    label=f"Curve fit : {popt[0]:.3f}x^3+{popt[1]:.3f}x^2+{popt[2]:.3f}x+{popt[3]:.3f}",
    color="red",
)

# Formatting the figure
ax.tick_params(axis="both", direction="in")
ax.set_xlabel("x values")
ax.set_ylabel("y values")
ax.legend(frameon=False)

# Show the figure
plt.show()
```

In this simple example, 38 lines of code were necessary to get the same result we obtained in only 20 lines of code with GraphingLib.

GraphingLib also simplifies the customization of figure looks by adding the option to specify custom themes. Those themes can then be used by adding a simple parameter the the Figure object. Here are two examples of themes that are predefined :

In the example above the theme is the "plain" one. If we instead use the "horrible" theme instead we get
```python
fig_1 = gl.Figure(x_label="x values", y_label="y values", figure_style="horrible")
```
![horrible theme image](https://github.com/GraphingLib/GraphingLib/blob/master/images/Horrible-theme.png?raw=true)
