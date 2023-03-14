from Graphing import Curve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class FitFromPolynomial(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a polynomial fit.
    """
    
    def __init__(self, curve_to_be_fit: Curve, degree: int, color: str, label: str):
        self.curve_to_be_fit = curve_to_be_fit
        inversed_coeffs, inversed_cov_matrix = np.polyfit(self.curve_to_be_fit.xdata, self.curve_to_be_fit.ydata, degree, cov=True)
        self.coeffs = inversed_coeffs[::-1]
        self.cov_matrix = np.flip(inversed_cov_matrix)
        self.function = self.get_polynomial_function()
        self.color = color
        self.label = label
        
    def __str__(self):
        coeff_chunks = []
        power_chunks = []
        ordered_rounded_coeffs = [round(coeff, 3) for coeff in self.coeffs[::-1]]
        for coeff, power in zip(ordered_rounded_coeffs, range(len(ordered_rounded_coeffs) - 1, -1, -1)):
            if coeff == 0:
                continue
            coeff_chunks.append(self.format_coeff(coeff))
            power_chunks.append(self.format_power(power))
        coeff_chunks[0] = coeff_chunks[0].lstrip("+ ")
        return 'f(x) = '+''.join([coeff_chunks[i] + power_chunks[i] for i in range(len(coeff_chunks))])

    @staticmethod
    def format_coeff(coeff):
        return " - {0}".format(abs(coeff)) if coeff < 0 else " + {0}".format(coeff)

    @staticmethod
    def format_power(power):
        return 'x^{0}'.format(power) if power != 0 else ''

    def get_polynomial_function(self):
        """
        Returns a linear function using the given coefficients.
        """
        return lambda x: sum(coeff * x**exponent for exponent, coeff in enumerate(self.coeffs))
    
    def plot_curve(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points)
        ydata = self.function(xdata)
        self.handle, = axes.plot(xdata, ydata, color=self.color, label=self.label)


class FitFromSine(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a sinusoidal fit.
    """
    
    def __init__(self, curve_to_be_fit: Curve, color: str, label: str):
        self.curve_to_be_fit = curve_to_be_fit
        self.calculate_parameters()
        self.function = self.get_sine_func_with_params()
        self.color = color
        self.label = label
    
    def __str__(self) -> str:
        return f"{self.amplitude:.3f}sin({self.frequency_rad:.3f}x + {self.phase:.3f}) + {self.vertical_shift:.3f}"
    
    def calculate_parameters(self):
        parameters, self.cov_matrix = curve_fit(self.sine_func_template,
                                                self.curve_to_be_fit.xdata,
                                                self.curve_to_be_fit.ydata)
        self.amplitude, self.frequency_rad, self.phase, self.vertical_shift = parameters
    
    @staticmethod
    def sine_func_template(x, a, b, c, d):
            return a * np.sin(b*x + c) + d
    
    def get_sine_func_with_params(self):
        return lambda x: self.amplitude * np.sin(self.frequency_rad * x + self.phase) + self.vertical_shift
    
    def plot_curve(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points)
        ydata = self.function(xdata)
        self.handle, = axes.plot(xdata, ydata, color=self.color, label=self.label)