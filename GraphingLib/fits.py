from .graph_elements import Curve
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


class FitFromPolynomial(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a polynomial fit.
    """
    
    def __init__(self, curve_to_be_fit: Curve, degree: int, label: str, color: str = 'default', line_width: int = 'default'):
        self.curve_to_be_fit = curve_to_be_fit
        inversed_coeffs, inversed_cov_matrix = np.polyfit(self.curve_to_be_fit.xdata, self.curve_to_be_fit.ydata, degree, cov=True)
        self.coeffs = inversed_coeffs[::-1]
        self.cov_matrix = np.flip(inversed_cov_matrix)
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))
        self.function = self.polynomial_func_with_params()
        self.color = color
        self.line_width = line_width
        self.label = label + ' : ' + 'f(x) = ' + str(self)
        
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
        return ''.join([coeff_chunks[i] + power_chunks[i] for i in range(len(coeff_chunks))])

    @staticmethod
    def format_coeff(coeff):
        return " - {0}".format(abs(coeff)) if coeff < 0 else " + {0}".format(coeff)

    @staticmethod
    def format_power(power):
        return 'x^{0}'.format(power) if power != 0 else ''

    def polynomial_func_with_params(self):
        """
        Returns a linear function using the class' coefficients.
        """
        return lambda x: sum(coeff * x**exponent for exponent, coeff in enumerate(self.coeffs))
    
    def plot_curve(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points)
        ydata = self.function(xdata)
        self.handle, = axes.plot(xdata, ydata, label=self.label, color=self.color, linewidth=self.line_width)


class FitFromSine(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object using a sinusoidal fit.
    """
    
    def __init__(self, curve_to_be_fit: Curve, color: str, label: str, guesses: npt.ArrayLike=None):
        self.curve_to_be_fit = curve_to_be_fit
        self.guesses = guesses
        self.calculate_parameters()
        self.function = self.sine_func_with_params()
        self.color = color
        self.label = label + ' : ' + 'f(x) = ' + str(self)
    
    def __str__(self) -> str:
        part1 = f"{self.amplitude:.3f} sin({self.frequency_rad:.3f}x"
        part2 = f" + {self.phase:.3f})" if self.phase >= 0 else f" - {abs(self.phase):.3f})"
        part3 = f" + {self.vertical_shift:.3f}" if self.phase >= 0 else f" - {abs(self.vertical_shift):.3f}"
        return part1 + part2 + part3
    
    def calculate_parameters(self):
        parameters, self.cov_matrix = curve_fit(self.sine_func_template,
                                                self.curve_to_be_fit.xdata,
                                                self.curve_to_be_fit.ydata, p0=self.guesses)
        self.amplitude, self.frequency_rad, self.phase, self.vertical_shift = parameters
        self.standard_deviation = np.sqrt(np.diag(self.cov_matrix))
    
    @staticmethod
    def sine_func_template(x, a, b, c, d):
            return a * np.sin(b*x + c) + d
    
    def sine_func_with_params(self):
        return lambda x: self.amplitude * np.sin(self.frequency_rad * x + self.phase) + self.vertical_shift
    
    def plot_curve(self, axes: plt.Axes):
        num_of_points = 500
        xdata = np.linspace(self.curve_to_be_fit.xdata[0], self.curve_to_be_fit.xdata[-1], num_of_points)
        ydata = self.function(xdata)
        self.handle, = axes.plot(xdata, ydata, color=self.color, label=self.label)
