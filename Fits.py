from Graphing import Curve
import numpy as np
import matplotlib.pyplot as plt

class FitFromPolynomial(Curve):
    """
    Create a curve fit (continuous Curve) from an existing curve object.
    """
    
    def __init__(self, curve_to_be_fit: Curve, degree: int, color: str, label: str):
        self.curve_to_be_fit = curve_to_be_fit
        self.coeffs = np.polyfit(self.curve_to_be_fit.xdata, self.curve_to_be_fit.ydata, degree)[::-1]
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