import numpy as np
from scipy.optimize import curve_fit


class Perin11:
    """
    This class fits a Gaussian function to the connection probability vs. inter-somatic
    distance among pairs of thick-tufted L5 pyramids in P14-16 Wistar rats, from Fig. 1 of [1].

    In the source figure, I would expect "overall" to be the sum of reciprical
    and non-reciprocal, but it isn't. It doesn't look like this much affects the spatial
    profile though, just the peak (which we don't use).

    [1] R. Perin, T. K. Berger, and H. Markram, “A synaptic organizing principle for cortical neuronal
    groups.,” Proc. Natl. Acad. Sci. U. S. A., vol. 108, no. 13, pp. 5419–24, Mar. 2011.
    """

    def __init__(self):
        connection_probability_vs_distance = [
            [17.441860465116307, 0.21723833429098494],
            [52.79069767441864, 0.1676015362748359],
            [87.44186046511628, 0.14761544742492516],
            [122.5581395348837, 0.12294674448846282],
            [157.67441860465118, 0.09515710527111632],
            [192.55813953488376, 0.10208848701121961],
            [227.44186046511635, 0.06337617564339071],
            [262.5581395348837, 0.03480630235582299],
            [297.44186046511635, 0.07021622765899538],
        ]

        def gaussian(x, peak, sigma):
            return peak * np.exp(-(x**2) / 2 / sigma**2)

        cp = np.array(connection_probability_vs_distance)
        popt, pcov = curve_fit(gaussian, cp[:, 0], cp[:, 1], p0=(0.2, 150))
        self.width_micrometers = popt[1]


class Billeh19:
    """
    Data from literature review by Yazan Billeh.
    TODO: further details and reference the paper once it's published.
    """

    def __init__(self):
        self._layers = ["2/3", "4", "5", "6"]
        self.probabilities = [
            [0.160, 0.016, 0.083, 0],
            [0.14, 0.243, 0.104, 0.032],
            [0.021, 0.007, 0.116, 0.047],
            [0, 0, 0.012, 0.026],
        ]

    def get_connection_probability(self, source, target):
        assert source in self._layers
        assert target in self._layers

        source_index = self._layers.index(source)
        target_index = self._layers.index(target)

        return self.probabilities[source_index][target_index]
