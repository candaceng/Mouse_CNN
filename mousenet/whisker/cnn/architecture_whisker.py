import numpy as np

class WhiskerArchitecture:
    def __init__(self, neurons_per_channel=200):
        self.neurons_per_channel = neurons_per_channel
        self.in_degree = 1000  # avg number of inputs per target neuron
        self.neuron_counts = {
            'SSp-bfd': {'2/3': 175481, '4': 135943, '5': 117751},
            'SSs': {'2/3': 95958, '4': 48397, '5': 65758},
            'VISrl': {'2/3': 11831, '4': 4927, '5': 8161}  # estimated from (density x SA) ratio with VISl
        }
        self.areas = ['SSp-bfd', 'SSs', 'VISrl']
        self.layers = ['2/3', '4', '5'] 

        self.d_w = {'SSp-bfd4 --> SSp-bfd2/3': 333.45187201506764,
            'SSp-bfd2/3 --> SSp-bfd5': 360.9601087973823,
            'SSs4 --> SSs2/3': 669.0915520825824,
            'SSs2/3 --> SSs5': 685.6151811397214,
            'SSp-bfd2/3 --> SSs4': 378.79212975098477,
            'SSp-bfd4 --> SSs4': 371.80719034383293,
            'SSp-bfd5 --> SSs4': 388.33602475942547,
            'VISrl4 --> VISrl2/3': 103.80973051521892,
            'VISrl2/3 --> VISrl5': 85.9964158059525,
            'SSs2/3 --> VISrl4': 676.663926314319,
            'SSs4 --> VISrl4': 634.205923982863,
            'SSs5 --> VISrl4': 672.7665540952046,
            'SSp-bfd2/3 --> VISrl4': 375.60246193309837,
            'SSp-bfd4 --> VISrl4': 375.3593497233801,
            'SSp-bfd5 --> VISrl4': 397.6682507518624
        }

        # estimate_area_surface_mm2() in util.py
        self.surface_areas = {
            'SSp-bfd': 9.289074661919935,
            'SSs': 16.87476087000696,
            'VISrl': 3.0787379412331273
        }

    def get_hierarchical_level(self, area):
        return {'SSp-bfd': 1, 'SSs': 2, 'VISrl': 3}[area]
    
    def get_channels(self, area, layer, max_channels=256):
        """
        Estimate the number of channels for a given area and layer
        using scaled neuron counts relative to the largest area-layer.

        Args:
            area (str): e.g. 'SSp-bfd'
            layer (str): e.g. '5'
            spatial_size (int): total number of spatial positions (e.g., 5*7)
            max_channels (int): maximum allowed number of channels

        Returns:
            int: number of channels for this area/layer
        """
        neurons = self.neuron_counts[area][layer]

        # Compute max neuron count across all areas/layers
        max_neurons = max(
            self.neuron_counts[a][l]
            for a in self.neuron_counts
            for l in self.neuron_counts[a]
        )

        # Scale neuron count relative to max and map to max_channels
        scaled_fraction = neurons / max_neurons
        raw_channels = round(scaled_fraction * max_channels)

        # Make sure at least 1 channel
        channels = max(1, raw_channels)

        return channels


    def get_pixels_per_micrometer(self, area, layer):
        n = self.neuron_counts[area][layer]
        channels = self.get_channels(area, layer)
        linear_pixels = np.sqrt(n / channels)
        linear_um = 1000 * self.surface_areas[area]  
        return linear_pixels / linear_um

    def get_connection_probability(self, source, target):
        layers = ['2/3', '4', '5', '6']
        probabilities = [
            [.160, .016, .083, 0],
            [.14, .243, .104, .032],
            [.021, .007, .116, .047],
            [0, 0, .012, .026]
        ]
        assert source in layers
        assert target in layers

        source_index = layers.index(source)
        target_index = layers.index(target)

        return probabilities[source_index][target_index]
    
    def get_hit_rate_width(self, source_layer, target_layer):
        """
        :param source_layer: name of presynaptic layer
        :param target_layer: name of postsynaptic layer
        :return: width of Gaussian approximation of fraction of excitatory neuron pairs with
            functional connection in this direction
        """

        # Levy & Reyes (2012; J Neurosci) report sigma 114 micrometers for probability of
        # functional connection between pairs of L4 pyramidal cells in mouse auditory cortex.
        # See their Table 3.
        l4_to_l4 = 114

        # Stepanyants, Hirsch, Martinez (2008; Cerebral Cortex) report variations in width
        # of potential connection probability depending on source and target layer in cat V1.
        # See their Figure 8B. Values below are rough manual estimates from their figure.
        cat = { # source -> target
            '2/3': {'2/3': 225, '4': 50, '5': 100, '6': 50},
            '4': {'2/3': 220, '4': 180, '5': 140, '6': 110},
            '5': {'2/3': 150, '4': 100, '5': 210, '6': 125},
            '6': {'2/3': 120, '4': 20, '5': 150, '6': 150}
        }

        return cat[source_layer][target_layer] / cat['4']['4'] * l4_to_l4
    
    def get_hit_rate_peak(self, source_layer, target_layer):
        """
        :param source_layer: name of presynaptic layer
        :param target_layer: name of postsynaptic layer
        :return: fraction of excitatory neuron pairs with functional connection in this
            direction, at zero horizontal offset
        """
        hit_rate = self.get_connection_probability(source_layer, target_layer)
        fraction_of_peak = np.exp(-75**2 / 2 / self.get_hit_rate_width(source_layer, target_layer)**2)
        return hit_rate / fraction_of_peak

    def get_kernel_width_pixels(self, source_area, source_layer, target_area, target_layer):
        if source_area == target_area: # from interlaminar hit rate spatial profile
            return self.get_hit_rate_width(source_layer, target_layer) * self.get_pixels_per_micrometer(source_area, source_layer)

        d_w_um = self.d_w[f'{source_area}{source_layer} --> {target_area}{target_layer}']

        pixels_per_um = self.get_pixels_per_micrometer(source_area, source_layer)
        # print(f'd_w from {source_area}{source_layer} to {target_area}{target_layer}: {d_w_um* pixels_per_um}')
        return d_w_um * pixels_per_um  # final result in pixels

    def get_kernel_peak_probability(self, source_area, source_layer, target_area, target_layer):
        if source_area == target_area: # from interlaminar hit rates
            return self.get_hit_rate_peak(source_layer, target_layer)
        
        d_w = self.get_kernel_width_pixels(source_area, source_layer, target_area, target_layer)
        source_channels = self.get_channels(source_area, source_layer)

        # Create a 2D grid around the center of the kernel, using 3σ radius
        support = int(np.ceil(3 * d_w))
        x = np.arange(-support, support + 1)
        X, Y = np.meshgrid(x, x)
        radius_squared = X**2 + Y**2

        gaussian = np.exp(-radius_squared / (2 * d_w ** 2))
        normalization = np.sum(gaussian)

        # Solve for d_p such that expected total input = e_ij
        d_p = self.in_degree / (source_channels * normalization)
        return d_p

    def get_kernel_size(self, source_area, source_layer, target_area, target_layer):
        d_w = self.get_kernel_width_pixels(source_area, source_layer, target_area, target_layer)
        
        return np.int(2 * np.floor(d_w) + 1)  # ensure odd size

    def get_padding(self, source_area, source_layer, target_area, target_layer):
        k = self.get_kernel_size(source_area, source_layer, target_area, target_layer)
        return k // 2
            