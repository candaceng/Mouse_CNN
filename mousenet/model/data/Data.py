import os
import pathlib

import numpy as np
import pandas as pd

from mousenet.model.data.connection_probability import Billeh19, Perin11

"""
Interface to mouse data sources.
"""


class Data:
    def __init__(self):
        self.p11 = Perin11()
        self.b19 = Billeh19()

    @staticmethod
    def get_areas():
        """
        :return: list of names of visual areas included in the model
        """
        return ["LGNd", "VISp", "VISl", "VISrl", "VISli", "VISpl", "VISal", "VISpor"]

    @staticmethod
    def get_layers():
        """
        :return: list of cortical layers included in model
        """
        return ["2/3", "4", "5"]

    @staticmethod
    def get_hierarchical_level(area):
        """
        :param area: Name of visual area
        :return: Hierarchical level number, from 0 (LGN) to 3 (VISpor) from Stefan's
            analysis
        """
        hierarchy = {
            "LGNd": 0,
            "VISp": 1,
            "VISl": 2,
            "VISrl": 2,
            "VISli": 2,
            "VISpl": 2,
            "VISal": 2,
            "VISpor": 3,
        }
        return hierarchy[area]

    @staticmethod
    def get_num_neurons(area, layer):
        """
        :param area: visual area name (e.g. 'VISp')
        :param layer: layer name (e.g. '2/3')
        :return: estimate of number of excitatory neurons in given area/layer
        """
        numbers = {
            "LGNd": 21200,
            "VISp2/3": 173253,
            "VISl2/3": 22299,
            "VISrl2/3": 22598,
            "VISli2/3": 9587,
            "VISpl2/3": 17924,
            "VISal2/3": 15760,
            "VISpor2/3": 30576,
            "VISp4": 108623,
            "VISl4": 15501,
            "VISrl4": 14360,
            "VISli4": 5620,
            "VISpl4": 3912,
            "VISal4": 9705,
            "VISpor4": 5952,
            "VISp5": 134530,
            "VISl5": 20826,
            "VISrl5": 19173,
            "VISli5": 11611,
            "VISpl5": 20041,
            "VISal5": 15939,
            "VISpor5": 30230,
        }
        if area == "LGNd":
            region = area
        else:
            region = "%s%s" % (area, layer)
        return numbers[region]

    def get_extrinsic_in_degree(self, target_area, target_layer):
        """
        :param target_area: visual area name (e.g. 'VISp')
        :param target_layer: layer name (e.g. '2/3')
        :return: estimate of mean number of neurons from OTHER AREAS that synapse onto a
            single excitatory neuron in given area / layer
        """
        return 1000  # TODO: replace with real estimate

    def get_hit_rate_peak(self, source_layer, target_layer):
        """
        :param source_layer: name of presynaptic layer
        :param target_layer: name of postsynaptic layer
        :return: fraction of excitatory neuron pairs with functional connection in this
            direction, at zero horizontal offset
        """
        hit_rate = self.b19.get_connection_probability(source_layer, target_layer)
        fraction_of_peak = np.exp(
            -(75**2) / 2 / self.get_hit_rate_width(source_layer, target_layer) ** 2
        )
        return hit_rate / fraction_of_peak

    @staticmethod
    def get_hit_rate_width(source_layer, target_layer):
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
        cat = {  # source -> target
            "2/3": {"2/3": 225, "4": 50, "5": 100, "6": 50},
            "4": {"2/3": 220, "4": 180, "5": 140, "6": 110},
            "5": {"2/3": 150, "4": 100, "5": 210, "6": 125},
            "6": {"2/3": 120, "4": 20, "5": 150, "6": 150},
        }

        return cat[source_layer][target_layer] / cat["4"]["4"] * l4_to_l4

    @staticmethod
    def get_visual_field_shape(area):
        """
        :param area: visual area name
        :return: (height, width) of visual field for that area, relative to a reference 64x64
        input video representing a width 90 and height 60 degrees of visual field.
        """
        # We return a constant for simplicity. This is based on the range of the scale
        # bars in Figure 9C,D of ﻿J. Zhuang et al., “An extended retinotopic map of mouse cortex,”
        # Elife, p. e18372, 2017. In fact different areas have different visual field shapes and
        # offsets, but we defer this aspect to future models.
        area = "".join([i for i in area if not i.isdigit() and i != "/"])
        area = area.lower()
        if area == "lgnd":
            return 0, 0, 64, 64
        project_root = pathlib.Path(__file__).parent.parent.resolve()
        df = pd.read_csv(os.path.join(project_root, "..", "config", "retinotopics", "retinomap.csv"))
        df["name"] = df["name"].apply(lambda x: x.lower())
        return [
            int(x)
            for x in df[df["name"] == area][["x1", "y1", "x2", "y2"]].values.tolist()[0]
        ]

    @staticmethod
    def get_visual_field(area):
        """
        Rectangular approximation of visual field of an area, based on Figure 8 of Zhuang et al.,
        “An extended retinotopic map of mouse cortex,” Elife, p. e18372, 2017. This area is the
        extent of RF centers.

        :param area: visual area name
        :return: [min_azimuth, max_azimuth, min_altitude, max_altitude]
        """
        # TODO: these are approximate numbers
        # TODO: changing these so they all overlap VISpor -- revisit this
        fields = {
            "VISp": [0, 85, -25, 35],
            "VISrl": [7, 53, -23, 12],
            "VISal": [17, 46, -5, 15],
            "VISl": [3, 46, -5, 28],
            "VISli": [26, 46, -5, 30],
            "VISpl": [
                26,
                85,
                -5,
                32,
            ],  # note this doesn't overlap VISpor at all without modification
            "VISpor": [26, 46, -5, 4],
            "VISam": [26, 60, -21, 5],
            "VISpm": [26, 82, -10, 6],
        }

        if area in fields:
            return fields[area]
        else:
            raise Exception("Unknown visual field for area" + area)
