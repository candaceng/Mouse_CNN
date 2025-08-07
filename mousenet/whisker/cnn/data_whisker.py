# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
from scipy.optimize import curve_fit
import pathlib
import pdb
import pandas as pd
"""
Interface to mouse data sources.
"""

class Data:
    def __init__(self):
        pass

    def get_areas(self):
        """
        :return: list of names of visual areas included in the model
        """
        return ['SSp-bfd', 'SSs', 'VISrl']
    
    def get_layers(self):
        """
        :return: list of cortical layers included in model
        """
        return ['2/3', '4', '5']

    def get_hierarchical_level(self, area):
        """
        :param area: Name of visual area
        :return: Hierarchical level number, from 0 (LGN) to 3 (VISpor) from Stefan's
            analysis
        """
        return {'SSpbfd': 1, 'SSs': 2, 'VISrl': 3}[area]

    def get_num_neurons(self, area, layer):
        """
        :param area: visual area name (e.g. 'VISp')
        :param layer: layer name (e.g. '2/3')
        :return: estimate of number of excitatory neurons in given area/layer
        """
        numbers = {
            'SSp-bfd': {'2/3': 175481, '4': 135943, '5': 117751},
            'SSs': {'2/3': 95958, '4': 48397, '5': 65758},
            'VISrl': {'2/3': 11831, '4': 4927, '5': 8161}  # estimated from (density x SA) ratio with VISl
        }
        return numbers[area][layer]

