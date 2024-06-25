from typing import Tuple
import os

import pandas as pd
import torch
import torch.nn as nn
import yaml

from mousenet.model.lgn_model.FilterParams import FilterParams
from mousenet.model.lgn_model.LGNConv3D import LGNConv3D


class LGNConv3DLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        path_neurons_per_filter_yaml,
        path_param_filer_lgn_file,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.neurons_per_filter = self._load_neurons_per_filter(
            path_neurons_per_filter_yaml
        )
        self.param_table = self._load_filter_parameters(path_param_filer_lgn_file)
        self.num_channels = sum(self.neurons_per_filter.values())
        self.convs = self._create_convolutional_layers(in_channels)
        self.activation = nn.ReLU()

    @staticmethod
    def _load_neurons_per_filter(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} was not found.")
        with open(filepath, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    @staticmethod
    def _load_filter_parameters(filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} was not found.")
        param_table = pd.read_csv(filepath)
        return param_table

    def get_num_out_channels(self):
        return self.num_channels

    def _get_parameters_for_filter_type(self, filter_type: str) -> FilterParams:
        params = self.param_table[self.param_table["model_id"] == filter_type].iloc[0]
        params = FilterParams(params)
        return params

    def _create_convolutional_layers(self, in_channels: int):
        conv_layers = nn.ModuleDict()
        for filter_type, num_cells in self.neurons_per_filter.items():
            params_for_filter_type = self._get_parameters_for_filter_type(filter_type)
            conv_layers[filter_type] = LGNConv3D(
                in_channels,
                num_cells,
                self.kernel_size,
                params_for_filter_type,
                stride=self.stride,
                padding=self.padding,
            )
        return conv_layers

    def forward(self, input: torch.Tensor):
        B, C, D, W, H = (
            input.shape
        )  # batch size, number of channels, depth, width, height

        output_shape = (
            B,
            self.num_channels,
            (D + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1,
            (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1,
            (H + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1,
        )
        output = torch.empty(output_shape, device=input.device)

        channel_index = 0

        for filter_type, num_cells in self.neurons_per_filter.items():
            conv_output = self.convs[filter_type](input)
            output[:, channel_index : channel_index + num_cells] = conv_output
            channel_index += num_cells

        return self.activation(output)
