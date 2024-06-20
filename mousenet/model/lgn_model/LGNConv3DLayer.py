import os

import pandas as pd
import torch
import torch.nn as nn
import yaml

from mousenet.config.config import DEBUG
from mousenet.model.lgn_model.FilterParams import FilterParams
from mousenet.model.lgn_model.LGNConv3D import LGNConv3D


class LGNConv3DLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        path_neurons_per_filter_yaml,
        path_param_filer_lgn_file,
    ):
        super().__init__()
        self.kernel_size = kernel_size
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
                in_channels, num_cells, self.kernel_size, params_for_filter_type
            )
        return conv_layers

    def forward(self, input: torch.Tensor):
        B, C, D, W, H = (
            input.shape
        )  # batch size, number of channels, depth, width, height

        output = torch.empty(
            (
                B,
                self.num_channels,
                D - self.kernel_size[0] + 1,
                W - self.kernel_size[1] + 1,
                H - self.kernel_size[2] + 1,
            ),
            device=input.device,
        )  # TODO: change for more complex param
        channel_index = 0

        # if DEBUG:
        #     print(
        #         f"B: {B}, C: {C}, D: {D}, W: {W}, H: {H}"
        #         f"\nnum_channels: {self.num_channels}"
        #         f"\noutput.shape: {output.shape}\n"
        #     )

        for filter_type, num_cells in self.neurons_per_filter.items():
            conv_output = self.convs[filter_type](input)
            # if DEBUG:
            #     print(f"conv_output.shape: {conv_output.shape}")
            #     print(
            #         f"output expected input shape: {output[:, channel_index:channel_index + num_cells].shape}"
            #     )
            output[:, channel_index : channel_index + num_cells] = conv_output
            channel_index += num_cells

        return self.activation(output)
