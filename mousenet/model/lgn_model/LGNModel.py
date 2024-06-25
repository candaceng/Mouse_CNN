import torch.nn as nn
import torch.nn.functional as F

from mousenet.config.config import DEBUG
from mousenet.model.lgn_model.LGNConv3DLayer import LGNConv3DLayer


class LGNModel(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        depth,
        path_neurons_per_filter_yaml,
        path_param_filer_lgn_file,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
    ):
        """
        depth: of the input image of size (depth, width, height)
        """

        super(LGNModel, self).__init__()

        self.lgn_layer = LGNConv3DLayer(
            in_channels,
            kernel_size,
            path_neurons_per_filter_yaml,
            path_param_filer_lgn_file,
            stride=stride,
            padding=padding,
        )

        expected_d = (depth + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        out_channels = self.lgn_layer.get_num_out_channels()

        in_channels = int(out_channels * expected_d)

        self.conv2d_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(64, 5, kernel_size=3, padding=1)

    def get_num_out_channels(self):
        return self.lgn_layer.get_num_out_channels()

    def forward(self, x):
        if DEBUG:
            i = 1
            print(f"\n{i}. shape: {x.shape}")

        x = self.lgn_layer(x)
        if DEBUG:
            i += 1
            print(f"{i}. shape: {x.shape}")

        B, C, D, H, W = x.shape
        x = x.view(B, C * D, H, W)  # reshape for 2D convolutions
        if DEBUG:
            i += 1
            print(f"{i}. shape: {x.shape}")

        x = self.conv2d_1(x)
        x = F.relu(x)
        if DEBUG:
            i += 1
            print(f"{i}. shape: {x.shape}")

        x = self.conv2d_2(x)
        x = F.relu(x)
        if DEBUG:
            i += 1
            print(f"{i}. shape: {x.shape}")

        return x
