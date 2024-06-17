import torch.nn as nn
from mousenet.model.lgn_model.LGNConv3DLayer import LGNConv3DLayer
import torch.nn.functional as F


class LGNModel(nn.Module):
    def __init__(self, in_channels, kernel_size, path_neurons_per_filter_yaml, path_param_filer_lgn_file):
        super(LGNModel, self).__init__()

        self.lgn_layer = LGNConv3DLayer(in_channels, kernel_size, path_neurons_per_filter_yaml,
                                        path_param_filer_lgn_file)

        out_channels = self.lgn_layer.get_num_out_channels()

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.conv2d_1 = nn.Conv2d(out_channels, 64, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 22 * 22, 2)  # Adjust dimensions as necessary

    def get_num_out_channels(self):
        return self.lgn_layer.get_num_out_channels()

    def forward(self, x):
        x = self.lgn_layer(x)
        # assume, kernel_size = (19, 21, 21)
        # output shape (B, out_channels, 2, 44, 44)
        x = self.pool(x)
        #  output shape (B, out_channels, 1, 22, 22)
        B, C, D, H, W = x.shape
        x = x.view(B, C * D, H, W)  # reshape for 2D convolutions
        #  output shape (B, out_channels*1, 22, 22)
        x = self.conv2d_1(x)
        x = F.relu(x)
        # output shape (B, 64, 22, 22)
        x = self.conv2d_2(x)
        x = F.relu(x)
        # output shape (B, 128, 22, 22)
        x = x.view(B, -1)  # Flatten
        # output shape (B, 128 * 22 * 22)
        x = self.fc(x)
        # output shape (B, 2)
        return x