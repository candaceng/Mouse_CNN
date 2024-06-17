import sys
sys.path.append("../../bmtk")

import torch

from mousenet.model.lgn_model.LGNConv3DLayer import LGNConv3DLayer
from mousenet.model.lgn_model.LGNModel import LGNModel

# Constants
PARAM_FILTER_LGN_FILE = "../mousenet/config/data_lgn/lgn_filters_mean.csv"
NUM_OF_NEURONS_PER_FILTER_LGN_FILE = (
    "../mousenet/config/data_lgn/num_neurons_per_filter_debug.yaml"
)
KERNEL_SIZE = (19, 21, 21)
DEBUG = False


def create_input_tensor(device: str) -> torch.Tensor:
    return torch.rand((2, 3, 20, 64, 64)).to(device)


# Test functions
def test_lgn_layer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = create_input_tensor(device)
    num_channels = x.shape[1]
    model = LGNConv3DLayer(
        in_channels=num_channels,
        kernel_size=KERNEL_SIZE,
        path_neurons_per_filter_yaml=NUM_OF_NEURONS_PER_FILTER_LGN_FILE,
        path_param_filer_lgn_file=PARAM_FILTER_LGN_FILE,
    ).to(device)

    out_channels = model.get_num_out_channels()

    output = model(x).detach().cpu().numpy()
    assert output.shape == (2, out_channels, 2, 44, 44), "Output shape mismatch"
    print("test_lgn_layer passed")


def test_lgn_full_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = create_input_tensor(device)
    num_channels = x.shape[1]

    model = LGNModel(
        num_channels,
        KERNEL_SIZE,
        NUM_OF_NEURONS_PER_FILTER_LGN_FILE,
        PARAM_FILTER_LGN_FILE,
    ).to(device)

    output = model(x).detach().cpu().numpy()
    assert output.shape == (2, 2), "Output shape mismatch"
    print("test_lgn_model passed")


def test_lgn_model_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = create_input_tensor(device)
    num_channels = x.shape[1]

    model = LGNConv3DLayer(
        in_channels=num_channels,
        kernel_size=KERNEL_SIZE,
        path_neurons_per_filter_yaml=NUM_OF_NEURONS_PER_FILTER_LGN_FILE,
        path_param_filer_lgn_file=PARAM_FILTER_LGN_FILE,
    ).to(device)

    output = model(x)
    assert output.device.type == device, "Device type mismatch"
    print("test_lgn_model_device passed")


if __name__ == "__main__":
    test_lgn_layer()
    test_lgn_full_model()
    test_lgn_model_device()
