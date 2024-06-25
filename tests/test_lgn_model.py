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
KERNEL_SIZE = (3, 3, 3)
STRIDE = (2, 2, 1)
PADDING = (1, 2, 1)
DEBUG = False


def create_input_tensor(device: str) -> torch.Tensor:
    return torch.rand((2, 3, 21, 53, 81)).to(device)


# Test functions
def test_lgn_layer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = create_input_tensor(device)
    b, c, d, h, w = x.shape

    num_channels = x.shape[1]
    kernel_size = KERNEL_SIZE
    stride = STRIDE
    padding = PADDING

    model = LGNConv3DLayer(
        in_channels=num_channels,
        kernel_size=KERNEL_SIZE,
        path_neurons_per_filter_yaml=NUM_OF_NEURONS_PER_FILTER_LGN_FILE,
        path_param_filer_lgn_file=PARAM_FILTER_LGN_FILE,
        stride=stride,
        padding=padding
    ).to(device)

    out_channels = model.get_num_out_channels()

    output = model(x).detach().cpu().numpy()

    expected_output_shape = (
        b,
        out_channels,
        (d + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,
        (h + 2 * padding[1] - kernel_size[1]) // stride[1] + 1,
        (w + 2 * padding[2] - kernel_size[2]) // stride[2] + 1,
    )

    assert output.shape == expected_output_shape, "Output shape mismatch"
    print("test_lgn_layer passed")


def test_lgn_full_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = create_input_tensor(device)
    b, c, d, h, w = x.shape
    num_channels = x.shape[1]
    stride = STRIDE
    padding = PADDING

    model = LGNModel(
        num_channels,
        KERNEL_SIZE,
        x.shape[2],
        NUM_OF_NEURONS_PER_FILTER_LGN_FILE,
        PARAM_FILTER_LGN_FILE,
        stride=stride,
        padding=padding
    ).to(device)

    output = model(x).detach().cpu().numpy()

    expected_h = (h + 2 * padding[1] - KERNEL_SIZE[1]) // stride[1] + 1
    expected_w = (w + 2 * padding[2] - KERNEL_SIZE[2]) // stride[2] + 1
    expected_shape = (b, 5, expected_h, expected_w)

    assert output.shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
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
