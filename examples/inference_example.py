import os
import random
import sys
import time

import numpy as np
import torch

print("Current Directory:", os.getcwd())
sys.path.append("../../bmtk")
sys.path.append("../mousenet/")

from mousenet.config.config import INPUT_SIZE
from mousenet.constructing_model import gen_network
from mousenet.model.Architecture import Architecture
from mousenet.model.MouseNetCompletePool import MouseNetCompletePool


def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)  # Python random seed
    torch.manual_seed(seed)  # PyTorch random seed
    np.random.seed(seed)  # NumPy random seed
    torch.backends.cudnn.deterministic = True


def initialize_network(input_size, regenerate=False):
    """Initialize the neural network."""
    net_name = "network_(%s,%s,%s)" % (input_size[0], input_size[1], input_size[2])
    architecture = Architecture()
    net = gen_network(net_name, architecture, regenerate)
    return net


def initialize_model(network):
    mousenet = MouseNetCompletePool(
        network  # , mask=MASK
    )
    return mousenet


def mousenet_inference(model, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        assert INPUT_SIZE[2] >= 51
        assert INPUT_SIZE[3] >= 81
        input_tensor = torch.rand((2, *INPUT_SIZE))

        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
    return output


def setup_device(mousenet, device):
    """Move the model to the specified device."""
    mousenet.to(device)


if __name__ == "__main__":
    # Constants and configurations
    SEED = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seeds
    start_time = time.time()
    set_random_seeds(SEED)
    print(
        f"Constants and configurations set up, t: {time.time() - start_time:.2f} seconds"
    )

    # Initialize network
    start_time = time.time()
    network = initialize_network(INPUT_SIZE, regenerate=False)
    print(f"Network initialized, t: {time.time() - start_time:.2f} seconds")

    # Initialize model
    start_time = time.time()
    mousenet = initialize_model(network)
    print(f"Model initialized, t: {time.time() - start_time:.2f} seconds")

    # Setup device
    start_time = time.time()
    setup_device(mousenet, DEVICE)
    print(f"Setup complete, t: {time.time() - start_time:.2f} seconds")

    # Inference
    start_time = time.time()
    output = mousenet_inference(mousenet, DEVICE)
    print(f"Inference complete, t: {time.time() - start_time:.2f} seconds")
    print("Output shape: ", output.shape)
