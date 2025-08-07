import torch
import torch.nn as nn
import torch.nn.functional as F
from whisker.cnn.architecture_whisker import *
from whisker.cnn.anatomy_whisker import *
import networkx as nx
    
class ConvParam:
    def __init__(self, in_channels, out_channels, gsh, gsw, kernel_size, padding, stride=1):
        """
        A container for all convolution parameters.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param gsh: Gaussian peak height
        :param gsw: Gaussian spread width (σ)
        :param kernel_size: kernel size (from anatomical calculation)
        :param padding: padding (usually kernel_size // 2)
        :param stride: convolution stride (default 1)
        """
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gsh = gsh
        self.gsw = gsw
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
class ConvLayer:
    def __init__(self, source_name, target_name, params, out_size):
        """
        :param params: ConvParam containing the parameters of the layer
        :param source_name: name of the source area, e.g. VISp4, VISp2/3, VISp5
        :param target_name: name of the target area
        :param out_size: output size of the layer
        """
        self.params = params
        self.source_name = source_name
        self.target_name = target_name
        self.out_size = out_size
    
class TemporalBarrelColumn(nn.Module):
    def __init__(self, in_dim=4, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

    def forward(self, x):    # x: (batch, 15, 4)
        x = self.encoder(x)  # (batch, 15, embed_dim)
        return x.mean(dim=1)  # (batch, embed_dim)
    
def index_whisker_grid(whisker_names):
    grid = [
        ["A0", "A1", "A2", "A3", "A4", None, None],
        ["B0", "B1", "B2", "B3", "B4", None, None],
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        ["D0", "D1", "D2", "D3", "D4", "D5", "D6"],
        [None, "E1", "E2", "E3", "E4", "E5", "E6"]
    ]
    whisker_to_index = {w: i for i, w in enumerate(whisker_names)}
    grid_indices = [[whisker_to_index.get(w) if w else None for w in row] for row in grid]
    return grid_indices


def gridify(x, grid_indices):  # x: (B, 30, D)
    B, _, D = x.shape
    H = len(grid_indices)
    W = len(grid_indices[0])
    grid = torch.zeros(B, D, H, W, device=x.device)
    for i, row in enumerate(grid_indices):
        for j, idx in enumerate(row):
            if idx is not None:
                grid[:, :, i, j] = x[:, idx, :]
    return grid

def make_gaussian_mask(peak, sigma, size):
    assert size % 2 == 1, "Kernel size must be odd"
    radius = size // 2
    x = torch.arange(-radius, radius + 1)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    probs = peak * torch.exp(-R**2 / (2 * sigma**2))
    mask = (torch.rand_like(probs) < probs).float()
    return mask  # shape: (size, size)

class WhiskerEncoder(nn.Module):
    def __init__(self, arch, num_whiskers=60, in_dim=4, output_dim=128):
        super().__init__()

        embed_dim = arch.get_channels('SSp-bfd', '4')

        self.whisker_names = [
            "A1", "A2", "A3", "A4", "A5",
            "B1", "B2", "B3", "B4", "B5",
            "C1", "C2", "C3", "C4", "C5", "C6", "C7",
            "D1", "D2", "D3", "D4", "D5", "D6", "D7",
            "E2", "E3", "E4", "E5", "E6", "E7"
        ]

        self.barrels = nn.ModuleList([
            TemporalBarrelColumn(in_dim, embed_dim) for _ in range(num_whiskers)
        ])

        self.grid_indices = index_whisker_grid(self.whisker_names)

        anet = gen_anatomy(arch)  
        self.construct_from_anatomy(anet, arch)

        self.output_fc = nn.Linear(self.area_channels['VISrl5'], output_dim)

    def construct_from_anatomy(self, anet, architecture):
        self.layers = []
        self.convs = nn.ModuleDict()
        self.area_size = {}
        self.area_channels = {}

        G, _ = anet.make_graph()
        Gtop = list(nx.topological_sort(G))
        root = Gtop[0]  # starting node

        for i, e in enumerate(nx.edge_bfs(G, root)):
            in_layer = e[0]
            out_layer = e[1]
            in_name = in_layer.area + in_layer.depth
            out_name = out_layer.area + out_layer.depth

            if out_name == 'SSp-bfd4':
                continue  # this is defined via TemporalBarrelColumn

            # First layer: use manual defaults
            if in_name not in self.area_size:
                in_size = 5  # grid resolution 
                in_channels = architecture.get_channels(in_layer.area, in_layer.depth)
                self.area_size[in_name] = in_size
                self.area_channels[in_name] = in_channels
            else:
                in_size = self.area_size[in_name]
                in_channels = self.area_channels[in_name]

            # Compute output size
            out_sigma = 1  # no spatial downsampling
            out_size = int(in_size * out_sigma)
            out_neurons = out_layer.num
            out_channels = max(1, int(out_neurons / (out_size ** 2)))

            architecture.neuron_counts[out_layer.area][out_layer.depth]
            self.area_size[out_name] = out_size
            self.area_channels[out_name] = out_channels

            # Compute anatomical kernel parameters
            gsw = architecture.get_kernel_width_pixels(in_layer.area, in_layer.depth, out_layer.area, out_layer.depth)
            gsh = architecture.get_kernel_peak_probability(in_layer.area, in_layer.depth, out_layer.area, out_layer.depth)
            ksize = architecture.get_kernel_size(in_layer.area, in_layer.depth, out_layer.area, out_layer.depth)
            pad = architecture.get_padding(in_layer.area, in_layer.depth, out_layer.area, out_layer.depth)

            print(f'gsw: {gsw}, gsh: {gsh}, ksize: {ksize}, pad: {pad}')

            param_count = in_channels * out_channels * (ksize ** 2)
            print(f"[{i}] {in_name} → {out_name}: {in_channels} → {out_channels} ({param_count / 1e6:.2f}M params)")

            # Store ConvLayer
            conv_param = ConvParam(in_channels, out_channels, gsh, gsw, ksize, pad)
            conv_layer = ConvLayer(in_name, out_name, conv_param, out_size)
            self.layers.append(conv_layer)

            self.convs[f"{in_name}_{out_name}"] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ksize,
                padding=pad
            )

    
    def forward(self, whisker_input):  # (B, 60, 15, 4)
        B = whisker_input.shape[0]
        barrel_outputs = [barrel(whisker_input[:, i]) for i, barrel in enumerate(self.barrels)]
        x = torch.stack(barrel_outputs, dim=1)  # (B, 60, embed_dim)

        left_embed = x[:, :30, :]
        right_embed = x[:, 30:, :]

        left_grid = gridify(left_embed, self.grid_indices)   # (B, D, H, W)
        right_grid = gridify(right_embed, self.grid_indices) # (B, D, H, W)
    
        x = left_grid + right_grid

        # print(f"Initial grid shape (after barrel + gridify): {x.shape}")  # e.g. (B, C, 5, 7)

        outputs = { 'SSp-bfd4': x }

        for layer in self.layers:
            name = f"{layer.source_name}_{layer.target_name}"
            src = outputs[layer.source_name]
            conv = self.convs[name]
            out = conv(src)
            # print(f"{name}: {src.shape} → {out.shape}")
            if layer.target_name in outputs:
                outputs[layer.target_name] = outputs[layer.target_name] + out 
            else:
                outputs[layer.target_name] = out

        # Final output from VISrl5 → spatial avg pool → FC
        x = outputs['VISrl5']  # (B, C, H, W)
        # print(f"Final VISrl5 shape: {x.shape}")
        x = torch.mean(x.view(B, x.size(1), -1), dim=2)  # (B, C)
        x = self.output_fc(x)  # (B, output_dim)

        return x
    


    
