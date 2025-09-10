from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from whisker.cnn.whisker_encoder import *
from contextlib import contextmanager

class PreprocessedTrialDataset(Dataset):
    def __init__(self, pt_folder):
        self.pt_files = sorted(Path(pt_folder).glob("trial_*.pt"))

    def __len__(self):
        return len(self.pt_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        image_left = data["image_left"]    # [1, 64, 64]
        image_right = data["image_right"]  # [1, 64, 64]
        whisker = torch.cat([data["whisker_L"], data["whisker_R"]], dim=0)  # [batch size, 60, 15, 4]
        return image_left, image_right, whisker
    
class WhiskerToVISRLFusion(nn.Module):
    """
    Make a spatial (H×W) whisker map and fuse it into VISrl with a 1×1 conv.
    Optional 'ports' let routing to coarse subregions (e.g., quadrants) to
    emulate 'projecting to different parts of VISrl'.
    """
    def __init__(self, whisker_dim: int, visrl_channels: int, H: int, W: int,
                 map_channels: int = None, num_ports: int = 0):
        super().__init__()
        self.H, self.W = H, W
        self.visrl_channels = visrl_channels
        self.map_channels = map_channels or max(8, visrl_channels // 2)
        self.num_ports = num_ports

        # Whisker embedding -> spatial map (B, C_map, H, W)
        self.z_to_map = nn.Sequential(
            nn.Linear(whisker_dim, 2 * self.map_channels),
            nn.GELU(),
            nn.Linear(2 * self.map_channels, self.map_channels * H * W)
        )

        # Optional mixture-of-ports (e.g., 4 quadrants) to localize injection
        if num_ports > 0:
            self.port_logits = nn.Linear(whisker_dim, num_ports)  # α = softmax(logits)
            self.register_buffer("port_masks", self._make_port_masks(num_ports, H, W))  # (K, 1, H, W)

        # Mix [VISrl, whisk_map] back to VISrl channels
        self.fuse = nn.Conv2d(visrl_channels + self.map_channels, visrl_channels, kernel_size=1)

    @staticmethod
    def _make_port_masks(K: int, H: int, W: int):
        # Simple 4-quadrant example; extend if K != 4
        masks = []
        if K == 4:
            halves = [(slice(0, H//2), slice(0, W//2)),
                      (slice(0, H//2), slice(W//2, W)),
                      (slice(H//2, H), slice(0, W//2)),
                      (slice(H//2, H), slice(W//2, W))]
            for hsl, wsl in halves:
                m = torch.zeros(1, H, W)
                m[:, hsl, wsl] = 1.0
                masks.append(m)
        else:
            # Fallback: uniform masks
            for _ in range(K):
                masks.append(torch.ones(1, H, W))
        return torch.stack(masks, dim=0)  # (K, 1, H, W)

    def forward(self, visrl: torch.Tensor, z_w: torch.Tensor):
        """
        visrl: (B, C, H, W)  feature map from VISrl
        z_w:   (B, D)        whisker embedding
        """
        B, C, H, W = visrl.shape

        # temperature (optional, improves selectivity)
        tau = getattr(self, "temperature", 1.0)
        alpha = None
        # 1) whisker → spatial map
        wmap = self.z_to_map(z_w).view(B, self.map_channels, self.H, self.W)

        # 2) ports
        if self.num_ports > 0:
            tau = getattr(self, "temperature", 1.0)
            logits = self.port_logits(z_w)
            alpha  = torch.softmax(logits / tau, dim=-1)           # use τ here
            masks  = torch.einsum("bk,kchw->bchw", alpha, self.port_masks)
            wmap   = wmap * masks

        # 3) Fuse (concat + 1x1 conv) with residual add
        fused = self.fuse(torch.cat([visrl, wmap], dim=1))

        # logging
        with torch.no_grad():
            if alpha is not None:
                self.last_alpha     = alpha.detach().mean(dim=0).cpu().tolist()
                self.last_alpha_ent = (-(alpha * (alpha+1e-8).log())
                                        .sum(dim=-1).mean().item())
            else:
                self.last_alpha, self.last_alpha_ent = None, None

        return visrl + fused
    
class InhibitoryFiLM(nn.Module):
    """
    Channelwise FiLM gate with guaranteed negative scale (inhibition).
    y = (1 + gamma(z)) ⊙ x + beta(z), with gamma(z) <= 0
    """

    def __init__(self, whisker_dim: int, n_channels: int, max_supp: float = 0.25, use_beta: bool = False):
        super().__init__()
        self.fc_gamma = nn.Linear(whisker_dim, n_channels)
        self.use_beta = use_beta
        if use_beta:
            self.fc_beta = nn.Linear(whisker_dim, n_channels)
            nn.init.zeros_(self.fc_beta.weight); nn.init.zeros_(self.fc_beta.bias)

        # init near identity: zero weights so sigmoid(h) ~ 0.5 -> scale ~ 1 - 0.5*max_supp
        nn.init.zeros_(self.fc_gamma.weight); nn.init.zeros_(self.fc_gamma.bias)
        self.max_supp = max_supp  # e.g., 0.3–0.5 => scale in [0.5..1]
    
    def forward(self, z, fmap):  # z: (B,D), fmap: (B,C,H,W)
        B, C, H, W = fmap.shape
        h = self.fc_gamma(z)                        # (B,C)
        scale = 1.0 - self.max_supp * torch.sigmoid(h)   # (B,C) in (1-max_supp, 1)
        y = scale.view(B, C, 1, 1) * fmap

        if self.use_beta:
            beta = self.fc_beta(z).view(B, C, 1, 1)
            y = y + beta

        # logging
        with torch.no_grad():
            self.last_scale_mean = scale.mean().item()
            self.last_scale_min  = scale.min().item()
            self.last_scale_max  = scale.max().item()
            self.last_supp_ratio = (y.abs().mean() / (fmap.abs().mean() + 1e-8)).item()
        return y
    
        
class MultimodalMouseModel(nn.Module):
    def __init__(self, visual_net, embed_dim=128, learnable_temp=True, temp=0.2):
        super().__init__()
        self.learnable_temp = learnable_temp
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(np.log(temp), dtype=torch.float32))
        else:
            self.register_buffer("log_temp", torch.tensor(np.log(temp), dtype=torch.float32))
        
        self.visual_net = visual_net
        self.whisker_encoder = WhiskerEncoder(num_whiskers=60, in_dim=4, arch=WhiskerArchitecture(), output_dim=128)
        self.retinotopic = visual_net.network.retinotopic
        self._visual_in_ch = 2 if self.retinotopic else 1

        dev   = next(self.visual_net.parameters()).device
        dtype = next(self.visual_net.parameters()).dtype
        dummy = torch.zeros(1, self._visual_in_ch, 64, 64, device=dev, dtype=dtype) # used to probe VISrl shape 
        visrl_map = self.visual_net.get_img_feature(dummy, ['VISrl5'], flatten=False)
        C_rl, H_rl, W_rl = visrl_map.shape[1:]

        # NEW: spatial fusion module (whisker -> VISrl)
        # num_ports=4 for quadrants, prob have to change to account for retinotopic = True
        self.visrl_fusion = WhiskerToVISRLFusion(
            whisker_dim=128,
            visrl_channels=C_rl,
            H=H_rl,
            W=W_rl,
            map_channels=max(8, C_rl // 2),
            num_ports=4
        )

        visual_feat = visrl_map.view(1, -1)
        self.visual_fc = nn.Linear(visual_feat.shape[1] * 2, embed_dim)  # ×2 for L + R

        visp_map = self.visual_net.get_img_feature(dummy, ['VISp2/3'], flatten=False)
        print(f'visp map shape: {visp_map.shape}')
        C_visp = visp_map.shape[1]
        self.visp_gate = InhibitoryFiLM(whisker_dim=128, n_channels=C_visp)

        # helper: find the module whose name contains 'visp' and '5'
        def _find_module_by_substrings(root, includes):
            cands = [(n, m) for n, m in root.named_modules()
                    if all(s in n.lower() for s in includes)]
            if not cands:
                # fallback: try just 'visp'
                cands = [(n, m) for n, m in root.named_modules() if 'visp' in n.lower()]
            name, module = cands[0]
            return name, module

        self._visp_name, self._visp_module = _find_module_by_substrings(self.visual_net, includes=('visp4visp2/3',))
        print("Hooking:", self._visp_name) 

        self._gate_enabled = True
        self._current_z = None

        self._hook_dbg_n = 0
        self._hook_dbg_max = 3   # only print on first 3 calls
        def _visp_hook(_mod, _inp, out):
            # out: (B, C_visp, H, W)
            if (not self._gate_enabled) or (self._current_z is None):
                return out
            
            pre = out.detach().abs().mean().item()          # mean magnitude before gating
            gated = self.visp_gate(self._current_z, out)    # apply FiLM (inhibition)
            post = gated.detach().abs().mean().item()       # after gating

            if self._hook_dbg_n < self._hook_dbg_max:
                print(f"[VISp hook] out={tuple(out.shape)} pre={pre:.4f} post={post:.4f}")
                self._hook_dbg_n += 1

            return gated

        # register once
        self._visp_hook_handle = self._visp_module.register_forward_hook(lambda m, i, o: _visp_hook(m, i, o))

    def set_fusion_enabled(self, enabled: bool):
        self._fusion_enabled = enabled

    def get_debug_metrics(self):
        m = {
            "visp_scale_mean": getattr(self.visp_gate, "last_scale_mean", None),
            "visp_scale_min":  getattr(self.visp_gate, "last_scale_min",  None),
            "visp_scale_max":  getattr(self.visp_gate, "last_scale_max",  None),
            "visp_supp_ratio": getattr(self.visp_gate, "last_supp_ratio", None),
            "visrl_alpha":     getattr(self.visrl_fusion, "last_alpha",    None),
            "visrl_alpha_ent": getattr(self.visrl_fusion, "last_alpha_ent",None),
        }
        return m
    
    def _to_on_off(self, x):  # x: (B, 1 or 3, H, W)
        # If RGB, convert to gray first 
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        assert x.shape[1] == 1, "ON/OFF expects 1-ch input (grayscale)"
        mu  = x.mean(dim=(2,3), keepdim=True)  # global mean
        on  = torch.relu(x - mu)
        off = torch.relu(mu - x)
        return torch.cat([on, off], dim=1)     # (B, 2, H, W)
    
    def forward(self, image_left, image_right, whisker):
        # 0) make images match the required input channels if retinotopic
        if self.retinotopic:
            image_left  = self._to_on_off(image_left)
            image_right = self._to_on_off(image_right)

        # 1) whisker latent first (so the VISp hook can use it)
        w_embed = self.whisker_encoder(whisker)
        self._current_z = w_embed

        # 2) run vision (hook fires inside)
        v_map_left  = self.visual_net.get_img_feature(image_left,  ['VISrl5'], flatten=False)
        v_map_right = self.visual_net.get_img_feature(image_right, ['VISrl5'], flatten=False)

        # 3) fuse into VISrl
        v_map_left  = self.visrl_fusion(v_map_left,  w_embed)
        v_map_right = self.visrl_fusion(v_map_right, w_embed)

        # 4) flatten + project
        v_feat_left  = v_map_left.view(image_left.size(0),  -1)
        v_feat_right = v_map_right.view(image_right.size(0), -1)
        v_embed = self.visual_fc(torch.cat([v_feat_left, v_feat_right], dim=1))

        self._current_z = None
        return v_embed, w_embed

    @property
    def temperature(self):
        return torch.clamp(self.log_temp.exp(), min=0.05, max=0.3)

    
def clip_loss(vision_embed, whisker_embed, temperature):
    vision_embed = F.normalize(vision_embed, dim=1)
    whisker_embed = F.normalize(whisker_embed, dim=1)
    logits = torch.matmul(vision_embed, whisker_embed.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2