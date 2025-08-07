from pathlib import Path
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from whisker.cnn.whisker_encoder import *

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

        dummy_input = torch.zeros(1, 1, 64, 64)
        visual_feat = self.visual_net.get_img_feature(dummy_input, ['VISrl5'],flatten=False).view(1, -1)
        self.visual_fc = nn.Linear(visual_feat.shape[1] * 2, embed_dim)  # ×2 for L + R
    
    def forward(self, image_left, image_right, whisker):
        # Visual VISrl embeddings
        v_feat_left = self.visual_net.get_img_feature(image_left, ['VISrl5'], flatten=False).view(image_left.size(0), -1)
        v_feat_right = self.visual_net.get_img_feature(image_right, ['VISrl5'], flatten=False).view(image_right.size(0), -1)

        v_combined = torch.cat([v_feat_left, v_feat_right], dim=1)
        v_embed = self.visual_fc(v_combined)

        # Whisker embedding
        w_embed = self.whisker_encoder(whisker)

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