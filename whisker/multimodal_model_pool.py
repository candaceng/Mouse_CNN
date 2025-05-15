import torch
import torch.nn as nn
# from mousenet.whisker.whisker_encoder import WhiskerEncoder
from mousenet.whisker.whisker_encoder import TemporalWhiskerEncoder
from mousenet.cmouse.mousenet_complete_pool import MouseNetCompletePool
import numpy as np

class MultimodalMouseModelPool(nn.Module):
    def __init__(self, visual_net, whisker_input_dim=4, embed_dim=128):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.2), dtype=torch.float32)) 
        self.visual_net = visual_net
        # self.whisker_encoder = WhiskerEncoder(input_dim=whisker_input_dim, embed_dim=embed_dim)
        self.whisker_encoder = TemporalWhiskerEncoder(input_dim=4, embed_dim=128)

        # Use dummy input to infer flattened visual feature size
        dummy_input = torch.zeros(1, 1, 64, 64)
        visual_feat = self.visual_net.get_img_feature(
            dummy_input, 
            ['VISp5', 'VISl5', 'VISrl5', 'VISli5', 'VISpl5', 'VISal5', 'VISpor5'],
            flatten=False
        )
        visual_feat = visual_feat.view(visual_feat.size(0), -1)
        self.visual_fc = nn.Linear(visual_feat.shape[1], embed_dim)

    def forward(self, image, whisker):
        visual_feat = self.visual_net.get_img_feature(image, ['VISp5', 'VISl5', 'VISrl5', 'VISli5', 'VISpl5', 'VISal5', 'VISpor5'], flatten=False)
        visual_feat = visual_feat.view(visual_feat.size(0), -1)  # Flatten to [B, 4384]
        visual_embed = self.visual_fc(visual_feat)
        whisker_embed = self.whisker_encoder(whisker)
        
        return visual_embed, whisker_embed
    
    @property
    def temperature(self):
        return torch.clamp(self.log_temp.exp(), min=0.05, max=0.3)

    
