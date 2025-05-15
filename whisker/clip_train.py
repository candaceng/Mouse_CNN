import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

# === Example Dataset ===
class WhiskerVisionDataset(Dataset):
    def __init__(self, image_tensor_file, whisker_npz_file):
        self.images = torch.load(image_tensor_file)  # shape: [N, 2, 64, 64]
        self.whiskers = torch.from_numpy(np.load(whisker_npz_file)['arr_0'])  # shape: [N, 60, 4]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.whiskers[idx]


# === CLIP-style Loss ===
def clip_loss(vision_embed, whisker_embed, temperature):
    vision_embed = F.normalize(vision_embed, dim=1)
    whisker_embed = F.normalize(whisker_embed, dim=1)

    logits = torch.matmul(vision_embed, whisker_embed.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)

    loss_v2w = F.cross_entropy(logits, labels)
    loss_w2v = F.cross_entropy(logits.T, labels)
    return (loss_v2w + loss_w2v) / 2


# === Training Loop ===
def train_clip_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, whiskers in dataloader:
        images, whiskers = images.to(device), whiskers.to(device)
        optimizer.zero_grad()

        z_v, z_w = model(images, whiskers)
        loss = clip_loss(z_v, z_w)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Avg Loss: {avg_loss:.4f}")


# === Usage Example ===
if __name__ == '__main__':
    import numpy as np
    from whisker.multimodal_model_pool import MultimodalMouseModelPool
    from mousenet_complete_pool import MouseNetCompletePool

    # Load prebuilt MouseNet model
    from some_module import network  # replace with actual path
    mousenet = MouseNetCompletePool(network)
    model = MultimodalMouseModelPool(visual_net=mousenet).cuda()

    # Dataset and loader
    dataset = WhiskerVisionDataset('image_tensor.pt', 'whisker_data.npz')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(10):
        print(f"Epoch {epoch+1}")
        train(model, dataloader, optimizer, device='cuda')
