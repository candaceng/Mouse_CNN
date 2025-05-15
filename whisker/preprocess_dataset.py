import torch
from pathlib import Path

class PreprocessedTrialDataset(torch.utils.data.Dataset):
    def __init__(self, pt_folder):
        self.pt_files = sorted(Path(pt_folder).glob("trial_*.pt"))

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        return torch.load(self.pt_files[idx])