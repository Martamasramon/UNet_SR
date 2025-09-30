# pythae_adapter.py
import torch
from torch.utils.data import Dataset
from pythae.models.base.base_utils import ModelOutput

class PythaeDataset(Dataset):
    """
    Wraps your MyDataset and exposes {'data': tensor(C,H,W)} for Pythae.
    key='highres' (default) trains the AE to reconstruct HR images.
    """
    def __init__(self, base_dataset, key: str = "lowres"):
        assert key in ("highres", "lowres")
        self.base = base_dataset
        self.key = key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]  # your dict: {'lowres', 'highres', optional 'T2W', ...}
        x      = sample[self.key]
        if isinstance(x, torch.Tensor):
            data = x
        else:
            data = torch.as_tensor(x)
        data = data.float()  # ensure float32
        # Must be (C,H,W); your transforms should already give that
        assert data.dim() == 3, f"Expected (C,H,W), got {tuple(data.shape)}"
        return ModelOutput(data=data)
