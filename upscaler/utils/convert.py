import numpy as np
import torch


def to_tensor(img):
    """
    Convert RGB uint8 image to tensor.
    """
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def to_rgb(rgb: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to RGB uint8 image.
    """
    rgb_img = rgb.detach().cpu().numpy()
    rgb_img = np.transpose(rgb_img, (1, 2, 0))  # HWC
    return np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)
