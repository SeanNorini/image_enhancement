from enum import Enum, auto
from pathlib import Path
import torch

from .io import image_to_tensor, tensor_to_rgb
from .model.network_scunet import SCUNet
from ..common.download_model import download_model

_MODELS = {}


class DenoiseModel(Enum):
    GRAY_15 = auto()
    GRAY_25 = auto()
    GRAY_50 = auto()
    COLOR_15 = auto()
    COLOR_25 = auto()
    COLOR_50 = auto()
    PSNR = auto()
    GAN = auto()


_MODEL_NAMES = {
    DenoiseModel.GRAY_15: "scunet_gray_15",
    DenoiseModel.GRAY_25: "scunet_gray_25",
    DenoiseModel.GRAY_50: "scunet_gray_50",
    DenoiseModel.COLOR_15: "scunet_color_15",
    DenoiseModel.COLOR_25: "scunet_color_25",
    DenoiseModel.COLOR_50: "scunet_color_50",
    DenoiseModel.PSNR: "scunet_color_real_psnr",
    DenoiseModel.GAN: "scunet_color_real_gan",
}


def load_model(model_name, model_path, device, n_channels: int = 3):
    global _MODELS
    if model_name in _MODELS:
        return _MODELS[model_name]

    model = SCUNet(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)

    if not model_path.exists():
        spec = {
            "download_url": f"https://github.com/cszn/KAIR/releases/download/v1.0/{model_name}.pth"
        }
        download_model(model_name, model_path, spec)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    _MODELS[model_name] = model.to(device)
    return model


def get_model_path(model: DenoiseModel) -> tuple[str, Path]:
    model_name = _MODEL_NAMES[model]
    return (
        model_name,
        Path(__file__).resolve().parent / f"model/weights/{model_name}.pth",
    )


def denoise(img, model: DenoiseModel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name, model_path = get_model_path(model)
    model = load_model(model_name, model_path, device)
    img = image_to_tensor(img).to(device)
    with torch.inference_mode():
        out = model(img)
    result = tensor_to_rgb(out)
    del img, out
    return result
