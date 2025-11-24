import os
from pathlib import Path

import torch
from spandrel import ModelLoader


class ExtendedModelLoader(ModelLoader):
    def __init__(
        self,
        model_specs: dict[str, dict[str, any]],
        device: torch.device,
        weights_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._registry = model_specs
        self._map_location = device
        self._weights_only = weights_only
        directory = Path(__file__).resolve().parent
        self._folder_path = Path.joinpath(directory, "weights")

    def load_from_file(self, model_name, **kwargs):
        data = self._registry[model_name]
        model_path = os.path.join(self._folder_path, model_name + data["ext"])
        if data["arch"] == "Spandrel":
            model = super().load_from_file(model_path).model

            model = model.to(torch.device(self._map_location)).eval()
            if data.get("is_half", False):
                model = model.half()

            return model
