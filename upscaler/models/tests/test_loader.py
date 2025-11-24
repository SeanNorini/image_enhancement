import pytest
import torch
from typing import Any

from upscaler.models.loader import ExtendedModelLoader


class DummyReturn:
    """Object returned by the mocked parent loader."""

    def __init__(self, model: Any):
        self.model = model


class FakeModel:
    """A minimal pytorch-like model for testing."""

    def __init__(self, arch):
        self.arch = arch
        self.to_called_with = None
        self.eval_called = False
        self.half_called = False

    def to(self, device):
        self.to_called_with = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def half(self):
        self.half_called = True
        return self


@pytest.fixture
def base_registry():
    return {
        "spandrel_model": {"arch": "Spandrel", "ext": ".safetensors"},
        "spandrel_half": {"arch": "Spandrel", "ext": ".safetensors", "is_half": True},
    }


@pytest.fixture
def loader(base_registry):
    device = torch.device("cpu")
    ld = ExtendedModelLoader(
        model_specs=base_registry,
        device=device,
        weights_only=True,
    )
    return ld


def test_load_spandrel(loader, mocker):
    fake_model = FakeModel("Spandrel")
    dummy_return = DummyReturn(fake_model)

    mocker.patch(
        "upscaler.models.loader.ModelLoader.load_from_file",
        return_value=dummy_return,
    )

    model = loader.load_from_file("spandrel_model")

    assert model.arch == "Spandrel"


@pytest.mark.parametrize(
    "model_name,expected", [("spandrel_model", False), ("spandrel_half", True)]
)
def test_loads_model_with_correctly(model_name, expected, loader, mocker):
    fake_model = FakeModel("Spandrel")
    dummy_return = DummyReturn(fake_model)

    mocker.patch(
        "upscaler.models.loader.ModelLoader.load_from_file",
        return_value=dummy_return,
    )

    model = loader.load_from_file(model_name)

    assert model.half_called is expected
    assert model.to_called_with == torch.device("cpu")
    assert model.eval_called is True
    assert model is fake_model
