import os
import shutil
from pathlib import Path

import pytest

from luxonis_train.core import Exporter, Trainer
from luxonis_train.utils import Config


@pytest.fixture(scope="function", autouse=True)
def clear_config():
    Config.clear_instance()
    shutil.rmtree("output", ignore_errors=True)


@pytest.mark.parametrize(
    "config_file", [path for path in os.listdir("configs") if "model" in path]
)
def test_sanity(config_file):
    opts = {
        "trainer.epochs": 1,
        "trainer.validation_interval": 1,
        "trainer.callbacks": [],
    }
    trainer = Trainer(f"configs/{config_file}", opts)
    trainer.train()
    Config.clear_instance()
    opts["model.weights"] = str(list(Path("output").rglob("*.ckpt"))[0])
    opts["exporter.onnx.opset_version"] = 11
    exporter = Exporter(f"configs/{config_file}", opts)
    exporter.export()
