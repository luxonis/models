import os

import pytest
from luxonis_train.core import Trainer


@pytest.mark.parametrize(
    "config_file", [path for path in os.listdir("configs") if "model" in path]
)
def test_train(config_file):
    opts = {
        "trainer.epochs": 1,
        "trainer.validation_interval": 1,
        "trainer.callbacks": [],
    }
    trainer = Trainer(f"configs/{config_file}", opts)
    trainer.train()
