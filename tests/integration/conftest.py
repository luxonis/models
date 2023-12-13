import os
import subprocess
from pathlib import Path

import pytest

TEST_LUXONIS_ML_BASE_PATH = Path("tests/luxonis_ml")


@pytest.fixture(scope="session")
def prepare_environment():
    TEST_LUXONIS_ML_BASE_PATH.mkdir(parents=True, exist_ok=True)
    os.environ["LUXONISML_BASE_PATH"] = str(TEST_LUXONIS_ML_BASE_PATH)


@pytest.fixture(scope="session", autouse=True)
def create_coco_dataset(prepare_environment):
    subprocess.run(["nbscript", "examples/COCO_people_dataset.ipynb"])


@pytest.fixture(scope="session", autouse=True)
def create_cifar10_dataset(prepare_environment):
    subprocess.run(["nbscript", "examples/CIFAR10_dataset.ipynb"])
