import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
def create_coco_dataset():
    subprocess.run(["nbscript", "examples/COCO_people_dataset.ipynb"])


@pytest.fixture(scope="session", autouse=True)
def create_cifar10_dataset():
    subprocess.run(["nbscript", "examples/CIFAR10_dataset.ipynb"])
