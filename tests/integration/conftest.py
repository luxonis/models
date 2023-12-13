import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
def create_coco_dataset():
    result = subprocess.run(
        ["nbscript", "examples/COCO_people_dataset.ipynb"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(result.stdout.decode("utf-8"))
    print(result.stderr.decode("utf-8"))


@pytest.fixture(scope="session", autouse=True)
def create_cifar10_dataset():
    result = subprocess.run(
        ["nbscript", "examples/CIFAR10_dataset.ipynb"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(result.stdout.decode("utf-8"))
    print(result.stderr.decode("utf-8"))
