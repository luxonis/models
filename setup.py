import io
from setuptools import setup

with open('requirements.txt') as f:
    required = f.readlines()

setup(
    name="luxonis-train",
    version="0.0.1",
    description="Luxonis training library for training lightweight models that run fast on OAK products.",
    long_description=io.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/luxonis/models-private",
    keywords="ml trainig luxonis",
    author="Luxonis",
    author_email="support@luxonis.com",
    license="MIT",
    packages=["luxonis_train"],
    package_dir={"":"."},
    package_data={"luxonis_train": ["core","datasets","models","utils"]},
    install_requires=required,
    include_package_data=True,
    project_urls={
        "Bug Tracker": "https://github.com/luxonis/models-private/issues",
        "Source Code": "https://github.com/luxonis/models-private/tree/train-refactor",
    },
    classifiers=[
        "License :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.8"
    ]
)
