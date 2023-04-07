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
    url="https://github.com/luxonis/models",
    keywords="ml trainig luxonis",
    author="Luxonis",
    author_email="support@luxonis.com",
    license="MIT",
    packages=["luxonis_train"],
    package_dir={"":"."},
    install_requires=required,
    include_package_data=True,
    project_urls={
        "Bug Tracker": "https://github.com/luxonis/models/issues",
        "Source Code": "https://github.com/luxonis/models/tree/dev",
    },
    classifiers=[
        "License :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.8"
    ]
)