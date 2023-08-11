import io
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.readlines()

export_packages = [
    "blobconverter>=1.3.0",
    # "openvino-dev==2022.1.0" # problematic because of numpy version
]

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
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # https://stackoverflow.com/a/67238346/5494277
    include_package_data=True,
    install_requires=required,
    extras_require={
        "export": export_packages,
    },
    project_urls={
        "Bug Tracker": "https://github.com/luxonis/models/issues",
        "Source Code": "https://github.com/luxonis/models/tree/dev",
    },
    classifiers=[
        "License :: MIT License",
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.8",
    ],
)
