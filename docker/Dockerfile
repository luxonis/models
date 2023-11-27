FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git wget python3-dev gcc s3fs ffmpeg libsm6 libxext6 curl libfreetype6-dev libssl-dev libpng-dev

RUN pip3 uninstall opencv PIL pillow -y

RUN useradd -ms /bin/bash luxonis

WORKDIR /home/luxonis

# Copy the whole library
COPY src src

# Copy package related files
COPY requirements.txt pyproject.toml README.md docker/entrypoint.sh .

# needed because of _imagingft C module error
RUN pip install --no-cache-dir pillow

# Install the library
RUN pip install --upgrade pip
RUN pip install --no-cache-dir .

RUN pip install awscli

# # Set the entrypoint command for the container
RUN chown -R luxonis:luxonis /home/luxonis
RUN chmod +x ./entrypoint.sh

USER luxonis
ENTRYPOINT "/home/luxonis/entrypoint.sh"
