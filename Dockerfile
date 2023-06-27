FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git wget python3-dev gcc s3fs ffmpeg libsm6 libxext6 curl libfreetype6-dev libssl-dev libpng-dev

RUN pip3 uninstall opencv PIL pillow -y

RUN useradd -ms /bin/bash luxonis
USER luxonis

WORKDIR /home/luxonis

# Copy the whole library and tool directory
COPY ../src ./src
COPY ../tools ./tools

# Copy package related files
COPY ["../requirements.txt", "../setup.py", "../entrypoint.sh", "../README.md", "../MANIFEST.in", "."]

# needed because of _imagingft C module error
RUN pip install --no-cache-dir pillow

# Install the library
RUN pip install --no-cache-dir .

# # Set the entrypoint command for the container
ENTRYPOINT ["/home/luxonis/entrypoint.sh"]