#!/bin/bash

mkdir /home/luxonis/.luxonis_mount
echo $AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY > /home/luxonis/.passwd-s3fs
chmod 600 /home/luxonis/.passwd-s3fs
s3fs $AWS_BUCKET /home/luxonis/.luxonis_mount \
  	-o passwd_file=/home/luxonis/.passwd-s3fs \
  	-o curldbg \
  	-o url=$S3_ENDPOINT \
 	-o use_path_request_style \
	-o umask=0000



# Check if no arguments are passed
if [ "$#" -eq 0 ]; then
    echo "Choose an action that you want to perform from supported actions: ['train', 'tune']"
    exit 1
fi

# Check the first argument
if [ "$1" = "train" ]; then
    echo "Starting training..."
    shift  # Remove the first argument from the list
    python3 /home/luxonis/tools/train.py "$@"  # Pass remaining arguments to start.py
elif [ "$1" = "tune" ]; then
    echo "Starting tunning..."
    shift  
    python3 /home/luxonis/tools/tune.py "$@"
else
    echo "Argument $1 doesn't match any action. Supported actions are ['train', 'tune']."
    exit 1
fi