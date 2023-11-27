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


function exit_with_help {
    echo "Choose an action that you want to perform from supported actions:"
    echo "['train', 'eval', 'export', 'tune']"
    exit 1
}

# Check if no arguments are passed
if [ "$#" -eq 0 ]; then
    exit_with_help
fi

case "$1" in
    train)
        echo "Starting training..."
        shift  # Remove the first argument from the list
        python3 -m luxonis_train.tools.train "$@"
        ;;
    tune)
        echo "starting tuning..."
        shift
        python3 -m luxonis_train.tools.tune "$@"
        ;;
    eval | evaluate)
        echo "starting evaluation..."
        shift
        python3 -m luxonis_train.tools.evaluate "$@"
        ;;
    export)
        echo "starting export..."
        shift
        python3 -m luxonis_train.tools.export "$@"
        ;;
    *)
        exit_with_help
        ;;
esac
