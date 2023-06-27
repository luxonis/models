#!/bin/bash
echo "test"
find / -name libcrypto.so.1.1 -print

# Check if no arguments are passed
# if [ "$#" -eq 0 ]; then
#     echo "Choose an action that you want to perform from supported actions: ['train', 'tune']"
#     exit 1
# fi

# # Check the first argument
# if [ "$1" = "train" ]; then
#     echo "Starting training..."
#     shift  # Remove the first argument from the list
#     python3 /app/tools/train.py "$@"  # Pass remaining arguments to start.py
# elif [ "$1" = "tune" ]; then
#     echo "Starting tunning..."
#     shift  
#     python3 /app/tools/tune.py "$@"
# else
#     echo "Argument $1 doesn't match any action. Supported actions are ['train', 'tune']."
#     exit 1
# fi