#!/bin/bash

# Directory to check (change this to your target directory)
TARGET_DIR="/home/yxh666/RE0"

# Check if TARGET_DIR exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Directory $TARGET_DIR does not exist."
  exit 1
fi

# Check using mypy
echo "Running mypy..."
mypy "$TARGET_DIR"

# Check using flake8
echo "Running flake8..."
flake8 "$TARGET_DIR"

# Check using pylint
echo "Running pylint..."
pylint "$TARGET_DIR"

echo "Finished checks."
