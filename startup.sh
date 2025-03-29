#!/bin/bash

# Update package list
echo "Updating package list..."
sudo apt update

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# Run the Python script nudify.py
echo "Running nudify.py..."
python3 nudify.py
