#!/bin/bash

git fetch
git pull origin master

# Install dependencies from requirements.txt
echo "Installing dependencies"
pip3 install torch==2.1.0+cu118 xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu118
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Run the Python script nudify.py
echo "Running nudify.py..."
python3 nudify.py
