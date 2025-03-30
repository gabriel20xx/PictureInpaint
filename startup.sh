#!/bin/bash

# Git stuff
echo "Resetting git"
git reset --hard
echo "Fetching repo"
git fetch
echo "Pulling changes"
git pull origin master

# Install dependencies from requirements.txt
echo "Installing dependencies"
pip3 install torch==2.1.0+cu118 xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
echo "Installing dependencies from requirements.txt..."
pip3 install -r requirements.txt

# Make file executable
echo "Making everything read, write and execute"
chmod -R 777 /app

# Print a warning if HF_TOKEN is not set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set!"
else
  echo "Shell script received HF_TOKEN."
fi

# Run the Python script nudify.py
echo "Running nudify.py..."
python3 nudify.py
