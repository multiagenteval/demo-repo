#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw
mkdir -p models/checkpoints
mkdir -p experiments/metrics
mkdir -p configs/experiments

echo "Setup complete! Activate your virtual environment with:"
echo "source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate" 