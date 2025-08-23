#!/bin/bash
# Install system dependencies for matplotlib
apt-get update
apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
