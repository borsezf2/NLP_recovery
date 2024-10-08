#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update package list and upgrade existing packages
apt update && apt upgrade -y

# Install Python 3 and pip
apt install python3 python3-pip python3-venv -y

# Create a virtual environment
python3 -m venv nlp_env

# Activate the virtual environment
source nlp_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version, for GPU version visit pytorch.org for the appropriate command)
pip install torch 

# Install Transformers and related Hugging Face libraries
pip install transformers datasets tokenizers huggingface-hub accelerate peft -y

# Install additional requested libraries
pip install bottleneck decorator ipykernel protobuf sentencepiece ollama -y

# Install some useful additional libraries
pip install numpy pandas matplotlib seaborn scikit-learn -y

# Install Jupyter for interactive development
pip install jupyter -y

echo "Installation complete! To activate this environment, use:"
echo "source nlp_env/bin/activate"

# Create a requirements.txt file
pip freeze > requirements.txt

echo "Requirements file created: requirements.txt"

# Deactivate the virtual environment
deactivate