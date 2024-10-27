#!/bin/bash

# Install anaconda (if needed)
if ! command -v conda &> /dev/null
then
    echo "Anaconda not found, installing..."
    bash anaconda.sh -b
    export PATH="$HOME/anaconda3/bin:$PATH"
    conda init
else
    echo "Anaconda found"
fi

# Create virtual environment
echo "Creating virtual environment 'unmaskx-env'"
conda create --name unmaskx-env python=3.8 -y
conda activate unmaskx-env

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete. Activate the environment with: conda activate unmaskx-env"
