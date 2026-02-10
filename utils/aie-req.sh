#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Installing lld..."
sudo apt update
sudo apt install -y lld

# echo "Adding deadsnakes PPA for Python 3.8..."
# sudo add-apt-repository -y ppa:deadsnakes/ppa
# sudo apt update

echo "Installing virtualenv using apt..."
sudo apt install -y python3-virtualenv python3-pip

echo "Installing CMake..."
sudo apt install -y cmake

echo "Installing Ninja..."
sudo apt install -y ninja-build

echo "Installing Python packages via apt..."
sudo apt install -y python3-psutil python3-rich python3-pybind11 python3-numpy

# echo "Adding toolchain PPA and installing GCC 13..."
# sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
# sudo apt update
# sudo apt install -y gcc-13 g++-13

echo "All packages installed successfully!"
