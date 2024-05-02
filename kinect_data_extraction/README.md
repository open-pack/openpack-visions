# Kinect Data Extraction

Here are collection of tools to extract data from the kinect recordings (i.e., mkv files).
Most of the tools are written in C++.

## Prerequisities

- Ubuntu (20.04)

## Common Setup

### Install Kinect SDK

Follow the instruction in ["Azure Kinect Sensor SDK download"](https://learn.microsoft.com/ja-jp/azure/kinect-dk/sensor-sdk-download)

```bash
# Add Linux Software Repository for Microsoft Products
curl -sSL -O https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb
sudo apt-get update

# Install Kinect SDK
sudo apt install k4a-tools
sudo apt install libk4a1.4-dev
```

### (Optional) VSCode Extentions

- [clang-format](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
- [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
- [clang-tidy](https://marketplace.visualstudio.com/items?itemName=notskm.clang-tidy)
- CMake
- CMake Tools
