#!/bin/bash
# TA-Lib Installation Script for WSL/Ubuntu
# Installs TA-Lib system library and Python wrapper

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "================================================================================"
echo -e "              ${BLUE}TA-LIB INSTALLATION SCRIPT${NC}"
echo "================================================================================"
echo ""

# Check if running on Linux/WSL
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script is for Linux/WSL only${NC}"
    echo "For macOS, use: brew install ta-lib"
    exit 1
fi

echo -e "${YELLOW}Step 1/6: Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y build-essential wget

echo ""
echo -e "${YELLOW}Step 2/6: Downloading TA-Lib source code...${NC}"
cd /tmp
if [ -f "ta-lib-0.4.0-src.tar.gz" ]; then
    echo "TA-Lib archive already exists, using cached version"
else
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
fi

echo ""
echo -e "${YELLOW}Step 3/6: Extracting TA-Lib...${NC}"
if [ -d "ta-lib" ]; then
    rm -rf ta-lib
fi
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

echo ""
echo -e "${YELLOW}Step 4/6: Updating config scripts for ARM64 compatibility...${NC}"
# Download updated config.guess and config.sub for ARM64 support
wget -O config.guess 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD'
wget -O config.sub 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD'
chmod +x config.guess config.sub

echo ""
echo -e "${YELLOW}Step 5/6: Configuring and compiling TA-Lib (this may take a few minutes)...${NC}"
./configure --prefix=/usr --build=aarch64-unknown-linux-gnu
make

echo ""
echo -e "${YELLOW}Step 5/6: Installing TA-Lib (requires sudo)...${NC}"
sudo make install
sudo ldconfig

echo ""
echo -e "${YELLOW}Step 6/6: Installing Python TA-Lib wrapper...${NC}"

# Go back to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run ./setup.sh first to create the virtual environment"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Install Python wrapper
pip install TA-Lib

# Verify installation
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"
python3 -c "import talib; print(f'TA-Lib version: {talib.__version__}')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo -e "${GREEN}                    TA-LIB INSTALLED SUCCESSFULLY${NC}"
    echo "================================================================================"
    echo ""
    echo "✅ TA-Lib system library installed"
    echo "✅ Python TA-Lib wrapper installed"
    echo ""
    echo "The system will now use 67+ technical indicators from TA-Lib"
    echo ""
    echo "You can now run: ./predict.sh"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo -e "${RED}                    INSTALLATION FAILED${NC}"
    echo "================================================================================"
    echo ""
    echo "There was an error installing TA-Lib Python wrapper"
    echo "The system will still work with manual indicator implementations"
    echo ""
fi

deactivate

exit 0
