#!/bin/bash
# AI Portfolio Manager - Complete Setup and Initialization Script

set -e  # Exit on error

echo "================================================================================"
echo "                    AI PORTFOLIO MANAGER - SETUP"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/8]${NC} Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Check pip
echo -e "\n${YELLOW}[2/8]${NC} Checking pip..."
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} pip3 found"
else
    echo -e "${RED}✗${NC} pip3 not found. Installing..."
    python3 -m ensurepip --upgrade
fi

# Create virtual environment
echo -e "\n${YELLOW}[3/8]${NC} Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠${NC} Virtual environment already exists. Removing..."
    rm -rf venv
fi
python3 -m venv venv
echo -e "${GREEN}✓${NC} Virtual environment created"

# Activate virtual environment
echo -e "\n${YELLOW}[4/8]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo -e "\n${YELLOW}[5/8]${NC} Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Upgraded successfully"

# Install dependencies
echo -e "\n${YELLOW}[6/8]${NC} Installing Python packages..."
echo "This may take several minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All packages installed successfully"
else
    echo -e "${RED}✗${NC} Some packages failed to install"
    echo "Please check the error messages above"
fi

# Create directory structure
echo -e "\n${YELLOW}[7/8]${NC} Creating directory structure..."
mkdir -p data/raw data/processed data/models logs notebooks tests results

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/models/.gitkeep
touch logs/.gitkeep
touch results/.gitkeep

echo -e "${GREEN}✓${NC} Directories created"

# Check TA-Lib installation
echo -e "\n${YELLOW}[8/8]${NC} Checking TA-Lib installation..."
python3 -c "import talib" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} TA-Lib is installed"
else
    echo -e "${YELLOW}⚠${NC} TA-Lib not found. Installing..."
    echo ""
    echo "For Ubuntu/Debian:"
    echo "  sudo apt-get install ta-lib"
    echo ""
    echo "For macOS:"
    echo "  brew install ta-lib"
    echo ""
    echo "For Windows:"
    echo "  Download from: https://github.com/TA-Lib/ta-lib-python"
    echo ""
    echo "After installing TA-Lib, run: pip install TA-Lib"
fi

# Summary
echo ""
echo "================================================================================"
echo -e "${GREEN}                         SETUP COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Configure API Keys:"
echo "   Edit config/config.yaml and add your API keys:"
echo "   - Alpha Vantage: https://www.alphavantage.co/support/#api-key"
echo "   - News API: https://newsapi.org/register"
echo ""
echo "2. Run a prediction:"
echo "   ./predict.sh"
echo ""
echo "3. Or specify more options:"
echo "   ./predict.sh AAPL MSFT GOOGL --budget 50000 --method black_litterman"
echo ""
echo "================================================================================"
echo ""
echo "For more information, see README.md"
echo ""

# Deactivate virtual environment
deactivate 2>/dev/null || true
