#!/bin/bash

# ToothFairy3 IAC Segmentation - Setup and Run Script
# This script creates a virtual environment, installs dependencies, and runs training or inference

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="toothfairy_env"
PYTHON_VERSION="python3"

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  ToothFairy3 IAC Segmentation Setup${NC}"
echo -e "${GREEN}==================================================${NC}"

# Check if Python is installed
if ! command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Python version:${NC}"
$PYTHON_VERSION --version

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_VERSION -m venv $VENV_NAME
    echo -e "${GREEN}Virtual environment created successfully!${NC}"
else
    echo -e "\n${YELLOW}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source $VENV_NAME/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo -e "\n${YELLOW}Installing dependencies from requirements.txt...${NC}"
    echo -e "${YELLOW}This may take several minutes...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
else
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi

# Check if necessary files exist
echo -e "\n${YELLOW}Checking for required files...${NC}"
if [ ! -f "train.py" ]; then
    echo -e "${RED}Error: train.py not found!${NC}"
    exit 1
fi

if [ ! -f "inference.py" ]; then
    echo -e "${RED}Error: inference.py not found!${NC}"
    exit 1
fi

if [ ! -f "clicks.json" ]; then
    echo -e "${YELLOW}Warning: clicks.json not found. This is required for inference.${NC}"
fi

echo -e "${GREEN}All required files found!${NC}"

# Prompt user for action
echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}  What would you like to do?${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "  ${YELLOW}1)${NC} Train the model"
echo -e "  ${YELLOW}2)${NC} Run inference"
echo -e "  ${YELLOW}3)${NC} Exit"
echo -e "${GREEN}==================================================${NC}"
read -p "Enter your choice (1, 2, or 3): " choice

case $choice in
    1)
        echo -e "\n${GREEN}Starting training...${NC}"
        echo -e "${YELLOW}Make sure you have configured the DATA_DIR path in train.py!${NC}"
        python train.py
        echo -e "\n${GREEN}Training completed!${NC}"
        ;;
    2)
        echo -e "\n${GREEN}Starting inference...${NC}"
        echo -e "${YELLOW}Make sure you have:${NC}"
        echo -e "  - best_model.pth in the current directory"
        echo -e "  - Input images in the 'input' directory"
        echo -e "  - clicks.json configured properly"
        python inference.py
        echo -e "\n${GREEN}Inference completed! Check the 'output' directory for results.${NC}"
        ;;
    3)
        echo -e "\n${YELLOW}Exiting...${NC}"
        deactivate
        exit 0
        ;;
    *)
        echo -e "\n${RED}Invalid choice. Please run the script again and enter 1, 2, or 3.${NC}"
        deactivate
        exit 1
        ;;
esac

echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}  Process completed successfully!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "\n${YELLOW}To deactivate the virtual environment, run:${NC} deactivate"
