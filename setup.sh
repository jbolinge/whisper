#!/bin/bash
# WhisperX Transcription Server - Setup Script
# Uses uv for fast, reliable dependency management

set -e  # Exit on error

echo "=================================================="
echo "  WhisperX Transcription Server - Setup Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root is not recommended.${NC}"
    echo "Consider running as a regular user with sudo access."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for FFmpeg
echo "Checking for FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ FFmpeg is installed${NC}"
else
    echo -e "${YELLOW}! FFmpeg not found. Installing...${NC}"
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y ffmpeg
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y ffmpeg
    else
        echo -e "${RED}✗ Could not install FFmpeg. Please install manually.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ FFmpeg installed${NC}"
fi

# Check for uv, install if not present
echo ""
echo "Checking for uv..."
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ uv is installed${NC}"
else
    echo -e "${YELLOW}! uv not found. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to current session PATH
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}✓ uv installed${NC}"
    else
        echo -e "${RED}✗ uv installation failed. Please install manually:${NC}"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv --python 3.11
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Install PyTorch CPU first (must be installed separately to avoid index conflicts)
echo ""
echo "Installing PyTorch (CPU version)..."
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Install remaining dependencies
echo ""
echo "Installing remaining dependencies..."
echo "This may take a few minutes on first run..."
uv sync --python 3.11
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Verify installation
echo ""
echo "Verifying installation..."
if uv run python -c "import whisperx; import gradio; print('OK')" &> /dev/null; then
    echo -e "${GREEN}✓ Installation verified successfully!${NC}"
else
    echo -e "${RED}✗ Installation verification failed. Check error messages above.${NC}"
    exit 1
fi

# Done
echo ""
echo "=================================================="
echo -e "${GREEN}  Setup Complete!${NC}"
echo "=================================================="
echo ""
echo "To start the server:"
echo "  uv run python app.py"
echo ""
echo "Then open in browser: http://localhost:7860"
echo ""
echo "For speaker diarization, you'll need a HuggingFace token."
echo "See README.md for details on getting one."
echo ""
