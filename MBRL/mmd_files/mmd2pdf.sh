#!/bin/bash

# Mermaid to Image Converter Script for LaTeX
# Usage: ./mmd2pdf.sh <input.mmd> [output_name] [format]
# If output_name is not provided, uses the input filename
# Format can be 'png' or 'pdf' (default: png, as it works more reliably)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGES_DIR="$SCRIPT_DIR/../LaTeX/doc/latex/tuda-ci/example/images"

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No input file specified${NC}"
    echo "Usage: $0 <input.mmd> [output_name] [format]"
    echo "Example: $0 trainingschedule.mmd training_pipeline png"
    exit 1
fi

INPUT_FILE="$1"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    # Try looking in the script directory
    if [ -f "$SCRIPT_DIR/$INPUT_FILE" ]; then
        INPUT_FILE="$SCRIPT_DIR/$INPUT_FILE"
    else
        echo -e "${RED}Error: Input file '$INPUT_FILE' not found${NC}"
        exit 1
    fi
fi

# Determine output filename
if [ $# -ge 2 ]; then
    OUTPUT_NAME="$2"
else
    # Extract filename without extension
    OUTPUT_NAME=$(basename "$INPUT_FILE" .mmd)
fi

# Determine output format
if [ $# -ge 3 ]; then
    FORMAT="$3"
else
    FORMAT="png"
fi

# Ensure images directory exists
mkdir -p "$IMAGES_DIR"

OUTPUT_FILE="$IMAGES_DIR/${OUTPUT_NAME}.${FORMAT}"

echo -e "${YELLOW}Converting mermaid diagram...${NC}"
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Format: $FORMAT"

# Check if npx is available
if ! command -v npx &> /dev/null; then
    echo -e "${RED}Error: npx not found. Please install Node.js and npm${NC}"
    exit 1
fi

# Convert mermaid to desired format
echo -e "${YELLOW}Converting .mmd to ${FORMAT}...${NC}"

if [ "$FORMAT" = "png" ]; then
    npx -p @mermaid-js/mermaid-cli mmdc \
        -i "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        -w 3000 \
        -b transparent 2>/dev/null
else
    # Try PDF format
    npx -p @mermaid-js/mermaid-cli mmdc \
        -i "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        -b transparent \
        --pdfFit 2>/dev/null
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to convert mermaid diagram${NC}"
    echo "Falling back to PNG format..."
    OUTPUT_FILE="$IMAGES_DIR/${OUTPUT_NAME}.png"
    FORMAT="png"

    npx -p @mermaid-js/mermaid-cli mmdc \
        -i "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        -w 3000 \
        -b transparent

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: PNG conversion also failed${NC}"
        exit 1
    fi
fi

# Success message
echo -e "${GREEN}✓ Success!${NC}"
echo -e "Image saved to: ${GREEN}$OUTPUT_FILE${NC}"
echo ""
echo "Add to your LaTeX document with:"
echo ""
echo -e "${YELLOW} \\\begin{figure}[ht]"
echo -e "\t\\centering"
echo -e "\t\\includegraphics[width=0.9\\textwidth]{images/${OUTPUT_NAME}.${FORMAT}}"
echo -e "\t\\caption{Your caption here}"
echo -e "\t\\label{fig:${OUTPUT_NAME}}"
echo -e " \\\end{figure}${NC}"

exit 0
