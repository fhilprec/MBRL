# Mermaid to Image Converter

This directory contains mermaid diagram files (.mmd) and a conversion script for LaTeX integration.

## Usage

Convert a mermaid diagram to an image for use in LaTeX:

```bash
./mmd2pdf.sh <input.mmd> [output_name] [format]
```

### Examples

```bash
# Convert trainingschedule.mmd to training_pipeline.png
./mmd2pdf.sh trainingschedule.mmd training_pipeline

# Convert with PDF format (requires ImageMagick)
./mmd2pdf.sh trainingschedule.mmd training_pipeline pdf

# Use input filename as output name
./mmd2pdf.sh mydiagram.mmd
```

## Output

- Images are automatically saved to: `../LaTeX/doc/latex/tuda-ci/example/images/`
- Default format: PNG (3000px width, transparent background)
- The script will print the LaTeX code needed to include the image

## LaTeX Integration

After running the script, add to your LaTeX document:

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{images/training_pipeline.png}
  \caption{Complete training pipeline for object-centric MBRL}
  \label{fig:training_pipeline}
\end{figure}
```

## Requirements

- Node.js and npx (for mermaid-cli)
- Optional: ImageMagick (for PDF conversion)

## Current Diagrams

- `trainingschedule.mmd` - Training pipeline flowchart (Section 3.3)
