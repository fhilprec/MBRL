#!/bin/bash
# Install all necessary LaTeX packages for TUDa thesis and dependencies

# Update package lists
sudo apt-get update

# Core TeX Live
sudo apt-get install -y texlive texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-extra

# Language support (for babel and class dependencies)
sudo apt-get install -y texlive-lang-german texlive-lang-english

# Bibliography and citation support
sudo apt-get install -y texlive-bibtex-extra biber

# Math and science packages
sudo apt-get install -y texlive-science

# Additional useful packages (if not included above)
sudo apt-get install -y texlive-xetex texlive-pictures texlive-pstricks

sudo apt install biber

# If urcls is not available via apt, download and install manually:
if ! kpsewhich URspecialopts.sty > /dev/null; then
    echo "URspecialopts.sty not found, installing urcls manually..."
    wget https://mirrors.ctan.org/macros/latex/contrib/urcls.zip
    unzip urcls.zip
    cp urcls/tex/*.sty /home/fhilprec/MBRL/MBRL/LaTeX/tex/latex/tuda-ci/
    rm -rf urcls urcls.zip
fi

echo "All required LaTeX packages should now be installed."







rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.bcf *.run.xml; TEXINPUTS=.:/home/fhilprec/MBRL/MBRL/LaTeX/tex/latex/tuda-ci//: latexmk -pdf -f DEMO-TUDaThesis.tex