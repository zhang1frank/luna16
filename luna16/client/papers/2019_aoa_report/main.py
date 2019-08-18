# %%

import subprocess

subprocess.call('pweave -f texpygments index.texw')
subprocess.call('pdflatex index.tex')
subprocess.call('bibtex index')
subprocess.call('pdflatex index.tex')
subprocess.call('pdflatex index.tex')