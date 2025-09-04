#!/bin/sh

set -eu

# answers.json -> answers.tex
python3 answers_to_tex.py > answers.tex

# answers.md -> answers.pdf
xelatex answers.tex
