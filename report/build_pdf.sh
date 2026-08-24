#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

build() {
  local src="$1" out="$2"
  pandoc "$src" -o "$out" --pdf-engine=xelatex \
    -V mainfont="Liberation Serif" -V monofont="DejaVu Sans Mono" -V lang="$3" \
    -V geometry:margin=2.5cm -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
    -V fontsize=10pt --include-in-header=preamble.tex
  echo "wrote $out"
}

build REPORT.md REPORT.pdf ru
[ -f REPORT_EN.md ] && build REPORT_EN.md REPORT_EN.pdf en
exit 0
