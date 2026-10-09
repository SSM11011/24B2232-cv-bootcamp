#!/usr/bin/env bash
# Package code/, predictions/, results.json, report.md and the analysis figures into <rollno>_cv_bootcamp.zip and check its format.
set -euo pipefail
cd "$(dirname "$0")/.."

ZIP=24B2232_cv_bootcamp.zip
rm -f "$ZIP"
zip -r -q "$ZIP" code predictions results.json report.md analysis/*.png analysis/*.json analysis/*.csv \
    -x "code/__pycache__/*"
echo "built $ZIP"
python3 check_format.py "$ZIP"
