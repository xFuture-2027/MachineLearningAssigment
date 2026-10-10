#!/bin/bash
# Run with: bash Start_Mac.command
set -e
cd -- "$(dirname -- "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
if [[ -x .venv/bin/python ]]; then
  project_python="$PWD/.venv/bin/python"
else
  project_python=""
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3,11) <= sys.version_info[:2] <= (3,13)))' 2>/dev/null; then
      project_python="$(command -v "$candidate")"
      break
    fi
  done
  if [[ -z "$project_python" ]]; then
    echo 'Python 3.11, 3.12 or 3.13 was not found. Install Python 3.12 from python.org, reopen Terminal, and try again.'
    exit 1
  fi
fi
if ! "$project_python" run_project.py setup; then
  echo 'Setup failed. Send startup_error.txt and the last lines from this Terminal window.'
  exit 1
fi
exec .venv/bin/python run_project.py notebook
