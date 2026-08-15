#!/usr/bin/env bash
set -euo pipefail

python3 "$(dirname "$0")/check_honest_fallbacks.py"
