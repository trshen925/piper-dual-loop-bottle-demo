#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-${repo_dir}/outputs/$(date -u +%Y%m%dT%H%M%SZ)}"
python_bin="${PYTHON:-python3}"
"${python_bin}" "${repo_dir}/demo.py" --cases "${repo_dir}/cases.json" --out "${output_dir}"
