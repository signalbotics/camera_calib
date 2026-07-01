#!/usr/bin/env bash
# Local wheel pipeline: build cp39-cp314 manylinux wheels ON DISK using Docker.
# Same wheels the GitHub workflow produces, but built here.
#
#   ./scripts/build_wheels.sh
#
# Builds from a clean export of committed HEAD in a temp dir, so it NEVER
# touches your working tree (build/, .venv, calibration_results/, ...).
# (Commit your changes first — uncommitted edits are not included.)
#
# Output: ./wheelhouse/*.whl
# Requires: Docker running, and cibuildwheel (pip install cibuildwheel).
set -euo pipefail
cd "$(dirname "$0")/.."
repo="$PWD"

if [ -x "$repo/.venv/bin/cibuildwheel" ]; then
    CIBW="$repo/.venv/bin/cibuildwheel"
elif command -v cibuildwheel >/dev/null 2>&1; then
    CIBW=cibuildwheel
else
    echo "cibuildwheel not found. Install it: pip install cibuildwheel" >&2
    exit 1
fi

mkdir -p "$repo/wheelhouse"

# Export the committed tree to a scratch dir and build there — leaves the
# working tree untouched and guarantees cibuildwheel sees no stale artifacts.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git archive --format=tar HEAD | tar -x -C "$tmp"

cd "$tmp"
"$CIBW" --platform linux --output-dir "$repo/wheelhouse" "$@"
