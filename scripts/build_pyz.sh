#!/usr/bin/env bash
# Build dist/lineage.pyz: a single-file app (code + pure-Python dependencies) that runs with
# any python3 >= 3.9 and needs no pip/installs. Requires pip on the *build* machine only.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
# Resolve wheels for the oldest supported Python so the bundle works on 3.9 .. 3.13+.
"$PY" -m pip install -q --target "$BUILD/app" --python-version 3.9 --only-binary=:all: --implementation py \
  httpx beautifulsoup4 typer rich
# pip evaluates environment markers against the build interpreter, so add 3.9/3.10-only deps explicitly.
"$PY" -m pip install -q --target "$BUILD/app" --python-version 3.9 --only-binary=:all: --implementation py \
  --no-deps exceptiongroup
cp -r src/paper_lineage "$BUILD/app/"
cat > "$BUILD/app/__main__.py" <<'PYEOF'
import sys

if sys.version_info < (3, 9):
    sys.exit("paper-lineage needs Python 3.9 or newer (found %d.%d)" % sys.version_info[:2])

from paper_lineage.cli import app

app(prog_name="lineage")
PYEOF
find "$BUILD/app" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf "$BUILD/app/bin"
mkdir -p dist
"$PY" -m zipapp "$BUILD/app" -o dist/lineage.pyz -p "/usr/bin/env python3" -c
chmod +x dist/lineage.pyz
echo "built dist/lineage.pyz ($(du -h dist/lineage.pyz | cut -f1))"
