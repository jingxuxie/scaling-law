#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/manual_push.sh /path/to/local/clone/of/jingxuxie/scaling-law

TARGET=${1:-}
if [[ -z "$TARGET" ]]; then
  echo "Provide a local clone path, e.g. ./scripts/manual_push.sh ../scaling-law" >&2
  exit 1
fi

rsync -av --exclude .git ./ "$TARGET"/
cd "$TARGET"
git add README.md notes experiments scripts
git commit -m "Add damped spectral preconditioning proof notes"
git push origin main
