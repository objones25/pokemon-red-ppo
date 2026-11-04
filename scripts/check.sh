#!/usr/bin/env bash
#
# check.sh: every gate a change must pass. Run it before calling any work done.
# Usage: scripts/check.sh
# Requires: bash 3.2+, uv.
# Exit status: 0 only if every gate passes.
# bash-target: 3.2

set -Eeuo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

uv run --quiet pyright
uv run --quiet ruff check .
uvx --quiet --from shellcheck-py shellcheck scripts/*.sh
uv run --quiet python scripts/audit_tests.py tests/
uv run --quiet python scripts/audit_negative_space.py src/pokered --min-assertions 1
uv run --quiet pytest -m "" -p randomly -n auto -q
