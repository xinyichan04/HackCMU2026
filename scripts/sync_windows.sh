#!/usr/bin/env bash
# Push the current branch from WSL and fast-forward the native-Windows clone
# to match, in one step.
#
# This project is split across two checkouts on the same machine: WSL does
# training, file conversion and ASR, while live mic/speaker conversion runs
# from a native-Windows clone (PortAudio enumerates no devices under WSL).
# Every code change therefore has to reach the Windows side before it can be
# tested live, and doing that by hand is easy to forget -- a live test then
# silently runs against stale code.
#
# Usage:
#   bash scripts/sync_windows.sh
#
# Override the Windows checkout location if it differs:
#   WIN_REPO='C:\path\to\HackCMU2026' bash scripts/sync_windows.sh
set -euo pipefail

WIN_REPO="${WIN_REPO:-C:\\Users\\jxie0\\OneDrive\\Desktop\\Projects\\HackCMU2026}"
# powershell.exe is reachable from WSL via the interop path but is not always
# on PATH, so fall back to the absolute location rather than failing.
PWSH="$(command -v powershell.exe || echo /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe)"

if [ ! -x "$PWSH" ]; then
  echo "Could not find powershell.exe -- is this running inside WSL?" >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "HEAD" ]; then
  echo "Detached HEAD; check out a branch before syncing." >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Warning: uncommitted changes will NOT be synced (only pushed commits are)." >&2
fi

echo "==> Pushing $BRANCH to origin"
git push origin "$BRANCH"

echo "==> Updating Windows clone at $WIN_REPO"
# --ff-only so a diverged Windows checkout fails loudly here instead of
# producing a surprise merge commit in a clone nobody edits directly.
"$PWSH" -NoProfile -Command "
  \$ErrorActionPreference = 'Stop'
  Set-Location '$WIN_REPO'
  git fetch origin 2>&1 | Out-Null
  git checkout $BRANCH 2>&1 | Out-Null
  git pull --ff-only origin $BRANCH 2>&1 | Out-Null
  git log --oneline -1
" 2>&1 | tr -d '\r' | tail -3

echo "==> Windows clone is now at the commit above"
