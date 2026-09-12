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
# cmd.exe rather than powershell.exe: git writes ordinary progress ("From
# https://...", "Switched to branch ...") to stderr, and PowerShell turns any
# native stderr output into a NativeCommandError, which aborts the run and
# hides the real result. cmd.exe passes it through untouched. It is reachable
# from WSL via interop but is not always on PATH, so fall back to the absolute
# location rather than failing.
CMD="$(command -v cmd.exe || echo /mnt/c/Windows/system32/cmd.exe)"

if [ ! -x "$CMD" ]; then
  echo "Could not find cmd.exe -- is this running inside WSL?" >&2
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
"$CMD" /c "cd /d $WIN_REPO && git fetch origin && git checkout $BRANCH && git pull --ff-only origin $BRANCH" \
  > /dev/null 2>&1 || {
    echo "Windows-side git failed. Check for uncommitted changes or a diverged branch there:" >&2
    echo "  cmd.exe /c \"cd /d $WIN_REPO && git status\"" >&2
    exit 1
  }

# Verify rather than trust: confirm the Windows clone really landed on the
# same commit as WSL. A silent no-op here means a live test runs stale code.
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$("$CMD" /c "cd /d $WIN_REPO && git rev-parse HEAD" 2>/dev/null | tr -d '\r\n')"

if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
  echo "==> In sync at $(git log --oneline -1)"
else
  echo "==> OUT OF SYNC: WSL at ${LOCAL_SHA:0:7}, Windows at ${REMOTE_SHA:0:7}" >&2
  exit 1
fi
