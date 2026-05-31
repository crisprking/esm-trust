#!/usr/bin/env bash
# Push this repo to GitHub. Run from the repo root on a machine where git can authenticate.
# Your token (the ghp_... one in your Kaggle secret 'esmc-secret-key') is a GitHub PAT with
# 'repo' scope — paste it when git prompts for a password, or embed it in the remote URL.
set -euo pipefail
USER="${1:-crisprking}"
REPO="${2:-esm-trust}"

git init -q 2>/dev/null || true
git add -A
git -c user.name="$USER" commit -m \
  "v0.2: reproducible benchmark runner + data-driven figures (results.csv as single source of truth), audited engine, 16 CPU tests + CI, honest README/DATA provenance" \
  || echo "nothing to commit"
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${USER}/${REPO}.git"
echo "Now run:  git push -u origin main   (paste the ghp_ token when asked for a password)"
