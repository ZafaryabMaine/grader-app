#!/usr/bin/env bash
set -e

user_msg="${*:-default}"

git add .

if git diff --cached --quiet; then
  echo "No staged changes to commit."
  exit 0
fi

git commit -m "$user_msg"
git push