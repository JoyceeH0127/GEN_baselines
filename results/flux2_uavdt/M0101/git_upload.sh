#!/usr/bin/env bash
set -Eeuo pipefail

# Commit and push a whole repository, one file, or one directory.
# Deletions are included only in `repo` mode. Path modes add the named path.

REPO_DIR="${REPO_DIR:-/home/qinma/yelo/GEN_baselines}"
REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-main}"
SSH_KEY="${GITHUB_SSH_KEY:-$HOME/.ssh/id_ed25519_github_flux2}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Update repository files}"

usage() {
  cat <<'EOF'
Usage:
  git_upload.sh repo [commit-message]
  git_upload.sh file <repository-relative-path> [commit-message]
  git_upload.sh dir  <repository-relative-directory> [commit-message]

Environment overrides:
  REPO_DIR, GIT_REMOTE, GIT_BRANCH, GITHUB_SSH_KEY, COMMIT_MESSAGE

Examples:
  git_upload.sh repo "Update FLUX.2 experiments"
  git_upload.sh file flux2_uavdt_inference.py "Update FLUX.2 inference"
  git_upload.sh dir results/flux2_uavdt "Add generated UAVDT images"
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_relative_path() {
  local path="$1"
  [[ "$path" != /* ]] || die "Use a repository-relative path, not an absolute path"
  [[ "$path" != ../* && "$path" != */../* ]] || die "Path traversal is not allowed"
}

[[ $# -ge 1 ]] || { usage; exit 2; }
[[ -d "$REPO_DIR/.git" ]] || die "Not a Git repository: $REPO_DIR"
[[ -f "$SSH_KEY" ]] || die "GitHub SSH key not found: $SSH_KEY"

git config --global --add safe.directory "$REPO_DIR"
export GIT_SSH_COMMAND="ssh -i $SSH_KEY -p 443 -o HostName=ssh.github.com -o IdentitiesOnly=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=6"

cd "$REPO_DIR"
git remote get-url "$REMOTE" >/dev/null 2>&1 || die "Remote not found: $REMOTE"

mode="$1"
case "$mode" in
  repo)
    [[ $# -le 2 ]] || die "repo mode accepts only an optional commit message"
    [[ $# -eq 1 ]] || COMMIT_MESSAGE="$2"
    echo "Staging the entire repository, including tracked deletions ..."
    git add -A
    ;;

  file)
    [[ $# -ge 2 && $# -le 3 ]] || die "file mode requires a path and optional message"
    target="$2"
    validate_relative_path "$target"
    [[ -f "$target" || -L "$target" ]] || die "File not found: $REPO_DIR/$target"
    [[ $# -eq 2 ]] || COMMIT_MESSAGE="$3"
    git add -- "$target"
    ;;

  dir)
    [[ $# -ge 2 && $# -le 3 ]] || die "dir mode requires a path and optional message"
    target="$2"
    validate_relative_path "$target"
    [[ -d "$target" ]] || die "Directory not found: $REPO_DIR/$target"
    [[ $# -eq 2 ]] || COMMIT_MESSAGE="$3"
    git add -- "$target"
    ;;

  -h|--help|help)
    usage
    exit 0
    ;;

  *)
    usage
    die "Unknown mode: $mode"
    ;;
esac

if git diff --cached --quiet; then
  echo "Nothing staged; there is nothing to upload."
  exit 0
fi

echo "Staged changes:"
git status --short
git diff --cached --stat

git commit -m "$COMMIT_MESSAGE"

echo "Synchronizing with $REMOTE/$BRANCH ..."
git fetch "$REMOTE" "$BRANCH"
git rebase "$REMOTE/$BRANCH"

echo "Pushing HEAD to $REMOTE/$BRANCH ..."
git push -u "$REMOTE" "HEAD:$BRANCH"
echo "Upload complete."

