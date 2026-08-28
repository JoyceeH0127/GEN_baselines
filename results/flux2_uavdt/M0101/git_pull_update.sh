#!/usr/bin/env bash
set -Eeuo pipefail

# Pull a whole repository or replace one local file with its remote version.
# Defaults are tailored for the GEN_baselines server checkout.

REPO_DIR="${REPO_DIR:-/home/qinma/yelo/GEN_baselines}"
REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-main}"
SSH_KEY="${GITHUB_SSH_KEY:-$HOME/.ssh/id_ed25519_github_flux2}"
BACKUP_DIR="${BACKUP_DIR:-$REPO_DIR/.git/file-backups}"

usage() {
  cat <<'EOF'
Usage:
  git_pull_update.sh repo
  git_pull_update.sh file <repository-relative-path>

Environment overrides:
  REPO_DIR, GIT_REMOTE, GIT_BRANCH, GITHUB_SSH_KEY, BACKUP_DIR

Examples:
  git_pull_update.sh repo
  git_pull_update.sh file flux2_uavdt_inference.py
  git_pull_update.sh file scripts/flux2_uavdt_inference.py
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
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
    [[ $# -eq 1 ]] || die "repo mode accepts no path"
    if [[ -n "$(git status --porcelain)" ]]; then
      die "Working tree is not clean. Commit, stash, or discard local changes first."
    fi
    echo "Updating entire repository from $REMOTE/$BRANCH ..."
    git fetch "$REMOTE" "$BRANCH"
    git pull --ff-only "$REMOTE" "$BRANCH"
    ;;

  file)
    [[ $# -eq 2 ]] || die "file mode requires one repository-relative path"
    target="$2"
    [[ "$target" != /* ]] || die "Use a repository-relative path, not an absolute path"
    [[ "$target" != ../* && "$target" != */../* ]] || die "Path traversal is not allowed"

    echo "Fetching $REMOTE/$BRANCH ..."
    git fetch "$REMOTE" "$BRANCH"
    git cat-file -e "$REMOTE/$BRANCH:$target" 2>/dev/null || \
      die "File does not exist on $REMOTE/$BRANCH: $target"

    if [[ -e "$target" ]]; then
      mkdir -p "$BACKUP_DIR/$(dirname "$target")"
      backup="$BACKUP_DIR/${target}.before-pull"
      cp -p "$target" "$backup"
      echo "Local backup: $backup"
    fi

    git restore --source="$REMOTE/$BRANCH" --worktree -- "$target"
    echo "Updated file: $REPO_DIR/$target"
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    usage
    die "Unknown mode: $mode"
    ;;
esac

git status --short
