#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TAG="v1.0.0"

valid_github_repo_slug() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]
}

canonical_github_repo_slug() {
  local remote_url="$1" path
  case "$remote_url" in
    git@github.com:*) path="${remote_url#git@github.com:}" ;;
    ssh://git@github.com/*) path="${remote_url#ssh://git@github.com/}" ;;
    https://*@github.com/*) path="${remote_url#https://*@github.com/}" ;;
    https://github.com/*) path="${remote_url#https://github.com/}" ;;
    *) return 1 ;;
  esac
  path="${path%/}"
  path="${path%.git}"
  valid_github_repo_slug "$path" || return 1
  printf '%s\n' "${path,,}"
}

github_repo_slug() {
  local fetch_url fetch_slug push_output push_slug
  fetch_url="$(git remote get-url origin 2>/dev/null)" || return 1
  fetch_slug="$(canonical_github_repo_slug "$fetch_url")" || return 1
  push_output="$(git remote get-url --push --all origin 2>/dev/null)" || return 1
  [[ -n "$push_output" && "$push_output" != *$'\n'* ]] || return 1
  push_slug="$(canonical_github_repo_slug "$push_output")" || return 1
  [[ "$fetch_slug" == "$push_slug" ]] || return 1
  printf '%s\n' "$fetch_slug"
}

require_locked_main() {
  local repo_slug lock_enabled
  repo_slug="$(github_repo_slug)" || {
    echo "Unable to determine a safe GitHub repository slug for branch-lock verification." >&2
    return 1
  }
  lock_enabled="$(gh api --hostname github.com "repos/$repo_slug/branches/main/protection" --jq '.lock_branch.enabled' 2>/dev/null)" || {
    echo "Unable to retrieve GitHub branch protection lock for main." >&2
    return 1
  }
  [[ "$lock_enabled" == "true" ]] || {
    echo "GitHub branch protection must have lock_branch.enabled=true for main." >&2
    return 1
  }
}

if [[ "${1:-}" == "--print-github-repo-slug" && "$#" == "1" ]]; then
  github_repo_slug
  exit $?
fi
if [[ "${1:-}" == "--check-github-main-lock" && "$#" == "1" ]]; then
  require_locked_main
  exit $?
fi

OLD_TAG_REF_SHA="${1:?usage: release_overwrite_preflight.sh OLD_TAG_REF_SHA EXPECTED_APPROVED_MAIN_SHA}"
EXPECTED_APPROVED_MAIN_SHA="${2:?usage: release_overwrite_preflight.sh OLD_TAG_REF_SHA EXPECTED_APPROVED_MAIN_SHA}"
[[ "$OLD_TAG_REF_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected a 40-hex old remote tag ref SHA" >&2; exit 2; }
[[ "$EXPECTED_APPROVED_MAIN_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected a 40-hex approved main SHA" >&2; exit 2; }
cd "$ROOT_DIR"
git fetch origin main --tags --force
branch="$(git symbolic-ref --quiet --short HEAD || true)"
[[ "$branch" == "main" ]] || { echo "Release overwrite preflight requires branch main, got: ${branch:-detached}" >&2; exit 3; }
status="$(git status --porcelain)"
[[ -z "$status" ]] || { echo "Release overwrite preflight requires a clean working tree." >&2; printf '%s\n' "$status" >&2; exit 4; }
require_locked_main || exit 5
head="$(git rev-parse HEAD)"
remote_main="$(git rev-parse origin/main)"
[[ "$head" == "$EXPECTED_APPROVED_MAIN_SHA" && "$remote_main" == "$EXPECTED_APPROVED_MAIN_SHA" ]] || {
  echo "HEAD and origin/main must equal approved main: HEAD=$head origin/main=$remote_main approved=$EXPECTED_APPROVED_MAIN_SHA" >&2
  exit 5
}
REMOTE_MAIN_SHA="$(git ls-remote origin refs/heads/main | awk 'NR==1{print $1}')"
[[ "$REMOTE_MAIN_SHA" == "$EXPECTED_APPROVED_MAIN_SHA" ]] || {
  echo "Remote main changed or is not approved: remote=$REMOTE_MAIN_SHA approved=$EXPECTED_APPROVED_MAIN_SHA" >&2
  exit 5
}
REMOTE_TAG_REF_SHA="$(git ls-remote --tags origin "refs/tags/$TAG" | awk 'NR==1{print $1}')"
[[ -n "$REMOTE_TAG_REF_SHA" ]] || { echo "Required existing remote tag is missing: $TAG" >&2; exit 6; }
[[ "$REMOTE_TAG_REF_SHA" == "$OLD_TAG_REF_SHA" ]] || {
  echo "Remote tag lease mismatch: expected=$OLD_TAG_REF_SHA actual=$REMOTE_TAG_REF_SHA" >&2
  exit 7
}
REMOTE_TAG_TARGET_SHA="$(git ls-remote --tags origin "refs/tags/$TAG^{}" | awk 'NR==1{print $1}')"
[[ -n "$REMOTE_TAG_TARGET_SHA" ]] || { echo "$TAG must remain an annotated tag before overwrite" >&2; exit 8; }
[[ "$REMOTE_TAG_TARGET_SHA" != "$head" ]] || { echo "$TAG already targets final main; refusing unnecessary overwrite" >&2; exit 9; }
python3 scripts/release_contract.py "$TAG"
bash scripts/test_unit.sh
python3 scripts/check_docs.py
python3 -m compileall -q src scripts tests clients/python
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
node --check clients/node/translategemma-client.mjs
python3 scripts/secret_scan.py .
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
bash scripts/build_release_artifacts.sh "$TAG" "$tmp/dist"
(cd "$tmp/dist" && sha256sum --check -- *.sha256 && md5sum --check -- *.md5)
echo "RELEASE_TAG=$TAG"
echo "RELEASE_HEAD=$head"
echo "EXPECTED_APPROVED_MAIN_SHA=$EXPECTED_APPROVED_MAIN_SHA"
echo "REMOTE_MAIN_SHA=$REMOTE_MAIN_SHA"
echo "REMOTE_TAG_REF_SHA=$REMOTE_TAG_REF_SHA"
echo "REMOTE_TAG_TARGET_SHA=$REMOTE_TAG_TARGET_SHA"
echo "RELEASE_OVERWRITE_PREFLIGHT=PASS"
