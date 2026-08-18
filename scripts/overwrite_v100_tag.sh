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

OLD_TAG_REF_SHA="${1:?usage: overwrite_v100_tag.sh OLD_TAG_REF_SHA EXPECTED_APPROVED_MAIN_SHA}"
EXPECTED_APPROVED_MAIN_SHA="${2:?usage: overwrite_v100_tag.sh OLD_TAG_REF_SHA EXPECTED_APPROVED_MAIN_SHA}"

cd "$ROOT_DIR"
bash scripts/release_overwrite_preflight.sh "$OLD_TAG_REF_SHA" "$EXPECTED_APPROVED_MAIN_SHA"
git fetch origin main --tags --force
HEAD_SHA="$(git rev-parse HEAD)"
ORIGIN_MAIN_SHA="$(git rev-parse origin/main)"
REMOTE_MAIN_SHA="$(git ls-remote origin refs/heads/main | awk 'NR==1{print $1}')"
[[ "$HEAD_SHA" == "$EXPECTED_APPROVED_MAIN_SHA" && "$ORIGIN_MAIN_SHA" == "$EXPECTED_APPROVED_MAIN_SHA" && "$REMOTE_MAIN_SHA" == "$EXPECTED_APPROVED_MAIN_SHA" ]] || {
  echo "Approved main changed before tag overwrite: HEAD=$HEAD_SHA origin/main=$ORIGIN_MAIN_SHA remote=$REMOTE_MAIN_SHA approved=$EXPECTED_APPROVED_MAIN_SHA" >&2
  exit 10
}
require_locked_main || exit 10
REMOTE_TAG_REF_SHA="$(git ls-remote --tags origin "refs/tags/$TAG" | awk 'NR==1{print $1}')"
[[ "$REMOTE_TAG_REF_SHA" == "$OLD_TAG_REF_SHA" ]] || {
  echo "Remote tag lease mismatch before atomic push: expected=$OLD_TAG_REF_SHA actual=$REMOTE_TAG_REF_SHA" >&2
  exit 11
}
if git show-ref --verify --quiet "refs/tags/$TAG"; then
  git tag -d "$TAG" >/dev/null
fi
git tag -a -f "$TAG" HEAD -m "TranslateGemma 27B IT v1.0.0 — final public Kaggle TPU v5e-8 Text + Vision release"
NEW_TAG_REF_SHA="$(git rev-parse "refs/tags/$TAG")"
NEW_TAG_TARGET_SHA="$(git rev-parse "refs/tags/$TAG^{}")"
[[ "$NEW_TAG_TARGET_SHA" == "$EXPECTED_APPROVED_MAIN_SHA" ]] || { echo "New annotated tag does not peel to approved main" >&2; exit 12; }
git push origin \
  --force-with-lease="refs/tags/$TAG:$OLD_TAG_REF_SHA" \
  "refs/tags/$TAG:refs/tags/$TAG"
FINAL_REMOTE_MAIN_SHA="$(git ls-remote origin refs/heads/main | awk 'NR==1{print $1}')"
FINAL_TAG_TARGET_SHA="$(git ls-remote --tags origin "refs/tags/$TAG^{}" | awk 'NR==1{print $1}')"
[[ "$FINAL_REMOTE_MAIN_SHA" == "$EXPECTED_APPROVED_MAIN_SHA" ]] || { echo "Remote main changed after atomic push: $FINAL_REMOTE_MAIN_SHA" >&2; exit 13; }
[[ "$FINAL_TAG_TARGET_SHA" == "$EXPECTED_APPROVED_MAIN_SHA" ]] || { echo "Remote tag does not peel to approved main: $FINAL_TAG_TARGET_SHA" >&2; exit 14; }
echo "TAG_OVERWRITE=PASS"
echo "TAG=$TAG"
echo "OLD_REMOTE_TAG_REF_SHA=$OLD_TAG_REF_SHA"
echo "EXPECTED_APPROVED_MAIN_SHA=$EXPECTED_APPROVED_MAIN_SHA"
echo "NEW_TAG_REF_SHA=$NEW_TAG_REF_SHA"
echo "NEW_TAG_TARGET_SHA=$NEW_TAG_TARGET_SHA"
echo "FINAL_REMOTE_MAIN_SHA=$FINAL_REMOTE_MAIN_SHA"
echo "FINAL_TAG_TARGET_SHA=$FINAL_TAG_TARGET_SHA"
echo "Existing GitHub Release v1.0.0 should now be refreshed by the tag workflow."
