#!/usr/bin/env bash
# Publish this delivery repository through a temporary GitHub device login.
# No credential, key, or credential-helper setting is persisted in the shared account.

set -euo pipefail

readonly REPOSITORY_URL="https://github.com/SparklingPumpkin/pbr-vehicle.git"
readonly BRANCH="main"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly MAX_FILE_BYTES=$((95 * 1024 * 1024))

temporary_gh_config=""

cleanup() {
  local exit_code=$?
  if [[ -n "$temporary_gh_config" && -d "$temporary_gh_config" ]]; then
    GH_CONFIG_DIR="$temporary_gh_config" gh auth logout --hostname github.com >/dev/null 2>&1 || true
    find "$temporary_gh_config" -depth -delete 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  }
}

require_command git
require_command gh
require_command curl

cd "$REPOSITORY_ROOT"
temporary_gh_config="$(mktemp -d /tmp/pbr-vehicle-gh.XXXXXX)"
export GH_CONFIG_DIR="$temporary_gh_config"
export GH_NO_UPDATE_NOTIFIER=1

# Optional, process-local outbound proxy. This is intended for a temporary
# reverse SOCKS tunnel such as ssh -R 127.0.0.1:17891 <server>. Use socks5,
# not socks5h: the installed GitHub CLI supports the former proxy scheme.
if [[ -n "${PBR_VEHICLE_PROXY:-}" ]]; then
  export HTTP_PROXY="$PBR_VEHICLE_PROXY"
  export HTTPS_PROXY="$PBR_VEHICLE_PROXY"
  export ALL_PROXY="$PBR_VEHICLE_PROXY"
  export http_proxy="$PBR_VEHICLE_PROXY"
  export https_proxy="$PBR_VEHICLE_PROXY"
  export all_proxy="$PBR_VEHICLE_PROXY"
fi

printf '%s\n' 'On your local computer, open https://github.com/login/device in a browser.'
printf '%s\n' 'GitHub CLI will print a one-time code below. Enter that code in the local browser; do not enter it on this server.'

# Fail before showing an unusable local-browser instruction when this SSH shell
# cannot reach GitHub. Proxy values are intentionally never printed.
if ! curl --silent --show-error --fail --connect-timeout 10 --max-time 20 \
  --output /dev/null https://github.com/login/device; then
  printf '%s\n' 'This server cannot currently reach GitHub over HTTPS.' >&2
  printf '%s\n' 'Configure a working proxy, or use PBR_VEHICLE_PROXY=socks5://127.0.0.1:<port>, then rerun.' >&2
  exit 1
fi

# BROWSER=echo prevents gh from trying to launch a browser on this remote server.
# Transient proxy routes can time out while requesting the code, so retry a
# small bounded number of times. Only a successfully issued code is displayed.
authenticated=false
for attempt in 1 2 3; do
  printf 'Requesting GitHub device code (attempt %s/3)...\n' "$attempt"
  if BROWSER=echo gh auth login --hostname github.com --git-protocol https --web --insecure-storage; then
    authenticated=true
    break
  fi
  if [[ "$attempt" -lt 3 ]]; then
    printf '%s\n' 'GitHub did not issue a device code. Retrying in 5 seconds...' >&2
    sleep 5
  fi
done
if [[ "$authenticated" != true ]]; then
  printf '%s\n' 'Unable to obtain a GitHub device code after three attempts.' >&2
  printf '%s\n' 'The server needs a working HTTPS route to github.com before it can upload.' >&2
  exit 1
fi

github_login="$(gh api user --jq .login)"
if [[ -z "$github_login" ]]; then
  printf '%s\n' 'GitHub login did not return an account name.' >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init --initial-branch="$BRANCH"
fi

current_branch="$(git branch --show-current)"
if [[ -z "$current_branch" ]]; then
  git checkout -B "$BRANCH"
elif [[ "$current_branch" != "$BRANCH" ]]; then
  printf 'Current branch is %s; expected %s. Switch branches before publishing.\n' "$current_branch" "$BRANCH" >&2
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  remote_url="$(git remote get-url origin)"
  if [[ "$remote_url" != "$REPOSITORY_URL" ]]; then
    printf 'Existing origin is %s, not %s. Refusing to change it automatically.\n' "$remote_url" "$REPOSITORY_URL" >&2
    exit 1
  fi
else
  git remote add origin "$REPOSITORY_URL"
fi

# Commit identity is local to .git and is never written to the shared account config.
git config user.name "$github_login"
git config user.email "${github_login}@users.noreply.github.com"

git add --all

large_files=()
while IFS= read -r -d '' path; do
  if (( $(stat --printf='%s' -- "$path") > MAX_FILE_BYTES )); then
    large_files+=("$path")
  fi
done < <(git ls-files -z)

if (( ${#large_files[@]} > 0 )); then
  printf '%s\n' 'Refusing to publish files above the 95 MiB safety limit:' >&2
  printf '  %s\n' "${large_files[@]}" >&2
  printf '%s\n' 'Move them to Git LFS or add an explicit ignore rule, then rerun.' >&2
  exit 1
fi

if ! git diff --cached --quiet; then
  git commit -m 'Publish pbr-vehicle delivery'
else
  printf '%s\n' 'No file changes to commit; pushing the current branch.'
fi

# The helper is process-local and reads only the temporary GH_CONFIG_DIR above.
git -c credential.helper='!gh auth git-credential' push --set-upstream origin "$BRANCH"

printf '\nPublished %s to %s as %s.\n' "$BRANCH" "$REPOSITORY_URL" "$github_login"
