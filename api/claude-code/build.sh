#!/bin/sh
# Builds the belt binaries this app ships (bin/belt-linux-<arch>) from the cli
# worktree. Each app's bin is a symlink to the shared api/claude-code/bin, so
# one build serves all four apps; the packager dereferences it on deploy.
#   CLI_DIR  belt sources (default /home/ok/worktrees/cli-remote-llm-direct)
#   GOWORK   go.work for its modules (default /home/ok/worktrees/go.work)
#   ARCHES   GOARCH list (default "amd64 arm64")
set -e
# The script's real location (api/claude-code), through the app dir's symlink.
out="$(dirname "$(readlink -f "$0")")/bin"
mkdir -p "$out"
CLI_DIR=${CLI_DIR:-/home/ok/worktrees/cli-remote-llm-direct}
GOWORK=${GOWORK:-/home/ok/worktrees/go.work}
ARCHES=${ARCHES:-"amd64 arm64"}
VERSION=${VERSION:-$(git -C "$CLI_DIR" describe --tags --always 2>/dev/null || echo dev)}
for arch in $ARCHES; do
    echo "building belt-linux-$arch ($VERSION) -> $out"
    (cd "$CLI_DIR" && GOWORK="$GOWORK" CGO_ENABLED=0 GOOS=linux GOARCH="$arch" \
        go build -trimpath -ldflags "-s -w -X 'inference.sh/cli/internal/version.Version=$VERSION'" \
        -o "$out/belt-linux-$arch" .)
done
ls -la "$out"
