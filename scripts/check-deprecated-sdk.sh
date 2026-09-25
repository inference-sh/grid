#!/usr/bin/env bash
set -euo pipefail

# Fail when a file outside the baseline uses deprecated SDK surface:
#   - *CapabilityMixin / ReasoningMixin / ToolCallsMixin (no-ops since sdk-py b7cd3e7;
#     subclass LLMInput/LLMOutput from inferencesh.models.llm directly)
#   - setup(self, ..., metadata) (BaseApp.setup() takes no metadata)
# Run from anywhere:  scripts/check-deprecated-sdk.sh [--update]
# --update rewrites the baseline to the current offenders (use after cleanups only).

cd "$(dirname "$0")/.."

pattern='CapabilityMixin|ReasoningMixin|ToolCallsMixin|def setup\(self,[^)]*\bmetadata\b'
baseline=scripts/deprecated-sdk-baseline.txt

current=$(git grep --untracked -lE "$pattern" -- '*.py' '*.sh' ':!.claude' ':!scripts/check-deprecated-sdk.sh' | sort)

if [ "${1:-}" = "--update" ]; then
    printf '%s\n' "$current" > "$baseline"
    echo "baseline: $(wc -l < "$baseline") files"
    exit 0
fi

new=$(comm -23 <(printf '%s\n' "$current") <(sort "$baseline"))
fixed=$(comm -13 <(printf '%s\n' "$current") <(sort "$baseline"))

if [ -n "$fixed" ]; then
    echo "no longer offending (run with --update to drop from the baseline):"
    printf '  %s\n' $fixed
fi

if [ -n "$new" ]; then
    echo "deprecated SDK mixins or setup(self, metadata) in files outside the baseline:" >&2
    for f in $new; do
        git grep --untracked -nE "$pattern" -- "$f" | sed 's/^/  /' >&2
    done
    exit 1
fi
echo "ok"
