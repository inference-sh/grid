#!/usr/bin/env bash
set -euo pipefail

# Replace copied helper files in app dirs with symlinks to the canonical originals.
# Run from grid/api/:  ../scripts/relink-helpers.sh [--force] [namespace...]
# No args = all namespaces.  --force overwrites diverged copies too.
#
# Handles three patterns:
#   1. Same-namespace:    bria/edit/bria_helper.py        -> ../bria_helper.py
#   2. Nested namespace:  bytedance/video/app/helper.py   -> ../../helper.py
#   3. Cross-namespace:   microsoft/app/fal_helper.py     -> ../../falai/fal_helper.py

cd "$(dirname "$0")/../api"

force=false
namespaces=()
for arg in "$@"; do
    case "$arg" in
        --force|-f) force=true ;;
        *) namespaces+=("$arg") ;;
    esac
done

# Build a lookup of all canonical helpers: filename -> path
# A canonical helper is a .py file that sits in a directory containing
# subdirectories with inf.yml files (i.e. it's a namespace-level file).
declare -A canonical
while IFS= read -r yml; do
    nsdir=$(dirname "$(dirname "$yml")")
    [ "$nsdir" = "." ] && continue
    for h in "$nsdir"/*.py; do
        [ -f "$h" ] || continue
        fname=$(basename "$h")
        [ "$fname" = "inference.py" ] || [ "$fname" = "__init__.py" ] && continue
        # Prefer the most specific (deepest) canonical version
        existing="${canonical[$fname]:-}"
        if [ -z "$existing" ] || [ "${#h}" -gt "${#existing}" ]; then
            canonical[$fname]="$h"
        fi
    done
done < <(find . -name 'inf.yml' -not -path './inf.yml')

# Also register all namespace-level helpers at every depth
while IFS= read -r helper; do
    fname=$(basename "$helper")
    [ "$fname" = "inference.py" ] || [ "$fname" = "__init__.py" ] && continue
    hdir=$(dirname "$helper")
    # Check if this dir has subdirs with inf.yml
    if find "$hdir" -mindepth 2 -maxdepth 2 -name 'inf.yml' -print -quit | grep -q .; then
        key="$hdir/$fname"
        canonical[$key]="$helper"
    fi
done < <(find . -name '*.py' ! -name 'inference.py' ! -name '__init__.py' -not -path '*/\.*')

linked=0
skipped=0
forced=0

# Process each app directory (identified by inf.yml)
while IFS= read -r yml; do
    appdir=$(dirname "$yml")

    # Filter by namespace if specified
    if [ ${#namespaces[@]} -gt 0 ]; then
        ns="${appdir%%/*}"
        match=false
        for n in "${namespaces[@]}"; do
            [ "$ns" = "$n" ] && match=true && break
        done
        $match || continue
    fi

    for target in "$appdir"/*.py; do
        [ -f "$target" ] || continue
        fname=$(basename "$target")
        [ "$fname" = "inference.py" ] || [ "$fname" = "__init__.py" ] && continue

        if [ -L "$target" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        # Find the canonical helper: walk parent dirs, then fall back to global lookup
        helper=""
        dir=$(dirname "$target")
        while [ "$dir" != "." ]; do
            parent=$(dirname "$dir")
            candidate="$parent/$fname"
            if [ -f "$candidate" ] && [ ! -L "$candidate" ] && [ "$candidate" != "$target" ]; then
                helper="$candidate"
                break
            fi
            dir="$parent"
        done

        # Cross-namespace fallback: search all namespace dirs for this filename
        if [ -z "$helper" ]; then
            while IFS= read -r candidate; do
                [ "$candidate" = "$target" ] && continue
                cdir=$(dirname "$candidate")
                # Must be a namespace dir (has subdirs with inf.yml)
                if find "$cdir" -mindepth 2 -maxdepth 2 -name 'inf.yml' -print -quit 2>/dev/null | grep -q .; then
                    helper="$candidate"
                    break
                fi
            done < <(find . -maxdepth 2 -name "$fname" -not -path "$target" ! -type l)
        fi

        [ -z "$helper" ] && continue

        rel=$(python3 -c "import os.path; print(os.path.relpath('$helper', '$(dirname "$target")'))")

        if ! cmp -s "$helper" "$target"; then
            if [ "$force" = true ]; then
                rm "$target"
                ln -s "$rel" "$target"
                echo "FORCE $target -> $rel"
                forced=$((forced + 1))
            else
                echo "SKIP $target (differs from $helper)"
                skipped=$((skipped + 1))
            fi
            continue
        fi

        rm "$target"
        ln -s "$rel" "$target"
        echo "LINK $target -> $rel"
        linked=$((linked + 1))
    done
done < <(find . -name 'inf.yml' -not -path './inf.yml' | sort)

echo ""
echo "Done: $linked linked, $forced forced, $skipped skipped"
