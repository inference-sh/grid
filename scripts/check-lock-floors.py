#!/usr/bin/env python3
"""Fail when an app's requirements.lock or requirements.txt.compiled pins a
package outside the range its requirements.txt declares.

A lock left behind by a skipped redeploy installs versions the code was not
written for (e.g. inferencesh 0.7.31 under a >= 0.7.33 floor fails at import).

Checks inferencesh by default; --all checks every package with a specifier.

Usage: scripts/check-lock-floors.py [--all] [dir...]   (default dirs: api native)
"""
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PINNED = ("requirements.lock", "requirements.txt.compiled")


def declared(path, only):
    reqs = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        if req.url or not req.specifier or (req.marker and not req.marker.evaluate()):
            continue
        name = canonicalize_name(req.name)
        if only is None or name == only:
            reqs[name] = req
    return reqs


def pins(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" not in line or line.startswith("-"):
            continue
        name, _, ver = line.partition("==")
        name = name.split("[", 1)[0].strip()
        try:
            out[canonicalize_name(name)] = Version(ver.split(";", 1)[0].strip())
        except InvalidVersion:
            continue
    return out


def main(argv):
    root = Path(__file__).resolve().parent.parent
    only = None if "--all" in argv else "inferencesh"
    dirs = [Path(a) for a in argv if a != "--all"] or [root / "api", root / "native"]
    bad = 0
    for d in dirs:
        for req_txt in sorted(d.rglob("requirements.txt")):
            reqs = declared(req_txt, only)
            if not reqs:
                continue
            for fname in PINNED:
                pinned_file = req_txt.parent / fname
                if not pinned_file.is_file():
                    continue
                pinned = pins(pinned_file)
                for name, req in reqs.items():
                    v = pinned.get(name)
                    if v is not None and not req.specifier.contains(v, prereleases=True):
                        print(f"{pinned_file}: {name}=={v} violates '{req.specifier}' in requirements.txt")
                        bad += 1
    if bad:
        print(f"{bad} pin(s) outside requirements.txt; redeploy those apps to regenerate their locks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
