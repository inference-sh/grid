#!/usr/bin/env python3
"""Fail when an app's requirements.lock or requirements.txt.compiled pins a
package outside the range its requirements.txt declares.

A lock left behind by a skipped redeploy installs versions the code was not
written for (e.g. inferencesh 0.7.31 under a >= 0.7.33 floor fails at import).

Checks inferencesh by default; --all checks every package with a specifier.

Also fails when any pinned file pins a package below SECURITY_FLOORS, whether
or not requirements.txt names it (transitive pins like certifi included).
Apps that cannot resolve a floor are listed in scripts/lock-floors-exceptions.txt
as "<app dir> <package>  # blocking constraint".

Usage: scripts/check-lock-floors.py [--all] [dir...]   (default dirs: api native)
"""
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PINNED = ("requirements.lock", "requirements.txt.compiled")

# Lowest release with the fix for an advisory the scanners report.
SECURITY_FLOORS = {
    "pillow": Version("13.0.0"),  # GHSA-v9pv-5gc8-w2m6, GHSA-5x8w-8jj5-q52h and the 13.0.0 advisories
    "certifi": Version("2026.10.23"),  # GHSA-6q48-6c7w-9gw2 (emSign roots)
    "pyjwt": Version("2.14.0"),  # GHSA-ffc3-869f-jxw9
    "brotli": Version("1.2.1"),  # GHSA-9v8h-qqgm-4m35
    "aiohttp": Version("3.14.4"),  # GHSA-2g87-pp6c-29x5 and the other 3.14.4 advisories
    "huggingface-hub": Version("1.32.0"),  # AIKIDO-2026-285573 and the path/credential advisories fixed through 1.32.0
    "anthropic": Version("1.10.0"),  # AIKIDO-2026-451947, -480854 (0.117.0), -859572 (1.10.0)
    "matplotlib": Version("3.10.9"),  # AIKIDO-2026-10536 (eval in axes.prop_cycle)
    "requests": Version("2.34.0"),  # AIKIDO-2026-106840, CVE-2026-25645
    "urllib3": Version("2.8.0"),  # CVE-2026-44431 (2.7.0), CVE-2026-97687/97688/97689 (2.8.0)
    "numpy": Version("2.4.5"),  # AIKIDO-2026-795744; the 2.5.3 fixes need Python 3.12
    "filelock": Version("4.2.1"),  # AIKIDO-2026-190770 etc.; CVE-2026-112909 etc. fixed in 4.2.1
}
EXCEPTIONS = "scripts/lock-floors-exceptions.txt"


def exceptions(root):
    out = set()
    path = root / EXCEPTIONS
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split("#", 1)[0].split()
            if len(parts) == 2:
                out.add((parts[0].rstrip("/"), canonicalize_name(parts[1])))
    return out


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
    allowed = exceptions(root)
    for d in dirs:
        for fname in PINNED:
            for pinned_file in sorted(d.rglob(fname)):
                parent = pinned_file.parent.resolve()
                app = parent.relative_to(root).as_posix() if parent.is_relative_to(root) else parent.as_posix()
                for name, v in pins(pinned_file).items():
                    floor = SECURITY_FLOORS.get(name)
                    if floor is not None and v < floor and (app, name) not in allowed:
                        print(f"{pinned_file}: {name}=={v} is below the security floor {floor}")
                        bad += 1
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
        print(f"{bad} pin(s) outside requirements.txt or below a security floor; "
              f"raise them or list the app in {EXCEPTIONS}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
