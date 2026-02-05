#!/usr/bin/env python3
"""Sync models.json with fal.ai API. Preserves implemented/direct flags."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.fal.ai/v1/models"
DB_PATH = Path(__file__).parent / "models.json"

# Direct integration patterns
DIRECT_PREFIXES = ["google", "xai", "bytedance"]
DIRECT_CONTAINS = ["veo", "nano-banana"]


def is_direct(endpoint_id: str) -> bool:
    eid = endpoint_id.lower()
    if any(eid.startswith(p) or f"/{p}" in eid for p in DIRECT_PREFIXES):
        return True
    if any(p in eid for p in DIRECT_CONTAINS):
        return True
    return False


def fetch_page(url: str, retries: int = 5) -> dict:
    """Fetch a single page with retry logic."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return {}


def fetch_all_models() -> list[dict]:
    """Fetch all models from fal.ai API with pagination."""
    models = []
    cursor = None
    page = 1

    while True:
        url = f"{API_URL}?limit=100"
        if cursor:
            url += f"&cursor={cursor}"

        data = fetch_page(url)
        batch = data.get("models", [])
        models.extend(batch)
        print(f"  Page {page}: {len(batch)} models (total: {len(models)})")

        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        page += 1
        time.sleep(1)

    return models


def load_existing() -> dict[str, dict]:
    """Load existing flags from database."""
    if not DB_PATH.exists():
        return {}
    with open(DB_PATH) as f:
        return {
            m["endpoint_id"]: {"implemented": m.get("implemented", False), "direct": m.get("direct", False)}
            for m in json.load(f)
        }


def main():
    print("Fetching models from fal.ai API...")
    raw_models = fetch_all_models()

    print("Merging with existing database...")
    existing = load_existing()

    models = []
    for m in raw_models:
        eid = m["endpoint_id"]
        prev = existing.get(eid, {})
        models.append({
            "endpoint_id": eid,
            "category": m["metadata"].get("category", "unknown"),
            "display_name": m["metadata"].get("display_name", ""),
            "status": m["metadata"].get("status", "unknown"),
            "implemented": prev.get("implemented", False),
            "direct": prev.get("direct", False) or is_direct(eid),
        })

    models.sort(key=lambda x: x["endpoint_id"])

    # Stats
    new_models = [m for m in models if m["endpoint_id"] not in existing]
    impl_count = sum(1 for m in models if m["implemented"])
    direct_count = sum(1 for m in models if m["direct"])

    print(f"  Total: {len(models)}")
    print(f"  New: {len(new_models)}")
    print(f"  Implemented: {impl_count}")
    print(f"  Direct: {direct_count}")

    if new_models:
        print("\nNew models:")
        for m in new_models:
            print(f"  + {m['endpoint_id']} ({m['category']})")

    with open(DB_PATH, "w") as f:
        json.dump(models, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
