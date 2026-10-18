"""Validate tip files against data/SPEC.md and optionally merge them into one bundle.

Usage:
    python onetip/pipeline/validate_tips.py                 # validate data/tips/*.json
    python onetip/pipeline/validate_tips.py path/a.json     # validate specific files
    python onetip/pipeline/validate_tips.py --bundle        # validate, then write data/tips.bundle.json
                                                            # and data/sources.bundle.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TIPS_DIR = DATA_DIR / "tips"
SPEC_PATH = DATA_DIR / "SPEC.md"
BUNDLE_PATH = DATA_DIR / "tips.bundle.json"
SOURCES_DIR = DATA_DIR / "sources"
SOURCES_BUNDLE_PATH = DATA_DIR / "sources.bundle.json"

CATEGORIES = {
    "language", "emotion", "discipline", "play", "selfcare",
    "sleep", "food", "safety", "social", "family",
}
REQUIRED = ("id", "category", "title", "body", "age_min_months", "age_max_months", "tags", "source_ids", "basis")
BODY_MIN, BODY_MAX = 100, 160
TITLE_MIN, TITLE_MAX = 4, 14
BASIS_MAX = 60
NEAR_DUP_RATIO = 0.6


def load_source_ids() -> set[str]:
    """Source ids are the backticked first column of the tables in SPEC.md section 5."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text.split("## 5.", 1)[1]
    return set(re.findall(r"^\| `([a-z0-9-]+)` \|", section, flags=re.M))


def build_sources_bundle(source_ids: set[str]) -> list[dict]:
    """App-facing source list: {id, name, url} for every id in SPEC.md, Chinese name preferred."""
    catalog: dict[str, dict] = {}
    for path in sorted(SOURCES_DIR.glob("*.json")):
        for src in json.loads(path.read_text(encoding="utf-8")).get("sources", []):
            catalog[src["id"]] = src
    missing = sorted(source_ids - catalog.keys())
    if missing:
        print(f"warning: no catalog entry in data/sources/ for {missing}")
    bundle = []
    for sid in sorted(source_ids):
        src = catalog.get(sid, {})
        name = src.get("name_zh") if src.get("region") != "CN" else src.get("name")
        urls = src.get("urls") or []
        bundle.append({"id": sid, "name": name or src.get("name") or sid, "url": urls[0] if urls else None})
    return bundle


def check_tip(tip: dict, source_ids: set[str]) -> list[str]:
    errs: list[str] = []
    missing = [k for k in REQUIRED if k not in tip]
    if missing:
        return [f"missing fields {missing}"]
    if tip["category"] not in CATEGORIES:
        errs.append(f"unknown category {tip['category']!r}")
    if not re.fullmatch(rf"{re.escape(str(tip['category']))}-\d{{3}}", str(tip["id"])):
        errs.append("id must be <category>-NNN")
    if not TITLE_MIN <= len(tip["title"]) <= TITLE_MAX:
        errs.append(f"title length {len(tip['title'])} not in [{TITLE_MIN},{TITLE_MAX}]")
    if not BODY_MIN <= len(tip["body"]) <= BODY_MAX:
        errs.append(f"body length {len(tip['body'])} not in [{BODY_MIN},{BODY_MAX}]")
    if not tip["basis"] or len(tip["basis"]) > BASIS_MAX:
        errs.append(f"basis length {len(tip['basis'])} not in [1,{BASIS_MAX}]")
    lo, hi = tip["age_min_months"], tip["age_max_months"]
    if not (isinstance(lo, int) and isinstance(hi, int) and 24 <= lo < hi <= 72):
        errs.append(f"bad age range [{lo},{hi}]")
    elif hi < 35:
        errs.append("age range ends before 35 months")
    if not (isinstance(tip["tags"], list) and 1 <= len(tip["tags"]) <= 4):
        errs.append("tags must be a list of 1-4 items")
    srcs = tip["source_ids"]
    if not (isinstance(srcs, list) and 1 <= len(srcs) <= 3):
        errs.append("source_ids must be a list of 1-3 items")
    else:
        unknown = [s for s in srcs if s not in source_ids]
        if unknown:
            errs.append(f"unknown source_ids {unknown}")
    return errs


def find_near_duplicates(tips: list[dict]) -> list[tuple[str, str, float]]:
    dups = []
    for i, a in enumerate(tips):
        for b in tips[i + 1:]:
            if a["title"] == b["title"]:
                dups.append((a["id"], b["id"], 1.0))
                continue
            ratio = SequenceMatcher(None, a["body"], b["body"]).quick_ratio()
            if ratio >= NEAR_DUP_RATIO:
                ratio = SequenceMatcher(None, a["body"], b["body"]).ratio()
                if ratio >= NEAR_DUP_RATIO:
                    dups.append((a["id"], b["id"], round(ratio, 2)))
    return dups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--bundle", action="store_true", help="write data/tips.bundle.json when valid")
    args = parser.parse_args()

    files = args.files or sorted(TIPS_DIR.glob("*.json"))
    source_ids = load_source_ids()
    all_tips: list[dict] = []
    n_errors = 0
    seen_ids: set[str] = set()

    for path in files:
        try:
            tips = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"{path}: invalid JSON: {e}")
            n_errors += 1
            continue
        for tip in tips:
            errs = check_tip(tip, source_ids)
            if tip.get("id") in seen_ids:
                errs.append("duplicate id")
            seen_ids.add(tip.get("id"))
            for e in errs:
                print(f"{path.name}:{tip.get('id')}: {e}")
            n_errors += len(errs)
        all_tips.extend(tips)

    dups = find_near_duplicates(all_tips)
    for a, b, r in dups:
        print(f"near-duplicate: {a} ~ {b} ({r})")

    by_cat: dict[str, int] = {}
    for t in all_tips:
        by_cat[t.get("category", "?")] = by_cat.get(t.get("category", "?"), 0) + 1
    lengths = [len(t.get("body", "")) for t in all_tips] or [0]
    print(f"\n{len(all_tips)} tips, {n_errors} errors, {len(dups)} near-duplicates")
    print("by category:", json.dumps(by_cat, ensure_ascii=False))
    print(f"body length: min {min(lengths)}, max {max(lengths)}, avg {sum(lengths) / len(lengths):.0f}")

    if n_errors:
        return 1
    if args.bundle:
        all_tips.sort(key=lambda t: t["id"])
        BUNDLE_PATH.write_text(json.dumps(all_tips, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {BUNDLE_PATH.relative_to(DATA_DIR.parent)}")
        sources = build_sources_bundle(source_ids)
        SOURCES_BUNDLE_PATH.write_text(json.dumps(sources, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {SOURCES_BUNDLE_PATH.relative_to(DATA_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
