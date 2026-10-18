"""Harvest source pages and draft candidate tips from them.

Two steps, both resumable:

    python onetip/pipeline/harvest.py fetch             # download + extract text for reusable sources
    python onetip/pipeline/harvest.py draft [--limit N] # LLM turns cached text into draft tips

`fetch` only downloads sources whose `reuse` is `can_adapt` (public domain /
official documents / permissive licences) and whose robots.txt allows it.
Copyrighted sources (`cite_only`, `book`) are never scraped; they stay as
references that human-written tips cite.

`draft` writes to data/drafts/<source>.json. Drafts are NOT shipped: review,
edit, then move good ones into data/tips/<category>.json and run
validate_tips.py. LLM credentials use the same env vars as agent_loop.py
(OPENROUTER_API_KEY, else OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = ROOT / "data" / "sources"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
DRAFTS_DIR = ROOT / "data" / "drafts"
SPEC_PATH = ROOT / "data" / "SPEC.md"

USER_AGENT = "OneTipHarvester/0.1 (+personal parenting-tips app; contact via repo)"
REQUEST_INTERVAL_S = 1.5
CHUNK_CHARS = 2500
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg"}
BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "td", "dd", "dt", "blockquote", "br", "div"}


class TextExtractor(HTMLParser):
    """Keeps readable body text, drops navigation/scripts. Stdlib only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t　]+", " ", line).strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if len(line) > 1)


def load_sources() -> list[dict]:
    sources: list[dict] = []
    for path in sorted(SOURCES_DIR.glob("*.json")):
        sources.extend(json.loads(path.read_text(encoding="utf-8")).get("sources", []))
    return sources


def robots_allows(url: str, cache: dict[str, urllib.robotparser.RobotFileParser]) -> bool:
    parts = urlparse(url)
    base = f"{parts.scheme}://{parts.netloc}"
    if base not in cache:
        rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
        try:
            rp.read()
        except OSError:
            rp = None  # unreachable robots.txt: treat as allowed, like most crawlers
        cache[base] = rp
    rp = cache[base]
    return rp is None or rp.can_fetch(USER_AGENT, url)


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        html = resp.read().decode(charset, errors="replace")
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def cmd_fetch(args: argparse.Namespace) -> int:
    robots: dict[str, urllib.robotparser.RobotFileParser] = {}
    fetched = skipped = failed = 0
    for src in load_sources():
        if src.get("reuse") != "can_adapt":
            continue
        if args.only and src["id"] not in args.only:
            continue
        for url in src.get("urls", []):
            out = CACHE_DIR / src["id"] / (hashlib.sha1(url.encode()).hexdigest()[:12] + ".json")
            if out.exists() and not args.refresh:
                skipped += 1
                continue
            if not robots_allows(url, robots):
                print(f"robots.txt disallows {url}")
                skipped += 1
                continue
            try:
                text = fetch_text(url)
            except Exception as e:  # network errors are per-URL; keep going
                print(f"FAIL {src['id']} {url}: {e}")
                failed += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            record = {"source_id": src["id"], "url": url, "fetched_at": int(time.time()), "text": text}
            out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"ok   {src['id']} {url} ({len(text)} chars)")
            fetched += 1
            time.sleep(REQUEST_INTERVAL_S)
    print(f"\nfetched {fetched}, skipped {skipped}, failed {failed}")
    return 0 if not failed else 1


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    out, buf = [], ""
    for para in text.split("\n"):
        if len(buf) + len(para) > size and buf:
            out.append(buf)
            buf = ""
        buf += para + "\n"
    if buf.strip():
        out.append(buf)
    return out


def build_client():
    from openai import OpenAI

    or_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if or_key:
        base = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        model = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "openrouter/auto"
        return OpenAI(api_key=or_key, base_url=base), model
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")), os.getenv("OPENAI_MODEL", "gpt-5")


def draft_prompt(source: dict, excerpt: str) -> list[dict]:
    spec = SPEC_PATH.read_text(encoding="utf-8").split("## 5.", 1)[0]
    system = (
        "你是一位儿童发展与家庭教育内容编辑。严格按照下面的内容规范，"
        "从给定的权威资料片段中提炼适合 33–48 个月幼儿家长的 tip。\n\n" + spec
    )
    user = (
        f"资料来源 id: {source['id']}（{source.get('name_zh') or source.get('name')}）\n"
        f"资料片段：\n<<<\n{excerpt}\n>>>\n\n"
        "要求：只基于片段中明确支持的内容；片段里没有适合 33–48 个月的可操作建议就返回空数组；"
        "最多 3 条；用自己的话写，不照抄；source_ids 填 "
        f'["{source["id"]}"]；id 先用 "<category>-000" 占位。\n'
        '只输出 JSON：{"tips": [ ... ]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_tips(text: str) -> list[dict]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return []
    try:
        return list(json.loads(match.group(0)).get("tips", []))
    except json.JSONDecodeError:
        return []


def cmd_draft(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_tips import check_tip, load_source_ids

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    client, model = build_client()
    by_id = {s["id"]: s for s in load_sources()}
    source_ids = load_source_ids()
    calls = 0
    for cached in sorted(CACHE_DIR.glob("*/*.json")):
        record = json.loads(cached.read_text(encoding="utf-8"))
        source = by_id.get(record["source_id"])
        if not source or (args.only and source["id"] not in args.only):
            continue
        out = DRAFTS_DIR / f"{source['id']}.json"
        drafts = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        done = {d.get("_chunk") for d in drafts}
        for i, excerpt in enumerate(chunks(record["text"])):
            key = f"{cached.stem}:{i}"
            if key in done:
                continue
            if args.limit and calls >= args.limit:
                break
            resp = client.chat.completions.create(model=model, messages=draft_prompt(source, excerpt))
            calls += 1
            for tip in parse_tips(resp.choices[0].message.content or ""):
                problems = [e for e in check_tip(tip, source_ids) if not e.startswith("id must be")]
                drafts.append({**tip, "_chunk": key, "_url": record["url"], "_problems": problems})
            if not any(d.get("_chunk") == key for d in drafts):
                drafts.append({"_chunk": key, "_empty": True})  # remember chunks with nothing usable
            DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(drafts, ensure_ascii=False, indent=1), encoding="utf-8")
        n = sum(1 for d in drafts if not d.get("_empty"))
        print(f"{source['id']}: {n} drafts")
    print(f"\n{calls} LLM calls")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--only", nargs="*", help="source ids to fetch")
    p_fetch.add_argument("--refresh", action="store_true", help="re-download cached pages")
    p_draft = sub.add_parser("draft")
    p_draft.add_argument("--only", nargs="*", help="source ids to draft from")
    p_draft.add_argument("--limit", type=int, default=0, help="max LLM calls this run (0 = no limit)")
    args = parser.parse_args()
    return cmd_fetch(args) if args.cmd == "fetch" else cmd_draft(args)


if __name__ == "__main__":
    sys.exit(main())
