"""Build the single-file web prototype: web/dist/index.html.

Inlines picker.js and the content bundles into template.html, so the page
works offline and can be opened straight from a phone. Run
`python3 onetip/pipeline/validate_tips.py --bundle` first.
"""

from __future__ import annotations

import json
from pathlib import Path

WEB = Path(__file__).resolve().parent
DATA = WEB.parent / "data"
OUT = WEB / "dist" / "index.html"


def inline_json(path: Path) -> str:
    # Re-serialise compactly; "</" must not appear inside a <script> element.
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def main() -> None:
    html = (WEB / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__TIPS__*/", inline_json(DATA / "tips.bundle.json"))
    html = html.replace("/*__SOURCES__*/", inline_json(DATA / "sources.bundle.json"))
    html = html.replace("/*__PICKER__*/", (WEB / "picker.js").read_text(encoding="utf-8"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(WEB.parent)} ({len(html.encode()) // 1024} KB)")


if __name__ == "__main__":
    main()
