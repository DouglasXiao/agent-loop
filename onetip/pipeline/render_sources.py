"""Render data/SOURCES.md (human-readable catalog) from data/sources/*.json.

Usage: python onetip/pipeline/render_sources.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "SOURCES.md"

REUSE_LABEL = {"can_adapt": "✅ 可改写引用", "cite_only": "🔗 仅参考+链接", "book": "📕 书籍，仅参考"}
TIER_LABEL = {"A": "A 政府/国际组织/学术", "B": "B 循证项目/权威机构", "C": "C 书籍/商业科普"}
VERIFY_LABEL = {
    "verified": "✅ 已打开原页核实",
    "search_index": "✅ 搜索索引中可见，要点与摘录一致",
    "partial": "🟡 部分核实",
    "unverified": "⚪ 未核实（凭已有知识，待复核）",
}
CATEGORY_ZH = {
    "language": "语言", "emotion": "情绪", "discipline": "管教", "play": "游戏", "selfcare": "自理",
    "sleep": "睡眠", "food": "饮食", "safety": "安全", "social": "社交", "family": "家庭",
}


def cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def display_name(src: dict) -> str:
    if src.get("region") == "CN":
        return src.get("name", src["id"])
    zh = src.get("name_zh")
    return f"{src.get('name', src['id'])}（{zh}）" if zh else src.get("name", src["id"])


def render_entry(src: dict) -> list[str]:
    urls = src.get("urls") or []
    access = src.get("access") or {}
    access_bits = []
    if access.get("rss"):
        access_bits.append("RSS")
    if access.get("sitemap"):
        access_bits.append(f"[sitemap]({access['sitemap']})" if str(access["sitemap"]).startswith("http") else "sitemap")
    if access.get("api"):
        access_bits.append("API")
    robots = access.get("robots_ok")
    access_bits.append({True: "robots 允许", False: "robots 禁止", None: "robots 未确认"}.get(robots, "robots 未确认"))
    lines = [
        f"### {display_name(src)}",
        "",
        f"- **id**: `{src['id']}` · **发布方**: {src.get('publisher', '—')} · **层级**: {TIER_LABEL.get(src.get('tier'), src.get('tier', '—'))}",
        f"- **使用方式**: {REUSE_LABEL.get(src.get('reuse'), src.get('reuse', '—'))} — {src.get('license', '未确认')}",
        f"- **获取**: {' · '.join(access_bits)}",
        f"- **擅长**: {'、'.join(CATEGORY_ZH.get(c, c) for c in src.get('categories', [])) or '—'}",
    ]
    status = (src.get("verification") or {}).get("status")
    if status:
        lines.append(f"- **核实状态**: {VERIFY_LABEL.get(status, status)}")
    if urls:
        lines.append("- **链接**: " + " · ".join(f"[{i + 1}]({u})" for i, u in enumerate(urls)))
    if src.get("why"):
        lines.append(f"- **为什么值得加**: {src['why']}")
    points = src.get("key_points") or []
    if points:
        lines.append("- **和 3–4 岁相关的要点**:")
        lines.extend(f"  - {p}" for p in points)
    unverified = src.get("key_points_unverified") or []
    if unverified:
        lines.append("- **待核实要点**:")
        lines.extend(f"  - {p}" for p in unverified)
    if src.get("notes"):
        lines.append(f"- **备注**: {src['notes']}")
    lines.append("")
    return lines


def main() -> None:
    sources: list[dict] = []
    candidates: list[dict] = []
    for path in sorted(DATA.glob("sources/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sources.extend(data.get("sources", []))
        candidates.extend(data.get("candidates", []))

    reuse_counts: dict[str, int] = {}
    for s in sources:
        reuse_counts[s.get("reuse", "?")] = reuse_counts.get(s.get("reuse", "?"), 0) + 1

    out = [
        "# 数据源清单",
        "",
        "> 由 `pipeline/render_sources.py` 从 `data/sources/*.json` 生成，请改 JSON 再重新生成。",
        "",
        f"已收录 **{len(sources)}** 个来源（tip 的 `source_ids` 只能引用这些），另有 **{len(candidates)}** 个候选来源待评估。",
        "",
        "| 使用方式 | 数量 | 含义 |",
        "|---|---|---|",
        f"| {REUSE_LABEL['can_adapt']} | {reuse_counts.get('can_adapt', 0)} | 公有领域 / 国家机关官方文件 / 宽松许可：可抓取、可改写，注明出处 |",
        f"| {REUSE_LABEL['cite_only']} | {reuse_counts.get('cite_only', 0)} | 有版权的网站：不抓取正文，只作依据，用自己的话写并附链接 |",
        f"| {REUSE_LABEL['book']} | {reuse_counts.get('book', 0)} | 有版权的书：只参考思路 |",
        "",
        "核实状态说明：本次在云端沙箱里整理，网络策略挡住了直接打开大多数来源网页，"
        "核实主要靠搜索引擎索引和摘录；标 ⚪ 的条目是凭已有知识写的，上线前要逐条打开原页复核。",
        "",
        "## 总览",
        "",
        "| id | 来源 | 地区 | 层级 | 使用方式 | 核实 | 擅长 |",
        "|---|---|---|---|---|---|---|",
    ]
    order = {"A": 0, "B": 1, "C": 2}
    ranked = sorted(sources, key=lambda s: (s.get("region") != "CN", order.get(s.get("tier"), 9), s["id"]))
    for s in ranked:
        cats = "、".join(CATEGORY_ZH.get(c, c) for c in s.get("categories", []))
        out.append(
            f"| `{s['id']}` | {cell(display_name(s))} | {s.get('region', '')} | {s.get('tier', '')} "
            f"| {REUSE_LABEL.get(s.get('reuse'), s.get('reuse', ''))} "
            f"| {VERIFY_LABEL.get((s.get('verification') or {}).get('status'), '—')[:1]} | {cats} |"
        )
    out += ["", "## 详情", ""]
    for s in ranked:
        out += render_entry(s)
    if candidates:
        out += ["## 候选来源（尚未用于 tip）", ""]
        for c in sorted(candidates, key=lambda s: (s.get("region") != "CN", order.get(s.get("tier"), 9), s["id"])):
            out += render_entry(c)
    OUT.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(DATA.parent)}: {len(sources)} sources, {len(candidates)} candidates")


if __name__ == "__main__":
    main()
