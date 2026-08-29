"""
create_chunks.py — Atomic chunk extractor for CATALOG_KNOWLEDGE.md
Version 2: Content-type auto-detection. No dependency on section titles.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# ─── PATHS ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_FILE = PROJECT_ROOT / "data" / "knowledge" / "CATALOG_KNOWLEDGE.md"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ─── HELPERS ───────────────────────────────────────────────────────

def make_chunk(chunk_id: str, title: str, content: str, **metadata) -> Dict[str, Any]:
    return {
        "id": chunk_id,
        "title": title,
        "content": content.strip(),
        "metadata": {"source": "CATALOG_KNOWLEDGE.md", **metadata}
    }


def clean_json_text(raw: str) -> str:
    """Fix common markdown JSON issues before parsing."""
    # Remove trailing commas before ] or }
    raw = re.sub(r",\s*([\]\}])", r"\1", raw)
    # Replace smart quotes
    raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return raw


def extract_json_blocks(text: str) -> List[Any]:
    """Extract ALL ```json ... ``` blocks and parse them."""
    pattern = r"```json\s*([\s\S]*?)\s*```"
    parsed = []
    for match in re.finditer(pattern, text):
        raw = match.group(1).strip()
        try:
            parsed.append(json.loads(clean_json_text(raw)))
        except json.JSONDecodeError as e:
            # Try to show context for debugging
            snippet = raw[:200].replace("\n", " ")
            print(f"  ⚠️  JSON parse error: {e} | Snippet: {snippet}...")
    return parsed


def detect_type(first_item: dict) -> str:
    """Auto-detect chunk type from JSON keys."""
    keys = set(first_item.keys())
    if "profile_reference" in keys:
        return "profile"
    if "spec_name" in keys:
        return "spec"
    if "question" in keys and "answer_summary" in keys:
        return "qa"
    if "id" in keys and "content" in keys and "keywords" in keys:
        return "rag"
    if "reference_code" in keys:
        return "accessory"
    if "name" in keys and "type" in keys and "compatible_profiles" in keys:
        return "accessory"
    if "family_name" in keys:
        return "family"
    return "unknown"


# ─── FORMATTERS ───────────────────────────────────────────────────

def format_profile(p: dict) -> dict:
    ref = p.get("profile_reference", "UNKNOWN")
    name = p.get("profile_name", "Unknown")
    series = p.get("series", "Unknown")
    category = p.get("category", "Unknown")
    dims = p.get("dimensions", {})
    dim_text = ", ".join(f"{k}={v}" for k, v in dims.items() if v and "Not" not in str(v))

    content = (
        f"Profile {ref}: {name}. Series: {series}. "
        f"Category: {category}. "
        f"Application: {p.get('application', 'N/A')}. "
        f"Dimensions: {dim_text or 'N/A'}. "
        f"Material: {p.get('material', 'N/A')}. "
        f"Finish: {p.get('finish_color', 'N/A')}. "
        f"Compatible with: {p.get('compatible_products', 'N/A')}. "
        f"Notes: {p.get('technical_notes', 'N/A')}. "
        f"Source page: {p.get('source_page', 'N/A')}."
    )

    # Extract refs from compatible_products for keyword boosting
    compat = str(p.get("compatible_products", ""))
    extra_refs = re.findall(r"\b\d{3,4}[A-Z]?\b", compat)

    keywords = [ref, series.replace(" ", ""), category.split("/")[0].strip()]
    keywords.extend(extra_refs)
    if "Coulifixe" in name:
        keywords.append("Coulifixe")

    return make_chunk(
        chunk_id=f"profile_{ref}",
        title=f"{ref} — {name}",
        content=content,
        category="profile",
        series=series,
        profile_ref=ref,
        keywords=list(set(keywords)),
        source_page=str(p.get("source_page", ""))
    )


def format_accessory(a: dict) -> dict:
    ref = a.get("reference_code", a.get("name", "UNKNOWN").replace(" ", "_")[:20])
    name = a.get("name", "Unknown")
    compat = str(a.get("compatible_profiles", ""))

    content = (
        f"Accessory {ref}: {name}. "
        f"Type: {a.get('type', 'N/A')}. "
        f"Purpose: {a.get('purpose', 'N/A')}. "
        f"Compatible profiles: {compat}. "
        f"Technical details: {a.get('technical_details', 'N/A')}. "
        f"Source page: {a.get('source_page', 'N/A')}."
    )

    # Detect series from compatible_profiles text
    series = []
    if "RUBIS 95" in compat or "95 Rubis" in compat:
        series.append("RUBIS 95")
    if "RUBIS 85" in compat or "85 Rubis" in compat:
        series.append("RUBIS 85")
    if not series:
        series.append("Both")

    # Extract numeric refs for keywords
    refs = re.findall(r"\b\d{3,4}[A-Z]?\b", ref + " " + compat)

    return make_chunk(
        chunk_id=f"accessory_{ref}",
        title=f"{ref} — {name}",
        content=content,
        category="accessory",
        series=", ".join(series),
        profile_ref=ref,
        keywords=list(set([ref, name.split()[0] if name else ""] + refs + ["accessory"])),
        source_page=str(a.get("source_page", ""))
    )


def format_spec(s: dict) -> dict:
    name = s.get("spec_name", "Unknown")
    value = s.get("value", "N/A")
    unit = s.get("unit", "")
    related = str(s.get("related_product", "N/A"))

    content = (
        f"Technical specification: {name}. "
        f"Value: {value} {unit}. "
        f"Applies to: {related}. "
        f"Note: {s.get('technical_note', 'N/A')}. "
        f"Source page: {s.get('source_page', 'N/A')}."
    )

    refs = re.findall(r"\b\d{3,4}[A-Z]?\b", related)

    return make_chunk(
        chunk_id=f"spec_{re.sub(r'[^a-z0-9_]', '_', name.lower())[:50]}",
        title=name,
        content=content,
        category="specification",
        series="Both",
        profile_ref=refs[0] if refs else "",
        keywords=[name.split()[0], value, unit] + refs,
        source_page=str(s.get("source_page", ""))
    )


def format_rag(r: dict) -> dict:
    """Use Section 11 RAG chunks directly, enriched."""
    cid = r.get("id", "rag_unknown")
    title = r.get("title", "Untitled")
    content = r.get("content", "")
    category = r.get("category", "general")

    # Detect series from content
    series = "Both"
    if "RUBIS 95" in content and "RUBIS 85" not in content:
        series = "RUBIS 95"
    elif "RUBIS 85" in content and "RUBIS 95" not in content:
        series = "RUBIS 85"

    # Extract profile refs from keywords and content
    refs = re.findall(r"\b\d{3,4}[A-Z]?\b", content + " " + " ".join(r.get("keywords", [])))

    return make_chunk(
        chunk_id=cid,
        title=title,
        content=content,
        category=category,
        series=series,
        profile_ref=", ".join(refs[:5]),
        keywords=r.get("keywords", []),
        source_page=str(r.get("source_page", ""))
    )


def format_qa(q: dict) -> dict:
    question = q.get("question", "")
    answer = q.get("answer_summary", "")
    content = f"Q: {question}\nA: {answer}"

    # Detect series
    series = "Both"
    if "RUBIS 95" in question + answer and "RUBIS 85" not in question + answer:
        series = "RUBIS 95"
    elif "RUBIS 85" in question + answer and "RUBIS 95" not in question + answer:
        series = "RUBIS 85"

    return make_chunk(
        chunk_id=f"qa_{re.sub(r'[^a-z0-9_]', '_', question.lower())[:40]}",
        title=question[:80],
        content=content,
        category="qa",
        series=series,
        profile_ref="",
        keywords=q.get("required_knowledge", []),
        source_page=""
    )


def format_family(f: dict) -> dict:
    name = f.get("family_name", "Unknown")
    series_list = f.get("available_series", ["Both"])
    series = ", ".join(series_list) if isinstance(series_list, list) else str(series_list)

    chars = f.get("main_characteristics", [])
    chars_text = " ".join(chars) if isinstance(chars, list) else str(chars)

    content = (
        f"Product family: {name}. "
        f"Series: {series}. "
        f"Purpose: {f.get('purpose', 'N/A')}. "
        f"Characteristics: {chars_text}. "
        f"Related pages: {f.get('related_pages', 'N/A')}."
    )

    return make_chunk(
        chunk_id=f"family_{re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]}",
        title=name,
        content=content,
        category="family",
        series=series,
        profile_ref="",
        keywords=[name.split()[0] if name else ""] + series_list,
        source_page=str(f.get("related_pages", ""))
    )


# ─── CUT-LIST TABLE PARSER ─────────────────────────────────────────

def parse_cutlist_tables(text: str) -> List[dict]:
    """Parse Section 8 markdown tables into individual config chunks."""
    chunks = []
    # Find each Table 8.X block
    blocks = re.split(r"(?=## Table 8\.\d+)", text)

    for block in blocks:
        block = block.strip()
        if not block or not block.startswith("## Table 8"):
            continue

        # Extract title line
        title_match = re.match(r"##\s+Table\s+8\.\d+\s+—\s+(.*)", block)
        title = title_match.group(1).strip() if title_match else "Cut List"

        # Detect series
        series = "RUBIS 95" if "RUBIS 95" in title else ("RUBIS 85" if "RUBIS 85" in title else "Unknown")

        # Detect dormant
        dorm_match = re.search(r"Dormant\s+(\d+)", title)
        dormant = dorm_match.group(1) if dorm_match else "Unknown"

        # Detect panel count
        panel_match = re.search(r"(\d+)-Vantaux", title, re.IGNORECASE)
        panels = panel_match.group(1) if panel_match else "Unknown"

        # Extract glazing formula
        glazing = ""
        for line in block.split("\n"):
            if "COTES DE VITRAGE:" in line or "GLAZING" in line.upper():
                glazing = line.strip()
                break

        # Build content
        content = f"Cut list configuration: {title}.\nSeries: {series}. Dormant: {dormant}. Panels: {panels}.\n{glazing}\n\nFull table:\n{block[:3500]}"

        chunks.append(make_chunk(
            chunk_id=f"cutlist_{panels}v_d{dormant}_{series.replace(' ', '').lower()}",
            title=title,
            content=content,
            category="cut_list",
            series=series,
            profile_ref=dormant,
            keywords=[dormant, f"{panels}vantaux", series.replace(" ", ""), "debitage", "cutlist"],
            source_page=re.search(r"page\s+(\d+)", title, re.IGNORECASE).group(1) if re.search(r"page\s+(\d+)", title, re.IGNORECASE) else ""
        ))

    return chunks


# ─── SAFETY CHUNKS ────────────────────────────────────────────────

def generate_safety_chunks() -> List[dict]:
    return [
        make_chunk(
            chunk_id="safety_24mm_85",
            title="SAFETY: 24mm Glazing NOT Compatible with RUBIS 85",
            content="RUBIS 85 maximum glazing thickness is 20mm. Gaskets 9520 and 9524 (20mm and 24mm) are EXCLUSIVELY for RUBIS 95. Using 24mm glazing in RUBIS 85 is NOT SUPPORTED and will cause sealing failure. Source: catalog page 4.",
            category="safety",
            series="RUBIS 85",
            profile_ref="9524",
            keywords=["24mm", "85Rubis", "incompatible", "maximum glazing", "safety", "limit", "9524", "9520"],
            source_page="4"
        ),
        make_chunk(
            chunk_id="safety_sash_85_in_95",
            title="SAFETY: RUBIS 85 Sash Profiles NOT for RUBIS 95",
            content="Sash profiles 856, 857, 858, 897, 898 are EXCLUSIVELY for RUBIS 85. For RUBIS 95, you MUST use 956, 957, 958, 997, 998, 999. Cross-series sash usage is incompatible. Source: catalog compatibility tables.",
            category="safety",
            series="Both",
            profile_ref="856,957",
            keywords=["856", "957", "incompatible", "cross-series", "safety", "sash", "ouvrant"],
            source_page="6A"
        ),
        make_chunk(
            chunk_id="safety_gasket_85_in_95",
            title="SAFETY: 85-Series Gaskets NOT for RUBIS 95",
            content="Glazing gaskets 8512, 8516, 8518 are EXCLUSIVELY for RUBIS 85 sash profiles. For RUBIS 95, you MUST use 9512, 9516, 9518, 9520, 9524. Source: catalog page 13.",
            category="safety",
            series="Both",
            profile_ref="8512,9512",
            keywords=["8512", "9512", "incompatible", "gasket", "safety", "joint"],
            source_page="13"
        ),
        make_chunk(
            chunk_id="safety_hook_85",
            title="SAFETY: FAPIM Hook 7029 NOT for RUBIS 85",
            content="FAPIM hook 7029 is for RUBIS 95. For RUBIS 85, you MUST use hook 7028. Source: catalog pages 11, 28-43.",
            category="safety",
            series="RUBIS 85",
            profile_ref="7028,7029",
            keywords=["7029", "7028", "FAPIM", "hook", "incompatible", "safety", "crochet"],
            source_page="11"
        ),
        make_chunk(
            chunk_id="safety_versus_85",
            title="SAFETY: VERSUS Hook 9033 NOT for RUBIS 85",
            content="VERSUS hook 9033 is for RUBIS 95. For RUBIS 85, you MUST use VERSUS hook 9024. Source: catalog page 11.",
            category="safety",
            series="RUBIS 85",
            profile_ref="9024,9033",
            keywords=["9033", "9024", "VERSUS", "incompatible", "safety"],
            source_page="11"
        ),
        make_chunk(
            chunk_id="safety_dormant_950_85",
            title="SAFETY: RUBIS 95 Frame Profiles NOT for RUBIS 85 Sashes",
            content="Frame profiles 950, 951, 952 are designed for RUBIS 95 (60mm pre-frame, 33mm sash). While dormant 851 is shared between series, 950/951/952 require RUBIS 95 sash profiles. Do not pair 950/951 with 856/857/858 sashes.",
            category="safety",
            series="RUBIS 95",
            profile_ref="950,951,952",
            keywords=["950", "951", "incompatible", "safety", "frame", "dormant"],
            source_page="2,5"
        ),
    ]


# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    print("Reading CATALOG_KNOWLEDGE.md...")
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    all_chunks: List[dict] = []

    # ── Step 1: Extract all JSON blocks and auto-classify ─────────
    print("Extracting JSON blocks...")
    json_blocks = extract_json_blocks(text)
    print(f"  Found {len(json_blocks)} JSON blocks")

    for block in json_blocks:
        if isinstance(block, list) and len(block) > 0:
            item_type = detect_type(block[0])
            print(f"  Processing list of {item_type} ({len(block)} items)")
            for item in block:
                if item_type == "profile":
                    all_chunks.append(format_profile(item))
                elif item_type == "accessory":
                    all_chunks.append(format_accessory(item))
                elif item_type == "spec":
                    all_chunks.append(format_spec(item))
                elif item_type == "rag":
                    all_chunks.append(format_rag(item))
                elif item_type == "qa":
                    all_chunks.append(format_qa(item))
                elif item_type == "family":
                    all_chunks.append(format_family(item))
                else:
                    # Unknown: store as generic
                    all_chunks.append(make_chunk(
                        chunk_id=f"unknown_{len(all_chunks)}",
                        title="Unknown JSON item",
                        content=json.dumps(item, ensure_ascii=False)[:2000],
                        category="unknown",
                        series="Both"
                    ))
        elif isinstance(block, dict):
            item_type = detect_type(block)
            if item_type == "rag":
                all_chunks.append(format_rag(block))
            elif item_type == "profile":
                all_chunks.append(format_profile(block))
            else:
                all_chunks.append(make_chunk(
                    chunk_id=f"dict_{len(all_chunks)}",
                    title="Dict item",
                    content=json.dumps(block, ensure_ascii=False)[:2000],
                    category="unknown",
                    series="Both"
                ))

    # ── Step 2: Parse markdown cut-list tables ───────────────────
    print("Parsing cut-list tables...")
    cutlist_chunks = parse_cutlist_tables(text)
    all_chunks.extend(cutlist_chunks)
    print(f"  → {len(cutlist_chunks)} cut-list chunks")

    # ── Step 3: Add compatibility matrix as a chunk ────────────────
    compat_match = re.search(r"(## SECTION 6.*?)(?=## SECTION 7|$)", text, re.DOTALL)
    if compat_match:
        all_chunks.append(make_chunk(
            chunk_id="compatibility_matrix",
            title="Compatibility Matrix — RUBIS 85 vs 95",
            content=compat_match.group(1)[:4000],
            category="compatibility",
            series="Both",
            keywords=["compatibility", "85Rubis", "95Rubis", "matrix"],
            source_page="6A,6B,6C"
        ))

    # ── Step 4: Add assembly guide ───────────────────────────────
    assembly_match = re.search(r"(## SECTION 7.*?)(?=## SECTION 8|$)", text, re.DOTALL)
    if assembly_match:
        all_chunks.append(make_chunk(
            chunk_id="assembly_guide",
            title="Assembly and Installation Guide",
            content=assembly_match.group(1)[:4000],
            category="assembly",
            series="Both",
            keywords=["assembly", "cutting", "formula", "debitage", "45", "90", "mitre"],
            source_page="3,14-43"
        ))

    # ── Step 5: Add safety chunks ────────────────────────────────
    safety = generate_safety_chunks()
    all_chunks.extend(safety)
    print(f"  → {len(safety)} safety chunks")

    # ── Step 6: Deduplicate by ID ────────────────────────────────
    seen = set()
    unique = []
    for c in all_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    all_chunks = unique

    # ── Save ─────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Created {len(all_chunks)} atomic chunks")
    print(f"   Saved to: {OUTPUT_FILE}")

    # ── Stats ────────────────────────────────────────────────────
    cats = Counter(c["metadata"]["category"] for c in all_chunks)
    print("\nBreakdown by category:")
    for cat, count in cats.most_common():
        print(f"   {cat:15s}: {count:3d}")

    sizes = [len(c["content"]) for c in all_chunks]
    print(f"\nContent size stats:")
    print(f"   Average: {sum(sizes)/len(sizes):.0f} chars")
    print(f"   Median:  {sorted(sizes)[len(sizes)//2]:.0f} chars")
    print(f"   Max:     {max(sizes)} chars")
    print(f"   Min:     {min(sizes)} chars")

    big = [c for c in all_chunks if len(c["content"]) > 4000]
    if big:
        print(f"\n⚠️  {len(big)} chunks >4000 chars:")
        for c in big[:5]:
            print(f"   - {c['id']} ({len(c['content'])} chars)")

    # ── Sanity checks ────────────────────────────────────────────
    profiles = [c for c in all_chunks if c["metadata"]["category"] == "profile"]
    print(f"\nSanity: {len(profiles)} profile chunks")
    if profiles:
        refs = [c["metadata"]["profile_ref"] for c in profiles]
        print(f"   Sample refs: {refs[:5]}")

    safety_count = len([c for c in all_chunks if c["metadata"]["category"] == "safety"])
    print(f"Sanity: {safety_count} safety chunks")

    rag_count = len([c for c in all_chunks if c["metadata"]["category"] == "rag"])
    print(f"Sanity: {rag_count} RAG chunks")


if __name__ == "__main__":
    main()