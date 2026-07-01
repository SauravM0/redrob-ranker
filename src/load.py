"""Load candidates and build per-candidate evidence documents.

Robust to real-world file variants: UTF-8, UTF-8-with-BOM, UTF-16 (the Windows
PowerShell / Notepad default), gzip, and either JSONL (one object per line) or a
single pretty-printed JSON array (like sample_candidates.json). Bad lines are
skipped with a clear error only if NOTHING parses.
"""
import json, gzip

def _read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    # gzip magic
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    # BOM-based encoding detection
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return raw.decode("utf-32")
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # last resort: never crash on a stray byte
        return raw.decode("latin-1")

def load_candidates(path):
    text = _read_text(path).lstrip("\ufeff").strip()
    if not text:
        raise ValueError(f"{path} is empty.")
    # Case 1: a single JSON array (pretty-printed sample)
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON is not a list of candidates.")
        return data
    # Case 2: JSONL — one object per line
    rows, bad = [], 0
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if not rows:
        raise ValueError(
            f"No valid JSON found in {path}. If you made this sample on Windows, "
            f"re-save it as UTF-8 (PowerShell/Notepad often save UTF-16 or add a BOM). "
            f"Each line must be one candidate JSON object."
        )
    return rows

def evidence_doc(c):
    """Concatenate the text a recruiter would actually read."""
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""),
             p.get("current_title", ""), p.get("current_industry", "")]
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("industry", ""))
        parts.append(job.get("description", ""))
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    for e in c.get("education", []):
        parts.append(e.get("field_of_study", ""))
    return " ".join(x for x in parts if x).lower()

def career_text(c):
    return " ".join(j.get("description", "") for j in c.get("career_history", [])).lower()
