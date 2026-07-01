"""Small Streamlit sandbox for ranking candidate samples."""
import json
import os
import shutil
import sys
import tempfile

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank import rank_candidates
from src.load import load_candidates

MAX_CANDIDATES = 100
LARGE_FILE_MB = 25


def _encoding_for(path):
    with open(path, "rb") as f:
        raw = f.read(4)
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    if raw[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        return "utf-32"
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    return "utf-8"


def write_first_jsonl_rows(src_path, out_path, limit):
    """Write up to limit valid JSONL objects without loading a large upload."""
    encoding = _encoding_for(src_path)
    rows = 0
    saw_data = False
    with open(src_path, "r", encoding=encoding, errors="replace") as src, open(
        out_path, "w", encoding="utf-8"
    ) as out:
        for line in src:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            saw_data = True
            if line.startswith("["):
                raise ValueError(
                    "Large JSON arrays must be sampled before upload. For large files, "
                    "upload JSONL so the app can stream the first 100 rows."
                )
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            rows += 1
            if rows >= limit:
                break
    if rows == 0:
        kind = "JSON array" if saw_data else "empty file"
        raise ValueError(f"No valid JSONL candidates found in this {kind}.")
    return rows


st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title('Redrob Candidate Ranker - sandbox (Team "AI is it")')
st.caption(
    "Upload candidates as JSONL or a small JSON array. The hosted sandbox ranks up to "
    "100 candidates; large JSONL uploads are sampled to the first 100 rows."
)

up = st.file_uploader("candidates sample (.jsonl / .json)", type=["jsonl", "json", "txt"])
if up is not None:
    tmp = tempfile.mkdtemp()
    raw_inp = os.path.join(tmp, "uploaded_candidates")
    inp = os.path.join(tmp, "candidates.jsonl")
    out = os.path.join(tmp, "submission.csv")
    upload_size_mb = up.size / (1024 * 1024)

    up.seek(0)
    with open(raw_inp, "wb") as f:
        shutil.copyfileobj(up, f)

    try:
        if upload_size_mb > LARGE_FILE_MB:
            rows_loaded = write_first_jsonl_rows(raw_inp, inp, MAX_CANDIDATES)
            st.info(
                f"Large upload detected ({upload_size_mb:.1f} MB). "
                f"Ranking the first {rows_loaded} valid JSONL candidates for this sandbox run."
            )
        else:
            candidates = load_candidates(raw_inp)
            if len(candidates) > MAX_CANDIDATES:
                st.info(
                    f"Uploaded {len(candidates)} candidates. "
                    f"Ranking the first {MAX_CANDIDATES} for this sandbox run."
                )
                candidates = candidates[:MAX_CANDIDATES]
            with open(inp, "w", encoding="utf-8") as f:
                for candidate in candidates:
                    f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

        with st.spinner("Ranking..."):
            rank_candidates(inp, out, topk=MAX_CANDIDATES, verbose=False)

        df = pd.read_csv(out)
        st.success(f"Ranked {len(df)} candidates.")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download ranked CSV",
            open(out, "rb"),
            file_name="submission.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Could not rank this file: {type(e).__name__}: {e}")
        st.info(
            "For full-dataset ranking, use "
            "`python rank.py --candidates ./candidates.jsonl --out ./submission.csv` locally. "
            "The hosted app is only a sandbox."
        )
else:
    st.write(
        "Tip: create a clean sample with "
        "`python -c \"import itertools; open('sample.jsonl','w',encoding='utf-8')"
        ".writelines(itertools.islice(open('candidates.jsonl',encoding='utf-8'),100))\"`"
    )
