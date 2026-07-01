"""Sandbox demo (Streamlit / HuggingFace Spaces). Runs the ranker IN-PROCESS on a
small uploaded sample (<=100) and returns a ranked CSV. CPU-only, no network.
Runs the ranking function directly (no subprocess), so errors are visible and it
works on Streamlit Cloud / HF Spaces without relying on a 'python' on PATH."""
import os, sys, tempfile
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank import rank_candidates

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob Candidate Ranker — sandbox (Team \"AI is it\")")
st.caption("Upload a small candidates sample (JSONL or JSON array, ≤100). Runs CPU-only, no network. "
           "Any encoding (UTF-8/UTF-16/BOM) is handled automatically.")

up = st.file_uploader("candidates sample (.jsonl / .json)", type=["jsonl", "json", "txt"])
if up is not None:
    tmp = tempfile.mkdtemp()
    inp = os.path.join(tmp, "candidates.jsonl")
    out = os.path.join(tmp, "submission.csv")
    with open(inp, "wb") as f:            # write raw bytes; loader handles encoding
        f.write(up.getvalue())
    try:
        with st.spinner("Ranking…"):
            rows = rank_candidates(inp, out, topk=100, verbose=False)
        df = pd.read_csv(out)
        st.success(f"Ranked {len(df)} candidates.")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download ranked CSV", open(out, "rb"),
                           file_name="submission.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Could not rank this file: {type(e).__name__}: {e}")
        st.info("If you built the sample on Windows, re-save it as UTF-8 — though this app "
                "should now handle UTF-16/BOM too. Each line must be one candidate JSON object, "
                "or the file can be a single JSON array of candidates.")
else:
    st.write("Tip: create a clean sample with "
             "`python -c \"import itertools; open('sample.jsonl','w',encoding='utf-8')"
             ".writelines(itertools.islice(open('candidates.jsonl',encoding='utf-8'),100))\"`")
