"""
monitor.py

Monitoring dashboard for the Finance Document Assistant.
Reads feedback.jsonl and displays 5 charts.

Run with:
    uv run streamlit run monitor.py
"""

import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime

# ── Config ──────────────────────────────────────────────────
FEEDBACK_FILE     = "data/feedback.jsonl"
EVAL_RESULTS_PATH = "data/ground_truth/eval_results.csv"

# ── Page setup ───────────────────────────────────────────────
st.set_page_config(
    page_title="Finance Assistant — Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Finance Assistant — Monitoring Dashboard")
st.caption("Live feedback and usage metrics from the app.")


# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════

def load_feedback() -> pd.DataFrame:
    """Load feedback from JSONL file into a DataFrame."""
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()

    records = []
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    df["hour"]      = df["timestamp"].dt.hour
    df["positive"]  = df["feedback"] == "positive"

    return df


def load_eval_results() -> pd.DataFrame:
    """Load evaluation results if available."""
    if not os.path.exists(EVAL_RESULTS_PATH):
        return pd.DataFrame()
    return pd.read_csv(EVAL_RESULTS_PATH)


# ══════════════════════════════════════════════════════════
# GENERATE SAMPLE DATA if no real feedback yet
# ══════════════════════════════════════════════════════════

def generate_sample_feedback() -> pd.DataFrame:
    """
    Generate sample feedback data for demo purposes.
    Remove this once you have real user feedback.
    """
    import random
    from datetime import timedelta

    questions = [
        "What is the IBAN number?",
        "What are the payment terms?",
        "How much is the total?",
        "What happens if I pay late?",
        "Who is this invoice billed to?",
        "What is the VAT number?",
        "When is payment due?",
    ]

    records = []
    base_time = datetime.now()

    for i in range(30):
        ts       = base_time - timedelta(hours=random.randint(0, 72))
        question = random.choice(questions)
        feedback = "positive" if random.random() > 0.2 else "negative"

        records.append({
            "timestamp": ts.isoformat(),
            "question":  question,
            "answer":    "Sample answer",
            "sources":   [{"filename": "1.pdf", "page": 1, "score": round(random.uniform(0.3, 0.9), 3)}],
            "feedback":  feedback
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    df["hour"]      = df["timestamp"].dt.hour
    df["positive"]  = df["feedback"] == "positive"

    return df


# ══════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════

df = load_feedback()

if df.empty:
    st.info("No real feedback yet — showing sample data for demo.")
    df = generate_sample_feedback()

eval_df = load_eval_results()

# ══════════════════════════════════════════════════════════
# TOP METRICS ROW
# ══════════════════════════════════════════════════════════

st.divider()
col1, col2, col3, col4 = st.columns(4)

total     = len(df)
positive  = df["positive"].sum()
negative  = total - positive
pos_rate  = round(positive / total * 100, 1) if total > 0 else 0

col1.metric("Total queries",     total)
col2.metric("Positive feedback", int(positive))
col3.metric("Negative feedback", int(negative))
col4.metric("Satisfaction rate", f"{pos_rate}%")

st.divider()

# ══════════════════════════════════════════════════════════
# CHART 1 — Query volume over time
# ══════════════════════════════════════════════════════════

st.subheader("📈 Chart 1 — Query Volume Over Time")

volume_by_date = (
    df.groupby("date")
    .size()
    .reset_index(name="queries")
)
volume_by_date["date"] = pd.to_datetime(volume_by_date["date"])

st.line_chart(volume_by_date.set_index("date")["queries"])

# ══════════════════════════════════════════════════════════
# CHART 2 — Positive vs Negative feedback
# ══════════════════════════════════════════════════════════

st.subheader("👍 Chart 2 — Feedback Distribution")

feedback_counts = df["feedback"].value_counts().reset_index()
feedback_counts.columns = ["feedback", "count"]

st.bar_chart(feedback_counts.set_index("feedback"))

# ══════════════════════════════════════════════════════════
# CHART 3 — Most common questions
# ══════════════════════════════════════════════════════════

st.subheader("❓ Chart 3 — Most Common Questions")

top_questions = (
    df["question"]
    .value_counts()
    .head(7)
    .reset_index()
)
top_questions.columns = ["question", "count"]

st.bar_chart(top_questions.set_index("question"))

# ══════════════════════════════════════════════════════════
# CHART 4 — Feedback trend over time
# ══════════════════════════════════════════════════════════

st.subheader("📉 Chart 4 — Satisfaction Rate Over Time")

trend = (
    df.groupby("date")["positive"]
    .mean()
    .reset_index()
)
trend.columns     = ["date", "satisfaction_rate"]
trend["date"]     = pd.to_datetime(trend["date"])
trend["satisfaction_rate"] = (trend["satisfaction_rate"] * 100).round(1)

st.line_chart(trend.set_index("date")["satisfaction_rate"])
st.caption("Satisfaction rate = % of positive feedback per day")

# ══════════════════════════════════════════════════════════
# CHART 5 — Source document usage
# ══════════════════════════════════════════════════════════

st.subheader("📄 Chart 5 — Source Document Usage")

# extract filename from sources
def get_filename(sources):
    try:
        if isinstance(sources, str):
            sources = json.loads(sources.replace("'", '"'))
        return sources[0]["filename"]
    except:
        return "unknown"

df["source_file"] = df["sources"].apply(get_filename)

source_counts = (
    df["source_file"]
    .value_counts()
    .reset_index()
)
source_counts.columns = ["filename", "queries"]

st.bar_chart(source_counts.set_index("filename"))

# ══════════════════════════════════════════════════════════
# EVAL RESULTS TABLE
# ══════════════════════════════════════════════════════════

if not eval_df.empty:
    st.divider()
    st.subheader("🧪 Answer Quality Evaluation")

    counts = eval_df["relevance"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("RELEVANT",        counts.get("RELEVANT", 0))
    c2.metric("PARTLY_RELEVANT", counts.get("PARTLY_RELEVANT", 0))
    c3.metric("NOT_RELEVANT",    counts.get("NOT_RELEVANT", 0))

    with st.expander("View full evaluation results"):
        st.dataframe(eval_df[["question", "answer", "relevance", "explanation"]])

# ══════════════════════════════════════════════════════════
# RAW FEEDBACK TABLE
# ══════════════════════════════════════════════════════════

st.divider()
with st.expander("📋 View raw feedback log"):
    display_df = df[["timestamp", "question", "feedback"]].sort_values(
        "timestamp", ascending=False
    )
    st.dataframe(display_df)

# ── Auto refresh every 30 seconds ──────────────────────────
st.caption("Dashboard auto-refreshes every 30 seconds.")
st.markdown(
    '<meta http-equiv="refresh" content="30">',
    unsafe_allow_html=True
)