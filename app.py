# app.py
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



# ---------- Paths ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "entries.csv"


analyzer = SentimentIntensityAnalyzer()

COLUMNS = [
    "timestamp", "date", "time",
    "mood", "energy", "sleep_hours", "stress",
    "tags", "journal", "sentiment_compound"  # <— NEW
]


# ---------- Helpers ----------
def load_entries() -> pd.DataFrame:
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            missing = [c for c in COLUMNS if c not in df.columns]
            if missing:
                for m in missing:
                    df[m] = None
                df = df[COLUMNS]
            return df
        except Exception:
            backup = CSV_PATH.with_suffix(".bak.csv")
            CSV_PATH.rename(backup)
            return pd.DataFrame(columns=COLUMNS)
    else:
        return pd.DataFrame(columns=COLUMNS)

def save_entry(row: dict):
    df = load_entries()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

# ---------- UI ----------
st.set_page_config(page_title="Health Mood Tracker", page_icon="🧠", layout="centered")
st.title("🧠 Health Mood Tracker")
st.markdown("""
Welcome to Health Mood Tracker!  
This app helps you understand how daily factors like sleep, energy, and stress affect your mood.  
Record short entries each day, and the app will visualize your trends and highlight meaningful insights over time.
""")

st.caption("Version 0.1 · Core Features – Input, Save & Preview")


with st.form("daily_checkin", clear_on_submit=True):
    st.subheader("Daily Check-in")
    mood = st.slider("Mood (1–10)", 1, 10, 7)
    energy = st.slider("Energy (1–10)", 1, 10, 7)
    sleep_hours = st.number_input("Sleep (hours)", min_value=0.0, step=0.1, value=7.0, help="Enter your total sleep in hours")
    stress = st.slider("Stress (1–5)", 1, 5, 2)
    tags = st.text_input("Tags (comma-separated, optional)", placeholder="exam, gym, coffee", help="Use short keywords to describe your day. They’ll be used later to find patterns")
    journal = st.text_area("Journal (optional)", placeholder="One or two lines about your day…",     help="A short reflection or summary of your day. The app analyzes this text for sentiment (positive/negative tone).")

    submitted = st.form_submit_button("💾 Save Entry")
    if submitted:
        # Basic validation
        if sleep_hours < 0:
            st.error("Sleep hours cannot be negative.")
        else:
            now = datetime.now()
            text = journal.strip()
            sent = analyzer.polarity_scores(text)["compound"] if text else None
            row = {
                "timestamp": now.isoformat(timespec="seconds"),
                "date": now.date().isoformat(),
                "time": now.strftime("%H:%M:%S"),
                "mood": int(mood),
                "energy": int(energy),
                "sleep_hours": float(sleep_hours),
                "stress": int(stress),
                "tags": tags.strip(),
                "journal": text,
                "sentiment_compound": sent,
            }
            save_entry(row)
            st.success("Entry saved!")

st.divider()

st.subheader("Recent Entries")
df = load_entries()
if df.empty:
    st.info("No entries yet. Add your first entry above.")
else:
    show = df.sort_values("timestamp", ascending=False).head(10)
    st.dataframe(show, use_container_width=True)

st.divider()
st.subheader("📈 Mood Trend")

if df.empty:
    st.info("No data yet to plot. Add a few entries first.")
else:
    df_plot = df.copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"])
    df_plot = df_plot.sort_values("date")

    # Convert mood to numeric safely
    df_plot["mood"] = pd.to_numeric(df_plot["mood"], errors="coerce")

    # Rolling 7-day average
    df_plot["mood_avg7"] = df_plot["mood"].rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_plot["date"], df_plot["mood"], marker="o", label="Daily Mood", linewidth=1.5)
    ax.plot(df_plot["date"], df_plot["mood_avg7"], color="orange", label="7-Day Average", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mood (1–10)")
    ax.set_title("Mood Trend Over Time")
    ax.legend()
    st.pyplot(fig)

    st.divider()
st.subheader("📊 Summary")

if df.empty:
    st.info("No data yet to summarize.")
else:
    dsum = df.copy()
    dsum["mood"] = pd.to_numeric(dsum["mood"], errors="coerce")
    dsum["date"] = pd.to_datetime(dsum["date"])

    last7 = dsum[dsum["date"] >= (dsum["date"].max() - pd.Timedelta(days=6))]
    avg_all = dsum["mood"].mean()
    avg_7d = last7["mood"].mean() if not last7.empty else None
    avg_sent = pd.to_numeric(dsum["sentiment_compound"], errors="coerce").mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Mood (all time)", f"{avg_all:.1f}" if pd.notna(avg_all) else "—")
    col2.metric("Avg Mood (last 7 days)", f"{avg_7d:.1f}" if pd.notna(avg_7d) else "—")
    col3.metric("Avg Sentiment", f"{avg_sent:.2f}" if pd.notna(avg_sent) else "—")

