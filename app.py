# app.py
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import numpy as np


# ---------- Paths ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "entries.csv"


analyzer = SentimentIntensityAnalyzer()

COLUMNS = [
    "timestamp", "date", "time",
    "mood", "energy", "sleep_hours", "stress",
    "tags", "journal", "sentiment_compound", "photo_path"  
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

    enable_cam = st.checkbox("Enable webcam (optional)")
    img_file = None
    if enable_cam:
        img_file = st.camera_input("Take a quick snapshot (optional)")

    submitted = st.form_submit_button("💾 Save Entry")
    if submitted:
        # Basic validation
        if sleep_hours < 0:
            st.error("Sleep hours cannot be negative.")
        else:
            now = datetime.now()
            text = journal.strip()
            sent = analyzer.polarity_scores(text)["compound"] if text else None
            photo_path = ""
            if img_file is not None:
                # save image to data/captures with timestamped filename
                img_bytes = img_file.getvalue()
                photo_name = f"{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                photo_path = str(DATA_DIR / "captures" / photo_name)
                with open(photo_path, "wb") as f:
                    f.write(img_bytes)

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

# ----- Correlation Matrix -----
st.divider()
st.subheader("🔗 Correlation Matrix (Mood vs. Factors)")

# numeric columns that we will analyze
target_cols = ["mood", "energy", "sleep_hours", "stress", "sentiment_compound"]
present = [c for c in target_cols if c in df.columns]

if len(present) < 2 or df.empty:
    st.info("Not enough data yet to compute correlations. Add more entries.")
else:
    corr_df = df.copy()

    for c in present:
        corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")

    corr_df = corr_df[present].dropna(how="all")

    if len(corr_df) < 2:
        st.info("Need at least two entries with numeric values to compute correlations.")
    else:
        corr = corr_df[present].corr(method="pearson")

        st.write("Correlation values (Pearson):")
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(corr.values)  

        # Axis ticks/labels
        ax.set_xticks(range(len(present)))
        ax.set_yticks(range(len(present)))
        ax.set_xticklabels(present, rotation=45, ha="right")
        ax.set_yticklabels(present)
        ax.set_title("Correlation Heatmap")

        for i in range(len(present)):
            for j in range(len(present)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center")

        fig.tight_layout()
        st.pyplot(fig)

        st.caption("Tip: Values near +1 mean a strong positive relationship; near −1 mean a strong negative relationship.")

# ----- Lag Analysis -----
st.divider()
st.subheader("⏳ Lag Analysis (Yesterday ➜ Today)")

if df.empty:
    st.info("Not enough data yet to analyze lags. Add more daily entries first.")
else:
    lag_df = df.copy()
    lag_df["date"] = pd.to_datetime(lag_df["date"])
    lag_df = lag_df.sort_values("date")

    for col in ["mood", "sleep_hours", "stress", "energy"]:
        lag_df[col] = pd.to_numeric(lag_df[col], errors="coerce")

    lag_df["sleep_lag1"] = lag_df["sleep_hours"].shift(1)
    lag_df["stress_lag1"] = lag_df["stress"].shift(1)
    lag_df["energy_lag1"] = lag_df["energy"].shift(1)

    lag_corrs = {
        "Sleep (yesterday) → Mood (today)": lag_df["sleep_lag1"].corr(lag_df["mood"]),
        "Stress (yesterday) → Mood (today)": lag_df["stress_lag1"].corr(lag_df["mood"]),
        "Energy (yesterday) → Mood (today)": lag_df["energy_lag1"].corr(lag_df["mood"]),
    }

    lag_table = pd.DataFrame(lag_corrs, index=["Pearson r"]).T
    st.dataframe(lag_table.style.format("{:.2f}"), use_container_width=True)

    st.subheader("💡 Insights from Lag Analysis")
    for name, corr in lag_corrs.items():
        if pd.notna(corr):
            if corr > 0.3:
                st.success(f"{name} shows a *positive* relationship (r = {corr:.2f}). "
                           f"Better {name.split(' (')[0].lower()} may improve next-day mood.")
            elif corr < -0.3:
                st.warning(f"{name} shows a *negative* relationship (r = {corr:.2f}). "
                           f"Higher values tend to lower next-day mood.")
            else:
                st.info(f"{name} has a weak or no clear relationship (r = {corr:.2f}).")
        else:
            st.info(f"{name}: not enough paired data yet.")

# ----- Insight Cards -----
st.divider()
st.subheader("🧭 Insight Cards")

if df.empty:
    st.info("Add more entries to generate insights.")
else:
    ins = df.copy()
    ins["date"] = pd.to_datetime(ins["date"])
    ins = ins.sort_values("date")

    for col in ["mood", "sleep_hours", "stress", "energy", "sentiment_compound"]:
        if col in ins.columns:
            ins[col] = pd.to_numeric(ins[col], errors="coerce")

    # Build lag features (yesterday → today) for comparisons
    ins["sleep_lag1"] = ins["sleep_hours"].shift(1)
    ins["stress_lag1"] = ins["stress"].shift(1)
    ins["energy_lag1"] = ins["energy"].shift(1)

    def insight_card(title: str, desc: str, delta: float | None, n_a: int, n_b: int):
        if delta is None or pd.isna(delta):
            st.info(f"**{title}**\n\n{desc}\n\n*Not enough data yet.*")
            return
        tone = st.success if abs(delta) >= 0.5 else (st.warning if abs(delta) >= 0.25 else st.info)
        symbol = "▲" if delta > 0 else ("▼" if delta < 0 else "■")
        tone(f"**{title}**  \n{desc}  \n**Δ = {delta:+.2f}** ({symbol})  \n*n₁ = {n_a}, n₂ = {n_b}*")

    # 1) Yesterday's Sleep: ≥7h vs <7h → Today's Mood
    comp = ins.dropna(subset=["mood", "sleep_lag1"])
    grp_hi = comp[comp["sleep_lag1"] >= 7]["mood"]
    grp_lo = comp[comp["sleep_lag1"] < 7]["mood"]
    delta_sleep = (grp_hi.mean() - grp_lo.mean()) if (len(grp_hi) >= 5 and len(grp_lo) >= 5) else None
    insight_card(
        "Sleep (Yesterday) vs Mood (Today)",
        "Comparing days after **≥7h** sleep vs **<7h** sleep.",
        delta_sleep, len(grp_hi), len(grp_lo)
    )

    # 2) Yesterday's Stress: ≤2 vs ≥4 → Today's Mood
    comp = ins.dropna(subset=["mood", "stress_lag1"])
    grp_low = comp[comp["stress_lag1"] <= 2]["mood"]
    grp_high = comp[comp["stress_lag1"] >= 4]["mood"]
    delta_stress = (grp_low.mean() - grp_high.mean()) if (len(grp_low) >= 5 and len(grp_high) >= 5) else None
    insight_card(
        "Stress (Yesterday) vs Mood (Today)",
        "Comparing days after **low stress (≤2)** vs **high stress (≥4)**.",
        delta_stress, len(grp_low), len(grp_high)
    )

    # 3) Energy (Yesterday): ≥7 vs <7 → Today's Mood
    comp = ins.dropna(subset=["mood", "energy_lag1"])
    grp_ehi = comp[comp["energy_lag1"] >= 7]["mood"]
    grp_elo = comp[comp["energy_lag1"] < 7]["mood"]
    delta_energy = (grp_ehi.mean() - grp_elo.mean()) if (len(grp_ehi) >= 5 and len(grp_elo) >= 5) else None
    insight_card(
        "Energy (Yesterday) vs Mood (Today)",
        "Comparing days after **high energy (≥7)** vs **lower energy (<7)**.",
        delta_energy, len(grp_ehi), len(grp_elo)
    )

    if "tags" in ins.columns:
        tags_norm = ins.copy()
        tags_norm["has_gym"] = tags_norm["tags"].fillna("").str.lower().str.contains(r"\bgym\b")
        grp_gym = tags_norm[tags_norm["has_gym"]]["mood"]
        grp_nogym = tags_norm[~tags_norm["has_gym"]]["mood"]
        delta_gym = (grp_gym.mean() - grp_nogym.mean()) if (len(grp_gym) >= 5 and len(grp_nogym) >= 5) else None
        insight_card(
            "Tag Insight: 'gym' Days",
            "Same-day mood on entries tagged **gym** vs days without that tag.",
            delta_gym, len(grp_gym), len(grp_nogym)
        )

    st.caption("Notes: Δ shows mean difference between groups. Insights shown only when each group has at least 5 samples to avoid noise.")

# ----- Tomorrow's Mood Prediction -----
st.divider()
st.subheader("🔮 Tomorrow's Mood Prediction")

if df.empty or len(df) < 8:
    st.info("Not enough data yet to train a prediction model. Add more daily entries.")
else:
    pred_df = df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df = pred_df.sort_values("date")

    # Ensure numeric types
    for col in ["mood", "sleep_hours", "stress", "energy", "sentiment_compound"]:
        if col in pred_df.columns:
            pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

    # Recent rolling averages (3-day)
    pred_df["mood_avg3"] = pred_df["mood"].rolling(3).mean()
    pred_df["stress_avg3"] = pred_df["stress"].rolling(3).mean()
    pred_df["sleep_avg3"] = pred_df["sleep_hours"].rolling(3).mean()

    # Day-of-week as number (0=Monday,...,6=Sunday)
    pred_df["weekday"] = pred_df["date"].dt.weekday

    # Target: tomorrow's mood (shift -1)
    pred_df["mood_tomorrow"] = pred_df["mood"].shift(-1)

    # Features taken from "today" (row D) to predict mood at D+1
    feature_cols = [
        "mood",
        "sleep_hours",
        "stress",
        "energy",
        "sentiment_compound",
        "mood_avg3",
        "stress_avg3",
        "sleep_avg3",
        "weekday",
    ]

    # Drop rows that don't have all needed values
    train_df = pred_df.dropna(subset=feature_cols + ["mood_tomorrow"])

    if len(train_df) < 8:
        st.info("Not enough complete rows to train the model yet. Keep logging for a few more days.")
    else:
        X = train_df[feature_cols].values
        y = train_df["mood_tomorrow"].values

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Evaluate basic fit on training data (R^2)
        r2 = model.score(X, y)

        # Use the most recent day as "today" to predict tomorrow
        latest_row = pred_df.sort_values("date").iloc[-1]
        latest_features = latest_row[feature_cols]

        if latest_features.isna().any():
            st.info("The latest entry is missing some values, so prediction cannot be made yet.")
        else:
            X_latest = latest_features.values.reshape(1, -1)
            pred_mood = float(model.predict(X_latest)[0])
            # Clip prediction to valid mood scale
            pred_mood = max(1.0, min(10.0, pred_mood))

            col1, col2 = st.columns(2)
            col1.metric(
                "Predicted Mood (Tomorrow)",
                f"{pred_mood:.1f}",
                help="Estimated mood score for your next day based on your recent patterns."
            )
            col2.metric(
                "Model Fit (R² on history)",
                f"{r2:.2f}",
                help="How well the model explains variation in your past mood data (1.0 = perfect, 0 = no fit)."
            )

            # Feature importance (absolute coefficient magnitude)
            coefs = model.coef_
            importance = pd.DataFrame({
                "feature": feature_cols,
                "coef": coefs,
                "importance": np.abs(coefs),
            }).sort_values("importance", ascending=False)

            st.markdown("**Which factors influence the prediction the most?**")
            st.dataframe(importance[["feature", "coef"]].style.format({"coef": "{:.2f}"}), use_container_width=True)

            # Optional: simple bar chart for importance
            fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
            ax_imp.barh(importance["feature"], importance["importance"])
            ax_imp.invert_yaxis()
            ax_imp.set_xlabel("Absolute Coefficient (Importance)")
            ax_imp.set_title("Feature Importance for Tomorrow's Mood")
            st.pyplot(fig_imp)

            st.caption(
                "Note: This is a simple linear regression model trained on your own history. "
                "It provides an approximate forecast, not a guaranteed outcome."
            )
