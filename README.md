# Health Mood Tracker 🧠📊

Health Mood Tracker is a data-driven web application built with **Python** and **Streamlit** as part of a university **Open Source Software** term project.

The app enables users to log their **daily mood, sleep, stress, energy, tags**, and **journal reflections**, then analyze this data using:

- Time-series visualization
- Sentiment analysis (VADER)
- Correlation and lag analysis
- Keyword & word cloud analysis
- **Machine learning to predict tomorrow’s mood**

All data is stored locally in a CSV file, keeping the project lightweight and easy to run.

---

## 1. Features

### 🔐 Daily Check-In Form

Users can record:

- Mood (1–10)
- Energy (1–10)
- Sleep hours (decimal supported)
- Stress level (1–5)
- Tags (free keywords such as `gym`, `exam`, `travel`)
- Journal entry (free text)
- Optional webcam snapshot (image saved locally, path stored in CSV)

Each submission is timestamped and appended to `data/entries.csv`.

---

### 📈 Mood & Health Visualization

#### 📋 Recent Entries

Shows the latest recorded entries in a clean table.

#### 📉 Mood Trend

A line chart showing:

- Mood over time
- A **7-day moving average** (trend smoothing)

#### 📊 Summary Statistics

- Average mood (overall)
- Average mood (last 7 days)
- Average journal sentiment score

---

### 🧠 Sentiment Analysis (VADER)

Journal text is processed using **VADER SentimentIntensityAnalyzer**, producing:

- `sentiment_compound` score (−1 to +1)

This is included in all downstream analysis: correlations, lag analysis, and prediction.

---

### 📊 Analytics & Insights

#### a. Correlation Matrix

Computes relationships between:

- mood
- energy
- sleep_hours
- stress
- sentiment_compound

Displayed both as:

- A correlation table
- A heatmap (matplotlib)

#### b. Lag Analysis (Yesterday → Today)

Measures how **yesterday’s factors** relate to **today’s mood**, including:

- sleep_lag1
- stress_lag1
- energy_lag1

This reveals patterns like:

> "Higher sleep yesterday tends to improve mood today."

#### c. Insight Cards

Automatically generated insights comparing groups, such as:

- High sleep (≥7h) vs low sleep
- High stress vs low stress
- Energy highs/lows
- Tag patterns (e.g., days tagged “gym”)

Each insight shows:

- Average mood difference (Δ)
- Sample counts per group

#### d. Word Cloud & Keyword Frequency

Creates a word cloud from journal entries and displays the most frequent terms.

---

### 🔮 Tomorrow’s Mood Prediction (Machine Learning)

A simple supervised learning model (**Linear Regression**) predicts **tomorrow's mood** based on:

#### Input Features (Day D)

- Today's mood
- Today's stress
- Today's energy
- Last night's sleep hours
- Today's journal sentiment
- 3-day rolling averages:
  - mood_avg3
  - stress_avg3
  - sleep_avg3
- Day of week (0–6)

#### Target

- mood on the next day (`mood_tomorrow`)

The app displays:

- **Predicted mood for tomorrow**
- **R² score** showing model fit on historical data
- **Feature importance table + visualization**

This transforms the app from simple tracking into a meaningful personal analytics tool.

---

## 2. Tech Stack

| Component          | Library                        |
| ------------------ | ------------------------------ |
| Web Framework      | Streamlit                      |
| Data               | pandas, numpy                  |
| Visualization      | matplotlib                     |
| Sentiment Analysis | vaderSentiment                 |
| Machine Learning   | scikit-learn                   |
| Text Processing    | wordcloud                      |
| Storage            | Local CSV (`data/entries.csv`) |

---

## 3. Project Structure

```

health-mood-tracker/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
├── .gitignore
├── Plan_and_scheduling.txt # Development notes
│
├── data/
│   └── entries.csv         # User data (created automatically)
│
└── src/
├── analysis.py         # optional helper scripts
├── charts.py
└── io_utils.py

```

---

## 4. Installation & Setup

### 4.1 Clone the repository

```
git clone https://github.com/badrelmb/health-mood-tracker.git
cd health-mood-tracker

```

### 4.2 Create & activate a virtual environment

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\activate
```

### 4.3 Install dependencies

```
pip install -r requirements.txt
```

---

## 5. Running the App

From the project root (and with the virtual environment activated):

```
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 6. How to Use

1. Launch the app
2. Fill in the **Daily Check-In** form
3. Press **Save Entry**
4. Explore:

   - Mood trends
   - Summary metrics
   - Correlation analysis
   - Lag analysis
   - Insight cards
   - Word cloud
   - **Tomorrow’s mood prediction**

The data is saved in `data/entries.csv` and can be exported or inspected manually.

---

## 7. Screenshots

Please refer to the screenshots under /assets (from 1 to 6) to see the full functionality of the app.

---

## 8. Open Source License

This project is licensed under the **MIT License**.
See the `LICENSE` file for full details.

External libraries retain their respective OSS licenses.

---

## 9. Future Improvements

- Multi-page navigation (Home / Analytics / Settings)
- Additional ML models (RandomForest, GradientBoosting)
- Anomaly detection for unusually high/low mood days
- Weekly PDF health reports
- Optional cloud storage or user login system

---
