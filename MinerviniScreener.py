import pandas as pd
import yfinance as yf
import numpy as np
import time
import os
import smtplib
import warnings
import random
from email.message import EmailMessage

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MIN_VOLUME = 300_000
MA_SHORT = 50
MA_MED = 150
MA_LONG = 200

BATCH_SIZE = 15
SLEEP_BETWEEN_BATCHES = 3
MAX_RETRIES = 3

RS_LOOKBACK = "18mo"

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")

if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not TO_EMAIL:
    raise ValueError("❌ Email environment variables not set")

# --------------------------------------------------
# LOAD TICKERS
# --------------------------------------------------
tickers_raw = pd.read_csv("validated_us_tickers.csv", header=None)[0].tolist()
tickers = [str(t).strip().upper() for t in tickers_raw if isinstance(t, str) and t.strip()]
print(f"Loaded {len(tickers)} valid tickers")

results = []
failed_tickers = []

# --------------------------------------------------
# HELPER
# --------------------------------------------------
def rate_of_change(prices, days):
    if len(prices) >= days:
        return (prices.iloc[-1] / prices.iloc[-days] - 1) * 100
    return np.nan

# --------------------------------------------------
# TREND TEMPLATE SCREEN (UNCHANGED LOGIC, FIXED GUARDS)
# --------------------------------------------------
for i in range(0, len(tickers), BATCH_SIZE):
    batch = tickers[i:i + BATCH_SIZE]

    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(
                batch,
                period="1y",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )
            break
        except Exception as e:
            wait = random.randint(10, 20)
            print(f"Retry {attempt+1}/{MAX_RETRIES}, waiting {wait}s")
            time.sleep(wait)
    else:
        for t in batch:
            failed_tickers.append({"Ticker": t, "Reason": "Download failed"})
        continue

    for ticker in batch:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    continue
                df = data[ticker]
            else:
                df = data

            if df.empty or len(df) < MA_LONG:
                continue

            close = df["Close"]
            volume = df["Volume"]

            ma50 = close.rolling(MA_SHORT).mean()
            ma150 = close.rolling(MA_MED).mean()
            ma200 = close.rolling(MA_LONG).mean()

            # ---------------- CRITICAL GUARD (RESTORED) ----------------
            if (
                pd.isna(ma50.iloc[-1]) or
                pd.isna(ma150.iloc[-1]) or
                pd.isna(ma200.iloc[-1])
            ):
                continue

            price = close.iloc[-1]

            # ---------------- TREND RULES (ORIGINAL STRICTNESS) ----------------
            if not (
                price > ma150.iloc[-1] > ma200.iloc[-1] and
                ma50.iloc[-1] > ma150.iloc[-1] > ma200.iloc[-1]
            ):
                continue

            if ma200.iloc[-20] >= ma200.iloc[-1]:
                continue

            low_52w = close.min()
            high_52w = close.max()

            if price / low_52w < 1.25:
                continue

            if price / high_52w < 0.75:
                continue

            if volume.tail(50).mean() < MIN_VOLUME:
                continue

            # ---------------- METADATA (SAFE) ----------------
            sector = "Unknown"
            industry = "Unknown"

            results.append({
                "Ticker": ticker,
                "Sector": sector,
                "Industry": industry,
                "Price": round(price, 2),
                "% From 52W High": round((1 - price / high_52w) * 100, 1),
                "% From 52W Low": round((price / low_52w - 1) * 100, 1),
                "MA50": round(ma50.iloc[-1], 2),
                "MA150": round(ma150.iloc[-1], 2),
                "MA200": round(ma200.iloc[-1], 2)
            })

        except Exception as e:
            failed_tickers.append({"Ticker": ticker, "Reason": str(e)})

    print(f"Processed batch {i // BATCH_SIZE + 1}")
    time.sleep(SLEEP_BETWEEN_BATCHES)

# --------------------------------------------------
# RS CALCULATION — ANNOTATION ONLY
# --------------------------------------------------
df = pd.DataFrame(results)
print(f"Trend survivors: {len(df)}")

if not df.empty:
    rs_data = yf.download(
        df["Ticker"].tolist(),
        period=RS_LOOKBACK,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False
    )

    rs_rows = []

    for ticker in df["Ticker"]:
        try:
            if ticker not in rs_data or "Adj Close" not in rs_data[ticker]:
                continue

            prices = rs_data[ticker]["Adj Close"].dropna()

            roc_3m  = rate_of_change(prices, 63)
            roc_6m  = rate_of_change(prices, 126)
            roc_9m  = rate_of_change(prices, 189)
            roc_12m = rate_of_change(prices, 252)

            if np.isnan([roc_3m, roc_6m, roc_9m, roc_12m]).any():
                continue

            strength = (
                0.40 * roc_3m +
                0.20 * roc_6m +
                0.20 * roc_9m +
                0.20 * roc_12m
            )

            rs_rows.append({
                "Ticker": ticker,
                "RS_Strength": round(strength, 2)
            })

        except Exception:
            continue

    rs_df = pd.DataFrame(rs_rows)

    df = df.merge(rs_df, on="Ticker", how="left")
    df["RS_Rating"] = (df["RS_Strength"].rank(pct=True) * 100).round().astype("Int64")

print(f"After RS merge: {len(df)}")

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
df = df.sort_values("RS_Rating", ascending=False)
df.to_csv("minervini_candidates.csv", index=False)

if failed_tickers:
    pd.DataFrame(failed_tickers).to_csv("failed_tickers.csv", index=False)

sector_report = (
    df.groupby(["Sector", "Industry"], dropna=False)
      .size()
      .reset_index(name="Count")
      .sort_values("Count", ascending=False)
)
sector_report.to_csv("sector_industry_report.csv", index=False)

# --------------------------------------------------
# EMAIL
# --------------------------------------------------
if len(df) > 0:
    msg = EmailMessage()
    msg["Subject"] = "Minervini Screener — Trend + RS"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content(f"Total candidates: {len(df)}")

    for file in ["minervini_candidates.csv", "sector_industry_report.csv"]:
        if os.path.exists(file):
            with open(file, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype="application",
                    subtype="csv",
                    filename=file
                )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print("📧 Email sent")
else:
    print("⚠️ No candidates found")


