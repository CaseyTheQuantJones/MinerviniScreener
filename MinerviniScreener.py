import pandas as pd
import yfinance as yf
import time
import os
import smtplib
from email.message import EmailMessage
import warnings
import random

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MIN_VOLUME = 300_000
MA_SHORT = 50
MA_MED = 150
MA_LONG = 200

BATCH_SIZE = 15       # small batch to reduce rate limiting
SLEEP_BETWEEN_BATCHES = 3  # seconds
MAX_RETRIES = 3

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")

if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not TO_EMAIL:
    raise ValueError("❌ Email environment variables not set")

# --------------------------------------------------
# LOAD AND CLEAN TICKERS
# --------------------------------------------------
tickers_raw = pd.read_csv("validated_us_tickers.csv", header=None)[0].tolist()
tickers = [str(t).strip() for t in tickers_raw if isinstance(t, str) and t.strip()]
invalid_tickers = [t for t in tickers_raw if not isinstance(t, str) or not str(t).strip()]

if invalid_tickers:
    print(f"Skipping invalid tickers: {invalid_tickers}")

print(f"Loaded {len(tickers)} valid tickers")

results = []
failed_tickers = [{"Ticker": t, "Reason": "Invalid ticker"} for t in invalid_tickers]

# --------------------------------------------------
# SCREENING LOOP WITH BATCH DOWNLOADS
# --------------------------------------------------
for i in range(0, len(tickers), BATCH_SIZE):
    batch = tickers[i:i + BATCH_SIZE]
    retries = 0
    while retries < MAX_RETRIES:
        try:
            data = yf.download(batch, period="1y", auto_adjust=True, group_by='ticker', threads=True)
            break
        except Exception as e:
            retries += 1
            wait_time = random.randint(10, 20)
            print(f"Batch download failed (attempt {retries}/{MAX_RETRIES}): {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    else:
        print(f"Batch {batch} failed after {MAX_RETRIES} retries")
        for t in batch:
            failed_tickers.append({"Ticker": t, "Reason": "Download failed after retries"})
        continue

    for ticker in batch:
        try:
            df = data[ticker] if len(batch) > 1 else data
            if df.empty or len(df) < MA_LONG:
                failed_tickers.append({"Ticker": ticker, "Reason": "Not enough data"})
                continue

            close = df["Close"]
            volume = df["Volume"]

            ma50 = close.rolling(MA_SHORT).mean()
            ma150 = close.rolling(MA_MED).mean()
            ma200 = close.rolling(MA_LONG).mean()

            price = close.iloc[-1]
            ma50_now = ma50.iloc[-1]
            ma150_now = ma150.iloc[-1]
            ma200_now = ma200.iloc[-1]

            # --------------------------------------------------
            # TREND RULES
            # --------------------------------------------------
            if not (price > ma150_now > ma200_now and ma50_now > ma150_now and ma50_now > ma200_now):
                failed_tickers.append({"Ticker": ticker, "Reason": "Trend MA rules failed"})
                continue

            ma200_slope = (ma200_now - ma200.iloc[-20]) / 20
            if ma200_slope <= 0:
                failed_tickers.append({"Ticker": ticker, "Reason": "MA200 not trending up"})
                continue

            # --------------------------------------------------
            # PRICE VS 52-WEEK HIGH/LOW
            # --------------------------------------------------
            low_52w = close.min()
            high_52w = close.max()

            if price / low_52w < 1.25:
                failed_tickers.append({"Ticker": ticker, "Reason": "Price too close to 52-week low"})
                continue

            if price / high_52w < 0.75:
                failed_tickers.append({"Ticker": ticker, "Reason": "Price too far from 52-week high"})
                continue

            # --------------------------------------------------
            # LIQUIDITY
            # --------------------------------------------------
            if volume.tail(50).mean() < MIN_VOLUME:
                failed_tickers.append({"Ticker": ticker, "Reason": "Low average volume"})
                continue

            # --------------------------------------------------
            # METADATA (lightweight)
            # --------------------------------------------------
            tkr = yf.Ticker(ticker)
            info = getattr(tkr, "fast_info", {})
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            results.append({
                "Ticker": ticker,
                "Sector": sector,
                "Industry": industry,
                "Price": round(price, 2),
                "% From 52W High": round((1 - price / high_52w) * 100, 1),
                "% From 52W Low": round((price / low_52w - 1) * 100, 1),
                "MA50": round(ma50_now, 2),
                "MA150": round(ma150_now, 2),
                "MA200": round(ma200_now, 2)
            })

        except Exception as e:
            failed_tickers.append({"Ticker": ticker, "Reason": f"Other error: {e}"})
            continue

    print(f"Processed batch {i // BATCH_SIZE + 1} / {len(tickers) // BATCH_SIZE + 1}")
    time.sleep(SLEEP_BETWEEN_BATCHES)

# --------------------------------------------------
# OUTPUT FILES
# --------------------------------------------------
df = pd.DataFrame(results)
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
# EMAIL RESULTS
# --------------------------------------------------
if len(df) > 0:
    body = f"""Minervini First-Pass Screen

Total candidates: {len(df)}

Top Sectors / Industries:
"""
    for _, row in sector_report.head(10).iterrows():
        body += f"{row['Sector']} | {row['Industry']} : {row['Count']}\n"

    msg = EmailMessage()
    msg["Subject"] = "Minervini Screener — Trend & Leadership"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content(body)

    for file in ["minervini_candidates.csv", "sector_industry_report.csv", "failed_tickers.csv"]:
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

    print("📧 Email sent successfully")
else:
    print("⚠️ No candidates found — email not sent")

