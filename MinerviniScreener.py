import pandas as pd
import yfinance as yf
import time
import os
import smtplib
from email.message import EmailMessage
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MIN_VOLUME = 300_000
MA_SHORT = 50
MA_MED = 150
MA_LONG = 200
SLEEP = 0.01  # only between batch downloads

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")

if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not TO_EMAIL:
    raise ValueError("❌ Email environment variables not set")

tickers = pd.read_csv("validated_us_tickers.csv", header=None)[0].tolist()
print(f"Loaded {len(tickers)} tickers")

results = []
failed_tickers = []

# --------------------------------------------------
# BATCH DOWNLOAD
# --------------------------------------------------
print("Downloading historical data in batches...")
batch_size = 50  # adjust if you hit rate limits
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    try:
        data = yf.download(batch, period="1y", auto_adjust=True, group_by='ticker', threads=True)
    except Exception as e:
        print(f"Failed to download batch {batch}: {e}")
        for t in batch:
            failed_tickers.append({"Ticker": t, "Reason": "Download failed"})
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
            # METADATA
            # --------------------------------------------------
            tkr = yf.Ticker(ticker)
            info = tkr.info if isinstance(tkr.info, dict) else {}

            results.append({
                "Ticker": ticker,
                "Sector": info.get("sector", "Unknown"),
                "Industry": info.get("industry", "Unknown"),
                "Price": round(price, 2),
                "% From 52W High": round((1 - price/high_52w) * 100, 1),
                "% From 52W Low": round((price/low_52w - 1) * 100, 1),
                "MA50": round(ma50_now, 2),
                "MA150": round(ma150_now, 2),
                "MA200": round(ma200_now, 2)
            })

        except Exception as e:
            failed_tickers.append({"Ticker": ticker, "Reason": f"Other error: {e}"})
            continue

    time.sleep(SLEEP)  # pause between batches

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


       




