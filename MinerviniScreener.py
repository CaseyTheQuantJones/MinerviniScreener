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
SLEEP = 0.01

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")

if not EMAIL_ADDRESS or not EMAIL_PASSWORD or not TO_EMAIL:
    raise ValueError("❌ Email environment variables not set")

tickers = pd.read_csv("validated_us_tickers.csv", header=None)[0].tolist()
print(f"Loaded {len(tickers)} tickers")

results = []
failed_tickers = []

# fallback for tqdm if not installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# --------------------------------------------------
# SCREENING LOOP
# --------------------------------------------------
for ticker in tqdm(tickers, desc="Screening stocks"):
    try:
        tkr = yf.Ticker(ticker)
        data = tkr.history(period="1y", auto_adjust=True)

        if data.empty or len(data) < MA_LONG:
            failed_tickers.append(ticker)
            continue

        close = data["Close"]
        volume = data["Volume"]

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
        # 1 + 2 + 4 + 8: price > ma150 > ma200, ma50 above both
        if not (price > ma150_now > ma200_now and ma50_now > ma150_now and ma50_now > ma200_now):
            failed_tickers.append(ticker)
            continue

        # 3: 200-day MA trending up over last ~20 trading days
        ma200_slope = (ma200_now - ma200.iloc[-20]) / 20
        if ma200_slope <= 0:
            failed_tickers.append(ticker)
            continue

        # --------------------------------------------------
        # PRICE VS 52-WEEK HIGH/LOW
        # --------------------------------------------------
        low_52w = close.min()
        high_52w = close.max()

        # 5: price at least 25% above 52-week low
        if price / low_52w < 1.25:
            failed_tickers.append(ticker)
            continue

        # 6: price within 25% of 52-week high
        if price / high_52w < 0.75:
            failed_tickers.append(ticker)
            continue

        # --------------------------------------------------
        # LIQUIDITY
        # --------------------------------------------------
        if volume.tail(50).mean() < MIN_VOLUME:
            failed_tickers.append(ticker)
            continue

        # --------------------------------------------------
        # METADATA
        # --------------------------------------------------
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

    except Exception:
        failed_tickers.append(ticker)
        continue

    time.sleep(SLEEP)

# --------------------------------------------------
# OUTPUT FILES
# --------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("minervini_candidates.csv", index=False)

if failed_tickers:
    pd.DataFrame(failed_tickers, columns=["Ticker"]).to_csv("failed_tickers.csv", index=False)

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
