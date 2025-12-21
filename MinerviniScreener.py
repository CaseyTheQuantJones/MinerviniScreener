import pandas as pd
import yfinance as yf
import time
import os
import smtplib
from email.message import EmailMessage
import warnings
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
warnings.filterwarnings("ignore")

MIN_VOLUME = 300_000
MA_SHORT = 50
MA_LONG = 200
HIGH_THRESHOLD = 0.90          # 90% of 52W high
MAX_EXT_MA50 = 0.20            # max 20% above MA50
SLEEP = 0.01

# Email credentials pulled from GitHub Secrets
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")

# --------------------------------------------------
# LOAD TICKERS
# --------------------------------------------------
tickers = pd.read_csv("validated_us_tickers.csv", header=None)[0].tolist()
print(f"Loaded {len(tickers)} tickers")

results = []

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
for ticker in tqdm(tickers, desc="Screening stocks"):
    try:
        tkr = yf.Ticker(ticker)
        data = tkr.history(period="1y", auto_adjust=True)

        if data.empty or len(data) < MA_LONG:
            continue

        close = data["Close"]
        volume = data["Volume"]

        ma50 = close.rolling(MA_SHORT).mean()
        ma200 = close.rolling(MA_LONG).mean()

        price = close.iloc[-1]
        ma50_now = ma50.iloc[-1]
        ma200_now = ma200.iloc[-1]

        # 1) TREND ALIGNMENT
        if not (price > ma50_now > ma200_now):
            continue

        if ma200.iloc[-1] <= ma200.iloc[-20]:
            continue

        # 2) LEADERSHIP POSITION
        high_52w = close.max()
        pct_from_high = price / high_52w
        if pct_from_high < HIGH_THRESHOLD:
            continue

        # 3) NOT EXTENDED
        pct_above_ma50 = (price - ma50_now) / ma50_now
        if pct_above_ma50 > MAX_EXT_MA50:
            continue

        # 4) LIQUIDITY
        if volume.tail(50).mean() < MIN_VOLUME:
            continue

        # 5) METADATA
        info = tkr.info if isinstance(tkr.info, dict) else {}

        results.append({
            "Ticker": ticker,
            "Sector": info.get("sector", "Unknown"),
            "Industry": info.get("industry", "Unknown"),
            "Price": round(price, 2),
            "% From 52W High": round((1 - pct_from_high) * 100, 1),
            "% Above MA50": round(pct_above_ma50 * 100, 1),
            "MA50": round(ma50_now, 2),
            "MA200": round(ma200_now, 2)
        })

    except Exception:
        continue

    time.sleep(SLEEP)

# --------------------------------------------------
# OUTPUT FILES
# --------------------------------------------------
df = pd.DataFrame(results)

candidates_file = "minervini_candidates.csv"
df.to_csv(candidates_file, index=False)

sector_report = (
    df.groupby(["Sector", "Industry"], dropna=False)
      .size()
      .reset_index(name="Count")
      .sort_values("Count", ascending=False)
)

sector_file = "sector_industry_report.csv"
sector_report.to_csv(sector_file, index=False)

print(f"\n✅ Screening complete")
print(f"Candidates found: {len(df)}")
print(f"Saved: {candidates_file}")
print(f"Saved: {sector_file}")

# --------------------------------------------------
# EMAIL RESULTS (GITHUB ACTIONS SAFE)
# --------------------------------------------------
if len(df) > 0 and EMAIL_ADDRESS and EMAIL_PASSWORD and TO_EMAIL:
    body = f"""Minervini First-Pass Screen

Total candidates: {len(df)}

Top Sectors / Industries:
"""

    for _, row in sector_report.head(10).iterrows():
        body += f"{row['Sector']} | {row['Industry']} : {row['Count']}\n"

    msg = EmailMessage()
    msg["Subject"] = "Minervini Screener — Leadership & Candidates"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg.set_content(body)

    for file in [candidates_file, sector_file]:
        with open(file, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="text",
                subtype="csv",
                filename=file
            )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("📧 Email sent successfully")

    except Exception as e:
        print(f"⚠️ Email failed: {e}")

else:
    print("⚠️ Email skipped (no results or missing credentials)")

