import pandas as pd
import yfinance as yf
import numpy as np
import time
import os
import smtplib
from email.message import EmailMessage
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==================================================
# CONFIG (UNCHANGED)
# ==================================================
MIN_VOLUME = 300_000
MA_SHORT = 50
MA_LONG = 200
HIGH_THRESHOLD = 0.90
MAX_EXT_MA50 = 0.20

RS_BATCH = 50
RS_SLEEP = 5
EPS_SLEEP = 1.0  # IMPORTANT FIX

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")

# ==================================================
# STEP 1 — MINERVINI TREND TEMPLATE
# ==================================================
tickers = pd.read_csv("validated_us_tickers.csv", header=None)[0].tolist()
results = []

for ticker in tqdm(tickers, desc="Minervini Trend Screen"):
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

        if not (price > ma50.iloc[-1] > ma200.iloc[-1]):
            continue

        if ma200.iloc[-1] <= ma200.iloc[-20]:
            continue

        high_52w = close.max()
        pct_from_high = price / high_52w
        if pct_from_high < HIGH_THRESHOLD:
            continue

        pct_above_ma50 = (price - ma50.iloc[-1]) / ma50.iloc[-1]
        if pct_above_ma50 > MAX_EXT_MA50:
            continue

        if volume.tail(50).mean() < MIN_VOLUME:
            continue

        info = tkr.info if isinstance(tkr.info, dict) else {}

        results.append({
            "Ticker": ticker,
            "Sector": info.get("sector", "Unknown"),
            "Industry": info.get("industry", "Unknown"),
            "Price": round(price, 2),
            "% From 52W High": round((1 - pct_from_high) * 100, 1),
            "% Above MA50": round(pct_above_ma50 * 100, 1),
            "MA50": round(ma50.iloc[-1], 2),
            "MA200": round(ma200.iloc[-1], 2)
        })

    except Exception:
        continue

df_trend = pd.DataFrame(results)
df_trend.to_csv("minervini_candidates.csv", index=False)

# ==================================================
# STEP 2 — RELATIVE STRENGTH (UNCHANGED)
# ==================================================
def roc(prices, days):
    return (prices.iloc[-1] / prices.iloc[-days] - 1) * 100

rs_results = []
tickers = df_trend["Ticker"].tolist()

for i in tqdm(range(0, len(tickers), RS_BATCH), desc="RS Screen"):
    batch = tickers[i:i + RS_BATCH]

    try:
        data = yf.download(
            batch,
            period="18mo",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception:
        continue

    for ticker in batch:
        try:
            if ticker not in data:
                continue

            prices = data[ticker]["Adj Close"].dropna()

            r3  = roc(prices, 63)
            r6  = roc(prices, 126)
            r9  = roc(prices, 189)
            r12 = roc(prices, 252)

            strength = (
                0.40 * r3 +
                0.20 * r6 +
                0.20 * r9 +
                0.20 * r12
            )

            rs_results.append({
                "Ticker": ticker,
                "RS_3M": round(r3, 2),
                "RS_6M": round(r6, 2),
                "RS_9M": round(r9, 2),
                "RS_12M": round(r12, 2),
                "Strength": round(strength, 2)
            })

        except Exception:
            continue

    time.sleep(RS_SLEEP)

df_rs = pd.DataFrame(rs_results)
df_rs["RS_Rating"] = (df_rs["Strength"].rank(pct=True) * 100).round().astype(int)

# ==================================================
# STEP 3 — EPS & SALES (MINIMAL FIX)
# ==================================================
df_final = df_trend.merge(df_rs, on="Ticker", how="inner")

df_final["EPS_Growth_YoY_%"] = None
df_final["Revenue_Growth_YoY_%"] = None

for i, row in tqdm(df_final.iterrows(), total=len(df_final), desc="EPS & Sales"):
    ticker = row["Ticker"]

    try:
        # FORCE FRESH OBJECT + DELAY
        time.sleep(EPS_SLEEP)
        tkr = yf.Ticker(str(ticker))
        info = tkr.info if isinstance(tkr.info, dict) else {}

        eps_growth = info.get("earningsQuarterlyGrowth")
        revenue_growth = info.get("revenueGrowth")

        if eps_growth is not None:
            df_final.at[i, "EPS_Growth_YoY_%"] = round(eps_growth * 100, 1)

        if revenue_growth is not None:
            df_final.at[i, "Revenue_Growth_YoY_%"] = round(revenue_growth * 100, 1)

    except Exception:
        continue

# ==================================================
# OUTPUT FILES
# ==================================================
df_final.sort_values("RS_Rating", ascending=False, inplace=True)
df_final.to_csv("final_stock_results.csv", index=False)

sector_report = (
    df_final.groupby(["Sector", "Industry"], dropna=False)
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
)

sector_report.to_csv("final_sector_industry_report.csv", index=False)

# ==================================================
# EMAIL RESULTS
# ==================================================
msg = EmailMessage()
msg["Subject"] = "Combined Minervini Screener Results"
msg["From"] = EMAIL_ADDRESS
msg["To"] = TO_EMAIL

msg.set_content(f"""
Combined Screener Complete

Total stocks: {len(df_final)}

Attached:
- final_stock_results.csv
- final_sector_industry_report.csv
""")

for file in ["final_stock_results.csv", "final_sector_industry_report.csv"]:
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

print("✅ Combined screener complete — email sent")

