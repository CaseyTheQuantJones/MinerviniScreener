# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
failed_tickers = []

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
        ma200 = close.rolling(MA_LONG).mean()

        price = close.iloc[-1]
        ma50_now = ma50.iloc[-1]
        ma200_now = ma200.iloc[-1]

        # --------------------------------------------------
        # 1) TREND ALIGNMENT
        # --------------------------------------------------
        if not (price > ma50_now > ma200_now):
            failed_tickers.append(ticker)
            continue

        # MA200 gentle uptrend over last 50 trading days
        ma200_slope = (ma200_now - ma200.iloc[-50]) / 50
        if ma200_slope <= 0:
            failed_tickers.append(ticker)
            continue

        # --------------------------------------------------
        # 2) LEADERSHIP (52W HIGH PROXIMITY)
        # --------------------------------------------------
        high_52w = close.max()
        pct_from_high = price / high_52w

        if pct_from_high < 0.85:  # relaxed from 90%
            failed_tickers.append(ticker)
            continue

        # --------------------------------------------------
        # 3) LIQUIDITY
        # --------------------------------------------------
        if volume.tail(50).mean() < MIN_VOLUME:
            failed_tickers.append(ticker)
            continue

        # --------------------------------------------------
        # 4) METADATA
        # --------------------------------------------------
        info = tkr.info if isinstance(tkr.info, dict) else {}

        results.append({
            "Ticker": ticker,
            "Sector": info.get("sector", "Unknown"),
            "Industry": info.get("industry", "Unknown"),
            "Price": round(price, 2),
            "% From 52W High": round((1 - pct_from_high) * 100, 1),
            "MA50": round(ma50_now, 2),
            "MA200": round(ma200_now, 2)
        })

    except Exception:
        failed_tickers.append(ticker)
        continue

    time.sleep(SLEEP)

# Save failed tickers
if failed_tickers:
    pd.DataFrame(failed_tickers, columns=["Ticker"]).to_csv("failed_tickers.csv", index=False)
