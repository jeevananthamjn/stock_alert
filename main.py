"""
Swing Trade Alert System — Nifty 100
Runs twice daily: 9:00 AM IST (pre-market) and 6:00 PM IST (post-market)
"""

import os
import json
import logging
import smtplib
import warnings
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

NIFTY_100 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS", "NESTLEIND.NS",
    "HCLTECH.NS", "ADANIPORTS.NS", "POWERGRID.NS", "NTPC.NS", "TECHM.NS",
    "ONGC.NS", "GRASIM.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "TATAMOTORS.NS",
    "HINDALCO.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS", "CIPLA.NS", "DRREDDY.NS",
    "DIVISLAB.NS", "EICHERMOT.NS", "BPCL.NS", "COALINDIA.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "UPL.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "TATACONSUM.NS",
    "BRITANNIA.NS", "PIDILITIND.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS",
    "TORNTPHARM.NS", "LUPIN.NS", "BIOCON.NS", "ALKEM.NS", "AUROPHARMA.NS",
    "MCDOWELL-N.NS", "GODREJCP.NS", "BERGEPAINT.NS", "AMBUJACEM.NS", "ACC.NS",
    "SHREECEM.NS", "DMART.NS", "NYKAA.NS", "PAYTM.NS", "ZOMATO.NS",
    "NAUKRI.NS", "MPHASIS.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS",
    "HAVELLS.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "CROMPTON.NS", "POLYCAB.NS",
    "TATAPOWER.NS", "ADANIGREEN.NS", "ADANIENT.NS", "SIEMENS.NS", "ABB.NS",
    "BHEL.NS", "HAL.NS", "BEL.NS", "IRCTC.NS", "CONCOR.NS",
    "MUTHOOTFIN.NS", "BAJAJ-AUTO.NS", "TVSMOTORS.NS", "M&M.NS", "ASHOKLEY.NS",
    "CUMMINSIND.NS", "THERMAX.NS", "SUNTV.NS", "ZEEL.NS", "PVR.NS",
    "INDIGO.NS", "SPICEJET.NS", "OBEROIRLTY.NS", "DLF.NS", "GODREJPROP.NS",
]

STATE_FILE = Path("watchlist_state.json")
DASHBOARD_FILE = Path("dashboard.html")


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_data(ticker: str, period_daily="6mo", period_weekly="2y"):
    """Fetch daily and weekly OHLCV data plus fundamentals."""
    try:
        stock = yf.Ticker(ticker)
        daily = stock.history(period=period_daily, interval="1d")
        weekly = stock.history(period=period_weekly, interval="1wk")
        info = stock.info
        if daily.empty or len(daily) < 30:
            return None
        return {"daily": daily, "weekly": weekly, "info": info, "ticker": ticker}
    except Exception as e:
        log.warning(f"[{ticker}] fetch failed: {e}")
        return None


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────

def rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_indicators(data: dict) -> dict | None:
    """Compute all technical indicators from raw data."""
    daily = data["daily"].copy()
    weekly = data["weekly"].copy()
    info = data["info"]

    close_d = daily["Close"]
    volume_d = daily["Volume"]

    # ── RSI ──
    daily["rsi"] = rsi(close_d)
    weekly["rsi"] = rsi(weekly["Close"])

    # ── MACD ──
    daily["macd"], daily["macd_signal"], daily["macd_hist"] = macd(close_d)

    # ── Moving Averages ──
    daily["ma20"] = close_d.rolling(20).mean()
    daily["ma50"] = close_d.rolling(50).mean()

    # ── Volume ──
    daily["vol_avg20"] = volume_d.rolling(20).mean()
    daily["vol_ratio"] = volume_d / daily["vol_avg20"]

    # ── Volatility ──
    daily["volatility"] = close_d.pct_change().rolling(14).std()

    # ── Recent Returns ──
    daily["ret_1w"] = close_d.pct_change(5)
    daily["ret_2w"] = close_d.pct_change(10)

    row = daily.iloc[-1]
    prev = daily.iloc[-2] if len(daily) > 1 else row
    w_row = weekly.iloc[-1] if not weekly.empty else None

    price = float(row["Close"])
    low_10d = float(daily["Low"].tail(10).min())

    # ── Fundamental Filters ──
    debt_equity = info.get("debtToEquity", 999)
    if debt_equity is None:
        debt_equity = 999
    pe = info.get("trailingPE", 999)
    if pe is None:
        pe = 999
    market_cap = info.get("marketCap", 0)
    if market_cap is None:
        market_cap = 0

    # large cap = market cap > ₹20,000 crore (~$2.4B USD)
    large_cap = market_cap > 20_000_000_000

    indicators = {
        "ticker": data["ticker"],
        "price": price,
        "rsi_daily": float(row["rsi"]) if not pd.isna(row["rsi"]) else 50,
        "rsi_weekly": float(w_row["rsi"]) if w_row is not None and not pd.isna(w_row["rsi"]) else 50,
        "macd": float(row["macd"]),
        "macd_signal": float(row["macd_signal"]),
        "macd_hist": float(row["macd_hist"]),
        "prev_macd": float(prev["macd"]),
        "prev_macd_signal": float(prev["macd_signal"]),
        "prev_macd_hist": float(prev["macd_hist"]),
        "ma20": float(row["ma20"]) if not pd.isna(row["ma20"]) else price,
        "ma50": float(row["ma50"]) if not pd.isna(row["ma50"]) else price,
        "vol_ratio": float(row["vol_ratio"]) if not pd.isna(row["vol_ratio"]) else 1.0,
        "volatility": float(row["volatility"]) if not pd.isna(row["volatility"]) else 0.02,
        "ret_1w": float(row["ret_1w"]) if not pd.isna(row["ret_1w"]) else 0,
        "ret_2w": float(row["ret_2w"]) if not pd.isna(row["ret_2w"]) else 0,
        "low_10d": low_10d,
        "prev_high": float(daily["High"].iloc[-2]) if len(daily) > 1 else price,
        "debt_equity": debt_equity,
        "pe": pe,
        "large_cap": large_cap,
        "market_cap": market_cap,
        # Recent swing low for stop loss
        "swing_low": float(daily["Low"].tail(20).min()),
        "daily_df": daily,
    }
    return indicators


# ─────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────

def generate_signal(ind: dict) -> dict:
    """Apply strategy filters and return signal result."""
    reasons = []
    warnings_list = []

    # ── Fundamental filter ──
    if not ind["large_cap"]:
        return {"pass": False, "status": "Avoid", "reasons": ["Not large cap"]}
    if ind["debt_equity"] > 0.5:
        return {"pass": False, "status": "Avoid", "reasons": [f"D/E too high ({ind['debt_equity']:.2f})"]}
    if ind["pe"] > 35:
        return {"pass": False, "status": "Avoid", "reasons": [f"PE too high ({ind['pe']:.1f})"]}

    # ── Primary RSI signal ──
    rsi_d_ok = ind["rsi_daily"] < 30
    rsi_w_ok = ind["rsi_weekly"] < 30
    if not (rsi_d_ok and rsi_w_ok):
        return {"pass": False, "status": "Avoid", "reasons": [
            f"RSI daily={ind['rsi_daily']:.1f}, weekly={ind['rsi_weekly']:.1f} (need both <30)"
        ]}

    reasons.append(f"RSI oversold: daily={ind['rsi_daily']:.1f}, weekly={ind['rsi_weekly']:.1f}")

    # ── MACD confirmation ──
    bullish_crossover = (ind["macd"] > ind["macd_signal"]) and (ind["prev_macd"] <= ind["prev_macd_signal"])
    hist_positive = ind["macd_hist"] > 0 and ind["prev_macd_hist"] <= 0

    macd_confirmed = bullish_crossover or hist_positive
    if bullish_crossover:
        reasons.append("MACD bullish crossover")
    elif hist_positive:
        reasons.append("MACD histogram turning positive")
    else:
        warnings_list.append("No MACD confirmation yet")

    # ── Volume confirmation ──
    vol_spike = ind["vol_ratio"] >= 1.5
    if vol_spike:
        reasons.append(f"Volume spike: {ind['vol_ratio']:.1f}x average")
    else:
        warnings_list.append(f"Volume ratio only {ind['vol_ratio']:.1f}x")

    vol_confirmed = vol_spike  # simplified; local bottom check could extend this

    # ── Avoid fresh breakdown ──
    if ind["price"] < ind["low_10d"]:
        return {"pass": False, "status": "Avoid", "reasons": ["Fresh breakdown below 10-day low"]}

    # ── Determine status ──
    if macd_confirmed and vol_confirmed:
        status = "Ready"
    else:
        status = "Prepare"

    return {
        "pass": True,
        "status": status,
        "macd_confirmed": macd_confirmed,
        "vol_confirmed": vol_confirmed,
        "reasons": reasons,
        "warnings": warnings_list,
    }


# ─────────────────────────────────────────────
# TRADE SETUP
# ─────────────────────────────────────────────

def build_trade_setup(ind: dict, signal: dict) -> dict:
    """Compute entry, stop loss, target, and RR ratio."""
    price = ind["price"]
    entry = ind["prev_high"]  # breakout level

    # Stop loss: max of swing low or fixed 5% below current
    sl_swing = ind["swing_low"]
    sl_fixed = price * 0.95
    stop_loss = max(sl_swing, sl_fixed)  # tighter of the two

    # Targets
    target_min = price * 1.05
    target_pref = price * 1.10

    # Risk/Reward
    risk = price - stop_loss
    reward = target_pref - entry
    rr = reward / risk if risk > 0 else 0

    return {
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target_min": round(target_min, 2),
        "target": round(target_pref, 2),
        "risk_pct": round((price - stop_loss) / price * 100, 2),
        "rr": round(rr, 2),
        "rr_valid": rr >= 1.8,
    }


# ─────────────────────────────────────────────
# ML PROBABILITY MODEL
# ─────────────────────────────────────────────

def _build_feature_row(ind: dict) -> list:
    return [
        ind["rsi_daily"],
        ind["rsi_weekly"],
        ind["macd"],
        ind["macd_signal"],
        ind["macd_hist"],
        (ind["price"] - ind["ma20"]) / ind["ma20"] if ind["ma20"] else 0,
        (ind["price"] - ind["ma50"]) / ind["ma50"] if ind["ma50"] else 0,
        ind["vol_ratio"],
        ind["volatility"],
        ind["ret_1w"],
        ind["ret_2w"],
    ]


def _prepare_training_data(daily_df: pd.DataFrame) -> tuple:
    """Build training rows from historical data of a single stock."""
    df = daily_df.copy()
    df["rsi_d"] = rsi(df["Close"])
    df["macd_l"], df["macd_s"], df["macd_h"] = macd(df["Close"])
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["volatility"] = df["Close"].pct_change().rolling(14).std()
    df["ret_1w"] = df["Close"].pct_change(5)
    df["ret_2w"] = df["Close"].pct_change(10)

    # Target: did price rise >= 5% within next 10 days?
    df["future_max"] = df["Close"].shift(-10).rolling(10, min_periods=1).max()
    df["target"] = ((df["future_max"] - df["Close"]) / df["Close"] >= 0.05).astype(int)

    feature_cols = ["rsi_d", "macd_l", "macd_s", "macd_h", "ma20", "ma50", "vol_ratio", "volatility", "ret_1w", "ret_2w"]
    df = df.dropna(subset=feature_cols + ["target"])

    X = df[feature_cols].values
    y = df["target"].values
    return X, y


def ml_probability(ind: dict) -> float:
    """Train a quick RF on the stock's own history and return current prob."""
    try:
        daily_df = ind.get("daily_df")
        if daily_df is None or len(daily_df) < 100:
            return 0.5  # not enough data

        X, y = _prepare_training_data(daily_df)
        if len(X) < 60 or y.sum() < 5:
            return 0.5

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        # Train on all but last 10 rows to avoid leakage
        clf.fit(X_scaled[:-10], y[:-10])

        # Predict on current row features
        current = scaler.transform([_build_feature_row(ind)])
        prob = clf.predict_proba(current)[0][1]
        return round(float(prob), 3)
    except Exception as e:
        log.warning(f"ML error for {ind['ticker']}: {e}")
        return 0.5


# ─────────────────────────────────────────────
# WATCHLIST STATE
# ─────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"active_trades": [], "watchlist": []}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def update_follow_up(state: dict, current_prices: dict) -> list:
    """Update active trades with current price and return follow-up list."""
    follow_up = []
    active = []
    for trade in state.get("active_trades", []):
        ticker = trade["ticker"]
        curr = current_prices.get(ticker, trade["entry"])
        ret_pct = (curr - trade["entry"]) / trade["entry"] * 100
        days = trade.get("days_held", 0) + 1

        trade["current_price"] = round(curr, 2)
        trade["return_pct"] = round(ret_pct, 2)
        trade["days_held"] = days

        # Remove if target or SL hit
        if curr >= trade["target"] or curr <= trade["stop_loss"]:
            status = "✅ Target hit" if curr >= trade["target"] else "🔴 SL hit"
            trade["closed"] = status
            follow_up.append(trade)
        else:
            active.append(trade)
            follow_up.append(trade)

    state["active_trades"] = active
    return follow_up


# ─────────────────────────────────────────────
# AI INSIGHT
# ─────────────────────────────────────────────

def get_ai_insight(setups: list) -> str:
    """Call Groq API for market insight, fallback to plain text."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or not setups:
        return _fallback_insight(setups)

    try:
        import urllib.request
        tickers = [s["ticker"] for s in setups]
        prompt = (
            f"You are a swing trading analyst. Today's oversold large-cap Nifty stocks with RSI < 30 "
            f"on both daily and weekly: {', '.join(tickers)}. "
            "In 3-4 sentences, comment on: (1) current broader market context, "
            "(2) likely sector trend driving this oversold condition, "
            "(3) quality of these setups. "
            "Do NOT predict exact prices. Be factual and concise."
        )
        payload = json.dumps({
            "model": "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250,
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"Groq API error: {e}")
        return _fallback_insight(setups)


def _fallback_insight(setups: list) -> str:
    if not setups:
        return "No high-quality setups found today. Market may be in distribution phase."
    n = len(setups)
    avg_rsi = sum(s["rsi_daily"] for s in setups) / n
    return (
        f"Found {n} oversold setup(s) with average daily RSI of {avg_rsi:.1f}. "
        "Both daily and weekly RSI below 30 suggests significant selling pressure has been absorbed. "
        "Wait for confirmed breakout above entry level before committing. "
        "Volume confirmation is critical — a dry-volume bounce is a trap."
    )


# ─────────────────────────────────────────────
# MAIN SCANNER
# ─────────────────────────────────────────────

def run_scanner() -> dict:
    """Full scan of Nifty 100 universe. Returns structured results."""
    log.info("Starting scanner...")
    results = []
    current_prices = {}

    for ticker in NIFTY_100:
        log.info(f"  Processing {ticker}...")
        data = fetch_data(ticker)
        if data is None:
            continue

        ind = calculate_indicators(data)
        if ind is None:
            continue

        current_prices[ticker] = ind["price"]
        signal = generate_signal(ind)

        if not signal["pass"]:
            continue

        prob = ml_probability(ind)
        if prob < 0.6:
            log.info(f"  [{ticker}] ML prob too low ({prob:.2f}), skipping")
            continue

        setup = build_trade_setup(ind, signal)
        if not setup["rr_valid"]:
            log.info(f"  [{ticker}] RR {setup['rr']} < 1.8, skipping")
            continue

        results.append({
            "ticker": ticker,
            "price": ind["price"],
            "rsi_daily": ind["rsi_daily"],
            "rsi_weekly": ind["rsi_weekly"],
            "macd_confirmed": signal["macd_confirmed"],
            "vol_confirmed": signal["vol_confirmed"],
            "vol_ratio": ind["vol_ratio"],
            "status": signal["status"],
            "reasons": signal["reasons"],
            "warnings": signal.get("warnings", []),
            "probability_upside": prob,
            **setup,
        })

    # Sort by probability desc
    results.sort(key=lambda x: x["probability_upside"], reverse=True)

    # Top 3 are trade setups; rest go to watchlist
    top_setups = [r for r in results if r["status"] in ("Ready", "Prepare")][:3]
    watchlist = [r for r in results if r not in top_setups]

    # Update state
    state = load_state()
    follow_up = update_follow_up(state, current_prices)

    # Add new Ready signals to active trades
    for s in top_setups:
        if s["status"] == "Ready":
            exists = any(t["ticker"] == s["ticker"] for t in state["active_trades"])
            if not exists:
                state["active_trades"].append({
                    "ticker": s["ticker"],
                    "entry": s["entry"],
                    "stop_loss": s["stop_loss"],
                    "target": s["target"],
                    "days_held": 0,
                    "current_price": s["price"],
                    "return_pct": 0,
                })
    save_state(state)

    ai_insight = get_ai_insight(top_setups)

    return {
        "timestamp": datetime.now().isoformat(),
        "top_setups": top_setups,
        "watchlist": watchlist,
        "follow_up": follow_up,
        "ai_insight": ai_insight,
    }


# ─────────────────────────────────────────────
# SINGLE STOCK ANALYZER
# ─────────────────────────────────────────────

def analyze_single(ticker: str) -> dict:
    """Analyze any ticker with full strategy logic."""
    if not ticker.endswith(".NS") and "." not in ticker:
        ticker = ticker.upper() + ".NS"
    data = fetch_data(ticker)
    if data is None:
        return {"error": f"Could not fetch data for {ticker}"}

    ind = calculate_indicators(data)
    if ind is None:
        return {"error": "Indicator calculation failed"}

    signal = generate_signal(ind)
    prob = ml_probability(ind)
    setup = build_trade_setup(ind, signal) if signal["pass"] else {}

    return {
        "ticker": ticker,
        "price": ind["price"],
        "rsi_daily": ind["rsi_daily"],
        "rsi_weekly": ind["rsi_weekly"],
        "macd": ind["macd"],
        "macd_signal": ind["macd_signal"],
        "macd_hist": ind["macd_hist"],
        "vol_ratio": ind["vol_ratio"],
        "signal": signal,
        "probability_upside": prob,
        "setup": setup,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from email_builder import build_and_send_email
    from dashboard_builder import build_dashboard

    results = run_scanner()

    # Save JSON snapshot
    Path("results.json").write_text(json.dumps(results, indent=2, default=str))
    log.info("JSON results saved.")

    # Build and send email
    build_and_send_email(results)

    # Build dashboard
    build_dashboard(results)
    log.info("Done.")
