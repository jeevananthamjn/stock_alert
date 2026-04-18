# 📈 Swing Trade Alert System — Nifty 100

Algorithmic swing trade scanner for large-cap Nifty 100 stocks.  
Sends HTML email alerts twice daily and generates a local dashboard.

---

## Features

| Feature | Detail |
|---|---|
| Universe | Nifty 100 (.NS tickers) |
| Primary signal | RSI < 30 on both Daily & Weekly |
| Confirmation | MACD crossover + volume spike |
| ML model | RandomForest — predicts +5% within 10 days |
| Filters | Large cap, D/E < 0.5, PE < 35 |
| Trade labels | Prepare / Ready / Avoid (no "Buy Now") |
| RR filter | Only setups with RR ≥ 1.8x |
| Email | HTML via Gmail SMTP (2×/day) |
| Dashboard | Self-contained HTML, no server needed |
| AI Insight | Groq API (optional), plain-text fallback |
| CI/CD | GitHub Actions ready |

---

## Quick Start

```bash
git clone <your-repo>
cd stock-alert
pip install -r requirements.txt

# Optional: set credentials
export GMAIL_USER="you@gmail.com"
export GMAIL_PASS="your-app-password"
export TO_EMAIL="alerts@yourdomain.com"
export GROQ_API_KEY="gsk_..."  # optional

python main.py
```

Outputs:
- `results.json` — full scan results
- `dashboard.html` — open in browser
- `email_output.html` — email preview (if no Gmail creds)

---

## GitHub Actions Setup

1. Fork/clone this repo on GitHub
2. Go to **Settings → Secrets → Actions** and add:
   - `GMAIL_USER` — your Gmail address
   - `GMAIL_PASS` — Gmail App Password (not your login password)
   - `TO_EMAIL` — where to receive alerts
   - `GROQ_API_KEY` — optional, for AI commentary
3. The workflow runs automatically:
   - **9:00 AM IST** (pre-market)
   - **6:00 PM IST** (post-market)
   - Or trigger manually from the Actions tab

---

## Gmail App Password

1. Enable 2-Factor Authentication on your Google account
2. Go to: https://myaccount.google.com/apppasswords
3. Create an app password for "Mail"
4. Use that 16-character password as `GMAIL_PASS`

---

## Project Structure

```
stock-alert/
├── main.py              # Entry point + core strategy logic
├── email_builder.py     # HTML email generation + SMTP send
├── dashboard_builder.py # HTML dashboard generation
├── requirements.txt
├── watchlist_state.json # Auto-generated, tracks active trades
├── .github/
│   └── workflows/
│       └── swing_alert.yml
└── README.md
```

---

## Strategy Logic

### Universe Filter
- Large cap (Market Cap > ₹20,000 Cr)
- Debt/Equity < 0.5
- Trailing PE < 35

### Signal (ALL required)
- RSI (14) < 30 on **daily** chart
- RSI (14) < 30 on **weekly** chart
- Price NOT at fresh 10-day breakdown

### Confirmation (ONE required)
- MACD bullish crossover OR histogram turning positive
- Volume > 1.5× 20-day average

### ML Filter
- RandomForest trained on 6 months of price history
- Predicts probability of +5% gain within 10 trading days
- Only proceed if `probability_upside >= 0.60`

### Trade Setup
- **Entry**: Previous day's high (breakout level)
- **Stop Loss**: max(20-day swing low, 5% below current price)
- **Target**: +10% from current price
- **Only take if RR ≥ 1.8x**

---

## Disclaimer

This is an algorithmic screening tool for educational purposes.  
It is **not financial advice**. Always do your own research.  
Past signals do not guarantee future results.  
Use a stop loss on every trade.
