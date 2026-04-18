"""
Email builder — generates HTML email and sends via Gmail SMTP.
"""

import os
import smtplib
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

STATUS_COLOR = {"Ready": "#16a34a", "Prepare": "#d97706", "Avoid": "#dc2626"}


def _status_badge(status: str) -> str:
    color = STATUS_COLOR.get(status, "#6b7280")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:700;">{status.upper()}</span>'


def _setup_card(s: dict, rank: int) -> str:
    reasons_html = "".join(f"<li>{r}</li>" for r in s["reasons"])
    warnings_html = ""
    if s.get("warnings"):
        warnings_html = "<li style='color:#b45309;'>" + "</li><li style='color:#b45309;'>".join(s["warnings"]) + "</li>"

    return f"""
    <div style="border:1px solid #e5e7eb;border-radius:12px;padding:20px;margin-bottom:16px;background:#fafafa;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div>
          <span style="font-size:11px;color:#6b7280;font-weight:600;">#{rank} SETUP</span>
          <h3 style="margin:2px 0;font-size:20px;font-weight:800;color:#111827;">{s['ticker'].replace('.NS','')}</h3>
          <span style="font-size:15px;color:#374151;">₹{s['price']:.2f}</span>
        </div>
        <div style="text-align:right;">
          {_status_badge(s['status'])}
          <div style="margin-top:6px;font-size:18px;font-weight:700;color:#1d4ed8;">{s['probability_upside']*100:.0f}%<span style="font-size:11px;font-weight:400;color:#6b7280;"> ML upside prob</span></div>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <tr>
          <td style="padding:4px 8px;color:#6b7280;">Entry (breakout)</td>
          <td style="padding:4px 8px;font-weight:600;">₹{s['entry']:.2f}</td>
          <td style="padding:4px 8px;color:#6b7280;">Stop Loss</td>
          <td style="padding:4px 8px;font-weight:600;color:#dc2626;">₹{s['stop_loss']:.2f} ({s['risk_pct']:.1f}%)</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;color:#6b7280;">Target</td>
          <td style="padding:4px 8px;font-weight:600;color:#16a34a;">₹{s['target']:.2f} (+10%)</td>
          <td style="padding:4px 8px;color:#6b7280;">Risk/Reward</td>
          <td style="padding:4px 8px;font-weight:600;">{s['rr']:.1f}x {'✅' if s['rr_valid'] else '⚠️'}</td>
        </tr>
        <tr>
          <td style="padding:4px 8px;color:#6b7280;">RSI Daily / Weekly</td>
          <td style="padding:4px 8px;">{s['rsi_daily']:.1f} / {s['rsi_weekly']:.1f}</td>
          <td style="padding:4px 8px;color:#6b7280;">Volume Ratio</td>
          <td style="padding:4px 8px;">{s.get('vol_ratio',1):.1f}x</td>
        </tr>
      </table>
      <div style="margin-top:12px;font-size:13px;">
        <strong>Why this setup:</strong>
        <ul style="margin:6px 0 0 0;padding-left:18px;color:#374151;">
          {reasons_html}{warnings_html}
        </ul>
      </div>
    </div>
    """


def _watchlist_row(s: dict) -> str:
    return f"""
    <tr style="border-bottom:1px solid #f3f4f6;">
      <td style="padding:8px;font-weight:600;">{s['ticker'].replace('.NS','')}</td>
      <td style="padding:8px;">₹{s['price']:.2f}</td>
      <td style="padding:8px;">{s['rsi_daily']:.1f}</td>
      <td style="padding:8px;">{s['rsi_weekly']:.1f}</td>
      <td style="padding:8px;">{s['probability_upside']*100:.0f}%</td>
      <td style="padding:8px;">{_status_badge(s['status'])}</td>
    </tr>
    """


def _follow_up_row(t: dict) -> str:
    ret = t.get("return_pct", 0)
    color = "#16a34a" if ret >= 0 else "#dc2626"
    sign = "+" if ret >= 0 else ""
    closed = t.get("closed", "")
    return f"""
    <tr style="border-bottom:1px solid #f3f4f6;">
      <td style="padding:8px;font-weight:600;">{t['ticker'].replace('.NS','')}</td>
      <td style="padding:8px;">₹{t['entry']:.2f}</td>
      <td style="padding:8px;">₹{t.get('current_price', t['entry']):.2f}</td>
      <td style="padding:8px;font-weight:700;color:{color};">{sign}{ret:.1f}%</td>
      <td style="padding:8px;">{t.get('days_held',0)}d</td>
      <td style="padding:8px;">{closed if closed else '🔄 Active'}</td>
    </tr>
    """


def build_email_html(results: dict, session: str) -> str:
    ts = datetime.now().strftime("%d %b %Y, %I:%M %p IST")
    session_label = "🌅 Pre-Market" if session == "morning" else "🌆 Post-Market"

    setups_html = ""
    for i, s in enumerate(results["top_setups"], 1):
        setups_html += _setup_card(s, i)
    if not setups_html:
        setups_html = '<p style="color:#6b7280;font-style:italic;">No high-confidence setups today. Patience is a position.</p>'

    watchlist_rows = "".join(_watchlist_row(s) for s in results.get("watchlist", []))
    follow_rows = "".join(_follow_up_row(t) for t in results.get("follow_up", []))

    ai_block = f"""
    <div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:16px;border-radius:0 8px 8px 0;margin-bottom:8px;">
      <strong style="color:#1e40af;">🤖 AI Insight</strong>
      <p style="margin:8px 0 0 0;color:#1e3a5f;font-size:14px;line-height:1.6;">{results.get('ai_insight','No insight available.')}</p>
    </div>
    """

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f9fafb;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="max-width:620px;margin:24px auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.07);">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#1e3a5f 0%,#1d4ed8 100%);padding:28px 28px 20px;">
      <div style="color:#93c5fd;font-size:12px;font-weight:600;letter-spacing:1px;">SWING TRADE ALERT</div>
      <h1 style="color:#fff;margin:4px 0;font-size:24px;font-weight:800;">Nifty 100 Scanner</h1>
      <div style="color:#bfdbfe;font-size:13px;">{session_label} &nbsp;·&nbsp; {ts}</div>
    </div>

    <div style="padding:24px 28px;">

      <!-- AI Insight -->
      <h2 style="font-size:16px;font-weight:700;color:#111827;margin:0 0 12px;">💡 Market Context</h2>
      {ai_block}

      <!-- Top Setups -->
      <h2 style="font-size:16px;font-weight:700;color:#111827;margin:24px 0 12px;">🎯 Top Trade Setups</h2>
      <p style="font-size:12px;color:#9ca3af;margin:0 0 12px;">Stocks with RSI&lt;30 (daily+weekly), ML prob ≥60%, RR ≥1.8x</p>
      {setups_html}

      <!-- Watchlist -->
      {'<h2 style="font-size:16px;font-weight:700;color:#111827;margin:24px 0 12px;">👁 Watchlist (Near Setups)</h2><table style="width:100%;border-collapse:collapse;font-size:13px;"><thead><tr style="background:#f3f4f6;"><th style="padding:8px;text-align:left;">Ticker</th><th style="padding:8px;text-align:left;">Price</th><th style="padding:8px;text-align:left;">RSI-D</th><th style="padding:8px;text-align:left;">RSI-W</th><th style="padding:8px;text-align:left;">ML%</th><th style="padding:8px;text-align:left;">Status</th></tr></thead><tbody>' + watchlist_rows + '</tbody></table>' if watchlist_rows else ''}

      <!-- Follow Up -->
      {'<h2 style="font-size:16px;font-weight:700;color:#111827;margin:24px 0 12px;">📊 Follow-Up Tracker</h2><table style="width:100%;border-collapse:collapse;font-size:13px;"><thead><tr style="background:#f3f4f6;"><th style="padding:8px;text-align:left;">Ticker</th><th style="padding:8px;text-align:left;">Entry</th><th style="padding:8px;text-align:left;">Current</th><th style="padding:8px;text-align:left;">Return</th><th style="padding:8px;text-align:left;">Days</th><th style="padding:8px;text-align:left;">Status</th></tr></thead><tbody>' + follow_rows + '</tbody></table>' if follow_rows else ''}

    </div>

    <!-- Footer -->
    <div style="background:#f9fafb;padding:16px 28px;border-top:1px solid #e5e7eb;">
      <p style="font-size:11px;color:#9ca3af;margin:0;line-height:1.6;">
        ⚠️ <strong>Disclaimer:</strong> This is an algorithmic screening tool. Not financial advice.
        All trade decisions are your own responsibility. Past signals do not guarantee future results.
        Always use a stop loss.
      </p>
    </div>

  </div>
</body>
</html>
"""


def build_and_send_email(results: dict, session: str = "auto"):
    if session == "auto":
        hour = datetime.now().hour
        session = "morning" if hour < 12 else "evening"

    html = build_email_html(results, session)

    gmail_user = os.getenv("GMAIL_USER", "")
    gmail_pass = os.getenv("GMAIL_PASS", "")
    to_email = os.getenv("TO_EMAIL", gmail_user)

    if not gmail_user or not gmail_pass:
        log.warning("Gmail credentials not set. Saving email to email_output.html only.")
        from pathlib import Path
        Path("email_output.html").write_text(html)
        return

    subject = f"[Swing Alert] Nifty 100 — {len(results['top_setups'])} setup(s) | {datetime.now().strftime('%d %b %Y')}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to_email, msg.as_string())
        log.info(f"Email sent to {to_email}")
    except Exception as e:
        log.error(f"Email send failed: {e}")
        from pathlib import Path
        Path("email_output.html").write_text(html)
        log.info("Email saved locally as email_output.html")
