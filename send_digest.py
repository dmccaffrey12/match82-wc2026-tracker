"""
send_digest.py — Daily Match 82 email digest sender.

Run by GitHub Actions each morning at 7 AM PT (14:00 UTC), AFTER the
nightly_precompute.yml job has already written results.json.

Workflow:
  1. Load pre-computed MC results from results.json
  2. Build structured context from generate_digest_paragraph()
  3. Pass context to gpt-4o-mini → get a punchy 3-4 sentence paragraph
  4. Render the 'Most Probable Exact Matchups' chart as an inline PNG
  5. Build a full dark-themed HTML email via generate_digest_email_html()
  6. Read subscriber list from subscribers.csv in the repo root
  7. Send one email per subscriber via Resend

Required environment variables (set as GitHub Actions secrets):
  OPENAI_API_KEY   — OpenAI API key (gpt-4o-mini, ~$0.001/run)
  RESEND_API_KEY   — Resend API key (free up to 3,000 emails/month)

Subscriber list:
  subscribers.csv in the repo root — two columns: email, subscribed_at
  This file is committed to the repo. New signups from the Streamlit app
  are written to /tmp/match82_subscribers.csv on Streamlit Cloud (ephemeral).
  Until you wire persistent storage, manually copy new signups here.
  See README.md for the Supabase upgrade path.

Usage:
  python send_digest.py              # normal run
  python send_digest.py --dry-run    # print email HTML, don't send
  python send_digest.py --to you@example.com  # send to one address only (testing)
"""

import argparse
import csv
import datetime
import os
import sys
import types

# ── Stub streamlit before importing app.py ────────────────────────────────────
_st = types.ModuleType("streamlit")

class _CacheStub:
    def __init__(self, *a, **kw): pass
    def __call__(self, fn): return fn

_st.cache_data      = _CacheStub
_st.set_page_config = lambda **kw: None
_st.markdown        = lambda *a, **kw: None
_st.info            = lambda *a, **kw: None
_st.warning         = lambda *a, **kw: None
_st.spinner         = lambda *a, **kw: types.SimpleNamespace(
    __enter__=lambda s: s, __exit__=lambda *a: None
)
_st.sidebar = types.SimpleNamespace(
    markdown=lambda *a, **kw: None,
    slider=lambda *a, **kw: None,
    checkbox=lambda *a, **kw: None,
    selectbox=lambda *a, **kw: None,
    button=lambda *a, **kw: None,
    toggle=lambda *a, **kw: False,
    caption=lambda *a, **kw: None,
    divider=lambda *a, **kw: None,
    expander=lambda *a, **kw: types.SimpleNamespace(
        __enter__=lambda s: s, __exit__=lambda *a: None
    ),
)
_st.session_state = {}
sys.modules["streamlit"] = _st

sys.path.insert(0, os.path.dirname(__file__))
from app import (  # noqa: E402
    load_precomputed_results,
    run_monte_carlo,
    generate_digest_paragraph,
    generate_digest_email_html,
    build_matchup_distribution,
    compute_chaos_index,
)

# ── Constants ─────────────────────────────────────────────────────────────────
FROM_ADDRESS  = "Match 82 Brief <match82@danielmccaffrey.io>"
REPLY_TO      = "noreply@danielmccaffrey.io"
SUBSCRIBERS_CSV = os.path.join(os.path.dirname(__file__), "subscribers.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_subscribers(filepath: str = SUBSCRIBERS_CSV) -> list[str]:
    """Return list of subscriber email addresses from CSV."""
    if not os.path.exists(filepath):
        print(f"[send_digest] No subscribers file found at {filepath}")
        return []
    emails = []
    with open(filepath, newline="") as f:
        for row in csv.reader(f):
            if row and "@" in row[0]:
                emails.append(row[0].strip().lower())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in emails:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


def write_llm_paragraph(context: str) -> str:
    """
    Call gpt-4o-mini to write a sharp, data-driven paragraph from the
    structured context string produced by generate_digest_paragraph().

    Falls back to the raw context string if the API call fails.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sharp soccer analyst writing a daily Match 82 odds brief "
                        "for smart fans who want real numbers, not hype. "
                        "Write exactly 3-4 sentences. "
                        "Be specific — include the actual percentages. "
                        "Sound like The Athletic, not ESPN. "
                        "No markdown formatting, no bullet points, plain prose only. "
                        "End with one sentence about what to watch in today's matches "
                        "that could shift the probabilities."
                    ),
                },
                {"role": "user", "content": context},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[send_digest] OpenAI call failed ({e}), falling back to template paragraph.")
        return context


def build_chart_png(mc: dict) -> bytes | None:
    """
    Renders the top-12 most probable Match 82 matchups as a PNG using
    matplotlib — no kaleido, no Chromium, works in any headless environment.
    Returns PNG bytes or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend, must be set before pyplot import
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import io

        joint = mc.get("match82_joint_prob", {})
        if not joint:
            return None

        sorted_pairs = sorted(joint.items(), key=lambda x: x[1], reverse=True)[:12]

        labels = [f"{a} vs {b}" for (a, b), _ in sorted_pairs]
        values = [v * 100 for _, v in sorted_pairs]

        # Dark theme colors
        BG      = "#080f1c"
        BAR     = "#3b82f6"
        TEXT    = "#cbd5e1"
        SUBTEXT = "#475569"
        GRID    = "#1e3a5f"

        # Emoji flags can't render in matplotlib PNG (bitmap font limitation)
        # Use clean team names only — chart still looks sharp
        plt.rcParams["font.family"] = "DejaVu Sans"

        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        fig.subplots_adjust(left=0.38, right=0.88, top=0.90, bottom=0.08)

        n = len(labels)
        y_pos = list(range(n))
        # Highest probability at top
        bars = ax.barh(y_pos, values[::-1], color=BAR, height=0.55,
                       zorder=3, alpha=0.92)
        # Subtle bar edge
        for bar in bars:
            bar.set_edgecolor("#1d4ed8")
            bar.set_linewidth(0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[::-1], color=TEXT, fontsize=9)
        ax.set_xlabel("Probability (%)", color=SUBTEXT, fontsize=9, labelpad=8)
        ax.set_title("Most Probable Exact Matchups — Match 82",
                     color="#f8fafc", fontsize=12, fontweight="bold", pad=14)

        # Value labels to the right of each bar
        max_v = max(values)
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + max_v * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", color="#f8fafc",
                    fontsize=8.5, fontweight="600")

        ax.tick_params(axis="both", colors=SUBTEXT, labelsize=9)
        ax.tick_params(axis="y", length=0)  # no y-tick marks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_edgecolor(GRID)
        ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--", zorder=0)
        ax.set_xlim(0, max_v * 1.22)
        ax.set_ylim(-0.6, n - 0.4)

        n_sims = mc.get("n_sims", 50000)
        fig.text(0.99, 0.01,
                 f"{n_sims:,}-trial Dixon-Coles / Elo Monte Carlo  ·  Match 82 · Lumen Field · July 1, 2026",
                 ha="right", color=SUBTEXT, fontsize=7)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"[build_chart_png] Failed: {e}")
        return None


def send_email(resend_client, to: str, subject: str, html: str,
               attachment_png: bytes | None = None,
               dry_run: bool = False) -> bool:
    """Send one email via Resend with optional PNG attachment. Returns True on success."""
    if dry_run:
        print(f"[DRY RUN] Would send to: {to}")
        return True
    try:
        import base64
        payload = {
            "from":     FROM_ADDRESS,
            "reply_to": REPLY_TO,
            "to":       to,
            "subject":  subject,
            "html":     html,
        }
        if attachment_png:
            payload["attachments"] = [{
                "filename": "match82_matchups.png",
                "content":  list(attachment_png),  # Resend expects list of ints
            }]
        resend_client.Emails.send(payload)
        return True
    except Exception as e:
        print(f"[send_digest] Failed to send to {to}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Send the daily Match 82 email digest.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print email HTML to stdout, don't send anything.")
    parser.add_argument("--to", type=str, default=None,
                        help="Send to a single address only (for testing).")
    parser.add_argument("--sims", type=int, default=50_000,
                        help="MC simulations to run if no results.json found.")
    args = parser.parse_args()

    today = datetime.date.today().strftime("%B %d, %Y")
    subject = f"Match 82 Brief — {today}"

    print(f"[send_digest] Starting digest run for {today}")

    # ── 1. Load MC results ────────────────────────────────────────────────────
    mc = load_precomputed_results()
    if mc:
        age = mc.get("_age_hours", "?")
        print(f"[send_digest] Loaded pre-computed snapshot ({age}h old)")
    else:
        print(f"[send_digest] No snapshot found — running live {args.sims:,}-trial simulation...")
        mc = run_monte_carlo(n_sims=args.sims, use_markets=False)
        print("[send_digest] Simulation complete")

    # ── 2. Build paragraph ────────────────────────────────────────────────────
    context = generate_digest_paragraph(mc)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print("[send_digest] Calling gpt-4o-mini for paragraph...")
        paragraph = write_llm_paragraph(context)
    else:
        print("[send_digest] No OPENAI_API_KEY — using template paragraph")
        paragraph = context

    print(f"[send_digest] Paragraph ({len(paragraph)} chars):\n{paragraph}\n")

    # ── 3. Build chart PNG attachment ────────────────────────────────────────
    print("[send_digest] Rendering chart as PNG attachment...")
    chart_png = build_chart_png(mc)
    if chart_png:
        print(f"[send_digest] Chart PNG ready ({len(chart_png):,} bytes)")
    else:
        print("[send_digest] Chart PNG failed — will send without attachment")

    # ── 4. Build HTML email body (clean, Gmail-safe) ──────────────────────────
    print("[send_digest] Building HTML email...")
    html_body = generate_digest_email_html(mc, paragraph)

    if args.dry_run:
        dry_run_path = os.path.join(os.path.dirname(__file__), "digest_preview.html")
        with open(dry_run_path, "w") as f:
            f.write(html_body)
        print(f"[DRY RUN] HTML written to {dry_run_path} — open in browser to preview.")
        return

    # ── 5. Load subscribers ───────────────────────────────────────────────────
    if args.to:
        subscribers = [args.to]
        print(f"[send_digest] Test mode — sending to {args.to} only")
    else:
        subscribers = load_subscribers()
        print(f"[send_digest] {len(subscribers)} subscriber(s) found")

    if not subscribers:
        print("[send_digest] No subscribers — nothing to send. Exiting.")
        return

    # ── 6. Send via Resend ────────────────────────────────────────────────────
    resend_key = os.environ.get("RESEND_API_KEY")
    if not resend_key:
        print("[send_digest] ERROR: RESEND_API_KEY not set. Exiting.")
        sys.exit(1)

    import resend
    resend.api_key = resend_key

    sent = 0
    failed = 0
    for email in subscribers:
        ok = send_email(resend, email, subject, html_body,
                        attachment_png=chart_png, dry_run=args.dry_run)
        if ok:
            sent += 1
            print(f"[send_digest] ✓ {email}")
        else:
            failed += 1

    print(f"\n[send_digest] Done. Sent: {sent} | Failed: {failed}")


if __name__ == "__main__":
    main()
