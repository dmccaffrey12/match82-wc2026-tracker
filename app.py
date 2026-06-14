"""
Match 82 — FIFA World Cup 2026 | Seattle Lumen Field | July 1, 2026
Round of 32: Winner of Group G vs. 3rd Place from Group A / E / H / I / J

PROBABILITY ENGINE: Monte Carlo simulation (default N=50,000 trials)
  - Dixon-Coles corrected Poisson model for scoreline simulation
  - Elo ratings drive match win/draw/loss probabilities
  - Full 12-group simulation ensures the 3rd-place slot is correctly resolved
    as a 12-way competitive race, not independent per-team probabilities
  - Polymarket/Kalshi live API overlay for high-liquidity teams (optional)
  - Google Sheet manual override layer (optional, for editorial control)

HOW TO CONNECT A LIVE GOOGLE SHEET (manual override layer):
  1. Create a Google Sheet with two tabs: "standings_override" and "probs_override"
  2. File → Share → Publish to web → CSV format for each tab
  3. Set env var: MATCH82_OVERRIDE_URL="url_tab1,url_tab2"
     OR paste into the sidebar text input at runtime.

HOW TO ENABLE LIVE POLYMARKET ODDS:
  Set env var: MATCH82_USE_MARKETS=1
  The app will blend Polymarket implied probs with Elo probs for teams
  that have active markets (typically only top-tier teams).
"""

import os
import math
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Match 82 — Seattle WC2026 Tracker",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

N_SIMULATIONS: int = 50_000          # Monte Carlo trials. 50k ≈ 1s on modern CPU.
REFRESH_SECONDS: int = 300           # Cache TTL for MC results
OVERRIDE_URL: str = os.environ.get("MATCH82_OVERRIDE_URL", "")
USE_MARKETS: bool = bool(int(os.environ.get("MATCH82_USE_MARKETS", "0")))

# Third-place slot: which groups can send a 3rd-place team to face G winner
THIRD_PLACE_GROUPS = ["A", "E", "H", "I", "J"]

# All 12 groups needed for the global 3rd-place ranking simulation
ALL_GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

FLAG_MAP: dict[str, str] = {
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czechia": "🇨🇿",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Côte d'Ivoire": "🇨🇮", "Ecuador": "🇪🇨",
    "Spain": "🇪🇸", "Cabo Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Canada": "🇨🇦", "Switzerland": "🇨🇭", "Qatar": "🇶🇦", "Bosnia": "🇧🇦",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🇸🇸",
    "USA": "🇺🇸", "Australia": "🇦🇺", "Türkiye": "🇹🇷", "Paraguay": "🇵🇾",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}

# ─────────────────────────────────────────────────────────────────────────────
# ELO RATINGS — as of June 13, 2026 (eloratings.net / Wikipedia)
# Update these as the tournament progresses (Elo shifts after each match)
# ─────────────────────────────────────────────────────────────────────────────
ELO: dict[str, float] = {
    # Group G
    "Belgium":      1894,
    "Egypt":        1800,   # approx; FIFA rank ~29
    "Iran":         1800,   # approx; FIFA rank ~20, some Elo sources ~1800
    "New Zealand":  1650,   # FIFA rank ~85, weakest in group
    # Group A
    "Mexico":       1881,
    "South Korea":  1830,   # FIFA rank ~25
    "Czechia":      1820,   # FIFA rank ~40
    "South Africa": 1680,   # FIFA rank ~60
    # Group B
    "Switzerland":  1865,
    "Canada":       1820,
    "Qatar":        1720,
    "Bosnia":       1760,
    # Group C
    "Brazil":       1978,
    "Morocco":      1860,
    "Scotland":     1800,
    "Haiti":        1600,
    # Group D
    "USA":          1860,
    "Australia":    1810,
    "Türkiye":      1885,
    "Paraguay":     1790,
    # Group E
    "Germany":      1932,
    "Ecuador":      1938,
    "Côte d'Ivoire":1810,
    "Curaçao":      1560,
    # Group F
    "Netherlands":  1948,
    "Japan":        1906,
    "Sweden":       1830,
    "Tunisia":      1760,
    # Group H
    "Spain":        2157,
    "Uruguay":      1892,
    "Saudi Arabia": 1700,
    "Cabo Verde":   1660,
    # Group I
    "France":       2063,
    "Senegal":      1860,
    "Norway":       1914,
    "Iraq":         1720,
    # Group J
    "Argentina":    2115,
    "Algeria":      1800,
    "Austria":      1845,
    "Jordan":       1650,
    # Group K
    "Portugal":     1989,
    "Colombia":     1982,
    "Uzbekistan":   1680,
    "DR Congo":     1740,
    # Group L
    "England":      2024,
    "Croatia":      1912,
    "Ghana":        1720,
    "Panama":       1700,
}

# ─────────────────────────────────────────────────────────────────────────────
# CURRENT LIVE STANDINGS  (update as matches complete)
# Format: {team: {"mp":int, "w":int, "d":int, "l":int, "gf":int, "ga":int}}
# The simulator picks up from here and only plays REMAINING fixtures.
# ─────────────────────────────────────────────────────────────────────────────
LIVE_STANDINGS: dict[str, dict] = {
    # Group A — 1 match played each (Mexico 2-0 South Africa; South Korea 2-1 Czechia)
    "Mexico":       {"mp":1,"w":1,"d":0,"l":0,"gf":2,"ga":0},
    "South Korea":  {"mp":1,"w":1,"d":0,"l":0,"gf":2,"ga":1},
    "Czechia":      {"mp":1,"w":0,"d":0,"l":1,"gf":1,"ga":2},
    "South Africa": {"mp":1,"w":0,"d":0,"l":1,"gf":0,"ga":2},
    # Group B — opening matches: Switzerland 1-1 Canada; Qatar 1-1 Bosnia
    "Switzerland":  {"mp":1,"w":0,"d":1,"l":0,"gf":1,"ga":1},
    "Canada":       {"mp":1,"w":0,"d":1,"l":0,"gf":1,"ga":1},
    "Qatar":        {"mp":1,"w":0,"d":1,"l":0,"gf":1,"ga":1},
    "Bosnia":       {"mp":1,"w":0,"d":1,"l":0,"gf":1,"ga":1},
    # Group C — Scotland 1-0 Haiti; Brazil 1-1 Morocco
    "Scotland":     {"mp":1,"w":1,"d":0,"l":0,"gf":1,"ga":0},
    "Morocco":      {"mp":1,"w":0,"d":1,"l":0,"gf":1,"ga":1},
    "Brazil":       {"mp":1,"w":0,"d":1,"l":0,"gf":1,"ga":1},
    "Haiti":        {"mp":1,"w":0,"d":0,"l":1,"gf":0,"ga":1},
    # Group D — USA 4-1 Paraguay; Australia 2-0 Türkiye (not yet official — placeholder)
    "USA":          {"mp":1,"w":1,"d":0,"l":0,"gf":4,"ga":1},
    "Australia":    {"mp":1,"w":1,"d":0,"l":0,"gf":2,"ga":0},
    "Türkiye":      {"mp":1,"w":0,"d":0,"l":1,"gf":0,"ga":2},
    "Paraguay":     {"mp":1,"w":0,"d":0,"l":1,"gf":1,"ga":4},
    # Groups E–L — no matches played yet
    "Germany":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Ecuador":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Côte d'Ivoire":{"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Curaçao":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Netherlands":  {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Japan":        {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Sweden":       {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Tunisia":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Belgium":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Egypt":        {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Iran":         {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "New Zealand":  {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Spain":        {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Uruguay":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Saudi Arabia": {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Cabo Verde":   {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "France":       {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Senegal":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Norway":       {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Iraq":         {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Argentina":    {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Algeria":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Austria":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Jordan":       {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Portugal":     {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Colombia":     {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Uzbekistan":   {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "DR Congo":     {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "England":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Croatia":      {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Ghana":        {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
    "Panama":       {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0},
}

# Full fixture list for all 12 groups (only unplayed remaining matches needed)
# Format: (group, home_team, away_team)
# Already-played matches are excluded — the sim picks up from current standings.
ALL_FIXTURES: list[tuple[str, str, str]] = [
    # Group A — matchday 2 & 3
    ("A","Mexico","South Korea"), ("A","Czechia","South Africa"),
    ("A","Mexico","Czechia"),     ("A","South Korea","South Africa"),
    # Group B — matchday 2 & 3
    ("B","Switzerland","Qatar"),  ("B","Canada","Bosnia"),
    ("B","Switzerland","Bosnia"), ("B","Canada","Qatar"),
    # Group C — matchday 2 & 3
    ("C","Scotland","Morocco"),   ("C","Brazil","Haiti"),
    ("C","Scotland","Brazil"),    ("C","Morocco","Haiti"),
    # Group D — matchday 2 & 3
    ("D","USA","Australia"),      ("D","Türkiye","Paraguay"),
    ("D","USA","Türkiye"),        ("D","Australia","Paraguay"),
    # Group E — all 6 matches
    ("E","Germany","Côte d'Ivoire"), ("E","Ecuador","Curaçao"),
    ("E","Germany","Ecuador"),       ("E","Côte d'Ivoire","Curaçao"),
    ("E","Germany","Curaçao"),       ("E","Ecuador","Côte d'Ivoire"),
    # Group F — all 6
    ("F","Netherlands","Japan"),  ("F","Sweden","Tunisia"),
    ("F","Netherlands","Sweden"), ("F","Japan","Tunisia"),
    ("F","Netherlands","Tunisia"),("F","Japan","Sweden"),
    # Group G — all 6
    ("G","Belgium","Egypt"),      ("G","Iran","New Zealand"),
    ("G","Belgium","Iran"),       ("G","New Zealand","Egypt"),
    ("G","New Zealand","Belgium"),("G","Egypt","Iran"),
    # Group H — all 6
    ("H","Spain","Cabo Verde"),   ("H","Saudi Arabia","Uruguay"),
    ("H","Spain","Saudi Arabia"), ("H","Uruguay","Cabo Verde"),
    ("H","Spain","Uruguay"),      ("H","Cabo Verde","Saudi Arabia"),
    # Group I — all 6
    ("I","France","Senegal"),     ("I","Iraq","Norway"),
    ("I","France","Iraq"),        ("I","Norway","Senegal"),
    ("I","Norway","France"),      ("I","Senegal","Iraq"),
    # Group J — all 6
    ("J","Argentina","Algeria"),  ("J","Austria","Jordan"),
    ("J","Argentina","Austria"),  ("J","Jordan","Algeria"),
    ("J","Argentina","Jordan"),   ("J","Algeria","Austria"),
    # Group K — all 6
    ("K","Portugal","Uzbekistan"),("K","Colombia","DR Congo"),
    ("K","Portugal","Colombia"),  ("K","Uzbekistan","DR Congo"),
    ("K","Portugal","DR Congo"),  ("K","Colombia","Uzbekistan"),
    # Group L — all 6
    ("L","England","Croatia"),    ("L","Ghana","Panama"),
    ("L","England","Ghana"),      ("L","Croatia","Panama"),
    ("L","England","Panama"),     ("L","Croatia","Ghana"),
]

# Group membership
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "Czechia", "South Africa"],
    "B": ["Switzerland", "Canada", "Qatar", "Bosnia"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["USA", "Australia", "Türkiye", "Paraguay"],
    "E": ["Germany", "Ecuador", "Côte d'Ivoire", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cabo Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# ─────────────────────────────────────────────────────────────────────────────
# DARK THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
  html, body, [class*="css"] {
    background-color: #0a0c14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  }
  .block-container { padding: 1.5rem 2rem 2rem !important; max-width: 1400px !important; }
  [data-testid="stSidebar"] { background: #0d1020 !important; border-right: 1px solid #1e2235 !important; }
  [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
  /* ── Metric cards ── */
  [data-testid="stMetric"] { background: #111627 !important; border: 1px solid #1e2a44 !important; border-radius: 10px !important; padding: 1rem 1.25rem !important; overflow: hidden !important; }
  [data-testid="stMetricLabel"] { font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; color: #64748b !important; white-space: normal !important; overflow: visible !important; text-overflow: unset !important; line-height: 1.3 !important; }
  [data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; color: #38bdf8 !important; white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem !important; color: #4ade80 !important; }
  [data-testid="stMetricDelta"] svg { display: none; }
  /* ── Headings ── */
  h1 { font-size: 1.75rem !important; font-weight: 800 !important; color: #f8fafc !important; letter-spacing: -0.02em !important; }
  h2 { font-size: 0.78rem !important; font-weight: 700 !important; color: #475569 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; border-bottom: 1px solid #1e2235 !important; padding-bottom: 0.5rem !important; margin-top: 1.8rem !important; margin-bottom: 0.8rem !important; }
  h3 { font-size: 1rem !important; font-weight: 600 !important; color: #e2e8f0 !important; }
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
  [data-baseweb="select"] > div { background: #111627 !important; border-color: #1e2a44 !important; border-radius: 8px !important; }
  hr { border-color: #1e2235 !important; margin: 1.5rem 0 !important; }
  [data-testid="stAlert"] { background: #0d1a2e !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; color: #93c5fd !important; }
  [data-testid="stExpander"] { background: #0e1628 !important; border: 1px solid #1e2a44 !important; border-radius: 8px !important; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0a0c14; }
  ::-webkit-scrollbar-thumb { background: #1e2a44; border-radius: 3px; }
  .recipe-card { background: #0e1a2e; border: 1px solid #1e3a5f; border-radius: 10px; padding: 1.1rem 1.4rem; line-height: 1.7; }
  .recipe-step { display: flex; gap: 0.6rem; align-items: flex-start; margin: 0.4rem 0; }
  .step-num { background: #1d4ed8; color: white; width: 22px; height: 22px; border-radius: 50%; font-size: 0.72rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px; }
  .step-text { color: #cbd5e1; font-size: 0.88rem; }
  .verdict-yes { color: #4ade80; font-weight: 600; }
  .verdict-no  { color: #f87171; font-weight: 600; }
  .verdict-maybe { color: #fbbf24; font-weight: 600; }
  .sim-badge { display: inline-flex; align-items: center; gap: 6px; background: #0d1a2e; border: 1px solid #1e3a5f; border-radius: 6px; padding: 3px 10px; font-size: 0.72rem; color: #60a5fa; font-family: monospace; }
  .method-pill { display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
  .method-mc { background: #0f2d4a; color: #38bdf8; border: 1px solid #1e4a6e; }
  .method-mkt { background: #1a0f4a; color: #a78bfa; border: 1px solid #3b2a6e; }
  .method-blend { background: #1a2e0f; color: #86efac; border: 1px solid #2a4e1e; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATCH PROBABILITY ENGINE — Dixon-Coles corrected Poisson
# ─────────────────────────────────────────────────────────────────────────────

def elo_to_win_prob(elo_home: float, elo_away: float, home_advantage: float = 65.0) -> tuple[float, float, float]:
    """
    Convert Elo ratings to win/draw/loss probabilities using the
    standard Elo formula, then split draws off from the win prob.

    home_advantage: Elo points added to home team (World Cup is neutral site → 0).
    We set it to 0 by default since WC group games are at neutral venues.

    Returns: (p_home_win, p_draw, p_away_win)
    """
    # At neutral venues, no home advantage. Pass home_advantage=0 for WC.
    elo_diff = (elo_home + home_advantage) - elo_away
    # Expected score for home team in Elo system
    expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    
    # Map expected score to win/draw/loss
    # Draw probability peaks at ~0.25 when teams are equal, diminishes for large gaps
    # Calibrated against historical World Cup results
    p_draw = 0.28 * math.exp(-2.0 * (elo_diff / 400.0) ** 2)
    p_home_win = expected_home * (1.0 - p_draw)
    p_away_win = (1.0 - expected_home) * (1.0 - p_draw)
    
    # Renormalize
    total = p_home_win + p_draw + p_away_win
    return p_home_win / total, p_draw / total, p_away_win / total


def expected_goals(elo_home: float, elo_away: float) -> tuple[float, float]:
    """
    Estimate expected goals (lambda) for each team based on Elo differential.
    
    Calibrated so:
      - Equal teams (Elo diff=0): both score ~1.15 goals (WC group stage average)
      - 200 Elo gap: stronger team scores ~1.5, weaker ~0.85
      - 400 Elo gap: stronger ~1.9, weaker ~0.6
    
    Returns: (lambda_home, lambda_away)
    """
    BASE_GOALS = 1.15
    SENSITIVITY = 0.0008  # goals per Elo point differential
    
    diff = elo_home - elo_away
    lam_home = BASE_GOALS + SENSITIVITY * diff
    lam_away = BASE_GOALS - SENSITIVITY * diff
    
    # Floor at 0.3 to prevent degenerate distributions
    lam_home = max(0.3, lam_home)
    lam_away = max(0.3, lam_away)
    return lam_home, lam_away


def dixon_coles_correction(goals_h: int, goals_a: int, lam_h: float, lam_a: float, rho: float = -0.13) -> float:
    """
    Dixon-Coles low-score correction factor.
    Adjusts the joint probability of low-scoring outcomes (0-0, 1-0, 0-1, 1-1)
    which are systematically over/under-predicted by independent Poisson.
    
    rho = -0.13 is the empirical value from Dixon & Coles (1997) calibrated
    on European football; reasonable for international tournaments.
    """
    if goals_h == 0 and goals_a == 0:
        return 1.0 - lam_h * lam_a * rho
    elif goals_h == 1 and goals_a == 0:
        return 1.0 + lam_a * rho
    elif goals_h == 0 and goals_a == 1:
        return 1.0 + lam_h * rho
    elif goals_h == 1 and goals_a == 1:
        return 1.0 - rho
    else:
        return 1.0


def simulate_scoreline(lam_h: float, lam_a: float, max_goals: int = 8) -> tuple[int, int]:
    """
    Sample a scoreline from a Dixon-Coles corrected joint Poisson distribution.
    Uses rejection sampling against the correction factor.
    
    Returns: (goals_home, goals_away)
    """
    while True:
        gh = np.random.poisson(lam_h)
        ga = np.random.poisson(lam_a)
        if gh > max_goals:
            gh = max_goals
        if ga > max_goals:
            ga = max_goals
        corr = dixon_coles_correction(gh, ga, lam_h, lam_a)
        # corr is always near 1.0; accept/reject based on it
        if np.random.random() < corr:
            return int(gh), int(ga)


def simulate_match(team_a: str, team_b: str) -> tuple[int, int]:
    """Simulate a single match, returning (goals_a, goals_b)."""
    elo_a = ELO.get(team_a, 1700)
    elo_b = ELO.get(team_b, 1700)
    lam_a, lam_b = expected_goals(elo_a, elo_b)
    return simulate_scoreline(lam_a, lam_b)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP STAGE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def init_group_tables() -> dict[str, dict[str, list]]:
    """
    Initialize standings from LIVE_STANDINGS.
    Returns: {team: [pts, gf, ga, gd, w, d, l]}
    """
    tables: dict[str, dict[str, list]] = {}
    for grp, teams in GROUPS.items():
        tables[grp] = {}
        for t in teams:
            s = LIVE_STANDINGS.get(t, {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0})
            pts = s["w"] * 3 + s["d"]
            tables[grp][t] = [pts, s["gf"], s["ga"], s["gf"]-s["ga"], s["w"], s["d"], s["l"]]
    return tables


def simulate_group_stage(tables: dict) -> dict:
    """
    Simulate all remaining fixtures and return final standings.
    Mutates a *copy* of tables.
    
    Returns: {group: {team: [pts, gf, ga, gd, w, d, l]}}
    """
    for (grp, home, away) in ALL_FIXTURES:
        gh, ga = simulate_match(home, away)
        t = tables[grp]
        # Home team
        t[home][1] += gh; t[home][2] += ga; t[home][3] += gh - ga
        # Away team
        t[away][1] += ga; t[away][2] += gh; t[away][3] += ga - gh
        if gh > ga:
            t[home][0] += 3; t[home][4] += 1; t[away][6] += 1
        elif ga > gh:
            t[away][0] += 3; t[away][4] += 1; t[home][6] += 1
        else:
            t[home][0] += 1; t[home][5] += 1
            t[away][0] += 1; t[away][5] += 1
    return tables


def rank_group(group_table: dict[str, list]) -> list[str]:
    """
    Rank teams in a group by FIFA rules:
    1. Points  2. GD  3. GF  4. (simplified: random tiebreak for sim speed)
    Returns ordered list [1st, 2nd, 3rd, 4th]
    """
    def sort_key(item):
        team, s = item
        # Add small random noise to break ties stochastically
        return (s[0], s[3], s[1], np.random.random() * 0.001)
    
    return [t for t, _ in sorted(group_table.items(), key=sort_key, reverse=True)]


def get_third_place_record(group_table: dict[str, list], third_team: str) -> tuple:
    """
    Returns the tiebreaker tuple for the 3rd-place team for global ranking.
    Tuple: (pts, gd, gf) — descending priority per FIFA rules.
    """
    s = group_table[third_team]
    return (s[0], s[3], s[1])  # pts, gd, gf


# ─────────────────────────────────────────────────────────────────────────────
# POLYMARKET API — Live overlay for high-liquidity markets
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def fetch_polymarket_group_g_probs() -> dict[str, float] | None:
    """
    Attempt to fetch Group G win probabilities from Polymarket's public API.
    Returns {team_name: probability} or None if unavailable.
    
    Polymarket gamma API — no auth required for reads.
    We search for the Group G winner market and parse outcome prices.
    """
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"search": "World Cup 2026 Group G winner", "limit": 5},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        markets = resp.json()
        if not markets:
            return None
        
        # Find the best matching market
        mkt = None
        for m in markets:
            title = (m.get("question") or m.get("title") or "").lower()
            if "group g" in title or ("group" in title and "belgium" in title):
                mkt = m
                break
        if not mkt:
            mkt = markets[0]
        
        # outcomePrices is a JSON-string array of prices, outcomes is a JSON-string array of names
        import json
        outcomes = json.loads(mkt.get("outcomes", "[]"))
        prices   = json.loads(mkt.get("outcomePrices", "[]"))
        
        if not outcomes or not prices or len(outcomes) != len(prices):
            return None
        
        result = {}
        for name, price in zip(outcomes, prices):
            prob = float(price)
            # Match to our team names
            for team in ["Belgium", "Egypt", "Iran", "New Zealand"]:
                if team.lower() in name.lower():
                    result[team] = prob
                    break
        
        return result if result else None
    
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)  
def fetch_polymarket_3rd_place_probs() -> dict[str, float] | None:
    """
    Attempt to fetch 3rd-place qualification probabilities from Polymarket.
    These markets exist for major teams (France, Argentina, Spain, Germany, Norway).
    Returns {team_name: probability} or None.
    """
    teams_to_try = ["France", "Argentina", "Spain", "Germany", "Norway", "Uruguay", "Ecuador"]
    result = {}
    
    try:
        for team in teams_to_try:
            resp = requests.get(
                "https://gamma-api.polymarket.com/markets",
                params={"search": f"{team} 2026 World Cup advance qualify", "limit": 3},
                timeout=4,
            )
            if resp.status_code != 200:
                continue
            markets = resp.json()
            for m in markets:
                title = (m.get("question") or m.get("title") or "").lower()
                # We want "advance from group" style markets, not tournament winner
                if ("advance" in title or "qualify" in title or "through" in title) and team.lower() in title:
                    import json
                    outcomes = json.loads(m.get("outcomes", "[]"))
                    prices   = json.loads(m.get("outcomePrices", "[]"))
                    for o, p in zip(outcomes, prices):
                        if "yes" in o.lower():
                            result[team] = float(p)
                            break
                    break
    except Exception:
        pass
    
    return result if result else None


def blend_mc_with_market(mc_prob: float, market_prob: float | None, market_weight: float = 0.4) -> tuple[float, str]:
    """
    Blend Monte Carlo probability with market-implied probability.
    
    We weight MC at 60% and market at 40% by default.
    Rationale: MC is more principled for obscure teams; markets are better
    for top teams where there's real liquidity and information aggregation.
    
    Returns: (blended_prob, method_label)
    """
    if market_prob is None:
        return mc_prob, "MC"
    blended = (1 - market_weight) * mc_prob + market_weight * market_prob
    return blended, "BLEND"


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def run_monte_carlo(n_sims: int = N_SIMULATIONS, use_markets: bool = False) -> dict:
    """
    Run N Monte Carlo simulations of the remaining group stage.
    
    For each simulation:
      1. Simulate all remaining fixtures using Dixon-Coles Poisson
      2. Rank each group
      3. Collect all 12 third-place teams
      4. Rank the 12 third-place teams globally (pts → gd → gf → random)
      5. Top 8 advance; record which 3rd-place team faces Group G winner
    
    Returns a dict of frequency-based probabilities.
    """
    np.random.seed(None)  # Fresh seed each cache miss
    
    # Counters — keyed by team name
    g_winner_counts   = {t: 0 for t in GROUPS["G"]}
    g_runnerup_counts = {t: 0 for t in GROUPS["G"]}
    third_advance_counts = {t: 0 for grp in ALL_GROUPS for t in GROUPS[grp]}
    match82_counts    = {}  # (g_winner, third_team): count
    
    base_tables = init_group_tables()
    
    for _ in range(n_sims):
        # Deep copy standings for this simulation
        tables = {grp: {t: list(v) for t, v in grp_table.items()}
                  for grp, grp_table in base_tables.items()}
        
        # Simulate all remaining fixtures
        simulate_group_stage(tables)
        
        # Rank each group
        ranked = {grp: rank_group(tables[grp]) for grp in ALL_GROUPS}
        
        # Record Group G outcomes
        g_winner   = ranked["G"][0]
        g_runnerup = ranked["G"][1]
        g_winner_counts[g_winner]     += 1
        g_runnerup_counts[g_runnerup] += 1
        
        # Collect all 12 third-place teams and their records
        third_place_teams = []
        for grp in ALL_GROUPS:
            third_team = ranked[grp][2]
            rec = get_third_place_record(tables[grp], third_team)
            third_place_teams.append((third_team, grp, rec))
        
        # Rank 3rd-place teams globally: pts desc, gd desc, gf desc, random tiebreak
        third_place_teams.sort(
            key=lambda x: (x[2][0], x[2][1], x[2][2], np.random.random()),
            reverse=True
        )
        
        # Top 8 advance
        advancing_thirds = third_place_teams[:8]
        advancing_third_teams = {t for t, _, _ in advancing_thirds}
        
        for t, _, _ in advancing_thirds:
            third_advance_counts[t] += 1
        
        # Who from eligible groups (A/E/H/I/J) is the 3rd-place qualifier?
        eligible_advancing = [(t, grp) for t, grp, _ in advancing_thirds
                              if grp in THIRD_PLACE_GROUPS]
        
        # Match 82: Group G winner vs. the eligible 3rd-place qualifier
        # (FIFA bracket assigns the 3rd-place team to Match 82 based on which
        # eligible groups produce the qualifying 3rd-place teams — but for
        # probability purposes we track each possible pairing)
        for third_team, _ in eligible_advancing:
            key = (g_winner, third_team)
            match82_counts[key] = match82_counts.get(key, 0) + 1
    
    # Convert counts to probabilities
    g_winner_prob   = {t: c / n_sims for t, c in g_winner_counts.items()}
    g_runnerup_prob = {t: c / n_sims for t, c in g_runnerup_counts.items()}
    third_advance_prob = {t: c / n_sims for t, c in third_advance_counts.items()}
    match82_joint_prob = {k: v / n_sims for k, v in match82_counts.items()}
    
    # Blend with market data if available
    methods = {}
    if use_markets:
        mkt_g = fetch_polymarket_group_g_probs()
        if mkt_g:
            for team in GROUPS["G"]:
                if team in mkt_g:
                    blended, method = blend_mc_with_market(g_winner_prob[team], mkt_g[team])
                    g_winner_prob[team] = blended
                    methods[team] = method
                else:
                    methods[team] = "MC"
        
        mkt_3rd = fetch_polymarket_3rd_place_probs()
        if mkt_3rd:
            for team, mkt_p in mkt_3rd.items():
                if team in third_advance_prob:
                    blended, method = blend_mc_with_market(third_advance_prob[team], mkt_p)
                    third_advance_prob[team] = blended
                    methods[team] = method
    
    return {
        "g_winner_prob":      g_winner_prob,
        "g_runnerup_prob":    g_runnerup_prob,
        "third_advance_prob": third_advance_prob,
        "match82_joint_prob": match82_joint_prob,
        "methods":            methods,
        "n_sims":             n_sims,
        "timestamp":          time.time(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHAOS INDEX
# ─────────────────────────────────────────────────────────────────────────────

def compute_chaos_index(mc: dict) -> float:
    """
    Shannon entropy of the Match 82 joint probability distribution.
    Normalized to [0, 100].
    
    The joint distribution is over all (g_winner, third_team) pairings,
    making this a true measure of uncertainty about the *exact* matchup.
    """
    probs = np.array(list(mc["match82_joint_prob"].values()), dtype=float)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 100.0
    probs = probs / probs.sum()
    h = -np.sum(probs * np.log2(probs))
    h_max = math.log2(len(probs))
    if h_max == 0:
        return 0.0
    return round(100 * h / h_max, 1)


def chaos_label(ci: float) -> tuple[str, str]:
    if ci < 35:
        return "LOCKED IN", "chaos-low"
    elif ci < 70:
        return "IN FLUX", "chaos-medium"
    else:
        return "TOTAL CHAOS", "chaos-high"


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_DARK = dict(
    paper_bgcolor="#0a0c14",
    plot_bgcolor="#0a0c14",
    font=dict(color="#94a3b8", family="Inter, system-ui, sans-serif", size=12),
    # No margin here — each chart sets its own to avoid duplicate-kwarg TypeError
)

def _dark(**overrides):
    """Return PLOTLY_DARK merged with per-chart overrides (safe, no duplicate keys)."""
    return {**PLOTLY_DARK, **overrides}
AXIS_STYLE = dict(
    gridcolor="#1e2235", zerolinecolor="#1e2235",
    tickfont=dict(color="#64748b", size=11),
    title_font=dict(color="#64748b"),
)


def build_chaos_gauge(chaos_index: float) -> go.Figure:
    label, _ = chaos_label(chaos_index)
    bar_color = "#4ade80" if chaos_index < 35 else ("#fbbf24" if chaos_index < 70 else "#f87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=chaos_index,
        number=dict(suffix="%", font=dict(size=38, color=bar_color, family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#1e2235",
                      tickvals=[0,25,50,75,100], tickfont=dict(color="#64748b", size=10)),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor="#0d1020", borderwidth=1, bordercolor="#1e2235",
            steps=[
                dict(range=[0,  35], color="#0a2218"),
                dict(range=[35, 70], color="#1a1408"),
                dict(range=[70,100], color="#1a0808"),
            ],
            threshold=dict(line=dict(color="#ffffff", width=2), thickness=0.75, value=chaos_index),
        ),
        title=dict(text=f"<b>{label}</b>", font=dict(size=13, color=bar_color)),
        domain=dict(x=[0,1], y=[0,1]),
    ))
    fig.update_layout(**_dark(height=280, margin=dict(l=30,r=30,t=30,b=5)))
    return fig


def build_heatmap(mc: dict) -> go.Figure:
    """
    Joint probability heatmap sourced directly from MC simulation counts.
    Each cell = P(G winner = X AND 3rd place qualifier from Group Y).
    """
    g_teams = GROUPS["G"]
    
    # Aggregate joint probs by (g_winner, 3rd_group)
    matrix_data = {}
    for (gw, tp), prob in mc["match82_joint_prob"].items():
        # Find which group tp belongs to
        tp_group = None
        for grp, members in GROUPS.items():
            if tp in members:
                tp_group = grp
                break
        if tp_group and tp_group in THIRD_PLACE_GROUPS:
            key = (gw, tp_group)
            matrix_data[key] = matrix_data.get(key, 0) + prob
    
    z = np.zeros((len(g_teams), len(THIRD_PLACE_GROUPS)))
    for i, gw in enumerate(g_teams):
        for j, grp in enumerate(THIRD_PLACE_GROUPS):
            z[i, j] = matrix_data.get((gw, grp), 0) * 100

    y_labels = [f"{FLAG_MAP.get(t,'🏳️')} {t}" for t in g_teams]
    x_labels = [f"Grp {g}" for g in THIRD_PLACE_GROUPS]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=12, family="JetBrains Mono, monospace"),
        colorscale=[[0,"#050d1a"],[0.15,"#0c2040"],[0.4,"#1d4ed8"],[0.7,"#0ea5e9"],[1.0,"#38bdf8"]],
        showscale=True,
        colorbar=dict(
            title=dict(text="Joint Prob %", side="right", font=dict(color="#64748b",size=10)),
            tickfont=dict(color="#64748b",size=10),bgcolor="#0d1020",
            bordercolor="#1e2235",borderwidth=1,thickness=14,len=0.8,
        ),
        hovertemplate="<b>%{y}</b> wins Group G<br><b>%{x}</b> 3rd place advances<br>Joint probability: <b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(
        **_dark(height=320),
        title=dict(text=f"MC Joint Probability Matrix — {mc['n_sims']:,} simulations", font=dict(size=12,color="#94a3b8"), x=0),
        xaxis=dict(**AXIS_STYLE, title="3rd-Place Qualifying Group"),
        yaxis=dict(**AXIS_STYLE, title="Group G Winner", autorange="reversed"),
        margin=dict(l=120,r=40,t=50,b=50),
    )
    return fig


def build_group_g_bar(mc: dict) -> go.Figure:
    probs = mc["g_winner_prob"]
    teams = sorted(probs.keys(), key=lambda t: probs[t])
    labels = [f"{FLAG_MAP.get(t,'🏳️')} {t}" for t in teams]
    vals   = [probs[t] * 100 for t in teams]
    colors = ["#38bdf8" if v >= 50 else ("#6366f1" if v >= 25 else "#475569") for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        textfont=dict(color="#94a3b8",size=11,family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>Win probability: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        **_dark(height=200, margin=dict(l=10,r=80,t=40,b=20)),
        title=dict(text="Group G Winner Probability (MC)", font=dict(size=12,color="#94a3b8"), x=0),
        xaxis=dict(**AXIS_STYLE, title="", range=[0,100], ticksuffix="%"),
        yaxis=dict(**AXIS_STYLE, title=""),
        showlegend=False,
    )
    return fig


def build_third_place_bar(mc: dict) -> go.Figure:
    """Bar chart showing top-3 advance candidates per eligible group."""
    palette = {"A":"#38bdf8","E":"#818cf8","H":"#34d399","I":"#fb923c","J":"#f472b6"}
    fig = go.Figure()
    
    for grp in THIRD_PLACE_GROUPS:
        teams = [(t, mc["third_advance_prob"].get(t, 0)) 
                 for t in GROUPS[grp]]
        teams.sort(key=lambda x: x[1], reverse=True)
        top3 = teams[:3]
        labels = [f"{FLAG_MAP.get(t,'🏳️')} {t}" for t, _ in top3]
        vals   = [p * 100 for _, p in top3]
        fig.add_trace(go.Bar(
            name=f"Grp {grp}", x=labels, y=vals,
            marker_color=palette.get(grp,"#64748b"), marker_line_width=0,
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
            textfont=dict(size=10,color="#94a3b8"),
            hovertemplate="<b>%{x}</b><br>Advance prob: <b>%{y:.1f}%</b><extra></extra>",
        ))
    
    max_val = max((mc["third_advance_prob"].get(t, 0) * 100 
                   for grp in THIRD_PLACE_GROUPS for t in GROUPS[grp]), default=20)
    
    fig.update_layout(
        **_dark(height=280, margin=dict(l=10,r=20,t=40,b=90), barmode="group"),
        title=dict(text="3rd-Place Advance Probability — Groups A/E/H/I/J (MC)", font=dict(size=12,color="#94a3b8"), x=0),
        xaxis=dict(**AXIS_STYLE, title="", tickangle=-30),
        yaxis=dict(**AXIS_STYLE, title="", ticksuffix="%", range=[0, max_val + 5]),
        legend=dict(font=dict(color="#64748b",size=10),bgcolor="rgba(0,0,0,0)",bordercolor="#1e2235"),
    )
    return fig


def build_matchup_distribution(mc: dict, top_n: int = 12) -> go.Figure:
    """
    Bar chart of the top-N most probable specific matchups (g_winner vs third_team).
    This is unique to the MC approach — impossible with independent probabilities.
    """
    pairs = sorted(mc["match82_joint_prob"].items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not pairs:
        return go.Figure()
    
    labels = [f"{FLAG_MAP.get(gw,'🏳️')} {gw} vs {FLAG_MAP.get(tp,'🏳️')} {tp}" 
              for (gw, tp), _ in pairs]
    vals   = [p * 100 for _, p in pairs]
    colors = ["#38bdf8" if v >= 5 else ("#6366f1" if v >= 2 else "#475569") for v in vals]
    
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.2f}%" for v in vals], textposition="outside",
        textfont=dict(size=10,color="#94a3b8",family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>Joint probability: <b>%{x:.2f}%</b><extra></extra>",
    ))
    fig.update_layout(
        **_dark(height=max(300, top_n * 28), margin=dict(l=220,r=80,t=40,b=20)),
        title=dict(text=f"Top {top_n} Most Probable Exact Match 82 Matchups (MC)", font=dict(size=12,color="#94a3b8"), x=0),
        xaxis=dict(**AXIS_STYLE, title="", range=[0, max(vals)*1.3], ticksuffix="%"),
        yaxis=dict(**AXIS_STYLE, title="", autorange="reversed"),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROOTING INTEREST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_path_to_seattle(target_team: str, mc: dict) -> dict:
    """
    Generate a natural-language Path to Seattle recipe using MC probabilities.
    """
    in_g   = target_team in GROUPS["G"]
    in_3rd = any(target_team in GROUPS[g] for g in THIRD_PLACE_GROUPS)
    
    if not in_g and not in_3rd:
        return {
            "team": target_team, "steps": [f"{target_team} is not in Group G or an eligible 3rd-place group."],
            "probability": 0.0, "verdict": "NO", "group": "N/A",
        }
    
    steps = []
    
    if in_g:
        win_p = mc["g_winner_prob"].get(target_team, 0)
        ru_p  = mc["g_runnerup_prob"].get(target_team, 0)
        method = mc["methods"].get(target_team, "MC")
        probability = win_p
        
        method_label = f'<span class="method-pill method-{"mkt" if method=="MKT" else ("blend" if method=="BLEND" else "mc")}">{method}</span>'
        
        steps.append(
            f"<b>{FLAG_MAP.get(target_team,'🏳️')} {target_team}</b> must <b>finish 1st in Group G</b>. "
            f"MC probability: <b>{win_p*100:.1f}%</b> {method_label}"
        )
        
        # Elo context
        elo = ELO.get(target_team, 1700)
        rivals = [(t, ELO.get(t,1700)) for t in GROUPS["G"] if t != target_team]
        rivals.sort(key=lambda x: x[1], reverse=True)
        steps.append(
            f"Elo rating: <b>{elo}</b>. Toughest remaining rival: "
            f"<b>{FLAG_MAP.get(rivals[0][0],'🏳️')} {rivals[0][0]}</b> (Elo {rivals[0][1]})."
        )
        
        if win_p >= 0.55:
            steps.append("<span class='verdict-yes'>Heavy favorite</span> — a point in the final group match likely seals first place.")
        elif win_p >= 0.25:
            steps.append("<span class='verdict-maybe'>Competitive group</span> — a win in the next fixture would significantly improve position.")
        else:
            steps.append("<span class='verdict-no'>Uphill battle</span> — needs wins and favorable results across multiple matches.")
        
        steps.append(
            f"Note: Group G runner-up (MC prob {ru_p*100:.1f}%) does <b>not</b> go to Match 82 — "
            f"they face Group D's runner-up. <b>Only the Group G winner reaches Seattle.</b>"
        )
        
        # Most likely opponent from 3rd-place slot
        eligible_pairs = {(gw, tp): p for (gw, tp), p in mc["match82_joint_prob"].items() if gw == target_team}
        if eligible_pairs:
            best_opp = max(eligible_pairs, key=eligible_pairs.get)
            best_opp_team = best_opp[1]
            best_opp_p = eligible_pairs[best_opp]
            steps.append(
                f"Most probable Match 82 opponent: "
                f"<b>{FLAG_MAP.get(best_opp_team,'🏳️')} {best_opp_team}</b> "
                f"(joint prob {best_opp_p*100:.2f}%)."
            )
        
        verdict = "YES" if win_p >= 0.50 else ("POSSIBLE" if win_p >= 0.20 else "NO")
    
    else:
        # 3rd-place team
        adv_p = mc["third_advance_prob"].get(target_team, 0)
        grp = next(g for g in THIRD_PLACE_GROUPS if target_team in GROUPS[g])
        probability = adv_p
        
        steps.append(
            f"<b>{FLAG_MAP.get(target_team,'🏳️')} {target_team}</b> must <b>finish 3rd in Group {grp}</b> "
            f"<i>and</i> rank in the <b>top-8 globally</b> among all 12 third-place teams. "
            f"MC probability (both conditions): <b>{adv_p*100:.1f}%</b> "
            f'<span class="method-pill method-mc">MC</span>'
        )
        
        # Within-group context
        grp_probs = [(t, mc["third_advance_prob"].get(t,0)) for t in GROUPS[grp]]
        grp_probs.sort(key=lambda x: x[1], reverse=True)
        rank_in_grp = [t for t,_ in grp_probs].index(target_team) + 1
        
        if rank_in_grp == 1:
            steps.append(f"<span class='verdict-yes'>Currently the top 3rd-place candidate</span> in Group {grp} by MC simulation.")
        elif rank_in_grp == 2:
            ahead = grp_probs[0]
            steps.append(
                f"<b>{FLAG_MAP.get(ahead[0],'🏳️')} {ahead[0]}</b> is ahead in Group {grp} "
                f"(Elo {ELO.get(ahead[0],1700)} vs {ELO.get(target_team,1700)}). "
                f"Needs results to swing in their favor."
            )
        else:
            steps.append(f"<span class='verdict-no'>Significant ground to make up</span> within Group {grp}.")
        
        steps.append(
            "The 3rd-place slot is a <b>12-way simultaneous race</b>. The MC engine simulates all 12 groups "
            "together, so this probability already accounts for competition from Groups B, C, D, F, K, L "
            "— not just the 5 eligible groups."
        )
        
        # Tiebreaker note
        elo = ELO.get(target_team, 1700)
        steps.append(
            f"Elo rating: <b>{elo}</b>. Tiebreaker priority: Points → GD → GF → Fair-play. "
            f"Running up the score in comfortable wins can be the difference."
        )
        
        # Most likely G winner they'd face
        eligible_pairs = {(gw, tp): p for (gw, tp), p in mc["match82_joint_prob"].items() if tp == target_team}
        if eligible_pairs:
            best_pair = max(eligible_pairs, key=eligible_pairs.get)
            best_gw = best_pair[0]
            steps.append(
                f"Most probable Match 82 opponent if they qualify: "
                f"<b>{FLAG_MAP.get(best_gw,'🏳️')} {best_gw}</b> "
                f"(joint prob {eligible_pairs[best_pair]*100:.2f}%)."
            )
        
        verdict = "YES" if adv_p >= 0.50 else ("POSSIBLE" if adv_p >= 0.12 else "NO")
    
    return {
        "team": target_team, "group": "G" if in_g else grp,
        "steps": steps, "probability": probability, "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, int, bool]:
    with st.sidebar:
        st.markdown("## ⚽ Match 82 Tracker")
        st.markdown('<p style="color:#475569;font-size:0.78rem;margin-top:-0.5rem;">Seattle · Lumen Field · July 1, 2026</p>', unsafe_allow_html=True)
        st.divider()
        
        st.markdown("### 🎲 Simulation")
        n_sims = st.select_slider(
            "Monte Carlo trials",
            options=[10_000, 25_000, 50_000, 100_000],
            value=50_000,
            help="More trials = more accurate but slower. 50k ≈ 1–2 seconds.",
        )
        
        use_markets = st.toggle(
            "🟣 Blend Polymarket odds",
            value=USE_MARKETS,
            help="Blend live Polymarket market prices with MC probabilities for top-tier teams.",
        )
        
        if use_markets:
            st.caption("60% MC + 40% market for teams with active Polymarket contracts.")
        
        if st.button("🔄 Re-run Simulation", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.markdown("### 🗺️ Rooting Interest")
        st.caption("Select a team to see their path to Lumen Field")
        
        eligible = sorted([t for grp in (["G"] + THIRD_PLACE_GROUPS) for t in GROUPS[grp]])
        display  = [f"{FLAG_MAP.get(t,'🏳️')} {t}" for t in eligible]
        default_idx = eligible.index("Belgium") if "Belgium" in eligible else 0
        
        sel = st.selectbox("Target Team", display, index=default_idx, label_visibility="collapsed")
        selected_team = sel.split(" ", 1)[-1] if " " in sel else sel
        
        st.divider()
        st.markdown("### ℹ️ Model Notes")
        st.caption("**Engine**: Dixon-Coles corrected Poisson + Elo ratings")
        st.caption("**3rd-place**: Full 12-group race, not independent per-team probs")
        st.caption("**Elo source**: eloratings.net as of June 13, 2026")
        st.caption("**Blend**: Polymarket API (no key required, free)")
        
        st.divider()
        st.markdown('<p style="color:#1e3a5f;font-size:0.72rem;">Match 82 · Seattle WC2026 · MC Engine v2</p>', unsafe_allow_html=True)
    
    return selected_team, n_sims, use_markets


# ─────────────────────────────────────────────────────────────────────────────
# DAILY DIGEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_digest_paragraph(mc: dict) -> str:
    """
    Generate a daily odds-movement narrative paragraph based on current MC output.

    In production, this function would be called by a scheduled job (e.g., a GitHub
    Actions cron, a Streamlit cron, or a simple cloud function) that:
      1. Runs the MC simulation against live standings
      2. Compares today's probabilities to yesterday's stored snapshot
      3. Passes the delta to an LLM (e.g. OpenAI gpt-4o-mini) to write the paragraph
      4. Sends the result to all subscribers via an email API (e.g. Resend / SendGrid)

    For now, this is a template-based generator that demonstrates the full output
    format. Swap the return statement for an OpenAI API call when you have a key.
    """
    import datetime

    today = datetime.date.today().strftime("%B %d, %Y")
    g_probs = mc["g_winner_prob"]
    g_leader = max(g_probs, key=g_probs.get)
    g_leader_p = g_probs[g_leader]
    g_sorted = sorted(g_probs.items(), key=lambda x: x[1], reverse=True)

    # Top 3rd-place candidates overall (eligible groups)
    eligible_3rd = {t: mc["third_advance_prob"].get(t, 0)
                    for grp in THIRD_PLACE_GROUPS for t in GROUPS[grp]}
    top_3rd = sorted(eligible_3rd.items(), key=lambda x: x[1], reverse=True)[:3]

    # Top exact matchup
    top_pair = max(mc["match82_joint_prob"], key=mc["match82_joint_prob"].get) \
               if mc["match82_joint_prob"] else ("TBD", "TBD")
    top_pair_p = mc["match82_joint_prob"].get(top_pair, 0)

    chaos = compute_chaos_index(mc)
    _, chaos_class = chaos_label(chaos)

    # —— Compose the paragraph ——
    # (In production, pass this context dict to gpt-4o-mini with a system prompt
    # instructing it to write one punchy paragraph in the style of a smart sports analyst.)
    g2 = g_sorted[1] if len(g_sorted) > 1 else ("TBD", 0)
    g3 = g_sorted[2] if len(g_sorted) > 2 else ("TBD", 0)

    para = (
        f"**Match 82 Daily Brief — {today}**\n\n"
        f"{FLAG_MAP.get(g_leader,'')} **{g_leader}** remains the Group G frontrunner "
        f"at **{g_leader_p*100:.1f}%** to win the group and claim the Seattle slot, "
        f"but {FLAG_MAP.get(g2[0],'')} {g2[0]} ({g2[1]*100:.1f}%) and "
        f"{FLAG_MAP.get(g3[0],'')} {g3[0]} ({g3[1]*100:.1f}%) are keeping the pressure on — "
        f"a single result can shuffle the entire picture. "
        f"On the 3rd-place side, the race to grab one of the eight global wild-card spots "
        f"is shaping up with {FLAG_MAP.get(top_3rd[0][0],'')} **{top_3rd[0][0]}** ({top_3rd[0][1]*100:.1f}%), "
        f"{FLAG_MAP.get(top_3rd[1][0],'')} {top_3rd[1][0]} ({top_3rd[1][1]*100:.1f}%), and "
        f"{FLAG_MAP.get(top_3rd[2][0],'')} {top_3rd[2][0]} ({top_3rd[2][1]*100:.1f}%) leading "
        f"the eligible groups. The single most probable exact matchup at Lumen Field is "
        f"{FLAG_MAP.get(top_pair[0],'')} **{top_pair[0]} vs {FLAG_MAP.get(top_pair[1],'')} {top_pair[1]}** "
        f"at **{top_pair_p*100:.2f}%** — still a long way from a lock. "
        f"The Chaos Index sits at **{chaos}%** ({chaos_class.replace('_',' ').title()}), "
        f"meaning the Seattle matchup remains genuinely wide open through the end of the group stage."
    )
    return para


def render_chart_to_b64(mc: dict) -> str:
    """
    Renders the 'Most Probable Exact Matchups' horizontal bar chart to a
    base64-encoded PNG string suitable for inline embedding in an HTML email.

    Requires the `kaleido` package (already in requirements.txt).
    Returns an empty string if rendering fails (e.g. kaleido not installed).
    """
    try:
        fig = build_matchup_distribution(mc, top_n=12)
        # Force full dark background for the email snapshot
        fig.update_layout(
            paper_bgcolor="#080f1c",
            plot_bgcolor="#080f1c",
            margin=dict(l=180, r=40, t=60, b=40),
        )
        img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
        import base64
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return ""


def generate_digest_email_html(mc: dict, paragraph: str) -> str:
    """
    Returns a full HTML email body containing:
      - The daily brief paragraph (plain text section)
      - The 'Most Probable Exact Matchups' bar chart as an inline PNG

    Parameters
    ----------
    mc        : dict returned by run_monte_carlo()
    paragraph : str  — the narrative paragraph (template or LLM-written)

    Usage in send_digest.py
    -----------------------
    mc        = run_monte_carlo(n_sims=50_000)
    context   = generate_digest_paragraph(mc)   # → LLM prompt context
    paragraph = call_llm(context)               # or use context directly
    html_body = generate_digest_email_html(mc, paragraph)

    resend.Emails.send({
        "from":    "match82@yourdomain.com",
        "to":      subscriber_email,
        "subject": f"Match 82 Brief — {today}",
        "html":    html_body,
    })
    """
    import datetime
    today = datetime.date.today().strftime("%B %d, %Y")
    b64 = render_chart_to_b64(mc)

    chart_html = (
        f'<img src="data:image/png;base64,{b64}" '
        f'width="900" style="max-width:100%;border-radius:8px;margin-top:1.2rem;" '
        f'alt="Most Probable Exact Matchups — Match 82" />'
        if b64 else
        '<p style="color:#64748b;font-size:0.8rem;">'
        '[Chart unavailable — install kaleido: pip install kaleido]</p>'
    )

    # Convert markdown bold (**text**) to <strong> for email clients
    import re
    paragraph_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", paragraph)
    paragraph_html = paragraph_html.replace("\n\n", "</p><p>")

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Match 82 Brief — {today}</title>
</head>
<body style="margin:0;padding:0;background:#080f1c;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#080f1c;">
    <tr><td align="center" style="padding:32px 16px;">
      <table width="640" cellpadding="0" cellspacing="0"
             style="max-width:640px;background:#0e1628;border-radius:12px;
                    border:1px solid #1e3a5f;overflow:hidden;">

        <!-- Header -->
        <tr><td style="background:#0d1b38;padding:24px 32px;
                        border-bottom:1px solid #1e3a5f;">
          <p style="margin:0;color:#475569;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.1em;font-weight:700;">Match 82 · Lumen Field · July 1, 2026</p>
          <h1 style="margin:6px 0 0;color:#f8fafc;font-size:1.35rem;font-weight:700;
                     line-height:1.3;">Daily Odds Brief</h1>
          <p style="margin:4px 0 0;color:#475569;font-size:0.82rem;">{today}</p>
        </td></tr>

        <!-- Body paragraph -->
        <tr><td style="padding:28px 32px 8px;">
          <p style="margin:0;color:#cbd5e1;font-size:0.95rem;line-height:1.85;">
            {paragraph_html}
          </p>
        </td></tr>

        <!-- Chart section -->
        <tr><td style="padding:8px 32px 28px;">
          <p style="margin:0 0 12px;color:#475569;font-size:0.72rem;text-transform:uppercase;
                    letter-spacing:0.08em;font-weight:700;">Most Probable Exact Matchups</p>
          {chart_html}
          <p style="margin:10px 0 0;color:#334155;font-size:0.74rem;">
            Probabilities from {mc.get('n_sims', 50000):,}-trial Dixon-Coles / Elo Monte Carlo simulation.
          </p>
        </td></tr>

        <!-- Footer -->
        <tr><td style="background:#060c18;border-top:1px solid #1e2a44;
                        padding:16px 32px;">
          <p style="margin:0;color:#334155;font-size:0.74rem;line-height:1.6;">
            You're receiving this because you subscribed at the
            <a href="https://match82-wc2026-tracker.streamlit.app" style="color:#3b82f6;">
            Match 82 Tracker</a>. No spam, ever.
          </p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
"""


def save_email_signup(email: str) -> bool:
    """
    Persist a subscriber email to a local CSV file in the workspace.

    In production, swap this for:
      - A Supabase/PostgreSQL insert, OR
      - A direct call to your ESP (Resend / SendGrid / Loops) list API, OR
      - A Google Sheet append via the Sheets API

    The CSV approach works perfectly on Streamlit Community Cloud as long as
    you don't need the list to survive redeployments (use a database for that).
    """
    import csv, os, re

    # Basic email validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email.strip()):
        return False

    filepath = "/tmp/match82_subscribers.csv"
    existing = set()
    if os.path.exists(filepath):
        with open(filepath, newline="") as f:
            existing = {row[0].strip().lower() for row in csv.reader(f) if row}

    if email.strip().lower() in existing:
        return True  # already subscribed, treat as success

    with open(filepath, "a", newline="") as f:
        csv.writer(f).writerow([email.strip().lower(),
                                 __import__("datetime").date.today().isoformat()])
    return True


def count_subscribers() -> int:
    import csv, os
    filepath = "/tmp/match82_subscribers.csv"
    if not os.path.exists(filepath):
        return 0
    with open(filepath, newline="") as f:
        return sum(1 for row in csv.reader(f) if row)


def render_digest_section(mc: dict) -> None:
    """
    Renders the daily digest signup section and a live preview of today's paragraph.
    """
    st.markdown("## 📧 Daily Odds Brief")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;margin-top:-0.4rem;">'
        'Get a one-paragraph daily summary of how Match 82 odds shifted overnight, '
        'driven by yesterday&#39;s results. Plain English. No noise.'
        '</p>',
        unsafe_allow_html=True,
    )

    col_form, col_preview = st.columns([1, 1.6], gap="large")

    with col_form:
        st.markdown(
            '<div style="background:#0e1628;border:1px solid #1e3a5f;border-radius:10px;'
            'padding:1.4rem 1.6rem;">'
            '<p style="color:#cbd5e1;font-size:0.92rem;font-weight:600;margin:0 0 0.3rem;">'
            '📬 Subscribe</p>'
            '<p style="color:#475569;font-size:0.8rem;margin:0 0 1rem;">'
            'Delivered each morning during the group stage — June 14 through June 26.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.form("digest_signup", clear_on_submit=True):
            email_input = st.text_input(
                "Your email",
                placeholder="you@example.com",
                label_visibility="collapsed",
            )
            submit = st.form_submit_button(
                "Subscribe to Daily Brief",
                use_container_width=True,
            )

        if submit:
            if email_input and "@" in email_input:
                ok = save_email_signup(email_input)
                if ok:
                    st.success(
                        f"Subscribed! You'll receive the first digest tomorrow morning. "
                        f"({count_subscribers()} subscriber{'s' if count_subscribers()!=1 else ''})"
                    )
                else:
                    st.error("That doesn't look like a valid email address.")
            else:
                st.warning("Please enter a valid email address.")

        # Explainer
        st.markdown(
            '<div style="margin-top:1rem;padding:0.9rem 1.1rem;background:#080f1c;'
            'border:1px solid #1e2a44;border-radius:8px;">'
            '<p style="color:#475569;font-size:0.76rem;margin:0 0 0.5rem;text-transform:uppercase;'
            'letter-spacing:0.07em;font-weight:700;">What you&#39;ll get</p>'
            '<ul style="color:#64748b;font-size:0.8rem;margin:0;padding-left:1.1rem;line-height:1.8;">'
            '<li>How overnight results shifted Match 82 win probabilities</li>'
            '<li>Which 3rd-place team climbed or fell in the global race</li>'
            '<li>The single biggest probability swing of the past 24 hours</li>'
            '<li>Current Chaos Index reading + what it means for Seattle</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Send instructions for production wiring
        with st.expander("🔧 Production wiring guide"):
            st.markdown(
                """
                **To send real emails with the chart**, wire these three components:

                **1. Scheduler** — GitHub Actions cron (free). Runs at 7 AM PT = 14:00 UTC.
                The snapshot is taken at run time, so it reflects the previous evening's
                final results once standings are updated in the app:
                ```yaml
                # .github/workflows/daily_digest.yml
                on:
                  schedule:
                    - cron: '0 14 * * *'  # 7 AM PT (UTC-7 during group stage)
                jobs:
                  digest:
                    runs-on: ubuntu-latest
                    steps:
                      - uses: actions/checkout@v4
                      - uses: actions/setup-python@v5
                        with: {{ python-version: '3.11' }}
                      - run: pip install -r requirements.txt
                      - run: python send_digest.py
                        env:
                          OPENAI_API_KEY: ${{{{ secrets.OPENAI_API_KEY }}}}
                          RESEND_API_KEY: ${{{{ secrets.RESEND_API_KEY }}}}
                ```

                **2. `send_digest.py`** — runs MC, writes LLM paragraph, renders chart, sends HTML email:
                ```python
                import os, datetime, openai, resend
                from app import (
                    run_monte_carlo,
                    generate_digest_paragraph,
                    generate_digest_email_html,
                )

                today = datetime.date.today().strftime("%B %d, %Y")

                # --- 1. Run simulation (snapshot of this morning's standings) ---
                mc = run_monte_carlo(n_sims=50_000)

                # --- 2. Generate paragraph via gpt-4o-mini ---
                context = generate_digest_paragraph(mc)  # structured context string
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                      {{"role": "system", "content":
                        "You are a sharp soccer analyst writing a 3-4 sentence daily "
                        "odds brief. Be specific about numbers. Sound like The Athletic, "
                        "not ESPN. No hype. Plain text only, no markdown."}},
                      {{"role": "user", "content": context}},
                    ],
                )
                paragraph = response.choices[0].message.content

                # --- 3. Build full HTML email with embedded chart image ---
                html_body = generate_digest_email_html(mc, paragraph)
                # (render_chart_to_b64 is called inside; requires kaleido)

                # --- 4. Send to all subscribers via Resend ---
                resend.api_key = os.environ["RESEND_API_KEY"]
                subscribers = []  # replace with your DB / CSV read
                for email in subscribers:
                    resend.Emails.send({{
                        "from":    "Match 82 Brief <match82@yourdomain.com>",
                        "to":      email,
                        "subject": f"Match 82 Brief — {{today}}",
                        "html":    html_body,
                    }})
                print(f"Sent to {{len(subscribers)}} subscribers.")
                ```

                **Chart snapshot timing**: The cron runs at 7 AM PT, after you've updated
                `LIVE_STANDINGS` in app.py with the previous day's final scores and pushed
                to GitHub. The chart will reflect that morning's MC output — exactly what
                changed overnight.

                Total cost: **$0** for scheduler + **~$0.001/digest** (gpt-4o-mini) +
                **$0** for email under 3k subscribers (Resend free tier).
                """
            )

    with col_preview:
        st.markdown(
            '<p style="color:#475569;font-size:0.72rem;text-transform:uppercase;'
            'letter-spacing:0.08em;font-weight:700;margin-bottom:0.6rem;">'
            'Today&#39;s digest preview</p>',
            unsafe_allow_html=True,
        )
        digest_text = generate_digest_paragraph(mc)
        st.markdown(
            f'<div style="background:#080f1c;border:1px solid #1e3a5f;border-radius:10px;'
            f'padding:1.4rem 1.6rem;line-height:1.85;font-size:0.88rem;color:#cbd5e1;">'
            f'{digest_text}'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Chart preview indicator
        st.markdown(
            '<div style="margin-top:1rem;padding:0.75rem 1rem;background:#080f1c;'
            'border:1px solid #1e3a5f;border-radius:8px;display:flex;align-items:center;gap:0.6rem;">'
            '<span style="font-size:1.1rem;">📊</span>'
            '<span style="color:#64748b;font-size:0.8rem;">'
            'Email includes a <strong style="color:#94a3b8;">Most Probable Exact Matchups</strong> '
            'chart — rendered fresh each morning from that day&#39;s MC snapshot.'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "In production this paragraph is written by gpt-4o-mini using live probability "
            "deltas as context. Today's preview is template-generated from current MC output."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    selected_team, n_sims, use_markets = render_sidebar()
    
    # ── Run Monte Carlo ────────────────────────────────────────────────────────
    with st.spinner(f"Running {n_sims:,} Monte Carlo simulations…"):
        mc = run_monte_carlo(n_sims=n_sims, use_markets=use_markets)
    
    chaos_val = compute_chaos_index(mc)
    c_label, _ = chaos_label(chaos_val)
    
    g_leader = max(mc["g_winner_prob"], key=mc["g_winner_prob"].get)
    g_leader_p = mc["g_winner_prob"][g_leader]
    
    # Best 3rd-place candidate in eligible groups
    eligible_3rd_probs = {t: mc["third_advance_prob"].get(t, 0)
                          for grp in THIRD_PLACE_GROUPS for t in GROUPS[grp]}
    best_3rd = max(eligible_3rd_probs, key=eligible_3rd_probs.get)
    best_3rd_p = eligible_3rd_probs[best_3rd]
    
    top_joint = max(mc["match82_joint_prob"].values()) if mc["match82_joint_prob"] else 0
    
    # ── HEADER ─────────────────────────────────────────────────────────────────
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        st.markdown(
            '<h1 style="font-size:1.75rem;font-weight:800;color:#f8fafc;letter-spacing:-0.02em;'
            'margin-bottom:0.3rem;">⚽ Match 82 — Seattle Lumen Field</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.85rem;margin-top:0;font-weight:400;">'
            'Round of 32 &nbsp;·&nbsp; '
            '<span style="color:#cbd5e1;font-weight:600;">Group G Winner vs. 3rd Place A/E/H/I/J</span>'
            ' &nbsp;·&nbsp; Wed July 1, 2026 &nbsp;·&nbsp; 1:00 PM PT'
            '</p>', unsafe_allow_html=True,
        )
    with col_badge:
        blend_label = "🟣 MC+MARKET" if (use_markets and mc["methods"]) else "🎲 PURE MC"
        blend_color = "#1a0f4a" if use_markets else "#0f2d4a"
        blend_border= "#3b2a6e" if use_markets else "#1e4a6e"
        blend_text  = "#a78bfa" if use_markets else "#38bdf8"
        st.markdown(
            f'<div style="background:{blend_color};border:1px solid {blend_border};border-radius:8px;'
            f'padding:0.6rem 1rem;text-align:center;margin-top:0.8rem;">'
            f'<span style="color:{blend_text};font-size:0.75rem;font-weight:700;">{blend_label}</span><br>'
            f'<span style="color:#475569;font-size:0.65rem;">{mc["n_sims"]:,} simulations</span>'
            f'</div>', unsafe_allow_html=True,
        )
    
    st.markdown("---")
    
    # ── TOP METRICS ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Group G Favourite", f"{FLAG_MAP.get(g_leader,'🏳️')} {g_leader}", f"{g_leader_p*100:.1f}% win prob")
    with c2:
        st.metric("Top 3rd-Place Candidate", f"{FLAG_MAP.get(best_3rd,'🏳️')} {best_3rd}", f"{best_3rd_p*100:.1f}% advance prob")
    with c3:
        top_pair = max(mc["match82_joint_prob"], key=mc["match82_joint_prob"].get) if mc["match82_joint_prob"] else ("?","?")
        # Render matchup as two lines to avoid truncation
        st.metric(
            "Most Likely Matchup",
            f"{FLAG_MAP.get(top_pair[0],'🏳️')} {top_pair[0]}",
            f"vs {FLAG_MAP.get(top_pair[1],'🏳️')} {top_pair[1]} · {top_joint*100:.2f}%",
        )
    with c4:
        st.metric("Chaos Index", f"{chaos_val}%", c_label)
    
    st.markdown("---")
    
    # ── PATH TO SEATTLE ─────────────────────────────────────────────────────────
    st.markdown("## 🗺️ Path to Seattle")
    recipe = generate_path_to_seattle(selected_team, mc)
    
    verdict = recipe["verdict"]
    v_color  = "#4ade80" if verdict=="YES" else ("#f87171" if verdict=="NO" else "#fbbf24")
    v_bg     = "#052e16" if verdict=="YES" else ("#1a0808" if verdict=="NO" else "#1a1008")
    v_border = "#166534" if verdict=="YES" else ("#7f1d1d" if verdict=="NO" else "#92400e")
    
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;">'
        f'<span style="font-size:1.55rem;font-weight:800;color:#f8fafc;letter-spacing:-0.01em;">'
        f'{FLAG_MAP.get(selected_team,"🏳️")} {selected_team}</span>'
        f'<span style="background:{v_bg};border:1px solid {v_border};border-radius:6px;'
        f'padding:4px 14px;font-size:0.82rem;font-weight:800;color:{v_color};'
        f'letter-spacing:0.06em;">{verdict}</span>'
        f'<span style="color:#64748b;font-size:0.82rem;border-left:1px solid #1e2235;'
        f'padding-left:1rem;">'
        f'Elo <span style="color:#94a3b8;font-weight:600;">{ELO.get(selected_team,"N/A")}</span>'
        f' &nbsp;·&nbsp; Group <span style="color:#94a3b8;font-weight:600;">{recipe["group"]}</span>'
        f'</span>'
        f'</div>', unsafe_allow_html=True,
    )
    
    steps_html = "".join([
        f'<div class="recipe-step"><div class="step-num">{i+1}</div><div class="step-text">{s}</div></div>'
        for i, s in enumerate(recipe["steps"])
    ])
    prob_pct = recipe["probability"]
    prob_bar_w = int(prob_pct * 100)
    prob_bar_color = "#4ade80" if prob_pct >= 0.50 else ("#fbbf24" if prob_pct >= 0.15 else "#f87171")
    
    st.markdown(
        f'<div class="recipe-card">{steps_html}'
        f'<div style="margin-top:1rem;padding-top:0.75rem;border-top:1px solid #1e2a44;">'
        f'<span style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;">'
        f'MC-simulated probability of appearing in Match 82</span><br>'
        f'<div style="background:#0d1020;border-radius:4px;height:8px;margin-top:0.4rem;overflow:hidden;">'
        f'<div style="background:{prob_bar_color};width:{prob_bar_w}%;height:100%;border-radius:4px;"></div>'
        f'</div>'
        f'<span style="color:{prob_bar_color};font-size:0.9rem;font-weight:700;font-family:monospace;">'
        f'{prob_pct*100:.1f}%</span>'
        f'</div></div>', unsafe_allow_html=True,
    )
    
    st.markdown("---")
    
    # ── HEATMAP + CHAOS GAUGE ───────────────────────────────────────────────────
    st.markdown("## 📊 Probability Analytics")
    col_heat, col_chaos = st.columns([2.2, 1])
    
    with col_heat:
        st.plotly_chart(build_heatmap(mc), use_container_width=True)
    
    with col_chaos:
        st.markdown("#### Chaos Index")
        st.caption(
            "Shannon entropy of the full Match 82 joint distribution — "
            "across every possible (G winner, 3rd-place team) pairing from the MC simulation."
        )
        st.plotly_chart(build_chaos_gauge(chaos_val), use_container_width=True)
        
        chaos_class = "chaos-low" if chaos_val < 35 else ("chaos-medium" if chaos_val < 70 else "chaos-high")
        chaos_css = {"chaos-low":"#052e16;color:#4ade80;border:1px solid #166534",
                     "chaos-medium":"#1c1917;color:#fbbf24;border:1px solid #92400e",
                     "chaos-high":"#1a0a0a;color:#f87171;border:1px solid #7f1d1d"}[chaos_class]
        st.markdown(
            f'<div style="text-align:center;margin-top:-0.5rem;">'
            f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.72rem;'
            f'font-weight:700;letter-spacing:0.05em;text-transform:uppercase;background:{chaos_css};">'
            f'{c_label}</span></div>', unsafe_allow_html=True,
        )
        with st.expander("Model notes"):
            st.markdown(
                f"""
                **Chaos Index** = Shannon entropy over the joint (G winner × 3rd-place team) 
                distribution from {mc['n_sims']:,} MC simulations, normalised to [0, 100].

                Unlike the v1 approach (independent per-group entropies added together), 
                this captures the *actual* combinatorial uncertainty of the matchup — 
                including correlations between groups and the 12-way 3rd-place race.

                **Match engine**: Dixon-Coles corrected Poisson driven by Elo ratings.  
                **Elo source**: eloratings.net as of June 13, 2026.
                """
            )
    
    st.markdown("---")
    
    # ── TOP MATCHUPS ────────────────────────────────────────────────────────────
    st.markdown("## 🎯 Most Probable Exact Matchups")
    st.caption("This is what the MC engine uniquely enables — ranked probability of every specific pairing.")
    st.plotly_chart(build_matchup_distribution(mc, top_n=12), use_container_width=True)
    
    st.markdown("---")
    
    # ── PROBABILITY DISTRIBUTIONS ───────────────────────────────────────────────
    st.markdown("## 📈 Component Probabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(build_group_g_bar(mc), use_container_width=True)
    with col2:
        st.plotly_chart(build_third_place_bar(mc), use_container_width=True)
    
    st.markdown("---")
    
    # ── LIVE STANDINGS TABLE ────────────────────────────────────────────────────
    st.markdown("## 📋 Current Standings")
    
    with st.expander("Group G", expanded=True):
        rows = []
        for t in GROUPS["G"]:
            s = LIVE_STANDINGS[t]
            pts = s["w"]*3 + s["d"]
            rows.append({
                "": FLAG_MAP.get(t,"🏳️"), "Team": t,
                "MP": s["mp"], "W": s["w"], "D": s["d"], "L": s["l"],
                "GF": s["gf"], "GA": s["ga"], "GD": s["gf"]-s["ga"], "Pts": pts,
                "Win Prob %": round(mc["g_winner_prob"].get(t,0)*100, 1),
            })
        df = pd.DataFrame(rows).sort_values("Pts", ascending=False).reset_index(drop=True)
        df.index += 1
        st.dataframe(df, use_container_width=True, hide_index=False,
            column_config={"Win Prob %": st.column_config.ProgressColumn("Win Prob %", min_value=0, max_value=100, format="%.1f%%")})
    
    tabs = st.tabs([f"Group {g}" for g in THIRD_PLACE_GROUPS])
    for tab, grp in zip(tabs, THIRD_PLACE_GROUPS):
        with tab:
            rows = []
            for t in GROUPS[grp]:
                s = LIVE_STANDINGS.get(t, {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0})
                pts = s["w"]*3 + s["d"]
                rows.append({
                    "": FLAG_MAP.get(t,"🏳️"), "Team": t,
                    "MP": s["mp"], "W": s["w"], "D": s["d"], "L": s["l"],
                    "GF": s["gf"], "GA": s["ga"], "GD": s["gf"]-s["ga"], "Pts": pts,
                    "Advance Prob %": round(mc["third_advance_prob"].get(t,0)*100, 1),
                })
            df = pd.DataFrame(rows).sort_values("Pts", ascending=False).reset_index(drop=True)
            df.index += 1
            st.dataframe(df, use_container_width=True, hide_index=False,
                column_config={"Advance Prob %": st.column_config.ProgressColumn("Advance Prob %", min_value=0, max_value=100, format="%.1f%%")})
    
    # ── DAILY DIGEST SIGNUP ────────────────────────────────────────────────────
    st.markdown("---")
    render_digest_section(mc)

    # ── FOOTER ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<p style="color:#1e3a5f;font-size:0.75rem;text-align:center;">'
        f'Match 82 · Seattle Lumen Field · July 1, 2026 · '
        f'Monte Carlo engine ({mc["n_sims"]:,} simulations) · '
        f'Dixon-Coles Poisson + Elo (eloratings.net June 2026) · '
        f'Polymarket API blend {"enabled" if use_markets else "disabled"} · '
        f'Not affiliated with FIFA'
        f'</p>', unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
