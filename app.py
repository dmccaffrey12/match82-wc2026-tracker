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
import json
import datetime
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

# Path to pre-computed results written by precompute.py / GitHub Actions
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "results.json")
RESULTS_MAX_AGE_HOURS = 25   # Accept snapshot up to 25 hours old before warning


def load_precomputed_results() -> dict | None:
    """
    Load the pre-computed Monte Carlo snapshot from results.json.

    Returns the deserialized MC dict if the file exists and is fresh
    (< RESULTS_MAX_AGE_HOURS old), otherwise returns None so the app
    falls back to a live simulation.
    """
    if not os.path.exists(RESULTS_JSON):
        return None
    try:
        with open(RESULTS_JSON) as f:
            data = json.load(f)
        # Restore tuple keys in match82_joint_prob
        joint = {}
        for key_str, v in data.get("match82_joint_prob", {}).items():
            parts = key_str.split(" vs ", 1)
            joint[tuple(parts) if len(parts) == 2 else (key_str, "TBD")] = v
        data["match82_joint_prob"] = joint
        # Check freshness
        computed_at_str = data.get("computed_at", "")
        if computed_at_str:
            computed_at = datetime.datetime.fromisoformat(computed_at_str.replace("Z", "+00:00"))
            age_hours = (datetime.datetime.now(datetime.timezone.utc) - computed_at).total_seconds() / 3600
            data["_age_hours"] = round(age_hours, 1)
            data["_stale"] = age_hours > RESULTS_MAX_AGE_HOURS
        else:
            data["_age_hours"] = None
            data["_stale"] = False
        data["_precomputed"] = True
        return data
    except Exception:
        return None

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
# TOURNAMENT ELO UPDATER
# Applies every played WC 2026 result to the base ELO ratings using the
# standard Elo update formula with K=60 (FIFA World Cup weight).
# This means Turkey (0W-0D-2L) falls ~80 pts; Norway (2W) rises ~80 pts.
# ─────────────────────────────────────────────────────────────────────────────

# All completed WC 2026 group-stage results (home, away, goals_home, goals_away)
# Update this list daily as matches finish.
PLAYED_RESULTS: list[tuple[str, str, int, int]] = [
    # Matchday 1 — June 11–16
    ("Mexico",       "South Africa",  2, 0),
    ("South Korea",  "Czechia",       2, 1),
    ("Canada",       "Bosnia",        1, 1),
    ("Qatar",        "Switzerland",   1, 1),
    ("USA",          "Paraguay",      4, 1),
    ("Australia",    "Türkiye",       2, 0),
    ("Scotland",     "Haiti",         1, 0),
    ("Brazil",       "Morocco",       1, 1),
    ("Germany",      "Curaçao",       7, 1),
    ("Côte d'Ivoire","Ecuador",       1, 0),
    ("Netherlands",  "Japan",         2, 2),
    ("Sweden",       "Tunisia",       5, 1),
    ("Belgium",      "Egypt",         1, 1),
    ("Iran",         "New Zealand",   2, 2),
    ("Spain",        "Cabo Verde",    0, 0),
    ("Saudi Arabia", "Uruguay",       1, 1),
    ("France",       "Senegal",       3, 1),
    ("Norway",       "Iraq",          4, 1),
    ("Argentina",    "Algeria",       3, 0),
    ("Austria",      "Jordan",        3, 1),
    ("Portugal",     "DR Congo",      1, 1),
    ("Colombia",     "Uzbekistan",    3, 1),
    ("England",      "Croatia",       4, 2),
    ("Ghana",        "Panama",        1, 0),
    # Matchday 2 — June 18–21
    ("Czechia",      "South Africa",  1, 1),
    ("Switzerland",  "Bosnia",        4, 1),
    ("Canada",       "Qatar",         6, 0),
    ("Mexico",       "South Korea",   1, 0),
    ("USA",          "Australia",     2, 0),
    ("Scotland",     "Morocco",       0, 1),
    ("Brazil",       "Haiti",         3, 0),
    ("Türkiye",      "Paraguay",      0, 1),
    ("Netherlands",  "Sweden",        5, 1),
    ("Germany",      "Côte d'Ivoire", 2, 1),
    ("Ecuador",      "Curaçao",       0, 0),
    ("Tunisia",      "Japan",         0, 4),
    # Matchday 2 — June 21 (FINAL)
    ("Belgium",     "Iran",          0, 0),   # BEL 0-0 IRA  (De Bruyne stumbles)
    ("New Zealand", "Egypt",         1, 3),   # NZL 1-3 EGY  (Egypt top of Group G)
    ("Spain",       "Saudi Arabia",  4, 0),   # ESP 4-0 KSA  (Yamal's 1st WC goal)
    ("Uruguay",     "Cabo Verde",    2, 2),   # URU 2-2 CPV  (Cape Verde's 1st WC goals)
    # Matchday 2 — June 22 (FINAL)
    ("France",      "Iraq",          3, 0),   # FRA 3-0 IRQ  (Mbappé brace, Dembélé; France through)
    ("Norway",      "Senegal",       3, 2),   # NOR 3-2 SEN  (Haaland brace; Norway through)
    ("Argentina",   "Austria",       2, 0),   # ARG 2-0 AUT  (Messi record brace; Argentina through)
    ("Jordan",      "Algeria",       1, 2),   # JOR 1-2 ALG  (Benbouali 69', Gouiri 82')
    # Matchday 2 — June 23 (FINAL)
    ("Portugal",    "Uzbekistan",    5, 0),   # POR 5-0 UZB  (Ronaldo brace; Portugal through)
    ("Colombia",    "DR Congo",      1, 0),   # COL 1-0 DRC  (Muñoz 76'; Colombia through)
    ("England",     "Ghana",         0, 0),   # ENG 0-0 GHA  (England held)
    ("Panama",      "Croatia",       0, 1),   # PAN 0-1 CRO  (Budimir)
    # Matchday 3 — June 24 (FINAL)
    ("South Africa", "South Korea",  1, 0),   # SAF 1-0 KOR  (Group A shock; South Africa 2nd)
    ("Czechia",      "Mexico",       0, 3),   # CZE 0-3 MEX  (Mexico perfect record, 9pts)
    ("Switzerland",  "Canada",       2, 1),   # SUI 2-1 CAN  (Switzerland Group B winners)
    ("Bosnia",       "Qatar",        3, 1),   # BOS 3-1 QAT  (Bosnia strong 3rd-place candidate)
    ("Scotland",     "Brazil",       0, 3),   # SCO 0-3 BRA  (Brazil through, Group C winners)
    ("Morocco",      "Haiti",        4, 2),   # MOR 4-2 HAI  (Morocco 2nd in Group C)
]

# Set of already-played fixtures as frozensets so simulate_group_stage
# can skip them — results are already baked into LIVE_STANDINGS / the API.
# Using frozenset so order doesn't matter (home/away is irrelevant here).
PLAYED_FIXTURES: set[frozenset] = {
    frozenset((h, a)) for h, a, _, _ in PLAYED_RESULTS
}


@st.cache_data(ttl=3600, show_spinner=False)
def compute_tournament_elos() -> dict[str, float]:
    """
    Apply all completed WC 2026 results to base ELO ratings.

    Uses standard Elo update formula:
        new_elo = old_elo + K * (actual - expected)

    where:
        K = 60  (FIFA World Cup weight, per eloratings.net)
        actual = 1.0 (win), 0.5 (draw), 0.0 (loss)
        expected = 1 / (1 + 10^(-diff/400))  for the home team

    Results are applied in chronological order so each match uses
    the rating at that point in the tournament.

    Returns a new dict with updated ratings — base ELO is never mutated.
    """
    K = 60.0
    ratings = dict(ELO)  # copy — never mutate the base dict
    for home, away, gh, ga in PLAYED_RESULTS:
        r_h = ratings.get(home, 1700.0)
        r_a = ratings.get(away, 1700.0)
        expected_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h) / 400.0))
        expected_a = 1.0 - expected_h
        actual_h = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        actual_a = 1.0 - actual_h
        ratings[home] = r_h + K * (actual_h - expected_h)
        ratings[away]  = r_a + K * (actual_a - expected_a)

    # ── Elimination suppression ────────────────────────────────────────────
    # Teams that Polymarket considers functionally eliminated (tourney-win
    # price ≤ 0.5% OR advance price ≤ 2%) have their Elo discounted to a
    # floor of 1200. This prevents eliminated teams like Türkiye from
    # accumulating unrealistically high simulated point totals and crowding
    # out genuine 3rd-place contenders in the Monte Carlo.
    # We skip this inside compute_tournament_elos itself (it's cached) and
    # let simulate_match apply the discount inline instead.
    return ratings


# ─────────────────────────────────────────────────────────────────────────────
# CURRENT LIVE STANDINGS  (update as matches complete)
# Format: {team: {"mp":int, "w":int, "d":int, "l":int, "gf":int, "ga":int}}
# The simulator picks up from here and only plays REMAINING fixtures.
#
# LIVE UPDATE: Set FOOTBALL_DATA_API_KEY in .streamlit/secrets.toml or as an
# environment variable to auto-fetch live standings from football-data.org.
# Free tier: 10 req/min — register at https://www.football-data.org/client/register
# If the key is absent the app falls back to the hardcoded LIVE_STANDINGS dict.
# ─────────────────────────────────────────────────────────────────────────────

FOOTBALL_DATA_API_KEY: str = (
    os.environ.get("FOOTBALL_DATA_API_KEY", "")
    or (st.secrets.get("FOOTBALL_DATA_API_KEY", "") if hasattr(st, "secrets") else "")
)


@st.cache_data(ttl=300)  # Re-fetch every 5 minutes
def fetch_standings_from_api() -> dict[str, dict] | None:
    """
    Fetch live group standings from football-data.org v4 API.

    Returns a dict of {team_name: {mp, w, d, l, gf, ga}} on success,
    or None if no API key is configured or the fetch fails (falls back
    to the hardcoded LIVE_STANDINGS dict).

    Setup (one-time):
      1. Register free at https://www.football-data.org/client/register
      2. Add to .streamlit/secrets.toml:  FOOTBALL_DATA_API_KEY = "your_key"
         OR add as a GitHub Actions secret and Streamlit Cloud secret.

    API endpoint: GET https://api.football-data.org/v4/competitions/WC/standings
    Auth header:  X-Auth-Token: {key}
    Free tier:    10 req/min — plenty for once-daily precompute.
    """
    if not FOOTBALL_DATA_API_KEY:
        return None
    import urllib.request
    url = "https://api.football-data.org/v4/competitions/WC/standings"
    req = urllib.request.Request(url, headers={"X-Auth-Token": FOOTBALL_DATA_API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        st.warning(f"⚠️ football-data.org fetch failed: {e}")
        return None
    # football-data.org uses different spellings for some teams.
    # Map their names -> our canonical GROUPS names.
    NAME_MAP = {
        # Ivory Coast / Côte d'Ivoire
        "Ivory Coast":              "Côte d'Ivoire",
        "Côte D'Ivoire":            "Côte d'Ivoire",
        "Cote d'Ivoire":            "Côte d'Ivoire",
        "Cote D'Ivoire":            "Côte d'Ivoire",
        # Cabo Verde (football-data.org uses "Cape Verde")
        "Cape Verde":               "Cabo Verde",
        "Cape Verde Islands":        "Cabo Verde",
        "Cabo Verde":               "Cabo Verde",  # already correct, belt-and-suspenders
        # Korea
        "Korea Republic":           "South Korea",
        "Republic of Korea":        "South Korea",
        "Korea DPR":                "North Korea",
        # Iran
        "IR Iran":                  "Iran",
        # Curaçao
        "Curacao":                  "Curaçao",
        # Türkiye
        "Turkey":                   "Türkiye",
        # USA
        "United States":            "USA",
        "United States of America": "USA",
        # Misc
        "Republic of Ireland":      "Ireland",
        "Bosnia & Herzegovina":     "Bosnia and Herzegovina",
        "North Macedonia":          "North Macedonia",
        "Trinidad & Tobago":        "Trinidad and Tobago",
    }
    standings: dict[str, dict] = {}
    for group_block in data.get("standings", []):
        for row in group_block.get("table", []):
            raw_name = row.get("team", {}).get("name", "").strip()
            if not raw_name:
                continue
            name = NAME_MAP.get(raw_name, raw_name)
            standings[name] = {
                "mp": int(row.get("playedGames", 0)),
                "w":  int(row.get("won",         0)),
                "d":  int(row.get("draw",        0)),
                "l":  int(row.get("lost",        0)),
                "gf": int(row.get("goalsFor",    0)),
                "ga": int(row.get("goalsAgainst",0)),
            }
    return standings if standings else None


# ── Hardcoded fallback standings (updated manually if no Sheet is configured) ─
LIVE_STANDINGS: dict[str, dict] = {
    # Group A — MD1: Mexico 2-0 RSA, SKor 2-1 CZE | MD2: CZE 1-1 RSA, Mexico 1-0 SKor
    # Group A — MD3 FINAL: SAF 1-0 KOR, CZE 0-3 MEX
    "Mexico":       {"mp":3,"w":3,"d":0,"l":0,"gf":7,"ga":0},
    "South Africa": {"mp":3,"w":1,"d":1,"l":1,"gf":2,"ga":4},
    "South Korea":  {"mp":3,"w":1,"d":0,"l":2,"gf":2,"ga":3},
    "Czechia":      {"mp":3,"w":0,"d":1,"l":2,"gf":2,"ga":6},
    # Group B — MD3 FINAL: Swi 2-1 Can, Bos 3-1 Qat
    "Switzerland":  {"mp":3,"w":2,"d":1,"l":0,"gf":7,"ga":3},
    "Canada":       {"mp":3,"w":1,"d":1,"l":1,"gf":8,"ga":3},
    "Bosnia":       {"mp":3,"w":1,"d":1,"l":1,"gf":5,"ga":6},
    "Qatar":        {"mp":3,"w":0,"d":1,"l":2,"gf":2,"ga":10},
    # Group C — MD3 FINAL: Sco 0-3 Bra, Mor 4-2 Hai
    "Brazil":       {"mp":3,"w":2,"d":1,"l":0,"gf":7,"ga":1},
    "Morocco":      {"mp":3,"w":2,"d":1,"l":0,"gf":6,"ga":3},
    "Scotland":     {"mp":3,"w":1,"d":0,"l":2,"gf":1,"ga":4},
    "Haiti":        {"mp":3,"w":0,"d":0,"l":3,"gf":2,"ga":8},
    # Group D — MD1: USA 4-1 Par, Aus 2-0 Tur | MD2: USA 2-0 Aus, Par 1-0 Tur
    "USA":          {"mp":2,"w":2,"d":0,"l":0,"gf":6,"ga":1},
    "Australia":    {"mp":2,"w":1,"d":0,"l":1,"gf":2,"ga":2},
    "Türkiye":      {"mp":2,"w":0,"d":0,"l":2,"gf":0,"ga":3},
    "Paraguay":     {"mp":2,"w":1,"d":0,"l":1,"gf":2,"ga":4},
    # Group E — MD1: Ger 7-1 Cur, CIV 1-0 Ecu | MD2: Ger 2-1 CIV, Ecu 0-0 Cur
    "Germany":      {"mp":2,"w":2,"d":0,"l":0,"gf":9,"ga":2},
    "Côte d'Ivoire":{"mp":2,"w":1,"d":0,"l":1,"gf":2,"ga":2},
    "Ecuador":      {"mp":2,"w":0,"d":1,"l":1,"gf":0,"ga":1},
    "Curaçao":      {"mp":2,"w":0,"d":1,"l":1,"gf":2,"ga":8},
    # Group F — MD1: Ned 2-2 Jpn, Swe 5-1 Tun | MD2: Ned 5-1 Swe, Jpn 4-0 Tun
    "Netherlands":  {"mp":2,"w":1,"d":1,"l":0,"gf":7,"ga":3},
    "Japan":        {"mp":2,"w":1,"d":1,"l":0,"gf":6,"ga":2},
    "Sweden":       {"mp":2,"w":1,"d":0,"l":1,"gf":6,"ga":6},
    "Tunisia":      {"mp":2,"w":0,"d":0,"l":2,"gf":1,"ga":9},
    # Group G — through MD2 (June 21 FINAL)
    # MD1: BEL 1-1 EGY, IRA 2-2 NZL
    # MD2: BEL 0-0 IRA, NZL 1-3 EGY
    # Egypt:       4pts, GF=4 (1+3), GA=2 (1+1), GD=+2
    # Belgium:     2pts, GF=1 (1+0), GA=1 (1+0), GD=0
    # Iran:        2pts, GF=2 (2+0), GA=2 (2+0), GD=0
    # New Zealand: 1pt,  GF=3 (2+1), GA=5 (2+3), GD=-2
    "Egypt":        {"mp":2,"w":1,"d":1,"l":0,"gf":4,"ga":2},
    "Belgium":      {"mp":2,"w":0,"d":2,"l":0,"gf":1,"ga":1},
    "Iran":         {"mp":2,"w":0,"d":2,"l":0,"gf":2,"ga":2},
    "New Zealand":  {"mp":2,"w":0,"d":1,"l":1,"gf":3,"ga":5},
    # Group H — through MD2 (June 21 FINAL)
    # Spain: 4pts (+4 GD), Uruguay: 2pts (0 GD), Cabo Verde: 2pts (0 GD), KSA: 1pt (-4 GD)
    "Spain":        {"mp":2,"w":1,"d":1,"l":0,"gf":4,"ga":0},
    "Uruguay":      {"mp":2,"w":0,"d":2,"l":0,"gf":3,"ga":3},
    "Cabo Verde":   {"mp":2,"w":0,"d":2,"l":0,"gf":2,"ga":2},
    "Saudi Arabia": {"mp":2,"w":0,"d":1,"l":1,"gf":1,"ga":5},
    # Group I — MD2 FINAL: Fra 3-0 Irq, Nor 3-2 Sen
    "France":       {"mp":2,"w":2,"d":0,"l":0,"gf":6,"ga":1},
    "Norway":       {"mp":2,"w":2,"d":0,"l":0,"gf":7,"ga":3},
    "Senegal":      {"mp":2,"w":0,"d":0,"l":2,"gf":3,"ga":6},
    "Iraq":         {"mp":2,"w":0,"d":0,"l":2,"gf":1,"ga":7},
    # Group J — MD2 FINAL: Arg 2-0 Aut, Jor 1-2 Alg
    "Argentina":    {"mp":2,"w":2,"d":0,"l":0,"gf":5,"ga":0},
    "Austria":      {"mp":2,"w":1,"d":0,"l":1,"gf":3,"ga":3},
    "Algeria":      {"mp":2,"w":1,"d":0,"l":1,"gf":2,"ga":4},
    "Jordan":       {"mp":2,"w":0,"d":0,"l":2,"gf":2,"ga":5},
    # Group K — MD2 FINAL: Por 5-0 Uzb, Col 1-0 DRC
    "Portugal":     {"mp":2,"w":1,"d":1,"l":0,"gf":6,"ga":1},
    "Colombia":     {"mp":2,"w":2,"d":0,"l":0,"gf":4,"ga":1},
    "Uzbekistan":   {"mp":2,"w":0,"d":0,"l":2,"gf":1,"ga":8},
    "DR Congo":     {"mp":2,"w":0,"d":1,"l":1,"gf":1,"ga":2},
    # Group L — MD2 FINAL: Eng 0-0 Gha, Pan 0-1 Cro
    "England":      {"mp":2,"w":1,"d":1,"l":0,"gf":4,"ga":2},
    "Ghana":        {"mp":2,"w":1,"d":1,"l":0,"gf":1,"ga":0},
    "Croatia":      {"mp":2,"w":1,"d":0,"l":1,"gf":3,"ga":4},
    "Panama":       {"mp":2,"w":0,"d":0,"l":2,"gf":0,"ga":2},
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
  .method-elim { background: #2d0f0f; color: #f87171; border: 1px solid #6e1e1e; }
  .method-blend { background: #1a2e0f; color: #86efac; border: 1px solid #2a4e1e; }

  /* ── Mobile responsiveness ── */
  @media (max-width: 768px) {
    /* Tighter page padding on small screens */
    .block-container { padding: 0.75rem 0.85rem 1.5rem !important; }

    /* Stack metric cards in a 2x2 grid instead of 4-across */
    [data-testid="stHorizontalBlock"] {
      flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
      min-width: 46% !important;
      flex: 1 1 46% !important;
    }

    /* Shrink metric values slightly so they don't overflow */
    [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.62rem !important; }

    /* Title sizing */
    h1 { font-size: 1.25rem !important; }
    h2 { font-size: 0.7rem !important; }

    /* Hide sidebar by default on mobile (user taps hamburger to open) */
    section[data-testid="stSidebar"] { width: 85vw !important; }

    /* Plotly charts — ensure they don't overflow horizontally */
    .js-plotly-plot, .plotly { max-width: 100% !important; overflow-x: hidden !important; }

    /* Path to Seattle card — tighter on mobile */
    .recipe-card { padding: 0.8rem 1rem !important; }
    .step-text { font-size: 0.82rem !important; }

    /* Digest form columns — stack vertically */
    /* Streamlit columns can't truly reflow but we can compress the form side */
    [data-testid="stForm"] { padding: 0 !important; }
  }

  @media (max-width: 480px) {
    /* Phone — single column feel */
    .block-container { padding: 0.5rem 0.6rem 1rem !important; }
    [data-testid="stMetricValue"] { font-size: 0.92rem !important; }
    h1 { font-size: 1.1rem !important; }
    /* Make the badge block stack below the title */
    [data-testid="stHorizontalBlock"]:first-of-type > [data-testid="stVerticalBlock"]:last-child {
      min-width: 100% !important;
    }
  }
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


# Module-level eliminated-team cache: refreshed once per MC run via TTL on
# the Polymarket fetchers. Avoids hitting the API inside the hot sim loop.
_ELIMINATED_CACHE: tuple[float, set[str]] = (0.0, set())
_ELIM_CACHE_TTL = 120.0  # seconds

def _get_eliminated_cache() -> set[str]:
    """Return cached set of eliminated teams (refreshes every 120s)."""
    global _ELIMINATED_CACHE
    ts, elim_set = _ELIMINATED_CACHE
    if time.time() - ts > _ELIM_CACHE_TTL:
        try:
            elim_set = get_eliminated_teams(use_markets=True)
        except Exception:
            elim_set = set()
        _ELIMINATED_CACHE = (time.time(), elim_set)
    return elim_set


def simulate_match(
    team_a: str,
    team_b: str,
    elo_ratings: dict[str, float] | None = None,
) -> tuple[int, int]:
    """Simulate a single match, returning (goals_a, goals_b).

    elo_ratings: pass compute_tournament_elos() to use tournament-updated
    ratings; falls back to the base ELO dict if None.
    """
    ratings = elo_ratings or ELO
    elo_a = ratings.get(team_a, ELO.get(team_a, 1700))
    elo_b = ratings.get(team_b, ELO.get(team_b, 1700))
    # Eliminate discount: teams the market has written off get a floor Elo
    # of 1200, making them heavy underdogs in every simulated match.
    # We use a module-level cache so we're not hitting Polymarket on
    # every single match simulation (millions of calls per MC run).
    _elim = _get_eliminated_cache()
    if team_a in _elim: elo_a = min(elo_a, 1200.0)
    if team_b in _elim: elo_b = min(elo_b, 1200.0)
    lam_a, lam_b = expected_goals(elo_a, elo_b)
    return simulate_scoreline(lam_a, lam_b)


# ─────────────────────────────────────────────────────────────────────────────
# GROUP STAGE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def init_group_tables() -> dict[str, dict[str, list]]:
    """
    Initialize standings from live API data (football-data.org) when available,
    falling back to the hardcoded LIVE_STANDINGS dict if no API key is set.
    Returns: {group: {team: [pts, gf, ga, gd, w, d, l]}}
    """
    live = fetch_standings_from_api() or LIVE_STANDINGS
    tables: dict[str, dict[str, list]] = {}
    for grp, teams in GROUPS.items():
        tables[grp] = {}
        for t in teams:
            s = live.get(t, {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0})
            pts = s["w"] * 3 + s["d"]
            tables[grp][t] = [pts, s["gf"], s["ga"], s["gf"]-s["ga"], s["w"], s["d"], s["l"]]
    return tables


def simulate_group_stage(tables: dict, elo_ratings: dict[str, float] | None = None) -> dict:
    """
    Simulate all remaining fixtures and return final standings.
    Mutates a *copy* of tables.

    elo_ratings: tournament-updated Elo dict from compute_tournament_elos().
    Returns: {group: {team: [pts, gf, ga, gd, w, d, l]}}
    """
    for (grp, home, away) in ALL_FIXTURES:
        # Skip fixtures whose result is already in LIVE_STANDINGS
        if frozenset((home, away)) in PLAYED_FIXTURES:
            continue
        gh, ga = simulate_match(home, away, elo_ratings)
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
    Fetch Group G win probabilities from Polymarket's CLOB API.
    Returns {team_name: probability} normalized to sum to 1.0, or None.

    Polymarket has individual Yes/No markets per team (not a single multi-outcome
    market). We fetch the 'Yes' price for each team directly via condition ID,
    then infer New Zealand's probability as the remainder.

    Condition IDs discovered via:
      https://clob.polymarket.com/sampling-markets?next_cursor=
    No auth required.
    """
    # Hardcoded condition IDs for Group G (Belgium, Egypt, Iran have liquid markets)
    GROUP_G_CONDITIONS = {
        "Belgium": "0x1e285f49c483634426c54834f840f6bfe780e0039eb0ad31357b936600b7c2d2",
        "Egypt":   "0xd5f291e4d2dfc44a42ee0eb1d83f2e42a90ce943a71f643cadf4e98c5a6ec3eb",
        "Iran":    "0x5bf2c54fb4ca9d63427816cd88ab8f7b5ede5b2b9166c366a940f1a0d6d0a290",
    }
    import urllib.request as _ur
    result: dict[str, float] = {}
    try:
        for team, cid in GROUP_G_CONDITIONS.items():
            url = f"https://clob.polymarket.com/markets/{cid}"
            req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with _ur.urlopen(req, timeout=6) as r:
                data = json.loads(r.read())
            if data.get("closed"):
                continue  # Market settled — skip
            tokens = data.get("tokens", [])
            yes_price = next(
                (float(t["price"]) for t in tokens if t.get("outcome", "").lower() == "yes"),
                None,
            )
            if yes_price is not None:
                result[team] = yes_price
    except Exception:
        return None

    if not result:
        return None

    # Infer New Zealand as remainder (no liquid market exists for them)
    nz_implied = max(0.0, 1.0 - sum(result.values()))
    result["New Zealand"] = nz_implied

    # Normalize to sum exactly to 1.0 (handles any over-round)
    total = sum(result.values())
    if total <= 0:
        return None
    return {t: p / total for t, p in result.items()}


# ─────────────────────────────────────────────────────────────────────────────
# POLYMARKET — ADVANCE-TO-KNOCKOUT MARKET  (primary signal)
# "World Cup: Team to advance to Knockout Stages" — 48 teams, $6.4M volume
# Resolves YES if team advances by any means (1st, 2nd, or qualifying 3rd)
# This is the most direct signal we have: Ecuador at 25.5% means the market
# collectively believes there's only a ~25% chance Ecuador makes the R32.
# ─────────────────────────────────────────────────────────────────────────────

# Condition IDs from gamma-api.polymarket.com event:
# slug=world-cup-team-to-advance-to-knockout-stages
ADVANCE_CONDITIONS: dict[str, str] = {
    # Group A
    "Mexico":                  "0x765c607f355d16dcab5ac2cdd29a37779d1428a071b866bf767badc62346ec6c",
    "South Korea":             "0x550d74a2fe47a7ce066e04e647da4010a2ac26e8aa1f0edcb9403ce90d0c9016",
    "Czechia":                 "0x390b496499262807b909b58a9aaa95f6ae0b322d4a324531a3e97b2937904eda",
    "South Africa":            "0x601d0d0b6e93e27832d2ecf4482e4fe9fed604cd0a29e7962b09a1d764fb4250",
    # Group B
    "Switzerland":             "0xeea6fafcf500f582bf1999d504b769befcceb645a7ed46c29aeb901d0ea29baf",
    "Canada":                  "0x655712b319987805e573590082fbe8bd688f64fb5fc53b8516d397763e6b3cf4",
    "Qatar":                   "0x3a6c34249d718a5bff78608e20b5cddefe722b73d5515524779747afb9c2a068",
    "Bosnia":                  "0x9a2b2aeccf873af12a6171722d70d86e458492424a7adb700b011ca3b7cc28e7",
    # Group C
    "Brazil":                  "0xbc94d393aa6c0c1a3c8c23f0ab2f45e95d05cfd50266a993c582e84c5117d984",
    "Morocco":                 "0x6ad7d53bc2a42daa2b9625eebf9651fd3aac1286078697182a3d1dc3bbd70173",
    "Scotland":                "0x73f6028060ff88ff0369307f2c45dde79a5e7d7c437c4987638def21d065bac7",
    "Haiti":                   "0x5ea45b0916e7ff547035047d0bd72502c1d1431c6196215f532741e14f997317",
    # Group D
    "USA":                     "0x6bac5bbce7d0ef0a0e036fea1eb9ec66835c9795a553e766fa305b3a8b065d93",
    "Australia":               "0x38f5caccf3b53ef5abeb0056838e4e5df3f000e97a4fad7eb55431781b427d2c",
    "Türkiye":                 "0x8c28874af6349a7e58f30909eecac5197bfdaa033b03ad98510853958ee41558",
    "Paraguay":                "0x0cbb73759bb6ad83040cc6edb908a7dbd51855618082d2f5dd3b58937a79b306",
    # Group E
    "Germany":                 "0xb8316d5b3cbd3a92e130a27a93c8eaf8faa29a09b3b3287787b621686889f9f5",
    "Ecuador":                 "0x8d84206ebda85fe26ac0f413f463c7029887a95c5bbc344e266bf4f0a0c2659d",
    "Côte d'Ivoire":           "0xf1b5eddb0ee3c3398c161f731d21f170342feb192f7c0263d9baa7cbb6c1c31b",
    "Curaçao":                 "0x62623e36db475ec25adf71f602ec64dab863e906365e7b27804f8e72a85e7dbe",
    # Group F
    "Netherlands":             "0x2deac4e9149d7933e977be2a572b46d427b5b52308553e385a42c8e9855ba536",
    "Japan":                   "0x9e91380150cd0739da220168cfc99a129a7b6f7dd89e28ae48ae60e5749ca699",
    "Sweden":                  "0xbea8b7504ec86c42abb39512b385c4dc695ba6c1d3f4d3aa9dd0ceddb0957338",
    "Tunisia":                 "0xebbdac391050303b03d01e3f53ee84050698d01a04dbeace9ecda45daef6fb4d",
    # Group G
    "Belgium":                 "0xc0d63027b98472e48bacae3726cb9cd1929f0c31f91a791b6d5e773b390a3238",
    "Egypt":                   "0x19eff468ad0ffce9ade9f41d3d323c42e48981a28772b47618d4ef77c7333c55",
    "Iran":                    "0xc2fcce9165ac160807304db5dc0ec730dfd6d17c02a23c0dafce10a471833725",
    "New Zealand":             "0x46d7512f30ec5b01e8194cf42457a041ca9bf675e4f58b32edcde8b3c431d18a",
    # Group H
    "Spain":                   "0xbdb9f8af2767fa217f65b2a970a9ec46f88fcf3a96e94421a3b51bc8cda1e12a",
    "Uruguay":                 "0xbafadf181195da28073877849e2a4601a2f4a99371bf94e24e8a6380c7baa072",
    "Saudi Arabia":            "0xe28714396c63822d0c1293f2aad16aaca02a143a32f20b71fd7fb58f078d6602",
    "Cabo Verde":              "0xf75a1084ea00a19ec34957397ea4d0a33258395ba6b50a2d4186ee0f77910c25",
    # Group I
    "France":                  "0x6f814a95b780d7e8b14e9a8ce9d34f7afa1b25be62830aefb3781a6aa9afbf16",
    "Senegal":                 "0x59ffd52a92c4afe5257997bbb6ecb38a260eb2ef4f8da31f0271fe48d35b8622",
    "Norway":                  "0x35e0edf13c676c05379882d01980fb360b7885884d56c57f228cd306018efb3d",
    "Iraq":                    "0xcb1bcc07313aefb781b5e6426625e525d4071c93d2115a558808ef4495da50a4",
    # Group J
    "Argentina":               "0x8e534d6f28c124e3d7414561be384e79c4b108420d1c43a9a965289e2ec25576",
    "Algeria":                 "0xa8cc82d418a2a52f1a8af46d00f8b87353b1926b1e59add939946c195e8f541f",
    "Austria":                 "0xbd0d83e891497ded91678ee5a6d58dede9b8f1adad52fa3e0b534359b737302c",
    "Jordan":                  "0xb2ddb90e1715deedf1fb1a1422aed272d2a1141f8e5e0ca6f1accb3a21a5eb77",
    # Group K
    "Portugal":                "0x94cac3a7ff4e968e68674c8dff21d74df39c9519291db5e4628486f977b1cad5",
    "Colombia":                "0x260688608f7d98c9cc5755228dbf4a04eb72c13308e603ba697b5f1ff4e8fe68",
    "Uzbekistan":              "0x77cdf10e0cbddb64775d735d882925631edcdbf84f23316e23d0d4be83636b30",
    "DR Congo":                "0xf53b6c5f8c269fb2c6fe8547a8356b6d57084a5d15502672165be922d58c46d6",
    # Group L
    "England":                 "0x1b37d3e123b994a315fc445ab4fdbb94bb2fa111437b6305a44a7fcd18c3d217",
    "Croatia":                 "0x339817a52eeb97cc6cf99afee47a61dbc83244a4266a947d19f9b21aa9d7bbd3",
    "Ghana":                   "0x2d93d3f277602a22f90c8ddcfd775945542d6b4a99449db03489de13609f3290",
    "Panama":                  "0x71c3fec5d09820f4bc86f4c3c0c4c750190fcdcd3009d3454629073c8f7e02ec",
}

# Tournament-win condition IDs ("Will X win the 2026 FIFA World Cup?")
# Used to detect functional elimination: price <= ELIM_THRESHOLD → treat as out
TOURNEY_WIN_CONDITIONS: dict[str, str] = {
    "Spain":        "0x7976b8dbacf9077eb1453a62bcefd6ab2df199acd28aad276ff0d920d6992892",
    "France":       "0x9b6fef249040fd17e9c107955b37ac2c3e923509b6b0ff01cc463a331ddeb894",
    "Argentina":    "0x0c4cd2055d6ea89354ffddc55d6dbcef9355748112ea952fc925f3db6a5c457f",
    "Germany":      "0x1595b4818eeb1ea1e0bec5de6f057218e557feee9b405a0e930d290384fa1d16",
    "Belgium":      "0x32cfa52198e85e070d1b17d1b53c5c3a6aaae7736cdc33fa6aa04d353f0c2811",
    "Norway":       "0x7b52405ad0e0d31bfe970940b67d77f24ecedeab8a2361c11148c02a006e325c",
    "Uruguay":      "0x7876851632c295043c66536150a304cb785abdf712ba8489d298c6e6926be106",
    "Mexico":       "0x5ccfe1b69a582d2985db08a8481a0d74c314b1fce9b4711ae2efb2c6467fe6aa",
    "Morocco":      "0x37a6de1b21803e5f3fb1965116218215d79963af4f7e51659696366267a63a03",
    "Egypt":        "0x7412d284c8f63791fec807f9b1f61c6fe61163621775a3dc8686cd2575272abe",
    "New Zealand":  "0x9e5f9d8c384f8fe368b195fa9a780be58643dff7360588a4e577012df8af00a7",
    "Iran":         "0x84edef36bded182da6a395ac6c785dba8f3e09b6c5ad041385b2042536cbef25",
    "South Korea":  "0x65307f30dce84ac35e41813035d3c04933da830dc4efbbb2fcdc4b282700ef3b",
    "Algeria":      "0x5a59d269c2b5108cd2f64c624e46ee2c8b5cfd88b882582565f927918315b6aa",
    "Senegal":      "0x6972edb1b3f8cd8192651a665fc424dff846efe1c4a2376f628d4b20c704144c",
    "Côte d'Ivoire":"0x289568d555ec620ed6fa33c936c5f42649d3a2e30748a1daf7079f42453fbea4",
    "Austria":      "0xfe230d510eaf545198c0d62bb17871e5fe8989f1b19aa54c0c062b858360987c",
    "Saudi Arabia": "0x3fb8a8de2ac275882d72b2c4f22d41776fcf033f9e413a77a84dd395c0d5257c",
    "Scotland":     "0xf950740bc71136155d6525cc0528a582c81f88812bff227803190c32ca25f54d",
    "Cabo Verde":   "0x3bc69cb672591e4fcd2ef856b64b219a906e15d4601b50066ac81a446574dfaf",
    "Qatar":        "0x4fe305a2ae995a52ff278895344895fe587b4fec3d5f04347b4dbf5e99bce99c",
    "Switzerland":  "0x3a26ca6425e2d98f14935670bc22cdb0744defc6f6d83c65f8c413a921c5c70c",
    "Japan":        "0x0189df05ed7bf84d799213b01a79571e305c03b2ac5359cfbb3a323448ba20fa",
    "Ecuador":      "0xbaf7780f9059e34b84301fd411f8dc573b4d56adfe6e0cda33daf304b1438da4",
    "Paraguay":     "0x675bba4df50fd123f7fbfbafa67e9b75f4092d85ce0f9148ce78fc945964c856",
    "Australia":    "0x098e2be3df8ab529940c567819f8ef007cf007820e9d627642a5bbfaa42af372",
    "Jordan":       "0x33a87d02fa01e958929385c74b8627d32cc4474e9ebd312d268865c5207147fa",
}

# A team is treated as functionally eliminated when its tournament-win price
# is at or below this threshold (0.5%). Türkiye, Qatar, Iraq, Curaçao, etc.
ELIM_THRESHOLD = 0.005  # 0.5%


@st.cache_data(ttl=120, show_spinner=False)
def fetch_polymarket_advance_probs() -> dict[str, float]:
    """
    Fetch P(advance to R32) from Polymarket's 48-team knockout market.
    Returns {team: probability} for every team with an active market.
    N/A / closed markets return 0.0 (resolved as eliminated).
    Falls back to empty dict on total failure.
    """
    import urllib.request as _ur
    result: dict[str, float] = {}
    try:
        for team, cid in ADVANCE_CONDITIONS.items():
            try:
                url = f"https://clob.polymarket.com/markets/{cid}"
                req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with _ur.urlopen(req, timeout=5) as r:
                    data = json.loads(r.read())
                if data.get("closed") or data.get("question", "") == "":
                    # Market closed/resolved → treat as 0 (eliminated) or 1 (advanced)
                    # We can't tell direction so skip; Elo + MC will handle it
                    continue
                tokens = data.get("tokens", [])
                yes_price = next(
                    (float(t["price"]) for t in tokens
                     if t.get("outcome", "").lower() == "yes"),
                    None,
                )
                if yes_price is not None:
                    result[team] = yes_price
            except Exception:
                continue
    except Exception:
        pass
    return result


@st.cache_data(ttl=120, show_spinner=False)
def fetch_polymarket_tourney_win_probs() -> dict[str, float]:
    """
    Fetch tournament-win prices. Used purely as an elimination signal:
    a team at <= ELIM_THRESHOLD (0.5%) is treated as functionally eliminated
    and its Elo contribution to simulations is discounted to near-zero.
    Returns {team: yes_price}. Missing teams are not in result.
    """
    import urllib.request as _ur
    result: dict[str, float] = {}
    try:
        for team, cid in TOURNEY_WIN_CONDITIONS.items():
            try:
                url = f"https://clob.polymarket.com/markets/{cid}"
                req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with _ur.urlopen(req, timeout=5) as r:
                    data = json.loads(r.read())
                tokens = data.get("tokens", [])
                yes_price = next(
                    (float(t["price"]) for t in tokens
                     if t.get("outcome", "").lower() == "yes"),
                    None,
                )
                if yes_price is not None:
                    result[team] = yes_price
            except Exception:
                continue
    except Exception:
        pass
    return result


def get_eliminated_teams(use_markets: bool = True) -> set[str]:
    """
    Return set of teams the market considers functionally eliminated.
    Criteria (either condition triggers elimination):
      1. Tournament-win price <= ELIM_THRESHOLD (0.5%)
      2. Advance-to-knockout price <= 0.02 (2%) — extreme long shot
    Only applied when use_markets=True.
    """
    if not use_markets:
        return set()
    eliminated: set[str] = set()
    win_probs = fetch_polymarket_tourney_win_probs()
    adv_probs = fetch_polymarket_advance_probs()
    for team, p in win_probs.items():
        if p <= ELIM_THRESHOLD:
            eliminated.add(team)
    for team, p in adv_probs.items():
        if p <= 0.02:
            eliminated.add(team)
    return eliminated


@st.cache_data(ttl=120, show_spinner=False)
def compute_market_third_place_probs() -> dict[str, dict] | None:
    """
    Infer P(finish 3rd AND advance as best 3rd) for every team in the
    five eligible groups (A/E/H/I/J) using the advance-market decomposition:

        P_3rd_qual(team) ≈ P_advance(team) - P_1st(team) - P_2nd(team)

    Where P_1st and P_2nd come from a quick internal MC run (5k sims).
    This is far more direct than the old Plackett-Luce group-win proxy:
    Ecuador at 25.5% advance price bakes in the full path (finish 3rd AND
    be one of the 8 best), exactly what we want.

    Falls back to Elo-only Plackett-Luce if market data unavailable.

    Returns {team: {"group": str, "p_advance": float, "p_3rd_qual": float,
                    "p_1st": float, "p_2nd": float, "source": str}}
    """
    adv_probs = fetch_polymarket_advance_probs()  # {team: P(advance by any means)}

    # ── Quick internal MC to get P(1st) and P(2nd) per team (5k sims) ──────
    # We need these to decompose P_advance into P_3rd_qual
    n_quick = 5_000
    p1_counts: dict[str, int] = {t: 0 for grp in THIRD_PLACE_GROUPS for t in GROUPS[grp]}
    p2_counts: dict[str, int] = {t: 0 for grp in THIRD_PLACE_GROUPS for t in GROUPS[grp]}
    base_tables = init_group_tables()
    live_elos = compute_tournament_elos()
    rng_state = np.random.get_state()
    np.random.seed(42)
    for _ in range(n_quick):
        tables = {grp: {t: list(v) for t, v in grp_table.items()}
                  for grp, grp_table in base_tables.items()
                  if grp in THIRD_PLACE_GROUPS}
        # Only simulate fixtures for THIRD_PLACE_GROUPS, skipping played ones
        for (grp, home, away) in ALL_FIXTURES:
            if grp not in THIRD_PLACE_GROUPS:
                continue
            if frozenset((home, away)) in PLAYED_FIXTURES:
                continue
            gh, ga = simulate_match(home, away, live_elos)
            t = tables[grp]
            t[home][1] += gh; t[home][2] += ga; t[home][3] += gh - ga
            t[away][1] += ga; t[away][2] += gh; t[away][3] += ga - gh
            if gh > ga:
                t[home][0] += 3; t[home][4] += 1; t[away][6] += 1
            elif ga > gh:
                t[away][0] += 3; t[away][4] += 1; t[home][6] += 1
            else:
                t[home][0] += 1; t[home][5] += 1
                t[away][0] += 1; t[away][5] += 1
        for grp in THIRD_PLACE_GROUPS:
            ranked = rank_group(tables[grp])
            if ranked[0] in p1_counts: p1_counts[ranked[0]] += 1
            if ranked[1] in p2_counts: p2_counts[ranked[1]] += 1
    np.random.set_state(rng_state)

    output: dict[str, dict] = {}
    for grp in THIRD_PLACE_GROUPS:
        for team in GROUPS[grp]:
            p_adv = adv_probs.get(team)  # None if no market
            p1 = p1_counts.get(team, 0) / n_quick
            p2 = p2_counts.get(team, 0) / n_quick
            if p_adv is not None:
                # Decompose: subtract P(1st) and P(2nd) to isolate 3rd-qual signal
                p3q = max(0.0, p_adv - p1 - p2)
                source = "MKT"
            else:
                # No market → use Elo-based residual (MC gives P(3rd) directly)
                p3q = max(0.0, 1.0 - p1 - p2) * 0.4  # rough: ~40% of 3rd-place sims qualify
                source = "ELO"
            output[team] = {
                "group": grp,
                "p_advance": p_adv if p_adv is not None else (p1 + p2 + p3q),
                "p_3rd_qual": p3q,
                "p_1st": p1,
                "p_2nd": p2,
                "source": source,
            }
    return output if output else None


# ── Legacy stub kept for backward compat (no longer used in MC blend) ────────
def fetch_polymarket_3rd_place_probs() -> dict[str, float] | None:
    """Deprecated: use fetch_polymarket_advance_probs() instead."""
    return None


# Maximum market weight when the group stage is fully complete (MD3 done).
# At 0 games played: weight = 0 (pure Elo/MC).
# After MD2 (4/6 games): weight = MAX_MARKET_WEIGHT * (4/6) ≈ 40%
# After MD3 (6/6 games): weight = MAX_MARKET_WEIGHT * (6/6) = 60%
MAX_MARKET_WEIGHT: float = 0.60


def dynamic_market_weight(group: str) -> float:
    """
    Compute the market weight for a given group based on how many
    group-stage games have been played.

    Formula:  w = MAX_MARKET_WEIGHT * (games_played / total_games)

    Total games per 4-team group = 6 (round-robin).
    games_played is inferred from the average matches-played across
    the four teams in LIVE_STANDINGS (each game increments two teams
    by 1, so avg_mp == games_played).

    Examples:
      MD0 (no games): weight = 0.60 * 0/6 = 0.00  → pure MC/Elo
      MD1 done:       weight = 0.60 * 2/6 = 0.20
      MD2 done:       weight = 0.60 * 4/6 = 0.40
      MD3 done:       weight = 0.60 * 6/6 = 0.60
    """
    teams = GROUPS.get(group, [])
    if not teams:
        return MAX_MARKET_WEIGHT
    # Use LIVE_STANDINGS mp if available; fall back to 0
    avg_mp = sum(
        LIVE_STANDINGS.get(t, {}).get("mp", 0) for t in teams
    ) / max(len(teams), 1)
    # avg_mp == games played (each match adds 1 mp to each of 2 teams;
    # summed over 4 teams and divided by 4 → games_played / 2; ×2 = games_played)
    # Actually: total_mp_sum = 2 * games_played, avg_mp = games_played/2 * ... 
    # Let's be explicit: sum of mp across all 4 teams = 2 * games_played
    total_mp = sum(LIVE_STANDINGS.get(t, {}).get("mp", 0) for t in teams)
    games_played = total_mp / 2  # each game adds 1 mp to 2 teams
    total_games = 6  # 4-team round robin = C(4,2) = 6
    frac = min(games_played / total_games, 1.0)
    return MAX_MARKET_WEIGHT * frac


def blend_mc_with_market(
    mc_prob: float,
    market_prob: float | None,
    market_weight: float | None = None,
    group: str | None = None,
) -> tuple[float, str]:
    """
    Blend Monte Carlo probability with market-implied probability.

    Weight is dynamic by default: scales with how much of the group
    stage has been played (0% at MD0 → 60% at MD3 complete).
    Pass an explicit market_weight to override, or a group name to
    auto-compute the dynamic weight for that group.

    Returns: (blended_prob, method_label)
    """
    if market_prob is None:
        return mc_prob, "MC"
    if market_weight is None:
        market_weight = dynamic_market_weight(group) if group else MAX_MARKET_WEIGHT * 0.67
    if market_weight <= 0:
        return mc_prob, "MC"
    blended = (1 - market_weight) * mc_prob + market_weight * market_prob
    label = "BLEND" if market_weight < 0.95 else "MKT"
    return blended, label


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_locked_outcomes() -> dict[str, dict[str, str]]:
    """
    Derive mathematically-locked group outcomes from current standings.
    After enough games have been played, some positions are already decided
    regardless of remaining results. We hard-clamp these in the MC so we
    don't waste simulations on impossible scenarios.

    Returns: {group: {"winner": team, "eliminated": [team, ...], ...}}
    Only includes groups/outcomes where something is truly decided.

    Rules used:
      - THROUGH (1st or 2nd guaranteed): max possible points for any rival
        cannot exceed this team's current points.
      - ELIMINATED (cannot finish top-2): even with max points from remaining
        games, team cannot reach the points of the 2nd-place team.
      - GROUP_WINNER: team already guaranteed 1st regardless of MD3.
    """
    live = fetch_standings_from_api() or LIVE_STANDINGS
    remaining_games: dict[str, int] = {}  # group -> remaining fixtures count
    for grp in ALL_GROUPS:
        played = sum(live.get(t, {}).get("mp", 0) for t in GROUPS[grp]) // 2
        remaining_games[grp] = 6 - played

    locked: dict[str, dict] = {}
    for grp in ALL_GROUPS:
        teams = GROUPS[grp]
        rem = remaining_games[grp]
        group_locked: dict = {"eliminated": [], "through": [], "winner": None}

        # Each team can earn at most 3 * remaining_games_for_that_team more pts
        # In a 4-team group with N games remaining, each team plays at most
        # ceil(N * 2/4) ≈ N/2 more games. Easier: track per-team remaining mp.
        team_mp   = {t: live.get(t, {}).get("mp", 0) for t in teams}
        team_pts  = {t: live.get(t, {}).get("w", 0) * 3 + live.get(t, {}).get("d", 0)
                     for t in teams}
        team_rem  = {t: 3 - team_mp[t] for t in teams}  # 3 games per team total
        team_max  = {t: team_pts[t] + 3 * team_rem[t] for t in teams}

        pts_sorted = sorted(team_pts.values(), reverse=True)

        for team in teams:
            # Eliminated: max possible pts < current 2nd-place pts
            # (can't reach top-2 even with perfect remaining record)
            if team_max[team] < pts_sorted[1]:
                group_locked["eliminated"].append(team)
            # Through: current pts > max possible for any team currently below 2nd
            # i.e., this team is guaranteed to finish top-2
            others_max = sorted([team_max[t] for t in teams if t != team], reverse=True)
            if team_pts[team] > others_max[1]:  # strictly better than 3rd-best-case
                group_locked["through"].append(team)
            # Group winner: current pts > max possible for every other team
            if all(team_pts[team] > team_max[t] for t in teams if t != team):
                group_locked["winner"] = team

        if any(group_locked[k] for k in group_locked):
            locked[grp] = group_locked

    return locked


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def run_monte_carlo(n_sims: int = N_SIMULATIONS, use_markets: bool = False) -> dict:
    """
    Run N Monte Carlo simulations of the remaining group stage.
    
    For each simulation:
      1. Simulate all remaining fixtures using Dixon-Coles Poisson
      2. Rank each group — applying locked outcomes as hard constraints
      3. Collect all 12 third-place teams
      4. Rank the 12 third-place teams globally via full FIFA tiebreaker
         (pts → GD → GF → random) across ALL 12 simultaneously
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
    # Compute tournament-updated Elo ratings once for the whole MC run
    live_elos = compute_tournament_elos()
    # Compute mathematically locked outcomes once (doesn't change during a run)
    locked = compute_locked_outcomes()

    for _ in range(n_sims):
        # Deep copy standings for this simulation
        tables = {grp: {t: list(v) for t, v in grp_table.items()}
                  for grp, grp_table in base_tables.items()}

        # Simulate remaining fixtures with tournament-adjusted Elos
        simulate_group_stage(tables, live_elos)
        
        # Rank each group
        ranked = {grp: rank_group(tables[grp]) for grp in ALL_GROUPS}

        # ── Apply locked outcomes as hard constraints ──────────────────────────
        # If a team is mathematically eliminated, force them to 3rd/4th.
        # If a team has clinched 1st, force them to the top slot.
        for grp, grp_lock in locked.items():
            r = ranked[grp]
            # Force confirmed winner to position 0
            winner = grp_lock.get("winner")
            if winner and r[0] != winner:
                r.remove(winner)
                r.insert(0, winner)
            # Force eliminated teams out of top-2
            for elim in grp_lock.get("eliminated", []):
                if elim in r[:2]:
                    idx = r.index(elim)
                    # Swap with the highest-ranked non-locked team outside top-2
                    for swap_idx in range(2, len(r)):
                        if r[swap_idx] not in grp_lock.get("eliminated", []):
                            r[idx], r[swap_idx] = r[swap_idx], r[idx]
                            break
        
        # Record Group G outcomes
        g_winner   = ranked["G"][0]
        g_runnerup = ranked["G"][1]
        g_winner_counts[g_winner]     += 1
        g_runnerup_counts[g_runnerup] += 1
        
        # ── Full FIFA cross-group 3rd-place ranking ────────────────────────────
        # Collect ALL 12 third-place teams with their full tiebreaker record.
        # FIFA ranks them simultaneously: pts → GD → GF → fair play → FIFA rank.
        # We approximate fair play / FIFA rank with a random tiebreak (rare).
        third_place_teams = []
        for grp in ALL_GROUPS:
            third_team = ranked[grp][2]
            s = tables[grp][third_team]
            # Full record: [pts, gd, gf, random_noise]
            rec = (s[0], s[3], s[1], np.random.random())
            third_place_teams.append((third_team, grp, rec))
        
        # Sort all 12 simultaneously — this IS the FIFA rule
        third_place_teams.sort(key=lambda x: x[2], reverse=True)
        
        # Top 8 advance to Round of 32
        advancing_thirds = third_place_teams[:8]
        
        for t, _, _ in advancing_thirds:
            third_advance_counts[t] += 1
        
        # Who from eligible groups (A/E/H/I/J) is the 3rd-place qualifier?
        eligible_advancing = [(t, grp) for t, grp, _ in advancing_thirds
                              if grp in THIRD_PLACE_GROUPS]
        
        # Match 82: Group G winner vs. the eligible 3rd-place qualifier
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
        # ── Group G blend ────────────────────────────────────────────────────
        # Advance market replaces old group-winner CLOB IDs (retired)
        adv_probs_g = fetch_polymarket_advance_probs()
        if adv_probs_g:
            for team in GROUPS["G"]:
                mkt_p = adv_probs_g.get(team)
                blended, method = blend_mc_with_market(g_winner_prob[team], mkt_p, group="G")
                g_winner_prob[team] = blended
                methods[team] = method
            # Re-normalize so Group G winner probs sum to 1
            g_total = sum(g_winner_prob[t] for t in GROUPS["G"])
            if g_total > 0:
                g_winner_prob = {t: p / g_total for t, p in g_winner_prob.items()}

        # ── Third-place blend — advance-market decomposition ─────────────────
        # Strategy: P_advance(team) = P_1st + P_2nd + P_3rd_qual
        # The Polymarket advance market resolves YES for ANY path through
        # (1st, 2nd, qualifying 3rd). So the advance price is the most direct
        # signal available. We use it as the market input to blend_mc_with_market,
        # which weights 60% MC / 40% market. This handles edge cases cleanly:
        #   Ecuador 25.5% advance: MC third_advance_prob ~20% → blend ≈ 23%
        #   Sweden 87.2% advance: NOT in THIRD_PLACE_GROUPS, so skipped here
        #   Türkiye: advance market closed/resolved, no price → stays at MC=0
        adv_probs = fetch_polymarket_advance_probs()  # {team: P(advance by any means)}
        if adv_probs:
            for grp in THIRD_PLACE_GROUPS:
                for team in GROUPS[grp]:
                    p_adv_mkt = adv_probs.get(team)
                    if p_adv_mkt is None:
                        methods.setdefault(team, "MC")
                        continue
                    mc_p3q = third_advance_prob.get(team, 0.0)
                    # Blend MC third-place-advance prob with market advance price.
                    # The market price is slightly higher than pure P_3rd_qual
                    # (it includes P_1st + P_2nd), but for genuine 3rd-place
                    # candidates those are near-zero, so the signal is accurate.
                    # For teams likely to finish 1st/2nd, market price >> mc_p3q,
                    # which would overstate their 3rd-place prob — so cap the
                    # market signal at 2x the MC estimate.
                    # Cap market signal at 2x MC to prevent 1st/2nd favorites
                    # from inflating their 3rd-place prob. No floor — if the
                    # market says 5% advance (Curaçao), don't manufacture signal.
                    p_adv_capped = min(p_adv_mkt, mc_p3q * 2.0) if mc_p3q > 0 else p_adv_mkt
                    blended, method = blend_mc_with_market(mc_p3q, p_adv_capped, group=grp)
                    third_advance_prob[team] = blended
                    methods[team] = method

        # ── Elimination suppression ─────────────────────────────────────────
        # Teams with tournament-win price ≤ 0.5% or advance price ≤ 2% are
        # treated as functionally eliminated. Zero out their third_advance_prob
        # so they can't appear as plausible Match 82 opponents.
        eliminated = get_eliminated_teams(use_markets=True)
        for team in eliminated:
            if team in third_advance_prob:
                third_advance_prob[team] = 0.0
                methods[team] = "ELIM"

        # ── Propagate blended Group G probs into match82_joint_prob ──────────
        # The joint counts were built from raw MC; re-weight by blended g_winner_prob
        if adv_probs_g:
            # Compute original MC g_winner_prob for rescaling
            mc_g_total = sum(g_winner_counts.values())
            mc_g_prob_raw = {t: c / mc_g_total for t, c in g_winner_counts.items()} if mc_g_total else {}
            new_joint: dict = {}
            for (gw, third), prob in match82_joint_prob.items():
                mc_raw = mc_g_prob_raw.get(gw, 0)
                if mc_raw > 0:
                    # Scale joint prob by ratio of blended to raw MC
                    scale = g_winner_prob.get(gw, mc_raw) / mc_raw
                    new_joint[(gw, third)] = prob * scale
                else:
                    new_joint[(gw, third)] = prob
            # Re-normalize joint distribution
            joint_total = sum(new_joint.values())
            if joint_total > 0:
                match82_joint_prob = {k: v / joint_total for k, v in new_joint.items()}

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
    Joint probability heatmap: Group G winner (rows) x 3rd-place team (columns).
    Each cell = P(G winner = X AND 3rd place qualifier = Y) from MC counts.
    Shows team-level matchups (not group aggregates) so every cell is a specific
    Match 82 scenario.
    """
    g_teams = sorted(GROUPS["G"],
                     key=lambda t: mc["g_winner_prob"].get(t, 0), reverse=True)

    # Build list of 3rd-place teams that appear in any joint prob cell,
    # sorted by their total advance probability (most likely first)
    tp_teams_raw: dict[str, float] = {}
    for (gw, tp), prob in mc["match82_joint_prob"].items():
        tp_grp = next((g for g, ms in GROUPS.items() if tp in ms), None)
        if tp_grp in THIRD_PLACE_GROUPS:
            tp_teams_raw[tp] = tp_teams_raw.get(tp, 0) + prob
    # Show top 10 3rd-place candidates to keep the chart readable
    tp_teams = sorted(tp_teams_raw, key=tp_teams_raw.get, reverse=True)[:10]

    z = []
    text = []
    for gw in g_teams:
        row_z, row_t = [], []
        for tp in tp_teams:
            p = mc["match82_joint_prob"].get((gw, tp), 0) * 100
            row_z.append(p)
            row_t.append(f"{p:.1f}%" if p >= 0.1 else "—")
        z.append(row_z)
        text.append(row_t)

    y_labels = [f"{FLAG_MAP.get(t,'🏳️')} {t}" for t in g_teams]
    x_labels = [f"{FLAG_MAP.get(t,'🏳️')} {t}" for t in tp_teams]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, family="JetBrains Mono, monospace"),
        colorscale=[[0,"#050d1a"],[0.1,"#0c2040"],[0.35,"#1d4ed8"],[0.65,"#0ea5e9"],[1.0,"#38bdf8"]],
        showscale=True,
        colorbar=dict(
            title=dict(text="Match 82 P%", side="right", font=dict(color="#64748b",size=10)),
            tickfont=dict(color="#64748b",size=10),bgcolor="#0d1020",
            bordercolor="#1e2235",borderwidth=1,thickness=14,len=0.8,
        ),
        hovertemplate=(
            "<b>%{y}</b> wins Group G<br>"
            "<b>%{x}</b> advances as 3rd place<br>"
            "Match 82 probability: <b>%{text}</b><extra></extra>"
        ),
    ))
    fig.update_layout(
        **_dark(height=max(280, 90 * len(g_teams))),
        title=dict(
            text=f"Match 82 Scenario Matrix — {mc['n_sims']:,} simulations",
            font=dict(size=12, color="#94a3b8"), x=0
        ),
        xaxis=dict(**AXIS_STYLE, title="3rd-Place Qualifier (top 10 candidates)",
                   tickangle=-30),
        yaxis=dict(**AXIS_STYLE, title="Group G Winner", autorange="reversed"),
        margin=dict(l=140, r=40, t=50, b=100),
    )
    return fig


def build_scenario_table(mc: dict, top_n: int = 12) -> str:
    """
    Return an HTML table of the top-N most probable Match 82 scenarios,
    showing Group G winner, 3rd-place opponent, and joint probability.
    """
    pairs = sorted(mc["match82_joint_prob"].items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not pairs: return ""
    top_pct = pairs[0][1] * 100
    rows = ""
    for rank, ((gw, tp), prob) in enumerate(pairs, 1):
        pct = prob * 100
        bar_w = int((pct / top_pct) * 56) if top_pct else 0
        gw_flag = FLAG_MAP.get(gw, "🏳️")
        tp_flag = FLAG_MAP.get(tp, "🏳️")
        rank_color = "#fbbf24" if rank == 1 else ("#94a3b8" if rank == 2 else ("#cd7f32" if rank == 3 else "#475569"))
        rows += (
            f'<tr style="border-bottom:1px solid #1e2235;">'
            f'<td style="color:{rank_color};font-weight:700;padding:5px 8px;width:28px;'
            f'font-family:monospace;">#{rank}</td>'
            f'<td style="padding:5px 8px;">{gw_flag} <b>{gw}</b></td>'
            f'<td style="padding:5px 8px;color:#64748b;">vs</td>'
            f'<td style="padding:5px 8px;">{tp_flag} <b>{tp}</b></td>'
            f'<td style="padding:5px 8px;font-family:monospace;color:#38bdf8;">'
            f'<span style="display:inline-block;background:#0c2040;height:6px;'
            f'width:{bar_w}px;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
            f'{pct:.1f}%</td>'
            f'</tr>'
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:0.82rem;color:#cbd5e1;">'
        '<thead><tr style="border-bottom:1px solid #334155;">'
        '<th style="padding:4px 8px;color:#64748b;text-align:left;"></th>'
        '<th style="padding:4px 8px;color:#64748b;text-align:left;">Grp G Winner</th>'
        '<th></th>'
        '<th style="padding:4px 8px;color:#64748b;text-align:left;">3rd-Place Opp.</th>'
        '<th style="padding:4px 8px;color:#64748b;text-align:left;">P(Match 82)</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


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
        
        _pill_cls = {"MKT": "mkt", "BLEND": "blend", "ELIM": "elim"}.get(method, "mc")
        method_label = f'<span class="method-pill method-{_pill_cls}">{method}</span>'
        
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
            # Show per-group dynamic weights so user can see how trust shifts
            g_w  = dynamic_market_weight("G")
            # Show a representative 3rd-place group weight (use the most-played one)
            tp_w = max(dynamic_market_weight(grp) for grp in THIRD_PLACE_GROUPS)
            st.caption(
                f"**Dynamic blend** — scales with games played.  \n"
                f"Group G: **{g_w*100:.0f}%** mkt / {(1-g_w)*100:.0f}% MC  \n"
                f"3rd-place groups: up to **{tp_w*100:.0f}%** mkt / {(1-tp_w)*100:.0f}% MC  \n"
                f"_(reaches 60% mkt when MD3 completes)_"
            )
        
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
        st.caption("**Engine**: Dixon-Coles corrected Poisson + Dynamic Elo")
        st.caption("**3rd-place**: Full 12-group race, not independent per-team probs")
        st.caption("**Elo base**: eloratings.net as of June 13, 2026")
        st.caption("**Elo updates**: K=60 applied after every WC result")
        st.caption("**Blend**: Dynamic — 0% mkt at MD0 → 60% mkt at MD3 (per group)")

        # ── Locked outcomes panel ─────────────────────────────────────────
        locked_outcomes = compute_locked_outcomes()
        grp_g_lock = locked_outcomes.get("G", {})
        any_locked = any(
            locked_outcomes.get(g, {}).get("winner") or
            locked_outcomes.get(g, {}).get("eliminated") or
            locked_outcomes.get(g, {}).get("through")
            for g in ["G"] + THIRD_PLACE_GROUPS
        )
        if any_locked:
            st.divider()
            st.markdown("### 🔐 Decided")
            st.caption("Mathematically locked outcomes — MC clamps these.")
            for grp in ["G"] + THIRD_PLACE_GROUPS:
                gl = locked_outcomes.get(grp, {})
                winner = gl.get("winner")
                through = gl.get("through", [])
                elim = gl.get("eliminated", [])
                if not (winner or through or elim):
                    continue
                st.markdown(f"**Group {grp}**")
                if winner:
                    f = FLAG_MAP.get(winner, "🏳️")
                    st.markdown(
                        f'<span style="color:#4ade80;font-size:0.78rem;">'
                        f'✅ {f} {winner} — group winner</span>',
                        unsafe_allow_html=True)
                for t in through:
                    if t != winner:
                        f = FLAG_MAP.get(t, "🏳️")
                        st.markdown(
                            f'<span style="color:#60a5fa;font-size:0.78rem;">'
                            f'→ {f} {t} — through</span>',
                            unsafe_allow_html=True)
                for t in elim:
                    f = FLAG_MAP.get(t, "🏳️")
                    st.markdown(
                        f'<span style="color:#f87171;font-size:0.78rem;">'
                        f'❌ {f} {t} — eliminated</span>',
                        unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📊 Elo Shifts")
        st.caption("Change from base rating after WC results")
        live_elos_sidebar = compute_tournament_elos()
        # Show all Group G teams + top movers from other groups
        key_teams = list(GROUPS["G"]) + [
            t for grp in THIRD_PLACE_GROUPS for t in GROUPS[grp]
        ]
        shifts = [
            (t, live_elos_sidebar.get(t, ELO.get(t, 1700)),
             live_elos_sidebar.get(t, ELO.get(t, 1700)) - ELO.get(t, 1700))
            for t in key_teams
        ]
        # Sort by absolute shift descending, show top 12
        # Mark eliminated teams from market signal
        _elim_set = get_eliminated_teams(use_markets=True) if True else set()
        shifts.sort(key=lambda x: -abs(x[2]))
        for team, new_elo, delta in shifts[:12]:
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
            color = "#22c55e" if delta > 0 else ("#ef4444" if delta < 0 else "#94a3b8")
            flag  = FLAG_MAP.get(team, "🏳️")
            elim_badge = (' <span style="font-size:0.65rem;color:#f87171;'
                          'background:#2d0f0f;border:1px solid #6e1e1e;'
                          'border-radius:3px;padding:0px 4px;">ELIM</span>'
                          if team in _elim_set else "")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;font-size:0.78rem;padding:2px 0;">'
                f'<span>{flag} {team}{elim_badge}</span>'
                f'<span style="color:{color};font-weight:600;">{arrow} {abs(delta):.0f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Show snapshot status
        snap = load_precomputed_results()
        if snap:
            age = snap.get("_age_hours")
            stale = snap.get("_stale", False)
            age_str = f"{age}h ago" if age is not None else "cached"
            color = "#f87171" if stale else "#4ade80"
            label = f"⚠ Snapshot {age_str} (stale)" if stale else f"✓ Snapshot {age_str}"
            st.markdown(
                f'<p style="color:{color};font-size:0.72rem;margin-top:0.3rem;">{label}</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="color:#475569;font-size:0.72rem;margin-top:0.3rem;">No snapshot — live sim</p>',
                unsafe_allow_html=True,
            )

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


def render_matchup_chart_html(mc: dict, top_n: int = 12) -> str:
    """
    Renders the top-N most probable Match 82 matchups as a pure HTML/CSS
    horizontal bar chart — no image rendering, no kaleido, works in every
    email client including Gmail.

    Returns an HTML string safe to embed directly in the email body.
    """
    joint = mc.get("match82_joint_prob", {})
    if not joint:
        return '<p style="color:#64748b;font-size:0.8rem;">[No matchup data available]</p>'

    # Sort and take top N
    sorted_pairs = sorted(joint.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_pairs:
        return ''

    max_prob = sorted_pairs[0][1]

    rows = []
    for (team_a, team_b), prob in sorted_pairs:
        pct = prob * 100
        bar_width = int((prob / max_prob) * 100) if max_prob > 0 else 0
        # No emoji flags — Gmail blocks rendering of tables containing emoji
        label = f"{team_a} vs {team_b}"
        rows.append(
            f'<tr>'
            f'<td style="padding:5px 10px 5px 0;color:#94a3b8;font-size:0.78rem;'
            f'white-space:nowrap;width:220px;font-family:Arial,sans-serif;">'
            f'{label}</td>'
            f'<td style="padding:5px 0;">'
            f'<table cellpadding="0" cellspacing="0" width="100%">'
            f'<tr>'
            f'<td width="{bar_width}%" style="background:#3b82f6;height:14px;'
            f'border-radius:3px;"></td>'
            f'<td width="{100 - bar_width}%"></td>'
            f'</tr></table>'
            f'</td>'
            f'<td style="padding:5px 0 5px 8px;color:#f8fafc;font-size:0.78rem;'
            f'white-space:nowrap;font-family:Arial,sans-serif;font-weight:700;">'
            f'{pct:.2f}%</td>'
            f'</tr>'
        )

    return (
        '<table cellpadding="0" cellspacing="0" '
        'style="width:100%;border-collapse:collapse;">'
        + "".join(rows)
        + "</table>"
    )


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
    import datetime, re
    today = datetime.date.today().strftime("%B %d, %Y")
    n_sims = mc.get("n_sims", 50000)

    # Strip emoji and markdown bold from paragraph — keep it clean plain text
    def strip_emoji(text: str) -> str:
        return "".join(
            c for c in text
            if not (0x1F300 <= ord(c) <= 0x1FAFF or
                    0x2600 <= ord(c) <= 0x27BF or
                    0x1F1E0 <= ord(c) <= 0x1F1FF)
        ).strip()

    clean = strip_emoji(paragraph)
    # Strip leading title line if present (e.g. "Match 82 Daily Brief — June X, 2026")
    lines = clean.strip().splitlines()
    if lines and ("Daily Brief" in lines[0] or "Match 82" in lines[0]):
        clean = "\n".join(lines[1:]).strip()
    # Convert **bold** to <strong>
    clean_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", clean)
    # Split on double newlines into paragraphs
    paras = [p.strip() for p in re.split(r"\n{2,}", clean_html) if p.strip()]
    paragraph_html = "".join(
        f'<p style="margin:0 0 14px 0;color:#1a1a2e;font-size:15px;'
        f'line-height:1.8;font-family:Arial,Helvetica,sans-serif;">{p}</p>'
        for p in paras
    )

    # Light-mode email — Gmail-safe, no dark backgrounds
    # Chart is sent as attachment, not embedded
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Match 82 Brief</title>
</head>
<body style="margin:0;padding:0;background:#f4f6f9;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f6f9;">
<tr><td align="center" style="padding:28px 12px;">

  <table width="580" cellpadding="0" cellspacing="0"
         style="max-width:580px;width:100%;background:#ffffff;
                border-radius:8px;border:1px solid #dde3ec;">

    <!-- Header band -->
    <tr>
      <td style="background:#0d1b38;padding:20px 28px;border-radius:8px 8px 0 0;">
        <p style="margin:0;color:#7aa8cc;font-size:11px;font-family:Arial,sans-serif;
                  text-transform:uppercase;letter-spacing:2px;font-weight:700;">
          MATCH 82 &middot; LUMEN FIELD &middot; JULY 1, 2026
        </p>
        <p style="margin:6px 0 2px;color:#ffffff;font-size:20px;font-weight:700;
                  font-family:Arial,sans-serif;">Daily Odds Brief</p>
        <p style="margin:0;color:#7aa8cc;font-size:13px;font-family:Arial,sans-serif;">{today}</p>
      </td>
    </tr>

    <!-- Body text -->
    <tr>
      <td style="padding:24px 28px 8px;">
        {paragraph_html}
      </td>
    </tr>

    <!-- Chart note -->
    <tr>
      <td style="padding:4px 28px 24px;">
        <p style="margin:0;color:#64748b;font-size:12px;font-family:Arial,sans-serif;
                  border-top:1px solid #e2e8f0;padding-top:14px;">
          See attached <strong>match82_matchups.png</strong> for the
          Most Probable Exact Matchups chart &mdash; based on a
          {n_sims:,}-trial Dixon-Coles / Elo Monte Carlo simulation.
        </p>
      </td>
    </tr>

    <!-- Footer -->
    <tr>
      <td style="background:#f8fafc;padding:14px 28px;border-radius:0 0 8px 8px;
                  border-top:1px solid #e2e8f0;">
        <p style="margin:0;color:#94a3b8;font-size:11px;font-family:Arial,sans-serif;">
          You signed up at the
          <a href="https://match82-wc2026-tracker.streamlit.app"
             style="color:#3b82f6;">Match 82 Tracker</a>. No spam, ever.
        </p>
      </td>
    </tr>

  </table>

</td></tr>
</table>
</body>
</html>"""


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

                **Chart snapshot timing**: The cron runs at 7 AM PT. With `FOOTBALL_DATA_API_KEY`
                configured, standings are fetched automatically from football-data.org — no
                manual code edits needed. The chart reflects that morning's MC output based
                on the previous day's final scores.

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

    # ── Load pre-computed snapshot or run live simulation ─────────────────────
    precomputed = load_precomputed_results()
    if precomputed and not use_markets:
        mc = precomputed
        age = mc.get("_age_hours")
        stale = mc.get("_stale", False)
        if stale:
            st.warning(
                f"⚠️ Pre-computed snapshot is {age}h old — standings may be outdated. "
                f"Click **Re-run Simulation** in the sidebar for fresh numbers.",
                icon="⏰",
            )
        else:
            age_str = f"{age}h ago" if age is not None else "recently"
            st.info(
                f"⚡ Loaded pre-computed snapshot ({age_str}) — instant load. "
                f"Use sidebar to re-run a live simulation.",
                icon="📊",
            )
    else:
        reason = "Polymarket blend requires live simulation" if use_markets else "No pre-computed snapshot found"
        with st.spinner(f"Running {n_sims:,} Monte Carlo simulations… ({reason})"):
            mc = run_monte_carlo(n_sims=n_sims, use_markets=use_markets)
    
    chaos_val = compute_chaos_index(mc)
    c_label, _ = chaos_label(chaos_val)
    
    g_leader = max(mc["g_winner_prob"], key=mc["g_winner_prob"].get)
    g_leader_p = mc["g_winner_prob"][g_leader]
    
    # Best 3rd-place candidate: sum joint prob over all Group G winners per opponent.
    # This correctly reflects both P(team reaches Match 82 as 3rd) AND
    # P(Group G produces a winner), giving a true matchup-weighted signal.
    eligible_3rd_joint = {}
    for (gw, third), prob in mc["match82_joint_prob"].items():
        eligible_3rd_joint[third] = eligible_3rd_joint.get(third, 0) + prob
    if eligible_3rd_joint:
        best_3rd = max(eligible_3rd_joint, key=eligible_3rd_joint.get)
        best_3rd_p = mc["third_advance_prob"].get(best_3rd, 0)  # show advance prob in label
    else:
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
    
    # ── HEATMAP + SCENARIO TABLE + CHAOS GAUGE ──────────────────────────────────
    st.markdown("## 📊 Probability Analytics")
    col_heat, col_chaos = st.columns([2.2, 1])

    with col_heat:
        st.plotly_chart(build_heatmap(mc), use_container_width=True)
        st.markdown(
            "<div style='font-size:0.75rem;color:#64748b;margin:-8px 0 4px;'>"
            "Top 12 most probable Match 82 matchups — ranked by joint probability"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(build_scenario_table(mc, top_n=12), unsafe_allow_html=True)
    
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

    # ── MARKET-IMPLIED 3RD PLACE RACE ───────────────────────────────────────────
    mkt_3rd = compute_market_third_place_probs()
    if mkt_3rd:
        st.markdown("## 🎰 3rd Place Race — Market View")
        st.caption(
            "Polymarket group-win prices fed into a Plackett-Luce model to infer "
            "P(finish 3rd) per team. Teams without a Polymarket market use Elo as a "
            "fallback weight. **MKT** = market-priced · **ELO** = Elo fallback."
        )

        # ── Per-group breakdown ──────────────────────────────────────────────
        grp_tabs = st.tabs([f"Group {g}" for g in THIRD_PLACE_GROUPS])
        for tab, grp in zip(grp_tabs, THIRD_PLACE_GROUPS):
            with tab:
                grp_teams = [
                    (t, mkt_3rd[t]) for t in GROUPS[grp] if t in mkt_3rd
                ]
                grp_teams.sort(key=lambda x: -x[1]["p_3rd_qual"])
                rows = []
                for t, d in grp_teams:
                    rows.append({
                        "Team": f"{FLAG_MAP.get(t,'🏳️')} {t}",
                        "Source": d["source"],
                        "P(Win Group) %": round(d["p_1st"] * 100, 1),
                        "P(Finish 3rd) %": round(d["p_3rd_qual"] * 100, 1),
                    })
                df_grp = pd.DataFrame(rows)
                st.dataframe(
                    df_grp,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "P(Win Group) %": st.column_config.ProgressColumn(
                            "P(Win Group) %", min_value=0, max_value=100, format="%.1f%%"
                        ),
                        "P(Finish 3rd) %": st.column_config.ProgressColumn(
                            "P(Finish 3rd) %", min_value=0, max_value=100, format="%.1f%%"
                        ),
                    },
                )

        # ── Cross-group leaderboard ──────────────────────────────────────────
        st.markdown("### Most Likely Match 82 Opponents (3rd Place)")
        st.caption(
            "Ranked by P(finish 3rd in their group). Top 8 third-place teams "
            "across all 12 groups advance — only Groups A/E/H/I/J are eligible "
            "to face the Group G winner at Match 82."
        )

        all_rows = []
        for t, d in mkt_3rd.items():
            all_rows.append({
                "Team": f"{FLAG_MAP.get(t,'🏳️')} {t}",
                "Group": d["group"],
                "Source": d["source"],
                "P(Win Group) %": round(d["p_1st"] * 100, 1),
                "P(Finish 3rd) %": round(d["p_3rd_qual"] * 100, 1),
            })
        all_rows.sort(key=lambda x: -x["P(Finish 3rd) %"])
        df_all = pd.DataFrame(all_rows)
        st.dataframe(
            df_all,
            use_container_width=True,
            hide_index=True,
            column_config={
                "P(Win Group) %": st.column_config.ProgressColumn(
                    "P(Win Group) %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "P(Finish 3rd) %": st.column_config.ProgressColumn(
                    "P(Finish 3rd) %", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

        # ── Small print on methodology ───────────────────────────────────────
        with st.expander("Methodology", expanded=False):
            st.markdown("""
**Model**: Plackett-Luce placement simulation (100k draws per group).

For each group, team strength is set equal to their Polymarket group-win
price ("Yes" price on the CLOB). Teams without a liquid market (typically
the weakest team in the group) receive a strength proportional to their
Elo rating scaled to the residual probability mass left after the known
markets are priced in.

The Gumbel-max trick converts these strengths into a full placement
distribution — P(1st), P(2nd), P(3rd), P(4th) — in a single vectorized
Monte Carlo pass. This is mathematically equivalent to a random-utility
model where each team's latent performance draw is `log(strength) + Gumbel(0,1)`.

**Important caveat**: P(finish 3rd in group) ≠ P(advance to Round of 32).
Only the top 8 third-place teams across all 12 groups advance. The full
Monte Carlo engine (left panel) models the global ranking step explicitly.
""")
    else:
        # Polymarket unavailable — show a quiet note
        if use_markets:
            st.info(
                "Polymarket data unavailable right now — toggle off to use pure MC.",
                icon="🟣",
            )

    st.markdown("---")

    # ── LIVE STANDINGS TABLE ────────────────────────────────────────────────────
    st.markdown("## 📋 Current Standings")
    
    with st.expander("Group G", expanded=True):
        _live = fetch_standings_from_api() or LIVE_STANDINGS
        rows = []
        for t in GROUPS["G"]:
            s = _live.get(t, {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0})
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
            _live = fetch_standings_from_api() or LIVE_STANDINGS
            rows = []
            for t in GROUPS[grp]:
                s = _live.get(t, {"mp":0,"w":0,"d":0,"l":0,"gf":0,"ga":0})
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
