"""
Match 82 — FIFA World Cup 2026 | Seattle Lumen Field | July 1, 2026
Round of 32: Winner of Group G vs. 3rd Place from Group A / E / H / I / J

Streamlit dashboard tracking live probabilities for the Seattle matchup.

HOW TO CONNECT A LIVE GOOGLE SHEET:
1. Create a Google Sheet with the schema below (or copy the provided template).
2. File → Share → Publish to web → CSV format.
3. Copy the CSV URL (looks like: https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=SHEET_GID)
4. Paste it into the GOOGLE_SHEET_URL constant below (or set env var MATCH82_SHEET_URL).
5. The app will auto-refresh every REFRESH_SECONDS seconds when "Live Mode" is toggled.

Required Google Sheet columns (two separate tabs / sheets recommended):
  Sheet 1 — "group_g"        : Team, MP, W, D, L, GF, GA, GD, Pts, GroupWinnerProb, RunnerUpProb, ThirdPlaceProb
  Sheet 2 — "third_place"    : Group, Team, MP, W, D, L, GF, GA, GD, Pts, AdvanceProb
  (AdvanceProb = probability this 3rd-place team qualifies as one of the 8 best 3rd-place finishers)

If the sheet is not configured the app falls back to embedded mock data.
"""

import os
import time
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Paste your public Google Sheet CSV export URL here, OR set the environment
# variable MATCH82_SHEET_URL before running.  Leave empty to use mock data.
GOOGLE_SHEET_URL: str = os.environ.get("MATCH82_SHEET_URL", "")

# How often (seconds) to re-fetch the sheet in live mode
REFRESH_SECONDS: int = 60

# 3rd-place slot: which groups can send a 3rd-place team to face the Group G winner
THIRD_PLACE_GROUPS: list[str] = ["A", "E", "H", "I", "J"]

# Group G teams (seed with real names; sheet data overrides when live)
GROUP_G_TEAMS: list[str] = ["Belgium", "Egypt", "Iran", "New Zealand"]

# All potential 3rd-place teams for Match 82 eligible groups
# (populated from mock data; sheet overrides when live)
ELIGIBLE_3RD_TEAMS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "E": ["Germany", "Curaçao", "Côte d'Ivoire", "Ecuador"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
}

# Flag emoji mapping for display (extend as needed)
FLAG_MAP: dict[str, str] = {
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czechia": "🇨🇿",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Côte d'Ivoire": "🇨🇮", "Ecuador": "🇪🇨",
    "Spain": "🇪🇸", "Cabo Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "USA": "🇺🇸",
}


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Match 82 — Seattle WC2026 Tracker",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# DARK-THEME CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────

DARK_CSS = """
<style>
  /* ── Global reset to dark surface ── */
  html, body, [class*="css"] {
    background-color: #0a0c14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  }

  /* ── Main container ── */
  .block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1400px !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1020 !important;
    border-right: 1px solid #1e2235 !important;
  }
  [data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background: #111627 !important;
    border: 1px solid #1e2a44 !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
  }
  [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #64748b !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
  }
  [data-testid="stMetricDelta"] svg { display: none; }

  /* ── Section headers ── */
  h1 { font-size: 1.6rem !important; font-weight: 800 !important; color: #f1f5f9 !important; }
  h2 { font-size: 1.15rem !important; font-weight: 700 !important; color: #94a3b8 !important;
       text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid #1e2235;
       padding-bottom: 0.4rem; margin-top: 1.5rem !important; }
  h3 { font-size: 1rem !important; font-weight: 600 !important; color: #e2e8f0 !important; }

  /* ── Dataframe / tables ── */
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
  .dataframe thead th { background: #131b2e !important; color: #64748b !important;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; }
  .dataframe tbody tr:nth-child(even) { background: #0e1425 !important; }
  .dataframe tbody tr:hover { background: #19243d !important; }

  /* ── Select boxes, dropdowns ── */
  [data-baseweb="select"] > div {
    background: #111627 !important;
    border-color: #1e2a44 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
  }

  /* ── Dividers ── */
  hr { border-color: #1e2235 !important; margin: 1.5rem 0 !important; }

  /* ── Info / alert boxes ── */
  [data-testid="stAlert"] {
    background: #0d1a2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #93c5fd !important;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: #0e1628 !important;
    border: 1px solid #1e2a44 !important;
    border-radius: 8px !important;
  }

  /* ── Toggle / checkbox ── */
  [data-baseweb="checkbox"] span { border-color: #38bdf8 !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0a0c14; }
  ::-webkit-scrollbar-thumb { background: #1e2a44; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #38bdf8; }

  /* ── Chaos badge ── */
  .chaos-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .chaos-low    { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .chaos-medium { background: #1c1917; color: #fbbf24; border: 1px solid #92400e; }
  .chaos-high   { background: #1a0a0a; color: #f87171; border: 1px solid #7f1d1d; }

  /* ── Path-to-Seattle recipe card ── */
  .recipe-card {
    background: #0e1a2e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    line-height: 1.7;
  }
  .recipe-step {
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    margin: 0.4rem 0;
  }
  .step-num {
    background: #1d4ed8;
    color: white;
    width: 22px; height: 22px;
    border-radius: 50%;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .step-text { color: #cbd5e1; font-size: 0.88rem; }
  .verdict-yes { color: #4ade80; font-weight: 600; }
  .verdict-no  { color: #f87171; font-weight: 600; }
  .verdict-maybe { color: #fbbf24; font-weight: 600; }

  /* ── Probability pill ── */
  .prob-pill {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }
  .prob-high   { background: #052e16; color: #4ade80; }
  .prob-medium { background: #172554; color: #60a5fa; }
  .prob-low    { background: #1a0a0a; color: #f87171; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER — Mock data + optional Google Sheet override
# ─────────────────────────────────────────────────────────────────────────────

def _mock_group_g() -> pd.DataFrame:
    """
    Returns a mock Group G standings DataFrame.
    Replace with live Google Sheet data when available.

    Column spec:
      Team               : str
      MP                 : int   — matches played
      W, D, L            : int   — wins, draws, losses
      GF, GA, GD         : int   — goals for, against, difference
      Pts                : int   — points
      GroupWinnerProb    : float — 0.0–1.0 implied probability (from prediction market)
      RunnerUpProb       : float — 0.0–1.0
      ThirdPlaceProb     : float — 0.0–1.0 (prob of finishing 3rd in group, not advancing)
    """
    data = {
        "Team":            ["Belgium",  "Egypt",  "Iran",   "New Zealand"],
        "MP":              [2,           2,         2,        2           ],
        "W":               [1,           1,         1,        0           ],
        "D":               [1,           0,         0,        0           ],
        "L":               [0,           1,         1,        2           ],
        "GF":              [3,           2,         1,        0           ],
        "GA":              [1,           3,         2,        2           ],
        "GD":              [2,          -1,        -1,       -2           ],
        "Pts":             [4,           3,         3,        0           ],
        # Prediction market contract prices mapped to probabilities
        # Source: e.g. Polymarket / Kalshi — update these cells in your Google Sheet
        "GroupWinnerProb": [0.62,        0.21,      0.15,     0.02        ],
        "RunnerUpProb":    [0.25,        0.39,      0.33,     0.03        ],
        "ThirdPlaceProb":  [0.10,        0.27,      0.38,     0.25        ],
    }
    return pd.DataFrame(data)


def _mock_third_place() -> pd.DataFrame:
    """
    Returns a mock DataFrame of all 3rd-place teams in groups eligible for Match 82.

    Column spec:
      Group              : str   — one of A, E, H, I, J
      Team               : str
      MP, W, D, L        : int
      GF, GA, GD, Pts    : int
      AdvanceProb        : float — prob this team ends up as the 3rd-place team of their
                                   group AND ranks in the top-8 third-place finishers globally
    """
    rows = []
    # Probabilities reflect both: (a) finishing 3rd in group AND (b) being top-8 globally
    seeds = {
        "A": [("Mexico", 3,2,1,0, 4,2, 2,7, 0.08),
              ("South Africa",3,1,1,1,3,3, 0,4, 0.04),
              ("South Korea",3,1,0,2,2,4,-2,3, 0.03),
              ("Czechia",3,0,0,3,1,4,-3,0, 0.01)],
        "E": [("Germany",3,2,1,0,5,1, 4,7, 0.07),
              ("Ecuador",3,1,1,1,3,3, 0,4, 0.05),
              ("Côte d'Ivoire",3,1,0,2,2,4,-2,3, 0.04),
              ("Curaçao",3,0,0,3,0,6,-6,0, 0.00)],
        "H": [("Spain",3,3,0,0,7,1, 6,9, 0.06),
              ("Uruguay",3,2,0,1,4,3, 1,6, 0.07),
              ("Saudi Arabia",3,0,1,2,2,5,-3,1, 0.02),
              ("Cabo Verde",3,0,1,2,1,5,-4,1, 0.01)],
        "I": [("France",3,2,1,0,5,2, 3,7, 0.09),
              ("Senegal",3,1,1,1,3,3, 0,4, 0.06),
              ("Norway",3,1,0,2,3,4,-1,3, 0.04),
              ("Iraq",3,0,0,3,1,5,-4,0, 0.01)],
        "J": [("Argentina",3,2,1,0,6,2, 4,7, 0.10),
              ("Algeria",3,1,1,1,3,3, 0,4, 0.05),
              ("Austria",3,1,0,2,2,4,-2,3, 0.04),
              ("Jordan",3,0,0,3,1,5,-4,0, 0.01)],
    }
    for grp, teams in seeds.items():
        for (team, mp, w, d, l, gf, ga, gd, pts, ap) in teams:
            rows.append({
                "Group": grp, "Team": team,
                "MP": mp, "W": w, "D": d, "L": l,
                "GF": gf, "GA": ga, "GD": gd, "Pts": pts,
                "AdvanceProb": ap,
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_data(sheet_url: str) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Load Group G and 3rd-place data.

    If sheet_url is provided, attempts to read two separate CSV exports:
      - sheet_url                      → group_g sheet (gid=0 or explicit gid param)
      - sheet_url with &gid=<second>   → third_place sheet

    Falls back to mock data on any error or when sheet_url is empty.

    Returns: (df_group_g, df_third_place, is_live)
    """
    if not sheet_url:
        return _mock_group_g(), _mock_third_place(), False

    try:
        # Expect sheet_url to be the export URL for the GROUP_G sheet.
        # Derive the 3rd-place sheet URL by appending &sheet=third_place
        # (user must configure two separate publish URLs for each tab).
        # For simplicity, we accept a comma-separated pair:
        #   "https://...&gid=0,https://...&gid=123456"
        if "," in sheet_url:
            url_g, url_3rd = [u.strip() for u in sheet_url.split(",", 1)]
        else:
            url_g = sheet_url
            url_3rd = None  # fallback to mock

        df_g = pd.read_csv(url_g)
        # Normalise column names: strip whitespace, title-case optional cols
        df_g.columns = [c.strip() for c in df_g.columns]
        # Cast numeric cols
        for col in ["MP","W","D","L","GF","GA","GD","Pts"]:
            if col in df_g.columns:
                df_g[col] = pd.to_numeric(df_g[col], errors="coerce").fillna(0).astype(int)
        for col in ["GroupWinnerProb","RunnerUpProb","ThirdPlaceProb"]:
            if col in df_g.columns:
                df_g[col] = pd.to_numeric(df_g[col], errors="coerce").fillna(0.0)

        if url_3rd:
            df_3rd = pd.read_csv(url_3rd)
            df_3rd.columns = [c.strip() for c in df_3rd.columns]
            for col in ["MP","W","D","L","GF","GA","GD","Pts"]:
                if col in df_3rd.columns:
                    df_3rd[col] = pd.to_numeric(df_3rd[col], errors="coerce").fillna(0).astype(int)
            if "AdvanceProb" in df_3rd.columns:
                df_3rd["AdvanceProb"] = pd.to_numeric(df_3rd["AdvanceProb"], errors="coerce").fillna(0.0)
            # Filter to eligible groups only
            df_3rd = df_3rd[df_3rd["Group"].isin(THIRD_PLACE_GROUPS)].copy()
        else:
            df_3rd = _mock_third_place()

        return df_g, df_3rd, True

    except Exception as e:
        st.sidebar.warning(f"Sheet load failed — showing mock data.\n\n`{e}`")
        return _mock_group_g(), _mock_third_place(), False


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def flag(team: str) -> str:
    return FLAG_MAP.get(team, "🏳️")


def prob_to_pct(p: float) -> str:
    return f"{p*100:.1f}%"


def prob_pill_html(p: float) -> str:
    cls = "prob-high" if p >= 0.50 else ("prob-medium" if p >= 0.20 else "prob-low")
    return f'<span class="prob-pill {cls}">{prob_to_pct(p)}</span>'


def compute_chaos_index(df_g: pd.DataFrame, df_3rd: pd.DataFrame) -> float:
    """
    Chaos Index (0–100):

    Measures how unsettled the Match 82 slot is by combining:
      1. Entropy of GroupWinnerProb distribution in Group G
      2. Entropy of AdvanceProb distribution across 3rd-place teams

    A uniform distribution (all equal odds) → chaos = 100.
    A fully locked distribution (one team at 100%) → chaos = 0.
    """
    def normalised_entropy(probs: np.ndarray) -> float:
        """Shannon entropy normalised to [0, 1] based on max possible entropy."""
        probs = np.array(probs, dtype=float)
        probs = probs[probs > 0]  # drop zeros before log
        if len(probs) == 0:
            return 0.0
        probs = probs / probs.sum()
        n = len(probs)
        if n == 1:
            return 0.0
        h = -np.sum(probs * np.log2(probs))
        h_max = math.log2(n)
        return h / h_max if h_max > 0 else 0.0

    winner_probs = df_g["GroupWinnerProb"].values
    third_probs  = df_3rd["AdvanceProb"].values

    e_winner = normalised_entropy(winner_probs)
    e_third  = normalised_entropy(third_probs)

    # Weight 50/50 between the two dimensions
    chaos = 100 * (0.5 * e_winner + 0.5 * e_third)
    return round(chaos, 1)


def chaos_label(ci: float) -> tuple[str, str]:
    """Returns (label, CSS class) for the chaos index."""
    if ci < 35:
        return "LOCKED IN", "chaos-low"
    elif ci < 70:
        return "IN FLUX", "chaos-medium"
    else:
        return "TOTAL CHAOS", "chaos-high"


def build_heatmap_data(df_g: pd.DataFrame, df_3rd: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a joint-probability matrix:
      Rows    = Group G teams (potential Group G winners)
      Columns = Eligible 3rd-place groups (A, E, H, I, J)

    Cell value = P(team wins Group G) × P(best 3rd from that group advances)
    The "best 3rd from group X" probability is the MAX AdvanceProb among teams in group X,
    representing the scenario where the strongest 3rd-place team from X qualifies.
    """
    # P(Group G winner = team)
    winner_probs = dict(zip(df_g["Team"], df_g["GroupWinnerProb"]))

    # P(a 3rd-place team from group X advances) = sum of individual AdvanceProbs in group X
    # (since exactly one team from each group can be the 3rd-place rep)
    group_advance_prob = {}
    for grp in THIRD_PLACE_GROUPS:
        sub = df_3rd[df_3rd["Group"] == grp]
        group_advance_prob[grp] = sub["AdvanceProb"].sum() if not sub.empty else 0.0

    teams  = list(winner_probs.keys())
    groups = THIRD_PLACE_GROUPS

    matrix = np.zeros((len(teams), len(groups)))
    for i, team in enumerate(teams):
        for j, grp in enumerate(groups):
            matrix[i, j] = winner_probs.get(team, 0.0) * group_advance_prob.get(grp, 0.0)

    df_heat = pd.DataFrame(matrix, index=teams, columns=[f"Grp {g}" for g in groups])
    return df_heat


def get_best_3rd_per_group(df_3rd: pd.DataFrame) -> dict[str, pd.Series]:
    """Returns the current leading (highest AdvanceProb) 3rd-place team per eligible group."""
    best = {}
    for grp in THIRD_PLACE_GROUPS:
        sub = df_3rd[df_3rd["Group"] == grp]
        if not sub.empty:
            best[grp] = sub.loc[sub["AdvanceProb"].idxmax()]
    return best


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_DARK = dict(
    paper_bgcolor="#0a0c14",
    plot_bgcolor="#0a0c14",
    font=dict(color="#94a3b8", family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
)

AXIS_STYLE = dict(
    gridcolor="#1e2235",
    zerolinecolor="#1e2235",
    tickfont=dict(color="#64748b", size=11),
    title_font=dict(color="#64748b"),
)


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_chaos_gauge(chaos_index: float) -> go.Figure:
    """Renders the Chaos Index as a Plotly gauge (speedometer style)."""
    label, _ = chaos_label(chaos_index)

    # Colour gradient: green → amber → red
    if chaos_index < 35:
        bar_color = "#4ade80"
    elif chaos_index < 70:
        bar_color = "#fbbf24"
    else:
        bar_color = "#f87171"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=chaos_index,
        number=dict(suffix="%", font=dict(size=38, color=bar_color, family="Inter")),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#1e2235",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0", "25", "50", "75", "100"],
                tickfont=dict(color="#64748b", size=10),
            ),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor="#0d1020",
            borderwidth=1,
            bordercolor="#1e2235",
            steps=[
                dict(range=[0,   35],  color="#0a2218"),
                dict(range=[35,  70],  color="#1a1408"),
                dict(range=[70,  100], color="#1a0808"),
            ],
            threshold=dict(
                line=dict(color="#ffffff", width=2),
                thickness=0.75,
                value=chaos_index,
            ),
        ),
        title=dict(
            text=f"<b>{label}</b>",
            font=dict(size=13, color=bar_color),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        height=280,
        margin=dict(l=30, r=30, t=30, b=5),
    )
    return fig


def build_heatmap(df_heat: pd.DataFrame) -> go.Figure:
    """
    Renders the joint-probability heatmap matrix.
    Rows = Group G teams; Columns = Eligible 3rd-place groups.
    """
    z     = df_heat.values * 100  # convert to percentages for readability
    teams = [f"{flag(t)} {t}" for t in df_heat.index]
    grps  = df_heat.columns.tolist()

    text_vals = [[f"{v:.1f}%" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=grps,
        y=teams,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=13, family="JetBrains Mono, monospace"),
        colorscale=[
            [0.0,  "#050d1a"],
            [0.15, "#0c2040"],
            [0.4,  "#1d4ed8"],
            [0.7,  "#0ea5e9"],
            [1.0,  "#38bdf8"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Joint Prob %", side="right", font=dict(color="#64748b", size=10)),
            tickfont=dict(color="#64748b", size=10),
            bgcolor="#0d1020",
            bordercolor="#1e2235",
            borderwidth=1,
            thickness=14,
            len=0.8,
        ),
        hovertemplate=(
            "<b>%{y}</b> wins Group G<br>"
            "<b>%{x}</b> 3rd place advances<br>"
            "Joint probability: <b>%{text}</b>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **PLOTLY_DARK,
        height=320,
        title=dict(
            text="Joint Probability Matrix — Who Meets in Seattle?",
            font=dict(size=13, color="#94a3b8"),
            x=0,
        ),
        xaxis=dict(**AXIS_STYLE, title="3rd-Place Qualifying Group"),
        yaxis=dict(**AXIS_STYLE, title="Group G Winner", autorange="reversed"),
        margin=dict(l=120, r=40, t=50, b=50),
    )
    return fig


def build_group_g_bar(df_g: pd.DataFrame) -> go.Figure:
    """Horizontal probability bar chart for Group G win probabilities."""
    df_sorted = df_g.sort_values("GroupWinnerProb", ascending=True)
    teams_labeled = [f"{flag(t)} {t}" for t in df_sorted["Team"]]

    colors = []
    for p in df_sorted["GroupWinnerProb"]:
        if p >= 0.50:
            colors.append("#38bdf8")
        elif p >= 0.25:
            colors.append("#6366f1")
        else:
            colors.append("#475569")

    fig = go.Figure(go.Bar(
        x=df_sorted["GroupWinnerProb"] * 100,
        y=teams_labeled,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{p*100:.1f}%" for p in df_sorted["GroupWinnerProb"]],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11, family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>Win probability: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_DARK,
        height=200,
        title=dict(text="Group G Winner Probability", font=dict(size=12, color="#94a3b8"), x=0),
        xaxis=dict(**AXIS_STYLE, title="", range=[0, 100], ticksuffix="%"),
        yaxis=dict(**AXIS_STYLE, title=""),
        showlegend=False,
        margin=dict(l=10, r=80, t=40, b=20),
    )
    return fig


def build_third_place_bar(df_3rd: pd.DataFrame) -> go.Figure:
    """
    Grouped bar showing 3rd-place advance probability by group + team,
    filtered to only the top-2 candidates per eligible group.
    """
    rows = []
    for grp in THIRD_PLACE_GROUPS:
        sub = df_3rd[df_3rd["Group"] == grp].nlargest(3, "AdvanceProb")
        for _, row in sub.iterrows():
            rows.append(row)
    df_plot = pd.DataFrame(rows)

    if df_plot.empty:
        return go.Figure()

    fig = go.Figure()
    palette = {"A": "#38bdf8", "E": "#818cf8", "H": "#34d399", "I": "#fb923c", "J": "#f472b6"}

    for grp in THIRD_PLACE_GROUPS:
        sub = df_plot[df_plot["Group"] == grp]
        if sub.empty:
            continue
        teams_lbl = [f"{flag(t)} {t}" for t in sub["Team"]]
        fig.add_trace(go.Bar(
            name=f"Grp {grp}",
            x=teams_lbl,
            y=sub["AdvanceProb"] * 100,
            marker_color=palette.get(grp, "#64748b"),
            text=[f"{p*100:.1f}%" for p in sub["AdvanceProb"]],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
            hovertemplate="<b>%{x}</b><br>Advance probability: <b>%{y:.1f}%</b><extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_DARK,
        height=260,
        title=dict(text="3rd-Place Advance Probability — Groups A/E/H/I/J", font=dict(size=12, color="#94a3b8"), x=0),
        xaxis=dict(**AXIS_STYLE, title="", tickangle=-25),
        yaxis=dict(**AXIS_STYLE, title="", ticksuffix="%", range=[0, max(df_plot["AdvanceProb"].max()*100 + 5, 20)]),
        legend=dict(
            font=dict(color="#64748b", size=10),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#1e2235",
        ),
        barmode="group",
        margin=dict(l=10, r=20, t=40, b=80),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROOTING INTEREST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_path_to_seattle(
    target_team: str,
    df_g: pd.DataFrame,
    df_3rd: pd.DataFrame,
) -> dict:
    """
    Analyse the current data and produce a structured 'Path to Seattle' recipe.

    Returns a dict with keys:
      is_group_g_team  : bool
      team             : str
      group            : str — "G" or the 3rd-place group
      steps            : list[str] — ordered natural-language instructions
      current_position : str — summary of where they stand
      probability      : float — estimated P(team appears in Match 82)
      verdict          : "YES" | "NO" | "POSSIBLE"
    """
    # Is the team in Group G?
    in_g = target_team in df_g["Team"].values
    in_3rd = target_team in df_3rd["Team"].values

    if not in_g and not in_3rd:
        return {
            "is_group_g_team": False,
            "team": target_team,
            "group": "N/A",
            "steps": [f"{target_team} is not in Group G or any of the eligible 3rd-place groups (A/E/H/I/J). They cannot appear in Match 82."],
            "current_position": "Not eligible",
            "probability": 0.0,
            "verdict": "NO",
        }

    steps = []
    probability = 0.0
    verdict = "POSSIBLE"

    if in_g:
        row = df_g[df_g["Team"] == target_team].iloc[0]
        win_prob = row["GroupWinnerProb"]
        ru_prob  = row["RunnerUpProb"]
        pts      = row["Pts"]
        gd       = row["GD"]
        mp       = row["MP"]

        current_pos = f"{flag(target_team)} {target_team} — {pts} pts, GD {gd:+d}, {mp} MP played"
        probability = win_prob

        steps.append(
            f"<b>{flag(target_team)} {target_team}</b> must <b>finish 1st in Group G</b> "
            f"(current win probability: {prob_to_pct(win_prob)})."
        )

        # Contextual advice based on current standing
        if win_prob >= 0.60:
            steps.append(
                "They are the <span class='verdict-yes'>heavy favorites</span> — a point in their "
                "final group match likely seals first place."
            )
        elif win_prob >= 0.30:
            steps.append(
                "The group is <span class='verdict-maybe'>still competitive</span>. "
                "A win in the next match would significantly lock in top spot."
            )
        else:
            steps.append(
                "They face an <span class='verdict-no'>uphill battle</span> — they likely need to "
                "win their remaining match(es) AND rely on results from other Group G games."
            )

        # Runner-up path? Not direct to Match 82 as group G runner-up goes vs Group D R/U
        steps.append(
            "Note: The Group G <i>runner-up</i> (prob "
            f"{prob_to_pct(ru_prob)}) plays against Group D's runner-up — not in Seattle. "
            "Only the Group G <b>winner</b> reaches Match 82."
        )

        # Opponent context — who is most likely waiting for them?
        best_3rd = get_best_3rd_per_group(df_3rd)
        most_likely_opponent = max(best_3rd.values(), key=lambda r: r["AdvanceProb"])
        steps.append(
            f"If they win Group G, their most probable Match 82 opponent is "
            f"<b>{flag(most_likely_opponent['Team'])} {most_likely_opponent['Team']}</b> "
            f"(Grp {most_likely_opponent['Group']}, 3rd-place advance prob "
            f"{prob_to_pct(most_likely_opponent['AdvanceProb'])})."
        )

        if win_prob >= 0.50:
            verdict = "YES"
        elif win_prob >= 0.20:
            verdict = "POSSIBLE"
        else:
            verdict = "NO"

    else:
        # Team is in a 3rd-place eligible group
        row = df_3rd[df_3rd["Team"] == target_team].iloc[0]
        adv_prob = row["AdvanceProb"]
        grp      = row["Group"]
        pts      = row["Pts"]
        gd       = row["GD"]
        mp       = row["MP"]

        current_pos = f"{flag(target_team)} {target_team} (Grp {grp}) — {pts} pts, GD {gd:+d}, {mp} MP played"
        probability = adv_prob

        # Which teams in the same group are ahead?
        same_grp = df_3rd[df_3rd["Group"] == grp].sort_values("AdvanceProb", ascending=False)
        rank_in_grp = same_grp["Team"].tolist().index(target_team) + 1

        steps.append(
            f"<b>{flag(target_team)} {target_team}</b> must <b>finish 3rd in Group {grp}</b> "
            f"(current 3rd-place advance probability: {prob_to_pct(adv_prob)})."
        )

        if rank_in_grp == 1:
            steps.append(
                f"They are currently the <span class='verdict-yes'>top-ranked 3rd-place candidate</span> "
                f"in Group {grp}. Hold this position."
            )
        elif rank_in_grp == 2:
            ahead_team = same_grp.iloc[0]
            steps.append(
                f"<b>{flag(ahead_team['Team'])} {ahead_team['Team']}</b> is ahead in Group {grp} — "
                f"{target_team} needs results to go their way, or to outperform them on "
                f"points / GD in remaining matches."
            )
        else:
            steps.append(
                f"<span class='verdict-no'>Significant ground to make up</span> — {target_team} "
                f"currently sits 3rd or lower among Group {grp} 3rd-place candidates."
            )

        steps.append(
            f"They must also rank in the <b>top-8 among all 12 third-place finishers</b> globally. "
            f"The tiebreaker order: (1) Points, (2) Goal Difference, (3) Goals Scored, "
            f"(4) Fair-play points."
        )

        # Competing 3rd-place groups — show which groups are hotly contested
        group_totals = {
            g: df_3rd[df_3rd["Group"] == g]["AdvanceProb"].sum()
            for g in THIRD_PLACE_GROUPS if g != grp
        }
        hardest_group = max(group_totals, key=group_totals.get)
        steps.append(
            f"Watch Group <b>{hardest_group}</b> closely — it has the most competitive "
            f"3rd-place battle and could squeeze {target_team} out of the top-8 globally."
        )

        # Group G opponent context
        winner_row = df_g.loc[df_g["GroupWinnerProb"].idxmax()]
        steps.append(
            f"If they reach Match 82, their most probable opponent is "
            f"<b>{flag(winner_row['Team'])} {winner_row['Team']}</b> "
            f"(Group G win prob: {prob_to_pct(winner_row['GroupWinnerProb'])})."
        )

        if adv_prob >= 0.50:
            verdict = "YES"
        elif adv_prob >= 0.15:
            verdict = "POSSIBLE"
        else:
            verdict = "NO"

    return {
        "is_group_g_team": in_g,
        "team": target_team,
        "group": "G" if in_g else row["Group"],
        "steps": steps,
        "current_position": current_pos,
        "probability": probability,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(df_g: pd.DataFrame, df_3rd: pd.DataFrame, is_live: bool) -> str:
    """Renders sidebar controls, returns selected target team."""
    with st.sidebar:
        st.markdown("## ⚽ Match 82 Tracker")
        st.markdown(
            '<p style="color:#475569;font-size:0.78rem;margin-top:-0.5rem;">Seattle · Lumen Field · July 1, 2026</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Live mode toggle
        live_mode = st.toggle("🔴 Live Mode", value=False,
                               help=f"Auto-refresh data from Google Sheet every {REFRESH_SECONDS}s")
        if live_mode and is_live:
            st.success(f"Connected · refreshes every {REFRESH_SECONDS}s")
        elif live_mode and not is_live:
            st.warning("No sheet URL — set MATCH82_SHEET_URL or paste below")

        sheet_input = st.text_input(
            "Google Sheet CSV URL",
            value=GOOGLE_SHEET_URL,
            placeholder="https://docs.google.com/spreadsheets/d/.../export?format=csv",
            help="Paste your public Google Sheet CSV export URL here.",
        )
        if sheet_input and sheet_input != GOOGLE_SHEET_URL:
            os.environ["MATCH82_SHEET_URL"] = sheet_input
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("### 🗺️ Rooting Interest")
        st.caption("Select a team to see their path to Lumen Field")

        # Build full eligible team list
        all_g_teams = df_g["Team"].tolist()
        all_3rd_teams = df_3rd["Team"].tolist()
        all_eligible = sorted(set(all_g_teams + all_3rd_teams))
        options_display = [f"{flag(t)} {t}" for t in all_eligible]

        default_idx = 0
        if "Belgium" in all_eligible:
            default_idx = all_eligible.index("Belgium")

        selected_display = st.selectbox(
            "Target Team",
            options=options_display,
            index=default_idx,
            label_visibility="collapsed",
        )
        # Strip flag emoji to get clean name
        selected_team = selected_display.split(" ", 1)[-1] if " " in selected_display else selected_display

        st.divider()
        st.markdown("### ℹ️ Format")
        st.caption(
            "Match 82 (R32) pits the **Group G winner** vs. "
            "the best 3rd-place team from **Groups A, E, H, I, or J**."
        )
        st.caption(
            "8 of 12 3rd-place teams advance. Tiebreaker: Points → GD → GF → Fair-play."
        )
        st.caption("Winner of Match 82 faces W81 in Match 94 on July 6.")

        st.divider()
        st.markdown(
            '<p style="color:#1e3a5f;font-size:0.72rem;">Data: Google Sheets / Prediction Markets · '
            'Built for WC2026 Seattle</p>',
            unsafe_allow_html=True,
        )

    return selected_team


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load data
    df_g, df_3rd, is_live = load_data(GOOGLE_SHEET_URL)

    # Sidebar (returns selected team)
    selected_team = render_sidebar(df_g, df_3rd, is_live)

    # ── HEADER ────────────────────────────────────────────────────────────────
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        st.markdown(
            "# ⚽ Match 82 — Seattle Lumen Field",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#475569;font-size:0.85rem;margin-top:-0.5rem;">'
            'Round of 32 · <b style="color:#64748b">Group G Winner vs. 3rd Place A/E/H/I/J</b> · '
            'Wed July 1, 2026 · 1:00 PM PT'
            '</p>',
            unsafe_allow_html=True,
        )

    with col_badge:
        data_src = "🟢 LIVE DATA" if is_live else "🟡 MOCK DATA"
        src_color = "#052e16" if is_live else "#2d1a00"
        src_border = "#166534" if is_live else "#92400e"
        src_text = "#4ade80" if is_live else "#fbbf24"
        st.markdown(
            f'<div style="background:{src_color};border:1px solid {src_border};border-radius:8px;'
            f'padding:0.6rem 1rem;text-align:center;margin-top:0.8rem;">'
            f'<span style="color:{src_text};font-size:0.75rem;font-weight:700;">{data_src}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── TOP METRICS ROW ───────────────────────────────────────────────────────
    chaos_val = compute_chaos_index(df_g, df_3rd)
    c_label, _ = chaos_label(chaos_val)

    # Find current leader / favourite
    g_leader = df_g.loc[df_g["GroupWinnerProb"].idxmax()]
    best_3rd_dict = get_best_3rd_per_group(df_3rd)
    best_3rd_overall = max(best_3rd_dict.values(), key=lambda r: r["AdvanceProb"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "G Winner Favourite",
            f"{flag(g_leader['Team'])} {g_leader['Team']}",
            f"{prob_to_pct(g_leader['GroupWinnerProb'])}",
        )
    with col2:
        # Dominant 3rd-place candidate overall
        st.metric(
            "Top 3rd-Place Candidate",
            f"{flag(best_3rd_overall['Team'])} {best_3rd_overall['Team']}",
            f"{prob_to_pct(best_3rd_overall['AdvanceProb'])} (Grp {best_3rd_overall['Group']})",
        )
    with col3:
        # Joint P of most likely exact matchup
        top_jp = build_heatmap_data(df_g, df_3rd).values.max()
        st.metric("Highest Joint Prob", f"{top_jp*100:.1f}%", "Most likely exact matchup")
    with col4:
        st.metric("Chaos Index", f"{chaos_val}%", c_label)

    st.markdown("---")

    # ── ROOTING INTEREST SECTION ───────────────────────────────────────────────
    st.markdown("## 🗺️ Path to Seattle")

    recipe = generate_path_to_seattle(selected_team, df_g, df_3rd)

    verdict = recipe["verdict"]
    v_color = "#4ade80" if verdict == "YES" else ("#f87171" if verdict == "NO" else "#fbbf24")
    v_bg    = "#052e16" if verdict == "YES" else ("#1a0808" if verdict == "NO" else "#1a1008")
    v_border= "#166534" if verdict == "YES" else ("#7f1d1d" if verdict == "NO" else "#92400e")

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.8rem;">'
        f'<span style="font-size:1.3rem;font-weight:800;color:#f1f5f9;">'
        f'{flag(selected_team)} {selected_team}</span>'
        f'<span style="background:{v_bg};border:1px solid {v_border};border-radius:6px;'
        f'padding:3px 12px;font-size:0.78rem;font-weight:800;color:{v_color};">{verdict}</span>'
        f'<span style="color:#475569;font-size:0.82rem;">{recipe["current_position"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    steps_html = "".join([
        f'<div class="recipe-step">'
        f'<div class="step-num">{i+1}</div>'
        f'<div class="step-text">{step}</div>'
        f'</div>'
        for i, step in enumerate(recipe["steps"])
    ])

    prob_pct = recipe["probability"]
    prob_bar_w = int(prob_pct * 100)
    prob_bar_color = "#4ade80" if prob_pct >= 0.50 else ("#fbbf24" if prob_pct >= 0.20 else "#f87171")

    st.markdown(
        f'<div class="recipe-card">'
        f'{steps_html}'
        f'<div style="margin-top:1rem;padding-top:0.75rem;border-top:1px solid #1e2a44;">'
        f'<span style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;">'
        f'Estimated probability of appearing in Match 82</span><br>'
        f'<div style="background:#0d1020;border-radius:4px;height:8px;margin-top:0.4rem;overflow:hidden;">'
        f'<div style="background:{prob_bar_color};width:{prob_bar_w}%;height:100%;border-radius:4px;'
        f'transition:width 0.6s ease;"></div>'
        f'</div>'
        f'<span style="color:{prob_bar_color};font-size:0.9rem;font-weight:700;font-family:monospace;">'
        f'{prob_pct*100:.1f}%</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── HEATMAP + CHAOS GAUGE ──────────────────────────────────────────────────
    st.markdown("## 📊 Probability Analytics")

    col_heat, col_chaos = st.columns([2.2, 1])

    with col_heat:
        df_heat = build_heatmap_data(df_g, df_3rd)
        fig_heat = build_heatmap(df_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_chaos:
        st.markdown("#### Chaos Index")
        st.caption(
            "Measures volatility of the Match 82 slot. "
            "High = many teams neck-and-neck. Low = matchup nearly locked."
        )
        fig_gauge = build_chaos_gauge(chaos_val)
        st.plotly_chart(fig_gauge, use_container_width=True)

        chaos_class = "chaos-low" if chaos_val < 35 else ("chaos-medium" if chaos_val < 70 else "chaos-high")
        st.markdown(
            f'<div style="text-align:center;margin-top:-0.5rem;">'
            f'<span class="chaos-badge {chaos_class}">{c_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Quick explanation
        with st.expander("How is this calculated?"):
            st.markdown(
                """
                **Chaos Index** uses Shannon entropy normalised to [0, 100]:

                - **Group G entropy**: How spread out are the Group G win probabilities?
                - **3rd-place entropy**: How spread out are the advance probabilities across all 20 eligible 3rd-place teams?

                The two components are weighted 50/50.

                **100%** = All teams have equal odds (maximum uncertainty).
                **0%** = One team is a mathematical lock (zero uncertainty).
                """
            )

    st.markdown("---")

    # ── PROBABILITY DISTRIBUTION CHARTS ────────────────────────────────────────
    st.markdown("## 📈 Group Standings & Probabilities")

    col_bar1, col_bar2 = st.columns(2)
    with col_bar1:
        fig_g_bar = build_group_g_bar(df_g)
        st.plotly_chart(fig_g_bar, use_container_width=True)
    with col_bar2:
        fig_3rd_bar = build_third_place_bar(df_3rd)
        st.plotly_chart(fig_3rd_bar, use_container_width=True)

    # ── GROUP G STANDINGS TABLE ─────────────────────────────────────────────────
    st.markdown("### Group G Standings")

    df_g_display = df_g.copy()
    df_g_display.insert(0, "Flag", df_g_display["Team"].map(flag))
    df_g_display["GroupWin%"]   = (df_g_display["GroupWinnerProb"] * 100).round(1)
    df_g_display["RunnerUp%"]   = (df_g_display["RunnerUpProb"]    * 100).round(1)
    df_g_display["3rdPlace%"]   = (df_g_display["ThirdPlaceProb"]  * 100).round(1)

    show_cols = ["Flag","Team","MP","W","D","L","GF","GA","GD","Pts","GroupWin%","RunnerUp%","3rdPlace%"]
    df_g_display = df_g_display[show_cols].sort_values("Pts", ascending=False).reset_index(drop=True)
    df_g_display.index += 1

    st.dataframe(
        df_g_display,
        use_container_width=True,
        hide_index=False,
        column_config={
            "Flag": st.column_config.TextColumn("", width="small"),
            "Team": st.column_config.TextColumn("Team"),
            "GroupWin%": st.column_config.ProgressColumn("Win Prob %", min_value=0, max_value=100, format="%.1f%%"),
            "RunnerUp%": st.column_config.ProgressColumn("Runner-Up %", min_value=0, max_value=100, format="%.1f%%"),
        }
    )

    # ── ELIGIBLE 3RD-PLACE STANDINGS ────────────────────────────────────────────
    st.markdown("### 3rd-Place Eligible Groups (A / E / H / I / J)")
    st.caption("Showing current 3rd-place teams in each group with their global advance probability.")

    tabs = st.tabs([f"Group {g}" for g in THIRD_PLACE_GROUPS])
    for tab, grp in zip(tabs, THIRD_PLACE_GROUPS):
        with tab:
            sub = df_3rd[df_3rd["Group"] == grp].copy()
            sub.insert(0, "Flag", sub["Team"].map(flag))
            sub["Advance%"] = (sub["AdvanceProb"] * 100).round(1)
            show = ["Flag","Team","MP","W","D","L","GF","GA","GD","Pts","Advance%"]
            sub = sub[show].sort_values("Pts", ascending=False).reset_index(drop=True)
            sub.index += 1
            st.dataframe(
                sub,
                use_container_width=True,
                hide_index=False,
                column_config={
                    "Flag": st.column_config.TextColumn("", width="small"),
                    "Advance%": st.column_config.ProgressColumn("Advance Prob %", min_value=0, max_value=100, format="%.1f%%"),
                }
            )

    # ── FOOTER ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="color:#1e3a5f;font-size:0.75rem;text-align:center;">'
        '2026 FIFA World Cup · Match 82 · Seattle Lumen Field · July 1, 2026 · '
        'Probabilities sourced from prediction market contract prices (update via Google Sheet) · '
        'Not affiliated with FIFA'
        '</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
