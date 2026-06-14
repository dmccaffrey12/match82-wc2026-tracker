# ⚽ Match 82 — FIFA World Cup 2026 Seattle Tracker

**Lumen Field · Seattle, WA · Wednesday July 1, 2026 · 1:00 PM PT**

Round of 32: **Group G Winner** vs. **3rd Place from Group A, E, H, I, or J**

A dark-themed Streamlit analytics dashboard tracking live probabilities for the Seattle World Cup slot.

---

## Features

| Feature | Description |
|---|---|
| **Path to Seattle Generator** | Select any eligible team → get a step-by-step natural-language "rooting recipe" showing exactly what has to happen for them to appear in Match 82 |
| **Imminent Matchup Heatmap** | Plotly 2D matrix of joint probabilities — Group G winners × 3rd-place qualifying groups, color-coded by likelihood |
| **Chaos Index Gauge** | 0–100% entropy-based gauge showing how settled or volatile the Seattle slot is right now |
| **Group Standings Tables** | Tabbed, live-updating standings for Group G + all 5 eligible 3rd-place groups (A/E/H/I/J) |
| **Probability Bar Charts** | Visual distribution of Group G win odds + 3rd-place advance odds by team |
| **Live Google Sheet Mode** | Connect a public Google Sheet for real-time updates; falls back to embedded mock data gracefully |

---

## Quickstart (Local)

```bash
# 1. Clone / copy files
cd match82_tracker

# 2. Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py

# 4. (Optional) Connect a live Google Sheet — see below
export MATCH82_SHEET_URL="https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv"
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Deploying to Streamlit Community Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo → branch → `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   MATCH82_SHEET_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv"
   ```
5. Click **Deploy** — takes ~2 minutes

---

## Connecting a Live Google Sheet

### Sheet Structure

The app expects **two separate published CSV tabs** from your Google Sheet.

#### Tab 1: `group_g` (Group G standings + market probabilities)

| Column | Type | Description |
|---|---|---|
| `Team` | str | Team name (must match exactly: Belgium, Egypt, Iran, New Zealand) |
| `MP` | int | Matches played |
| `W` | int | Wins |
| `D` | int | Draws |
| `L` | int | Losses |
| `GF` | int | Goals for |
| `GA` | int | Goals against |
| `GD` | int | Goal difference |
| `Pts` | int | Points |
| `GroupWinnerProb` | float | 0.0–1.0 — probability of winning Group G (from prediction market) |
| `RunnerUpProb` | float | 0.0–1.0 — probability of finishing runner-up |
| `ThirdPlaceProb` | float | 0.0–1.0 — probability of finishing 3rd in group |

#### Tab 2: `third_place` (All 3rd-place candidates in Groups A/E/H/I/J)

| Column | Type | Description |
|---|---|---|
| `Group` | str | One of: A, E, H, I, J |
| `Team` | str | Team name |
| `MP` | int | Matches played |
| `W, D, L` | int | Wins, draws, losses |
| `GF, GA, GD` | int | Goals for/against/difference |
| `Pts` | int | Points |
| `AdvanceProb` | float | 0.0–1.0 — probability this team ends as 3rd in their group **and** ranks in the top-8 globally |

### Publishing the Sheet as CSV

1. Open your Google Sheet
2. **File → Share → Publish to web**
3. Select the tab (e.g., `group_g`) → choose **Comma-separated values (.csv)**
4. Click **Publish** and copy the URL
5. Repeat for the `third_place` tab
6. Pass **both URLs as a comma-separated pair**:

```bash
export MATCH82_SHEET_URL="https://docs.google.com/spreadsheets/d/SHEET_ID/pub?gid=0&single=true&output=csv,https://docs.google.com/spreadsheets/d/SHEET_ID/pub?gid=123456789&single=true&output=csv"
```

Or paste into the **Google Sheet CSV URL** text input in the sidebar at runtime.

### Updating the Probabilities

The `GroupWinnerProb`, `RunnerUpProb`, and `AdvanceProb` columns should be updated with implied probabilities from prediction markets:

- **[Polymarket](https://polymarket.com)** — search for "2026 World Cup Group G winner"
- **[Kalshi](https://kalshi.com)** — similar contracts
- **[Metaculus](https://metaculus.com)** — community forecasts (no real money)

**Converting contract prices to probabilities:**
- If a Polymarket contract for "Belgium wins Group G" is trading at **$0.62**, enter `0.62` in `GroupWinnerProb`
- Probabilities within each team's row do **not** need to sum to 1.0 (the app normalises internally for display)

---

## Probability Model Notes

### Chaos Index

Uses Shannon entropy over the probability distributions:

```
H = -Σ p_i × log₂(p_i)   (normalised to [0, 1] by dividing by log₂(n))
Chaos = 100 × (0.5 × H_GroupG_winner + 0.5 × H_3rd_place_all_teams)
```

- **0%** = mathematically locked (one team guaranteed)
- **35-70%** = in flux, multiple realistic outcomes
- **>70%** = total chaos, no clear favourite

### Joint Probability Heatmap

Each cell = `P(Team X wins Group G) × P(A 3rd-place team from Group Y qualifies)`

The group-level 3rd-place probability is the **sum** of all individual team AdvanceProbs within that group (since exactly one team per group can be the 3rd-place representative).

### Path to Seattle

- **Group G teams**: Path requires finishing **1st** in Group G (runner-up goes to a different bracket position)
- **3rd-place teams**: Path requires (a) finishing 3rd in their group AND (b) ranking in the top-8 globally among all 12 third-place teams

---

## Architecture

```
app.py                  ← Single-file Streamlit app (pure Python)
requirements.txt        ← Dependencies (streamlit, pandas, numpy, plotly)
README.md               ← This file

Key sections in app.py:
  CONFIGURATION         ← Sheet URL, team lists, refresh interval
  DATA LAYER            ← _mock_*() functions + load_data() with @st.cache_data
  HELPER FUNCTIONS      ← Chaos index, heatmap builder, probability utils
  PLOTLY CHART BUILDERS ← Gauge, heatmap, bar charts
  ROOTING INTEREST      ← generate_path_to_seattle() engine
  SIDEBAR               ← render_sidebar() with live mode toggle
  MAIN RENDER           ← main() — layout, metrics, all sections
```

---

## Customisation

| What to change | Where |
|---|---|
| Refresh interval | `REFRESH_SECONDS` constant at top of `app.py` |
| Add a team flag emoji | `FLAG_MAP` dict |
| Change which groups are eligible | `THIRD_PLACE_GROUPS` list |
| Adjust Chaos Index weights | `compute_chaos_index()` function |
| Change chart color palette | `PLOTLY_DARK` dict + chart builder functions |
| Override group stage teams | `ELIGIBLE_3RD_TEAMS` dict |

---

## Match Context

- **Match 82** — Round of 32, July 1 2026, Lumen Field, Seattle
- **Group G teams**: Belgium, Egypt, Iran, New Zealand
- **Eligible 3rd-place groups**: A (Mexico, South Africa, South Korea, Czechia), E (Germany, Curaçao, Côte d'Ivoire, Ecuador), H (Spain, Cabo Verde, Saudi Arabia, Uruguay), I (France, Senegal, Iraq, Norway), J (Argentina, Algeria, Austria, Jordan)
- **Winner of Match 82** plays in Match 94 (Round of 16) at Lumen Field on July 6
- **3rd-place advancement tiebreaker**: Points → Goal Difference → Goals Scored → Fair-play points → Drawing of lots
