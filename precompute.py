"""
precompute.py — Nightly Monte Carlo pre-computation script.

Run by GitHub Actions each morning at 7 AM PT (14:00 UTC).
Writes results.json to the repo root so the Streamlit app can
load it instantly instead of running the simulation on page load.

Usage:
    python precompute.py              # 50k sims (default)
    python precompute.py --sims 100000

The Streamlit app checks for results.json at startup:
  - Found + fresh (< 25 hours old): loads instantly, shows "Pre-computed" badge
  - Found + stale: loads it but shows a warning + offers live re-run
  - Not found: falls back to live simulation (original behavior)
"""

import argparse
import json
import os
import sys
import time
import datetime

# ── Make sure we can import from app.py in the same directory ──────────────
sys.path.insert(0, os.path.dirname(__file__))

# We need to stub out streamlit before importing app.py, because app.py
# calls st.set_page_config() and st.markdown() at module level.
# The stub intercepts those calls silently so the engine functions
# (run_monte_carlo, etc.) are importable in a headless environment.
import types

_st_stub = types.ModuleType("streamlit")

class _CacheStub:
    """Minimal stub for @st.cache_data so decorated functions still work."""
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, fn):
        return fn

_st_stub.cache_data      = _CacheStub
_st_stub.set_page_config = lambda **kw: None
_st_stub.markdown        = lambda *a, **kw: None
_st_stub.sidebar         = types.SimpleNamespace(
    markdown=lambda *a, **kw: None,
    slider=lambda *a, **kw: None,
    checkbox=lambda *a, **kw: None,
    selectbox=lambda *a, **kw: None,
    button=lambda *a, **kw: None,
    expander=lambda *a, **kw: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: None),
)
_st_stub.session_state   = {}
_st_stub.secrets         = {}  # Prevent AttributeError; API keys come from os.environ in Actions
_st_stub.spinner         = lambda *a, **kw: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: None)
sys.modules["streamlit"] = _st_stub

from app import run_monte_carlo, compute_chaos_index  # noqa: E402  (after stub)


OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "results.json")


def serialize_mc(mc: dict) -> dict:
    """
    Convert the MC result dict to a JSON-serializable form.
    Tuple keys (team_a, team_b) in match82_joint_prob become "TeamA vs TeamB" strings.
    """
    joint = {
        f"{k[0]} vs {k[1]}": v
        for k, v in mc["match82_joint_prob"].items()
    }
    return {
        "g_winner_prob":      mc["g_winner_prob"],
        "g_runnerup_prob":    mc["g_runnerup_prob"],
        "third_advance_prob": mc["third_advance_prob"],
        "match82_joint_prob": joint,
        "methods":            mc["methods"],
        "n_sims":             mc["n_sims"],
        "timestamp":          mc["timestamp"],
        "computed_at":        datetime.datetime.utcnow().isoformat() + "Z",
    }


def deserialize_mc(data: dict) -> dict:
    """
    Restore the MC result dict from JSON, converting string keys back to tuples.
    """
    joint = {}
    for key_str, v in data.get("match82_joint_prob", {}).items():
        parts = key_str.split(" vs ", 1)
        if len(parts) == 2:
            joint[tuple(parts)] = v
        else:
            joint[(key_str, "TBD")] = v
    data["match82_joint_prob"] = joint
    return data


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Match 82 Monte Carlo probabilities.")
    parser.add_argument("--sims", type=int, default=50_000,
                        help="Number of Monte Carlo simulations (default: 50000)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output JSON file path")
    args = parser.parse_args()

    print(f"[precompute] Starting {args.sims:,}-trial Monte Carlo simulation...")
    t0 = time.time()

    mc = run_monte_carlo(n_sims=args.sims, use_markets=False)

    elapsed = time.time() - t0
    print(f"[precompute] Simulation complete in {elapsed:.1f}s")

    # Print a quick summary
    g_probs = mc["g_winner_prob"]
    g_leader = max(g_probs, key=g_probs.get)
    chaos = compute_chaos_index(mc)
    top_pair = max(mc["match82_joint_prob"], key=mc["match82_joint_prob"].get) \
               if mc["match82_joint_prob"] else ("TBD", "TBD")
    top_pair_p = mc["match82_joint_prob"].get(top_pair, 0)

    print(f"[precompute] Group G leader:  {g_leader} ({g_probs[g_leader]*100:.1f}%)")
    print(f"[precompute] Top matchup:     {top_pair[0]} vs {top_pair[1]} ({top_pair_p*100:.2f}%)")
    print(f"[precompute] Chaos Index:     {chaos}%")
    print(f"[precompute] Writing → {args.output}")

    serialized = serialize_mc(mc)
    with open(args.output, "w") as f:
        json.dump(serialized, f, indent=2)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"[precompute] Done. File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
