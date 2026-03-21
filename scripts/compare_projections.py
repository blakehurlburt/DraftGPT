"""
Compare our model's 2026 NFL projections against Sleeper's projections.

Fetches Sleeper projections via the API, matches players, and produces
a comprehensive analysis highlighting the largest disagreements and
potential reasons for divergence.

Usage:
    python -m scripts.compare_projections
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftsim.players import load_players
from draftassist.bridge import build_player_index
from draftassist.scoring import default_ppr_scoring, sleeper_stats_to_fantasy_points
from draftassist.sleeper import fetch_all_players, fetch_projections


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

async def _fetch_sleeper_data() -> tuple[dict, dict]:
    """Fetch Sleeper player database and 2026 projections."""
    async with httpx.AsyncClient(timeout=30) as client:
        players = await fetch_all_players(client)
        projections = await fetch_projections(client, season=2026)
    return players, projections


def load_comparison_data() -> pd.DataFrame:
    """Load our model and Sleeper projections into a single DataFrame."""
    players = load_players(sport="nfl", min_total=0.1)
    sleeper_players, sleeper_proj = asyncio.run(_fetch_sleeper_data())
    build_player_index(players, sleeper_players)

    scoring = default_ppr_scoring()

    rows = []
    for p in players:
        if not p.sleeper_id:
            continue
        stats = sleeper_proj.get(p.sleeper_id)
        if not stats:
            continue

        sleeper_pts = sleeper_stats_to_fantasy_points(stats, scoring)
        sleeper_ppr = float(stats.get("pts_ppr", 0) or 0)

        # Break down Sleeper stat components for analysis
        row = {
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "pos_rank": p.pos_rank,
            # Our model
            "model_total": p.projected_total,
            "model_ppg": p.projected_ppg,
            "model_games": p.projected_games,
            "model_floor": p.total_floor,
            "model_ceiling": p.total_ceiling,
            # Sleeper
            "sleeper_total": sleeper_pts,
            "sleeper_ppr_native": sleeper_ppr,  # Sleeper's own PPR calculation
            "sleeper_games": float(stats.get("gp", 0) or 0),
            # Sleeper raw stats for diagnosis
            "slp_pass_yd": float(stats.get("pass_yd", 0) or 0),
            "slp_pass_td": float(stats.get("pass_td", 0) or 0),
            "slp_pass_int": float(stats.get("pass_int", 0) or 0),
            "slp_rush_yd": float(stats.get("rush_yd", 0) or 0),
            "slp_rush_td": float(stats.get("rush_td", 0) or 0),
            "slp_rec": float(stats.get("rec", 0) or 0),
            "slp_rec_yd": float(stats.get("rec_yd", 0) or 0),
            "slp_rec_td": float(stats.get("rec_td", 0) or 0),
            "slp_fum_lost": float(stats.get("fum_lost", 0) or 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["diff"] = df["model_total"] - df["sleeper_total"]
    df["diff_pct"] = (df["diff"] / df["sleeper_total"].replace(0, float("nan"))) * 100
    df["abs_diff"] = df["diff"].abs()

    # Sleeper PPG (using their GP of 18, or fall back)
    df["sleeper_ppg"] = df.apply(
        lambda r: r["sleeper_total"] / r["sleeper_games"]
        if r["sleeper_games"] > 0
        else 0,
        axis=1,
    )
    df["ppg_diff"] = df["model_ppg"] - df["sleeper_ppg"]

    # Games difference
    df["games_diff"] = df["model_games"] - df["sleeper_games"]

    # Points attributable to games difference vs PPG difference
    df["pts_from_games_diff"] = df["ppg_diff"].clip(lower=0) * 0 + df["games_diff"] * df["sleeper_ppg"]
    df["pts_from_ppg_diff"] = df["ppg_diff"] * df["model_games"]

    return df.sort_values("abs_diff", ascending=False)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


def analyze(df: pd.DataFrame) -> None:
    """Run the full comparison analysis and print results."""

    # ── 1. Overview ──────────────────────────────────────────────────────
    print_section("OVERVIEW: Model vs Sleeper Projections")

    matched = len(df)
    print(f"Players matched:  {matched}")
    print(f"Mean model total: {df['model_total'].mean():.1f}")
    print(f"Mean sleeper total: {df['sleeper_total'].mean():.1f}")
    print(f"Mean difference (model - sleeper): {df['diff'].mean():.1f}")
    print(f"Median difference: {df['diff'].median():.1f}")
    print(f"Std of differences: {df['diff'].std():.1f}")
    print(f"Correlation (totals): {df['model_total'].corr(df['sleeper_total']):.4f}")

    for pos in ["QB", "RB", "WR", "TE"]:
        sub = df[df["position"] == pos]
        if sub.empty:
            continue
        print(f"\n  {pos} ({len(sub)} players):")
        print(f"    Mean diff: {sub['diff'].mean():+.1f}  |  Median: {sub['diff'].median():+.1f}  |  Corr: {sub['model_total'].corr(sub['sleeper_total']):.3f}")
        print(f"    Model avg: {sub['model_total'].mean():.1f}  |  Sleeper avg: {sub['sleeper_total'].mean():.1f}")
        print(f"    Model avg PPG: {sub['model_ppg'].mean():.2f}  |  Sleeper avg PPG: {sub['sleeper_ppg'].mean():.2f}")
        print(f"    Model avg GP: {sub['model_games'].mean():.1f}  |  Sleeper avg GP: {sub['sleeper_games'].mean():.1f}")

    # ── 2. Games played analysis ─────────────────────────────────────────
    print_section("GAMES PLAYED: Systematic Differences")

    for pos in ["QB", "RB", "WR", "TE"]:
        sub = df[df["position"] == pos]
        if sub.empty:
            continue
        print(f"  {pos}: Model avg GP = {sub['model_games'].mean():.1f}, Sleeper avg GP = {sub['sleeper_games'].mean():.1f}, diff = {sub['games_diff'].mean():+.1f}")

    big_gp_diff = df[df["games_diff"].abs() > 3].sort_values("games_diff")
    if not big_gp_diff.empty:
        print(f"\n  Players with games diff > 3:")
        for _, r in big_gp_diff.head(20).iterrows():
            print(f"    {r['name']:25s} {r['position']:3s}  model={r['model_games']:.1f}  sleeper={r['sleeper_games']:.0f}  diff={r['games_diff']:+.1f}")

    # ── 3. Largest positive disagreements (we're higher) ─────────────────
    print_section("TOP 30: Model HIGHER Than Sleeper (we're more bullish)")

    higher = df[df["diff"] > 0].nlargest(30, "diff")
    for _, r in higher.iterrows():
        flag = ""
        if abs(r["games_diff"]) > 2:
            flag = f" [GP diff: {r['games_diff']:+.1f}]"
        print(
            f"  {r['name']:25s} {r['position']:3s} (#{r['pos_rank']:<3.0f})  "
            f"Model: {r['model_total']:6.1f}  Sleeper: {r['sleeper_total']:6.1f}  "
            f"Diff: {r['diff']:+7.1f} ({r['diff_pct']:+5.1f}%){flag}"
        )

    # ── 4. Largest negative disagreements (we're lower) ──────────────────
    print_section("TOP 30: Model LOWER Than Sleeper (we're more bearish)")

    lower = df[df["diff"] < 0].nsmallest(30, "diff")
    for _, r in lower.iterrows():
        flag = ""
        if abs(r["games_diff"]) > 2:
            flag = f" [GP diff: {r['games_diff']:+.1f}]"
        print(
            f"  {r['name']:25s} {r['position']:3s} (#{r['pos_rank']:<3.0f})  "
            f"Model: {r['model_total']:6.1f}  Sleeper: {r['sleeper_total']:6.1f}  "
            f"Diff: {r['diff']:+7.1f} ({r['diff_pct']:+5.1f}%){flag}"
        )

    # ── 5. PPG vs Games decomposition ────────────────────────────────────
    print_section("PPG vs GAMES DECOMPOSITION (Top 20 abs diff)")

    top = df.nlargest(20, "abs_diff")
    print(f"  {'Name':25s} {'Pos':3s}  {'Diff':>7s}  {'PPG Diff':>8s}  {'GP Diff':>7s}  {'Pts from PPG':>12s}  {'Pts from GP':>11s}")
    print(f"  {'-'*25} {'-'*3}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*12}  {'-'*11}")
    for _, r in top.iterrows():
        print(
            f"  {r['name']:25s} {r['position']:3s}  {r['diff']:+7.1f}  "
            f"{r['ppg_diff']:+8.2f}  {r['games_diff']:+7.1f}  "
            f"{r['pts_from_ppg_diff']:+12.1f}  {r['pts_from_games_diff']:+11.1f}"
        )

    # ── 6. Stat-level breakdown for top disagreements ────────────────────
    print_section("STAT-LEVEL BREAKDOWN: Top 10 Absolute Disagreements")

    top10 = df.nlargest(10, "abs_diff")
    for _, r in top10.iterrows():
        print(f"\n  {r['name']} ({r['position']}, {r['team']})")
        print(f"    Model: {r['model_total']:.1f} total ({r['model_ppg']:.2f} PPG x {r['model_games']:.1f} GP)  |  Floor: {r['model_floor']:.1f}  Ceiling: {r['model_ceiling']:.1f}")
        print(f"    Sleeper: {r['sleeper_total']:.1f} total ({r['sleeper_ppg']:.2f} PPG x {r['sleeper_games']:.0f} GP)")
        print(f"    Diff: {r['diff']:+.1f} ({r['diff_pct']:+.1f}%)")
        if r["position"] == "QB":
            print(f"    Sleeper stats: {r['slp_pass_yd']:.0f} pass yd, {r['slp_pass_td']:.0f} pass TD, {r['slp_pass_int']:.0f} INT, {r['slp_rush_yd']:.0f} rush yd, {r['slp_rush_td']:.0f} rush TD")
        elif r["position"] == "RB":
            print(f"    Sleeper stats: {r['slp_rush_yd']:.0f} rush yd, {r['slp_rush_td']:.0f} rush TD, {r['slp_rec']:.0f} rec, {r['slp_rec_yd']:.0f} rec yd, {r['slp_rec_td']:.0f} rec TD, {r['slp_fum_lost']:.0f} fum")
        elif r["position"] in ("WR", "TE"):
            print(f"    Sleeper stats: {r['slp_rec']:.0f} rec, {r['slp_rec_yd']:.0f} rec yd, {r['slp_rec_td']:.0f} rec TD, {r['slp_rush_yd']:.0f} rush yd")

    # ── 7. Positional bias analysis ──────────────────────────────────────
    print_section("POSITIONAL BIAS")

    for pos in ["QB", "RB", "WR", "TE"]:
        sub = df[df["position"] == pos].copy()
        if len(sub) < 5:
            continue
        # Tier analysis
        tiers = [
            ("Elite (top 5)", sub.nsmallest(5, "pos_rank")),
            ("Starters (6-12)", sub[(sub["pos_rank"] >= 6) & (sub["pos_rank"] <= 12)]),
            ("Mid (13-24)", sub[(sub["pos_rank"] >= 13) & (sub["pos_rank"] <= 24)]),
            ("Deep (25+)", sub[sub["pos_rank"] >= 25]),
        ]
        print(f"\n  {pos}:")
        for label, tier in tiers:
            if tier.empty:
                continue
            print(f"    {label:20s}  N={len(tier):3d}  Mean diff: {tier['diff'].mean():+6.1f}  PPG diff: {tier['ppg_diff'].mean():+5.2f}  GP diff: {tier['games_diff'].mean():+4.1f}")

    # ── 8. Floor/ceiling vs Sleeper ──────────────────────────────────────
    print_section("FLOOR/CEILING ANALYSIS")

    in_range = df[
        (df["sleeper_total"] >= df["model_floor"])
        & (df["sleeper_total"] <= df["model_ceiling"])
    ]
    above = df[df["sleeper_total"] > df["model_ceiling"]]
    below = df[df["sleeper_total"] < df["model_floor"]]

    print(f"  Sleeper total within our floor-ceiling: {len(in_range):4d} ({100*len(in_range)/len(df):.1f}%)")
    print(f"  Sleeper total ABOVE our ceiling:        {len(above):4d} ({100*len(above)/len(df):.1f}%)")
    print(f"  Sleeper total BELOW our floor:          {len(below):4d} ({100*len(below)/len(df):.1f}%)")

    if not above.empty:
        print(f"\n  Sleeper above our CEILING (top 10):")
        for _, r in above.nlargest(10, "diff").iterrows():
            print(f"    {r['name']:25s} {r['position']:3s}  Ceiling: {r['model_ceiling']:.1f}  Sleeper: {r['sleeper_total']:.1f}  Over by: {r['sleeper_total'] - r['model_ceiling']:.1f}")

    if not below.empty:
        print(f"\n  Sleeper below our FLOOR (top 10):")
        for _, r in below.nsmallest(10, "diff").iterrows():
            print(f"    {r['name']:25s} {r['position']:3s}  Floor: {r['model_floor']:.1f}  Sleeper: {r['sleeper_total']:.1f}  Under by: {r['model_floor'] - r['sleeper_total']:.1f}")

    # ── 9. Team-level patterns ───────────────────────────────────────────
    print_section("TEAM-LEVEL BIAS (avg model-sleeper diff per team)")

    team_stats = (
        df.groupby("team")
        .agg(
            n=("diff", "size"),
            mean_diff=("diff", "mean"),
            total_model=("model_total", "sum"),
            total_sleeper=("sleeper_total", "sum"),
        )
        .sort_values("mean_diff")
    )
    team_stats = team_stats[team_stats["n"] >= 3]

    print(f"  {'Team':5s}  {'N':>3s}  {'Avg Diff':>8s}  {'Sum Model':>9s}  {'Sum Sleeper':>11s}")
    print(f"  {'-'*5}  {'-'*3}  {'-'*8}  {'-'*9}  {'-'*11}")
    for team, r in team_stats.iterrows():
        marker = " <<<" if abs(r["mean_diff"]) > 20 else ""
        print(f"  {team:5s}  {r['n']:3.0f}  {r['mean_diff']:+8.1f}  {r['total_model']:9.0f}  {r['total_sleeper']:11.0f}{marker}")

    # ── 10. Summary & recommendations ────────────────────────────────────
    print_section("SUMMARY & RECOMMENDATIONS")

    # Compute systematic patterns
    mean_games_diff = df["games_diff"].mean()
    mean_ppg_diff = df["ppg_diff"].mean()

    print("Systematic patterns detected:\n")

    # Games played bias
    if abs(mean_games_diff) > 0.5:
        direction = "fewer" if mean_games_diff < 0 else "more"
        print(f"  1. GAMES PLAYED BIAS: Our model projects {abs(mean_games_diff):.1f} {direction} games on average.")
        print(f"     Sleeper defaults to 18 GP (all regular season games), while our model")
        print(f"     learns historical availability. This is the single largest source of")
        print(f"     total-point differences. Consider whether our GP model is too conservative")
        print(f"     or Sleeper's 18-game default is unrealistic.\n")

    # PPG bias by position
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = df[df["position"] == pos]
        if sub.empty:
            continue
        ppg_bias = sub["ppg_diff"].mean()
        if abs(ppg_bias) > 0.5:
            direction = "higher" if ppg_bias > 0 else "lower"
            print(f"  {pos} PPG BIAS: Our model is {abs(ppg_bias):.2f} PPG {direction} on average.")
            # Check if it's top-heavy or bottom-heavy
            top = sub.nsmallest(12, "pos_rank")["ppg_diff"].mean()
            rest = sub[sub["pos_rank"] > 12]["ppg_diff"].mean()
            if abs(top - rest) > 0.5:
                print(f"     Top-12 avg PPG diff: {top:+.2f},  Rest: {rest:+.2f}")
                if top > rest:
                    print(f"     → Model may be inflating elite {pos}s relative to depth.")
                else:
                    print(f"     → Model may be compressing the {pos} talent distribution.")
            print()

    # Floor/ceiling calibration
    pct_in_range = 100 * len(in_range) / len(df)
    if pct_in_range < 60:
        print(f"  FLOOR/CEILING CALIBRATION: Only {pct_in_range:.0f}% of Sleeper projections")
        print(f"  fall within our floor-ceiling range. Our uncertainty bands may be too narrow.\n")
    elif pct_in_range > 90:
        print(f"  FLOOR/CEILING CALIBRATION: {pct_in_range:.0f}% of Sleeper projections")
        print(f"  fall within our range — bands may be too wide (not discriminating enough).\n")

    # Biggest team disagreements
    extreme_teams = team_stats[team_stats["mean_diff"].abs() > 20]
    if not extreme_teams.empty:
        print(f"  TEAM-LEVEL OUTLIERS:")
        for team, r in extreme_teams.iterrows():
            direction = "bullish" if r["mean_diff"] > 0 else "bearish"
            print(f"    {team}: We are {direction} by avg {abs(r['mean_diff']):.1f} pts/player")
        print(f"     → Check for offseason moves, coaching changes, or scheme shifts")
        print(f"       that one model may have captured and the other missed.\n")

    print("Actionable recommendations:\n")
    print("  a) GAMES MODEL: Compare our GP predictions against actual 2024/2025")
    print("     availability. If we're systematically low, the injury/games model")
    print("     may be over-penalizing. Sleeper's 18-game assumption is aggressive")
    print("     but being too conservative costs more in fantasy drafts.\n")
    print("  b) ROOKIE/BREAKOUT DETECTION: Check if the biggest positive diffs")
    print("     (where we're higher) are rookies or breakout candidates where our")
    print("     model learned from college/trajectory data that Sleeper's simpler")
    print("     methodology may miss, or vice versa.\n")
    print("  c) REGRESSION TO MEAN: If Sleeper is higher on players coming off")
    print("     career years, and lower on bounceback candidates, their model may")
    print("     be doing less regression. Compare which approach backtests better.\n")
    print("  d) POSITIONAL SCORING: Verify both models use the same PPR scoring.")
    print("     Small differences in assumed scoring (e.g., bonus receptions,")
    print("     fumbles, 2pt conversions) compound across a season.\n")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, path: str = "data/analysis/model_vs_sleeper.csv") -> None:
    """Save the comparison dataset for further analysis."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nDataset exported to {path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading model projections and fetching Sleeper data...")
    df = load_comparison_data()
    analyze(df)
    export_csv(df)


if __name__ == "__main__":
    main()
