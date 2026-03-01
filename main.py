import nfldata


def main():
    seasons = [2023, 2024]

    # 1. Player game log
    print("=" * 60)
    print("Josh Allen - Game Log (2023-2024)")
    print("=" * 60)
    game_log = nfldata.get_player_game_log("Josh Allen", seasons)
    cols = ["season", "week", "team", "opponent_team",
            "completions", "attempts", "passing_yards", "passing_tds",
            "passing_interceptions", "rushing_yards", "rushing_tds",
            "fantasy_points_ppr"]
    print(game_log.select([c for c in cols if c in game_log.columns]))

    # 2. Season summary
    print("\n" + "=" * 60)
    print("Josh Allen - Season Summary")
    print("=" * 60)
    summary = nfldata.player_season_summary("Josh Allen", seasons)
    key_cols = ["season", "games", "passing_yards", "passing_tds",
                "passing_interceptions", "rushing_yards", "rushing_tds",
                "fantasy_points_ppr", "passing_yards_per_game",
                "passing_tds_per_game"]
    print(summary.select([c for c in key_cols if c in summary.columns]))

    # 3. Compare players
    print("\n" + "=" * 60)
    print("QB Comparison: Allen vs Mahomes vs Hurts (2023-2024)")
    print("=" * 60)
    comparison = nfldata.compare_players(
        ["Josh Allen", "Patrick Mahomes", "Jalen Hurts"],
        seasons,
        stat_columns=["passing_yards", "passing_tds", "passing_interceptions",
                       "rushing_yards", "rushing_tds", "fantasy_points_ppr"],
    )
    print(comparison)

    # 4. Week 1 top passers
    print("\n" + "=" * 60)
    print("Top 10 Passers - 2024 Week 1")
    print("=" * 60)
    top = nfldata.top_performers(2024, 1, "passing_yards", n=10)
    print(top.select(["player_display_name", "position", "team",
                       "opponent_team", "passing_yards"]))


if __name__ == "__main__":
    main()
