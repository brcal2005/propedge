"""Download NBA player game logs into propedge's CSV format.

    pip install nba_api pandas
    python scripts/fetch_nba_logs.py --season 2026-27 --out nba_logs.csv

Team names are written in full ("Boston Celtics") so they match The Odds API event names,
which lets `scan --logs` apply the opponent adjustment automatically.
"""
import argparse

TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="e.g. 2026-27")
    ap.add_argument("--out", default="nba_logs.csv")
    ap.add_argument("--season-type", default="Regular Season")
    a = ap.parse_args()

    from nba_api.stats.endpoints import playergamelogs

    df = playergamelogs.PlayerGameLogs(season_nullable=a.season,
                                       season_type_nullable=a.season_type).get_data_frames()[0]
    opp = df["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[-1]
    out = df.assign(
        player=df["PLAYER_NAME"],
        date=df["GAME_DATE"].str[:10],
        team=df["TEAM_ABBREVIATION"].map(TEAMS).fillna(df["TEAM_ABBREVIATION"]),
        opponent=opp.map(TEAMS).fillna(opp),
        minutes=df["MIN"].astype(float),
        pts=df["PTS"], reb=df["REB"], ast=df["AST"], fg3m=df["FG3M"],
        stl=df["STL"], blk=df["BLK"], tov=df["TOV"],
    )[["player", "date", "team", "opponent", "minutes", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]]
    out = out[out["minutes"] > 0].sort_values(["player", "date"])
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
