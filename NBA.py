import pandas as pd

player_index = pd.read_csv("player_index.csv")
player_scoring_rs = pd.read_csv("player_stats_traditional_rs.csv")
player_scoring_post = pd.read_csv("player_stats_traditional_po.csv")
df = pd.merge(player_scoring_post, player_scoring_rs, on=["PLAYER_ID", "SEASON"], how="left", suffixes=("_po", "_rs"))

df.columns = df.columns.str.lower()

print(df.columns)
selected_columns = ["player_id", "player_name_po", "team_id_po", "team_id_rs", "age_po", "fg_pct_rs", "fg_pct_po", "fg3_pct_po", "fg3_pct_rs", "pts_po", "pts_rs", "plus_minus_rs", "plus_minus_po", "season"]
df = df[selected_columns]
print(df.head(10))

