import pandas as pd
player_index = pd.read_csv("player_index.csv")
player_scoring_rs = pd.read_csv("player_stats_traditional_rs.csv")
player_scoring_post = pd.read_csv("player_stats_traditional_po.csv")
print(player_scoring_rs.head(5))
print(player_scoring_post.head(5))

df = pd.merge(player_scoring_post, player_scoring_rs, on="PLAYER_ID", how="right", suffixes=("_po", "_rs"))

df.columns = df.columns.str.lower()

print(df.columns)
selected_columns = ["player_id", "player_name_po", "age"]