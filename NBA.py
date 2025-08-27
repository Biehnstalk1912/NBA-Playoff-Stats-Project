import pandas as pd
import matplotlib.pyplot as plt

player_index = pd.read_csv("player_index.csv")
player_scoring_rs = pd.read_csv("player_stats_traditional_rs.csv")
player_scoring_post = pd.read_csv("player_stats_traditional_po.csv")

df = pd.merge(player_scoring_post, player_scoring_rs, on=["PLAYER_ID", "SEASON"], how="left", suffixes=("_po", "_rs"))

df.columns = df.columns.str.lower()
df["age_rs"] = df["age_rs"].fillna(0).astype(int)
df["age_po"] = df["age_po"].fillna(0).astype(int)

selected_columns = ["player_id", "player_name_po", "team_id_po", "team_id_rs", "age_po", "fg_pct_rs", "fg_pct_po", "fg3_pct_po", "fg3_pct_rs", "pts_po", "pts_rs", "plus_minus_rs", "plus_minus_po", "season"]
df = df[selected_columns]

avg_pts_rs = df.groupby("age_po")["pts_rs"].mean().reset_index()
avg_pts_po = df.groupby("age_po")["pts_po"].mean().reset_index()
plt.figure(figsize=(10,6))
plt.plot(avg_pts_rs["age_po"], avg_pts_rs["pts_rs"], marker='o', label="Regular Season")
plt.plot(avg_pts_po["age_po"], avg_pts_po["pts_po"], marker='s', label="Postseason")
plt.title("Average Points per Game by Age")
plt.xlabel("Age")
plt.ylabel("Average Points per Game")
plt.legend()
plt.grid(True)
plt.show()
