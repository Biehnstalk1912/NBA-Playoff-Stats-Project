# Reading in python libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_squared_error, r2_score

# Reading in datasets using pandas
player_scoring_rs = pd.read_csv("player_stats_traditional_rs.csv")
player_scoring_post = pd.read_csv("player_stats_traditional_po.csv")

# Merging/Joining the datasets using primary_id and season as primary keys, adding _po and _rs to differniate between post and regular season
df = pd.merge(player_scoring_post, player_scoring_rs, on=["PLAYER_ID", "SEASON"], how="left", suffixes=("_po", "_rs"))

# Cleaning data and selecting variables of interest
df.columns = df.columns.str.lower()
df["age_rs"] = df["age_rs"].fillna(0).astype(int)
df["age_po"] = df["age_po"].fillna(0).astype(int)
selected_columns = [
    "player_id", "player_name_po", "team_id_po", "team_id_rs",
    "age_po", "age_rs", "fg_pct_rs", "fg_pct_po", "team_abbreviation_rs",
    "team_abbreviation_po", "fg3_pct_po", "fg3_pct_rs", "pts_po", "pts_rs",
    "plus_minus_rs", "plus_minus_po", "min_rs", "gp_rs",
    "w_pct_rs", "blk_rs", "blka_rs", "pf_rs", "pfd_rs",
    "min_po", "gp_po", "w_pct_po", "blk_po", "blka_po",
    "pf_po", "pfd_po", "season"
]
df = df[selected_columns]

# Explatory data analysis
avg_pts_rs = df.groupby("age_rs")["pts_rs"].mean().reset_index()
avg_pts_po = df.groupby("age_po")["pts_po"].mean().reset_index()

plt.figure(figsize=(10,6))
plt.plot(avg_pts_rs["age_rs"], avg_pts_rs["pts_rs"], marker='o', label="Regular Season")
plt.plot(avg_pts_po["age_po"], avg_pts_po["pts_po"], marker='s', label="Postseason")
plt.title("Average Points per Game by Age")
plt.xlabel("Age")
plt.ylabel("Average Points per Game")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

average_fg_rs = df.groupby("age_rs")["fg_pct_rs"].mean().reset_index()
average_fg_po = df.groupby("age_po")["fg_pct_po"].mean().reset_index()

plt.figure(figsize=(10,6))
plt.plot(average_fg_rs["age_rs"], average_fg_rs["fg_pct_rs"]*100, marker='o', label="Regular Season")
plt.plot(average_fg_po["age_po"], average_fg_po["fg_pct_po"]*100, marker='s', label="Postseason")
plt.title("Average Field Goal Percentage by Age")
plt.xlabel("Age")
plt.ylabel("Average Field Goal Percentage (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Creating a simple linear regression model predicting points for the regular season
features_simp_rs = ["age_rs"]
target_simp_rs = "pts_rs"
X_simp_rs = df[features_simp_rs]
y_simp_rs = df[target_simp_rs]
df_simp_rs = df[["age_rs", "pts_rs"]].dropna()

X_simp_rs = df_simp_rs[["age_rs"]]
y_simp_rs = df_simp_rs["pts_rs"]
y_simp_rs = y_simp_rs[X_simp_rs.index]
X_simp_rs_train, X_simp_rs_test, y_simp_rs_train, y_simp_rs_test = train_test_split(X_simp_rs, y_simp_rs, test_size=0.2, random_state=42)
model_simp_rs = LinearRegression()
model_simp_rs.fit(X_simp_rs_train, y_simp_rs_train)
y_rs_simp_pred = model_simp_rs.predict(X_simp_rs_test)
# Creating a simple linear regression model predicting points for the post season
features_simp_po = ["age_po"] 
target_simp_po = "pts_po"
X_simp_po = df[features_simp_po]
y_simp_po = df[target_simp_po]
X_simp_po = X_simp_po.dropna()
y_simp_po = y_simp_po.dropna()
y_simp_po = y_simp_po[X_simp_po.index]
X_simp_po_train, X_simp_po_test, y_simp_po_train, y_simp_po_test = train_test_split(X_simp_po, y_simp_po, test_size=0.2, random_state=42)
model_simp_po = LinearRegression()
model_simp_po.fit(X_simp_po_train, y_simp_po_train)
y_po_simp_pred = model_simp_po.predict(X_simp_po_test)

# Plotting the two simple regression lines together
plt.scatter(X_simp_rs_test['age_rs'], y_simp_rs_test, color='blue', alpha=0.5, label='RS Actual Points')
plt.scatter(X_simp_po_test['age_po'], y_simp_po_test, color='green', alpha=0.5, label='PO Actual Points')
plt.plot(X_simp_rs_test['age_rs'], y_rs_simp_pred, color='blue', linewidth=2, label='RS Regression Line')
plt.plot(X_simp_po_test['age_po'], y_po_simp_pred, color='green', linewidth=2, label='PO Regression Line')
plt.xlabel('Age')
plt.ylabel('Total Points')
plt.title('Simple Linear Regression: Age vs Total Points')
plt.legend()
plt.show()
# Creating a linear regression model predicting points for the regular season
features_rs = ['age_rs', 'min_rs', 'gp_rs', 'w_pct_rs', 'blk_rs', 'blka_rs', 'pf_rs', 'pfd_rs', 'team_abbreviation_rs']
target_rs = "pts_rs"
X_rs = df[features_rs]
y_rs = df[target_rs]
X_rs = X_rs.dropna()
y_po_ = y_rs.dropna()
y_rs = y_rs[X_rs.index]
X_rs = pd.get_dummies(X_rs, columns = ['team_abbreviation_rs'], drop_first = True)
X_rs_train, X_rs_test, y_rs_train, y_rs_test = train_test_split(X_rs, y_rs, test_size=0.2, random_state=42)
model_rs = LinearRegression()
model_rs.fit(X_rs_train, y_rs_train)
y_rs_pred = model_rs.predict(X_rs_test)
mse_rs = mean_squared_error(y_rs_test, y_rs_pred)
r2_rs = r2_score(y_rs_test, y_rs_pred)
print(f"Mean Squared Error: {mse_rs}")
print(f"R^2 Score: {r2_rs}")
print("Intercept:", model_rs.intercept_)
coefficients_rs = pd.Series(model_rs.coef_, index=X_rs_train.columns)
print(coefficients_rs.sort_values(ascending=False))
print(model_rs.coef_)

plt.scatter(X_rs['min_rs'], y_rs, color='blue', label='Actual Points')
plt.plot(X_rs['min_rs'], model_rs.predict(X_rs), color='red', label='Regression Line')
plt.xlabel('Minutes Played (MIN)')
plt.ylabel('Total Points')
plt.title('Linear Regression: MIN vs Total Points')
plt.legend()
plt.show()
# Creating a linear regression model predicting points for the post season
features_po = ['age_po', 'min_po', 'gp_po', 'w_pct_po', 'blk_po', 'blka_po', 'pf_po', 'pfd_po', "team_abbreviation_po"]
target_po = "pts_po"
X_po = df[features_po]
y_po = df[target_po]
X_po = X_po.dropna()
y_po_ = y_po.dropna()

y_po = y_po[X_po.index]

X_po = pd.get_dummies(X_po, columns = ["team_abbreviation_po"], drop_first = True)
X_po_train, X_po_test, y_po_train, y_po_test = train_test_split(X_po, y_po, test_size=0.2, random_state=42)
model_po = LinearRegression()
model_po.fit(X_po_train, y_po_train)
y_po_pred = model_po.predict(X_po_test)
mse_po = mean_squared_error(y_po_test, y_po_pred)
r2_po = r2_score(y_po_test, y_po_pred)
print(f"Mean Squared Error: {mse_po}")
print(f"R^2 Score: {r2_po}")
print("Intercept:", model_rs.intercept_)
coefficients_po = pd.Series(model_po.coef_, index=X_po_train.columns)
print(coefficients_po.sort_values(ascending=False))

plt.scatter(X_po['min_po'], y_po, color='blue', label='Actual Points')
plt.plot(X_po['min_po'], model_po.predict(X_po), color='red', label='Regression Line')
plt.xlabel('Minutes Played (MIN)')
plt.ylabel('Total Points')
plt.title('Linear Regression: MIN vs Total Points')
plt.legend()
plt.show()

