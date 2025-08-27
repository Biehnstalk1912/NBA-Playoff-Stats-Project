# ==============================
# Libraries
# ==============================
# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# ==============================
# Data Loading & Cleaning
# ==============================
player_scoring_rs = pd.read_csv("player_stats_traditional_rs.csv")
player_scoring_post = pd.read_csv("player_stats_traditional_po.csv")

# Merging/joining datasets by player_id and season
df = pd.merge(player_scoring_post, player_scoring_rs,
              on=["PLAYER_ID", "SEASON"], how="left",
              suffixes=("_po", "_rs"))

# Cleaning Columns
df.columns = df.columns.str.lower()
df["age_rs"] = df["age_rs"].fillna(0).astype(int)
df["age_po"] = df["age_po"].fillna(0).astype(int)

# Selecting relevant columns for models
selected_columns = [
    "player_id", "player_name_po", "team_id_po", "team_id_rs",
    "age_po", "age_rs", "fg_pct_rs", "fg_pct_po",
    "fg3_pct_po", "fg3_pct_rs", "pts_po", "pts_rs",
    "plus_minus_rs", "plus_minus_po", "min_rs", "gp_rs",
    "w_pct_rs", "blk_rs", "blka_rs", "pf_rs", "pfd_rs",
    "min_po", "gp_po", "w_pct_po", "blk_po", "blka_po",
    "pf_po", "pfd_po", "season"
]
df = df[selected_columns]

# ==============================
# Exploratory Data Analysis
# ==============================
df = df.dropna(subset=['pts_rs', 'pts_po'])

# Compute five-number summary for regular and post season points
five_num_rs = df['pts_rs'].describe()[['min', '25%', '50%', '75%', 'max']]
five_num_po = df['pts_po'].describe()[['min', '25%', '50%', '75%', 'max']]

# Combine into a single table
five_num_table = pd.DataFrame({
    'Regular Season': five_num_rs,
    'Playoffs': five_num_po
})

# Rename the index for clarity
five_num_table.index = ['Min', 'Q1', 'Median', 'Q3', 'Max']

print(five_num_table)

plt.figure(figsize=(10, 6))

# Boxplot for both regular season and playoff points
plt.boxplot([df['pts_rs'], df['pts_po']], labels=['Regular Season', 'Playoffs'],
            patch_artist=True,
            boxprops=dict(facecolor='skyblue'),
            medianprops=dict(color='red'))

plt.ylabel('Points')
plt.title('Distribution of Points: Regular Season vs Playoffs')
plt.show()

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

# ==============================
# Simple Linear Regression (RS & PO)
# ==============================
df_simp_rs = df[["age_rs", "pts_rs"]].dropna()
X_simp_rs = df_simp_rs[["age_rs"]]
y_simp_rs = df_simp_rs["pts_rs"]
X_simp_rs_train, X_simp_rs_test, y_simp_rs_train, y_simp_rs_test = train_test_split(
    X_simp_rs, y_simp_rs, test_size=0.2, random_state=42
)
model_simp_rs = LinearRegression()
model_simp_rs.fit(X_simp_rs_train, y_simp_rs_train)
y_rs_simp_pred = model_simp_rs.predict(X_simp_rs_test)

df_simp_po = df[["age_po", "pts_po"]].dropna()
X_simp_po = df_simp_po[["age_po"]]
y_simp_po = df_simp_po["pts_po"]
X_simp_po_train, X_simp_po_test, y_simp_po_train, y_simp_po_test = train_test_split(
    X_simp_po, y_simp_po, test_size=0.2, random_state=42
)
model_simp_po = LinearRegression()
model_simp_po.fit(X_simp_po_train, y_simp_po_train)
y_po_simp_pred = model_simp_po.predict(X_simp_po_test)

# Plot simple regression
plt.scatter(X_simp_rs_test, y_simp_rs_test, color='blue', alpha=0.5, label='RS Actual Points')
plt.scatter(X_simp_po_test, y_simp_po_test, color='green', alpha=0.5, label='PO Actual Points')
x_range_rs = np.linspace(X_simp_rs.min(), X_simp_rs.max(), 100).reshape(-1,1)
y_line_rs = model_simp_rs.predict(x_range_rs)
x_range_po = np.linspace(X_simp_po.min(), X_simp_po.max(), 100).reshape(-1,1)
y_line_po = model_simp_po.predict(x_range_po)
plt.plot(x_range_rs, y_line_rs, color='blue', linewidth=2, label='RS Regression Line')
plt.plot(x_range_po, y_line_po, color='green', linewidth=2, label='PO Regression Line')
plt.xlabel('Age')
plt.ylabel('Total Points')
plt.title('Simple Linear Regression: Age vs Total Points')
plt.legend()
plt.show()

# ==============================
# Polynomial Regression Function
# ==============================
def poly_regression(X, y, degree=2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, poly, X_train, X_test, y_train, y_test, y_pred, mse, r2

# Polynomial RS & PO
model_rs, poly_rs, X_train_rs, X_test_rs, y_train_rs, y_test_rs, y_pred_rs, mse_rs, r2_rs = poly_regression(X_simp_rs, y_simp_rs, degree=2)
model_po, poly_po, X_train_po, X_test_po, y_train_po, y_test_po, y_pred_po, mse_po, r2_po = poly_regression(X_simp_po, y_simp_po, degree=2)

# Plot polynomial regression
plt.scatter(X_test_rs, y_test_rs, color='blue', alpha=0.5, label='RS Actual Points')
plt.scatter(X_test_po, y_test_po, color='green', alpha=0.5, label='PO Actual Points')
x_range_rs = np.linspace(X_simp_rs.min(), X_simp_rs.max(), 100).reshape(-1,1)
y_line_rs = model_rs.predict(poly_rs.transform(x_range_rs))
x_range_po = np.linspace(X_simp_po.min(), X_simp_po.max(), 100).reshape(-1,1)
y_line_po = model_po.predict(poly_po.transform(x_range_po))
plt.plot(x_range_rs, y_line_rs, color='blue', linewidth=2, label=f'RS Poly Reg (R²={r2_rs:.2f})')
plt.plot(x_range_po, y_line_po, color='green', linewidth=2, label=f'PO Poly Reg (R²={r2_po:.2f})')
plt.xlabel('Age')
plt.ylabel('Total Points')
plt.title('Polynomial Regression: Age vs Total Points')
plt.legend()
plt.show()

# ==============================
# Multivariate Linear Regression (RS & PO)
# ==============================
features_rs = ['age_rs', 'min_rs', 'gp_rs', 'w_pct_rs', 'blk_rs', 'blka_rs', 'pf_rs', 'pfd_rs']
X_rs = df[features_rs].dropna()
y_rs = df.loc[X_rs.index, "pts_rs"]
X_rs_train, X_rs_test, y_rs_train, y_rs_test = train_test_split(X_rs, y_rs, test_size=0.2, random_state=42)
model_rs = LinearRegression()
model_rs.fit(X_rs_train, y_rs_train)
y_rs_pred = model_rs.predict(X_rs_test)

features_po = ['age_po', 'min_po', 'gp_po', 'w_pct_po', 'blk_po', 'blka_po', 'pf_po', 'pfd_po']
X_po = df[features_po].dropna()
y_po = df.loc[X_po.index, "pts_po"]
X_po_train, X_po_test, y_po_train, y_po_test = train_test_split(X_po, y_po, test_size=0.2, random_state=42)
model_po = LinearRegression()
model_po.fit(X_po_train, y_po_train)
y_po_pred = model_po.predict(X_po_test)

# ==============================
# STEP 1: Residual Analysis
# ==============================
def plot_residuals(y_true, y_pred, title_prefix=""):
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Points")
    plt.ylabel("Residuals")
    plt.title(f"{title_prefix} Residual Plot")
    plt.show()
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title(f"{title_prefix} Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()

plot_residuals(y_rs_test, y_rs_pred, "Regular Season")
plot_residuals(y_po_test, y_po_pred, "Postseason")

# ==============================
# STEP 2: Ridge & Lasso
# ==============================
def fit_regularization_models(X_train, y_train, X_test, y_test, prefix="RS", base_model=None):
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3,3,100), cv=5))
    lasso = make_pipeline(StandardScaler(), LassoCV(alphas=np.logspace(-3,3,100), cv=5, max_iter=10000))
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    ridge_r2 = ridge.score(X_test, y_test)
    lasso_r2 = lasso.score(X_test, y_test)
    ridge_mse = mean_squared_error(y_test, ridge.predict(X_test))
    lasso_mse = mean_squared_error(y_test, lasso.predict(X_test))
    coeffs = pd.DataFrame({
        "OLS": base_model.coef_,
        "Ridge": ridge.named_steps['ridgecv'].coef_,
        "Lasso": lasso.named_steps['lassocv'].coef_
    }, index=X_train.columns)
    print(f"\n--- {prefix} Ridge & Lasso Results ---")
    print("Ridge R^2:", ridge_r2, "MSE:", ridge_mse)
    print("Lasso R^2:", lasso_r2, "MSE:", lasso_mse)
    return ridge, lasso, ridge_r2, ridge_mse, lasso_r2, lasso_mse, coeffs

ridge_rs, lasso_rs, ridge_r2_rs, ridge_mse_rs, lasso_r2_rs, lasso_mse_rs, coeffs_rs = fit_regularization_models(
    X_rs_train, y_rs_train, X_rs_test, y_rs_test, prefix="RS", base_model=model_rs
)
ridge_po, lasso_po, ridge_r2_po, ridge_mse_po, lasso_r2_po, lasso_mse_po, coeffs_po = fit_regularization_models(
    X_po_train, y_po_train, X_po_test, y_po_test, prefix="PO", base_model=model_po
)

# ==============================
# STEP 2B: Plot Coefficients
# ==============================
def plot_coefficients(coeffs, title=""):
    coeffs_plot = coeffs.copy()
    coeffs_plot.plot(kind="bar", figsize=(12,6))
    plt.title(f"Coefficient Comparison ({title})")
    plt.ylabel("Coefficient Value")
    plt.tight_layout()
    plt.show()

plot_coefficients(coeffs_rs, "Regular Season")
plot_coefficients(coeffs_po, "Postseason")

# ==============================
# STEP 3: Model Comparison Table
# ==============================
results = pd.DataFrame({
    "Model": [
        "Simple Linear RS", "Polynomial RS", "OLS RS", "Ridge RS", "Lasso RS",
        "Simple Linear PO", "Polynomial PO", "OLS PO", "Ridge PO", "Lasso PO"
    ],
    "R^2": [
        r2_score(y_simp_rs_test, y_rs_simp_pred),
        r2_rs,
        r2_score(y_rs_test, y_rs_pred),
        ridge_r2_rs,
        lasso_r2_rs,
        r2_score(y_simp_po_test, y_po_simp_pred),
        r2_po,
        r2_score(y_po_test, y_po_pred),
        ridge_r2_po,
        lasso_r2_po
    ],
    "MSE": [
        mean_squared_error(y_simp_rs_test, y_rs_simp_pred),
        mse_rs,
        mean_squared_error(y_rs_test, y_rs_pred),
        ridge_mse_rs,
        lasso_mse_rs,
        mean_squared_error(y_simp_po_test, y_po_simp_pred),
        mse_po,
        mean_squared_error(y_po_test, y_po_pred),
        ridge_mse_po,
        lasso_mse_po
    ]
})

print("\n===== Model Comparison Table =====")
print(results)
