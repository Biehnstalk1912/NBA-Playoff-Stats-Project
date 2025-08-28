# ==============================
# Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set(style="whitegrid")

# ==============================
# Data Loading & Cleaning
# ==============================
player_scoring_rs = pd.read_csv("player_stats_traditional_rs.csv")
player_scoring_post = pd.read_csv("player_stats_traditional_po.csv")

df = pd.merge(player_scoring_post, player_scoring_rs,
              on=["PLAYER_ID", "SEASON"], how="left",
              suffixes=("_po", "_rs"))

df.columns = df.columns.str.lower()
df["age_rs"] = df["age_rs"].fillna(0).astype(int)
df["age_po"] = df["age_po"].fillna(0).astype(int)

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
df = df.dropna(subset=['pts_rs', 'pts_po'])

# ==============================
# Feature Engineering
# ==============================
df["pts_per_min_rs"] = df["pts_rs"] / df["min_rs"].replace(0, np.nan)
df["pts_per_min_po"] = df["pts_po"] / df["min_po"].replace(0, np.nan)
df["fg3_rate_rs"] = df["fg3_pct_rs"] / df["fg_pct_rs"].replace(0, np.nan)
df["fg3_rate_po"] = df["fg3_pct_po"] / df["fg_pct_po"].replace(0, np.nan)
df["age_min_interact_rs"] = df["age_rs"] * df["min_rs"]
df["age_min_interact_po"] = df["age_po"] * df["min_po"]

# ==============================
# 1. Exploratory Data Analysis
# ==============================
# Five-number summary
five_num_rs = df['pts_rs'].describe()[['min','25%','50%','75%','max']]
five_num_po = df['pts_po'].describe()[['min','25%','50%','75%','max']]
five_num_table = pd.DataFrame({'Regular Season': five_num_rs, 'Playoffs': five_num_po})
five_num_table.index = ['Min','Q1','Median','Q3','Max']
print(five_num_table)

# Boxplots
plt.figure(figsize=(10,6))
plt.boxplot([df['pts_rs'], df['pts_po']],
            labels=['Regular Season','Playoffs'],
            patch_artist=True,
            boxprops=dict(facecolor='skyblue'),
            medianprops=dict(color='red'))
plt.ylabel('Points')
plt.title('Distribution of Points: Regular Season vs Playoffs')
plt.show()

# Average points by age
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

# Smoothed curve & peak scoring age
def smooth_avg_points(df, age_col, pts_col):
    age_pts = df.groupby(age_col)[pts_col].mean().reset_index()
    age_pts['smoothed'] = age_pts[pts_col].rolling(3, center=True).mean()
    peak_age = age_pts.loc[age_pts['smoothed'].idxmax(), age_col]
    peak_pts = age_pts['smoothed'].max()
    return age_pts, peak_age, peak_pts

rs_age_pts, peak_age_rs, peak_pts_rs = smooth_avg_points(df, "age_rs", "pts_rs")
po_age_pts, peak_age_po, peak_pts_po = smooth_avg_points(df, "age_po", "pts_po")

plt.figure(figsize=(10,6))
plt.plot(rs_age_pts['age_rs'], rs_age_pts['smoothed'], marker='o', label=f'RS (Peak Age: {peak_age_rs})')
plt.plot(po_age_pts['age_po'], po_age_pts['smoothed'], marker='s', label=f'PO (Peak Age: {peak_age_po})')
plt.xlabel("Age")
plt.ylabel("Average Points")
plt.title("Smoothed Average Points by Age")
plt.legend()
plt.grid(True)
plt.show()

print(f"Peak scoring age RS: {peak_age_rs} ({peak_pts_rs:.2f} pts)")
print(f"Peak scoring age PO: {peak_age_po} ({peak_pts_po:.2f} pts)")

# ==============================
# 2. Model Creation: Looking to see what kind of model fits this data the best
# ==============================
# Simple linear regression
df_simp_rs = df[["age_rs","pts_rs"]].dropna()
X_simp_rs = df_simp_rs[["age_rs"]]
y_simp_rs = df_simp_rs["pts_rs"]
X_simp_rs_train, X_simp_rs_test, y_simp_rs_train, y_simp_rs_test = train_test_split(
    X_simp_rs, y_simp_rs, test_size=0.2, random_state=42)
model_simp_rs = LinearRegression()
model_simp_rs.fit(X_simp_rs_train, y_simp_rs_train)
y_rs_simp_pred = model_simp_rs.predict(X_simp_rs_test)

df_simp_po = df[["age_po","pts_po"]].dropna()
X_simp_po = df_simp_po[["age_po"]]
y_simp_po = df_simp_po["pts_po"]
X_simp_po_train, X_simp_po_test, y_simp_po_train, y_simp_po_test = train_test_split(
    X_simp_po, y_simp_po, test_size=0.2, random_state=42)
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

# Polynomial Regression
def poly_regression(X, y, degree=2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, poly, X_train, X_test, y_train, y_test, y_pred, mse, r2

model_rs, poly_rs, X_train_rs, X_test_rs, y_train_rs, y_test_rs, y_pred_rs, mse_rs, r2_rs = poly_regression(X_simp_rs, y_simp_rs, degree=2)
model_po, poly_po, X_train_po, X_test_po, y_train_po, y_test_po, y_pred_po, mse_po, r2_po = poly_regression(X_simp_po, y_simp_po, degree=2)

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

# Multivariate Linear Regression
features_rs = ['age_rs','min_rs','gp_rs','w_pct_rs','blk_rs','blka_rs','pf_rs','pfd_rs']
X_rs = df[features_rs].dropna()
y_rs = df.loc[X_rs.index, "pts_rs"]
X_rs_train, X_rs_test, y_rs_train, y_rs_test = train_test_split(X_rs, y_rs, test_size=0.2, random_state=42)
model_rs = LinearRegression()
model_rs.fit(X_rs_train, y_rs_train)
y_rs_pred = model_rs.predict(X_rs_test)

features_po = ['age_po','min_po','gp_po','w_pct_po','blk_po','blka_po','pf_po','pfd_po']
X_po = df[features_po].dropna()
y_po = df.loc[X_po.index, "pts_po"]
X_po_train, X_po_test, y_po_train, y_po_test = train_test_split(X_po, y_po, test_size=0.2, random_state=42)
model_po = LinearRegression()
model_po.fit(X_po_train, y_po_train)
y_po_pred = model_po.predict(X_po_test)

# Multicollinearity Check (VIF: Variance Inflation Factor)
vif_rs = pd.DataFrame({'feature':X_rs.columns, 'VIF':[variance_inflation_factor(X_rs.values,i) for i in range(X_rs.shape[1])]})
vif_po = pd.DataFrame({'feature':X_po.columns, 'VIF':[variance_inflation_factor(X_po.values,i) for i in range(X_po.shape[1])]})
print("\nRS VIF:\n", vif_rs)
print("\nPO VIF:\n", vif_po)

# Residual Analysis
def plot_residuals(y_true, y_pred, title_prefix=""):
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0,color='red',linestyle='--')
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

# Ridge & Lasso,,helps improve model stability
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
    X_rs_train, y_rs_train, X_rs_test, y_rs_test, prefix="RS", base_model=model_rs)
ridge_po, lasso_po, ridge_r2_po, ridge_mse_po, lasso_r2_po, lasso_mse_po, coeffs_po = fit_regularization_models(
    X_po_train, y_po_train, X_po_test, y_po_test, prefix="PO", base_model=model_po)

# Standardized Coefficients
def plot_standardized_coeffs(model, X, title=""):
    coefs_std = pd.Series(model.coef_ * X.std(), index=X.columns).sort_values()
    coefs_std.plot(kind='barh', figsize=(10,6), color='skyblue')
    plt.title(f"Standardized Coefficients ({title})")
    plt.xlabel("Coefficient Value")
    plt.show()

plot_standardized_coeffs(model_rs, X_rs, "Regular Season")
plot_standardized_coeffs(model_po, X_po, "Postseason")
# ==============================
# 3. Model Evaluation: Looking to see what model is the best for this data
# ==============================
# Cross-Validationm, tests skill of machine learning on unseen data
cv_r2_rs = cross_val_score(model_rs, X_rs, y_rs, cv=5, scoring='r2')
cv_r2_po = cross_val_score(model_po, X_po, y_po, cv=5, scoring='r2')
print(f"\nCross-Validated R^2 RS: {cv_r2_rs.mean():.3f}")
print(f"Cross-Validated R^2 PO: {cv_r2_po.mean():.3f}")

# Plotting Predicted vs Actual Values
def plot_pred_vs_actual(y_true, y_pred, title=""):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title(f"{title} Predicted vs Actual Points")
    plt.show()

plot_pred_vs_actual(y_rs_test, y_rs_pred, "RS")
plot_pred_vs_actual(y_po_test, y_po_pred, "PO")

# Model Comparison Table
results = pd.DataFrame({
    "Model":[
        "Simple Linear RS","Polynomial RS","OLS RS","Ridge RS","Lasso RS",
        "Simple Linear PO","Polynomial PO","OLS PO","Ridge PO","Lasso PO"
    ],
    "R^2":[
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
    "MSE":[
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
# ==============================
# Final Insights
# ==============================

# 1. Exploratory data analysis
# - Regular Season (RS) scoring is consistently higher than Playoff (PO) scoring.
# - Median RS points = 7.8 vs. PO = 6.3.
# - Outliers exist in both RS (max 36.1) and PO (max 37.4).
# - Both RS and PO peak scoring occur at age 28.
# - RS peak = 10.68 points, PO peak = 9.33 points.
# - Suggests age 28 is the “prime” scoring age.

# 3. Model Evaluation
# a. Simple Linear and Polynomial regression: R² ≈ 0 → age alone is not a good predictor.
# - Multivariate OLS, Ridge, and Lasso: R² ≈ 0.84–0.85 with low MSE
# - RS models outperform PO models slightly in both accuracy and error
# b. Ridge & Lasso Regularization
# - Ridge RS R² = 0.846, Lasso RS R² = 0.846 → nearly identical to OLS.
# - PO Ridge & Lasso also ≈ 0.841
# - Confirms that adding regularization does not improve much, but ensures model stability
# c. Cross-Validation
# - RS CV R² ≈ 0.831, PO CV R² ≈ 0.814.
# - Confirms models generalize well to new or unssen data
# d. Multicollinearity
# - RS features show strong collinearity (Age, Minutes, Games, Win % all VIF > 10)
# - PO features are less collinear, making the models more stable
# - Regularization (Ridge/Lasso) is useful for RS to reduce variance

# ------------------------------
# Final Takeaways
# ------------------------------
# - Age alone does not explain scoring; context variables (minutes, games, win %) are critical
# - Peak scoring occurs around age 28 in both RS and PO
# - RS players generally score more than in PO
# - Multivariate regression (OLS, Ridge, Lasso) provides strong predictive accuracy (R² ≈ 0.84–0.85)
# - Regular season stats are slightly easier to model than playoff stats

