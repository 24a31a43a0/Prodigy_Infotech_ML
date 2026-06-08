# ============================================================
# TASK 01 - House Price Prediction using Linear Regression
# Prodigy Infotech - Machine Learning Internship
# Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────
train_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'train.csv':
            train_path = os.path.join(root, f)
            break
    if train_path:
        break

if train_path is None:
    raise FileNotFoundError("train.csv not found! Please add the House Prices dataset.")

df = pd.read_csv(train_path)
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ── 2. Select Key Features ────────────────────────────────────
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
            'TotalBsmtSF', 'GarageArea', 'YearBuilt', 'OverallQual']
target = 'SalePrice'

df_model = df[features + [target]].dropna()
print(f"\nUsing {len(df_model)} rows after dropping nulls")

# ── 3. EDA - Visualizations ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(df_model['GrLivArea'], df_model['SalePrice'], alpha=0.4, color='steelblue')
axes[0].set_xlabel('Living Area (sq ft)')
axes[0].set_ylabel('Sale Price ($)')
axes[0].set_title('Living Area vs Sale Price')

axes[1].scatter(df_model['BedroomAbvGr'], df_model['SalePrice'], alpha=0.4, color='coral')
axes[1].set_xlabel('Number of Bedrooms')
axes[1].set_ylabel('Sale Price ($)')
axes[1].set_title('Bedrooms vs Sale Price')

axes[2].scatter(df_model['FullBath'], df_model['SalePrice'], alpha=0.4, color='green')
axes[2].set_xlabel('Full Bathrooms')
axes[2].set_ylabel('Sale Price ($)')
axes[2].set_title('Bathrooms vs Sale Price')

plt.tight_layout()
plt.savefig('task1_eda.png', dpi=100)
plt.show()
print("EDA plot saved!")

# ── 4. Correlation Heatmap ────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.heatmap(df_model.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('task1_correlation.png', dpi=100)
plt.show()

# ── 5. Train/Test Split ───────────────────────────────────────
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── 6. Train Linear Regression Model ─────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# ── 7. Evaluate Model ─────────────────────────────────────────
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n── Model Performance ──")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  R²   : {r2:.4f}")
print(f"  Accuracy (R²): {r2*100:.2f}%")

coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("\n── Feature Coefficients ──")
print(coef_df.sort_values('Coefficient', ascending=False).to_string(index=False))

# ── 8. Actual vs Predicted Plot ───────────────────────────────
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.tight_layout()
plt.savefig('task1_predictions.png', dpi=100)
plt.show()

# ── 9. Sample Prediction ──────────────────────────────────────
print("\n── Sample Custom Prediction ──")
sample = pd.DataFrame({
    'GrLivArea':     [2000],
    'BedroomAbvGr':  [3],
    'FullBath':      [2],
    'HalfBath':      [1],
    'TotalBsmtSF':   [1000],
    'GarageArea':    [400],
    'YearBuilt':     [2005],
    'OverallQual':   [7]
})
predicted_price = model.predict(sample)[0]
print(f"  Input  : 2000 sqft | 3 bed | 2 bath | Built 2005")
print(f"  Predicted Price: ${predicted_price:,.2f}")
