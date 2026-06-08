# ============================================================
# TASK 02 - Customer Segmentation using K-Means Clustering
# Prodigy Infotech - Machine Learning Internship
# Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────
mall_path = None
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        if f == 'Mall_Customers.csv':
            mall_path = os.path.join(root, f)
            break
    if mall_path:
        break

if mall_path is None:
    raise FileNotFoundError("Mall_Customers.csv not found! Please add the Customer Segmentation dataset.")

df = pd.read_csv(mall_path)
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ── 2. EDA ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df['Age'], bins=20, color='steelblue', edgecolor='white')
axes[0].set_title('Age Distribution')

axes[1].hist(df['Annual Income (k$)'], bins=20, color='coral', edgecolor='white')
axes[1].set_title('Annual Income Distribution')

axes[2].hist(df['Spending Score (1-100)'], bins=20, color='green', edgecolor='white')
axes[2].set_title('Spending Score Distribution')

plt.tight_layout()
plt.savefig('task2_eda.png', dpi=100)
plt.show()

# Gender column name check
gender_col = 'Genre' if 'Genre' in df.columns else 'Gender'
print(f"\nGender Distribution:\n{df[gender_col].value_counts()}")

# ── 3. Feature Selection & Scaling ───────────────────────────
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Elbow Method ───────────────────────────────────────────
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertia, 'bo-', markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].axvline(x=5, color='red', linestyle='--', label='Optimal K=5')
axes[0].legend()

axes[1].plot(K_range, silhouette_scores, 'ro-', markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs K')

plt.tight_layout()
plt.savefig('task2_elbow.png', dpi=100)
plt.show()

# ── 5. Train K-Means ──────────────────────────────────────────
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

# ── 6. Visualize Clusters ─────────────────────────────────────
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

plt.figure(figsize=(9, 6))
for i in range(5):
    mask = df['Cluster'] == i
    plt.scatter(df[mask]['Annual Income (k$)'],
                df[mask]['Spending Score (1-100)'],
                c=colors[i], s=80, alpha=0.7, label=f'Cluster {i}')

centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers_orig[:, 0], centers_orig[:, 1],
            c='black', marker='X', s=200, zorder=5, label='Centroids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments — K-Means Clustering (K=5)')
plt.legend()
plt.tight_layout()
plt.savefig('task2_clusters.png', dpi=100)
plt.show()

# ── 7. Summary ────────────────────────────────────────────────
print("\n── Cluster Summary ──")
summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)
print(summary)

silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
print(f"\nSilhouette Score (K=5): {silhouette_avg:.4f}")

