"""
Day 3 - Customer Segmentation with K-Means Clustering
--------------------------------------------------------
First unsupervised learning project so far (Day 1 and Day 2 were both
supervised - the model was given the "right answer" during training).
Here, there are no labels at all. The model just looks at customer data
(annual income and spending score) and groups similar customers together
on its own.

Concepts covered:
- Unsupervised learning vs supervised learning
- K-Means clustering
- The Elbow Method (how to pick a good number of clusters)
- Silhouette Score (a proper number to measure clustering quality)
- Visualizing clusters with matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it saves a file instead of trying to open a window
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------------------
# 1. Small built-in dataset
# Columns: annual_income (in thousands), spending_score (1-100)
# ---------------------------
data = np.array([
    [15, 39], [16, 81], [17, 6],  [18, 77], [19, 40],
    [20, 76], [21, 6],  [23, 94], [24, 3],  [25, 72],
    [28, 14], [29, 99], [30, 15], [33, 77], [35, 13],
    [37, 75], [40, 50], [42, 52], [45, 48], [46, 55],
    [50, 49], [54, 62], [58, 43], [60, 50], [62, 45],
    [65, 88], [67, 12], [70, 91], [75, 10], [78, 90],
])

feature_names = ["annual_income", "spending_score"]

# ---------------------------
# 2. Scale the data (K-Means uses distance, so features need to be on a similar scale)
# ---------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# ---------------------------
# 3. Elbow Method: try different numbers of clusters and see where the
# improvement starts to flatten out
# ---------------------------
inertia_values = []
silhouette_values = []
k_range = range(1, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(data_scaled)
    inertia_values.append(km.inertia_)
    # silhouette score needs at least 2 clusters to make sense, so skip k=1
    if k >= 2:
        silhouette_values.append(silhouette_score(data_scaled, labels))
    else:
        silhouette_values.append(None)

plt.figure(figsize=(7, 5))
plt.plot(list(k_range), inertia_values, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (lower = tighter clusters)")
plt.title("Elbow Method - finding the right number of clusters")
plt.tight_layout()
plt.savefig("elbow_plot.png")
print("Saved elbow_plot.png")

print("\n=== Silhouette Score for each k (higher is better, max 1.0) ===")
for k, score in zip(k_range, silhouette_values):
    if score is not None:
        print(f"k={k}: silhouette score = {round(score, 3)}")

# ---------------------------
# 4. Based on the elbow plot, 5 clusters looks like a good choice for this data
# ---------------------------
k_final = 5
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(data_scaled)

final_silhouette = silhouette_score(data_scaled, cluster_labels)

print(f"\nUsing k = {k_final} clusters")
print("Cluster assignments:", cluster_labels)
print(f"Final silhouette score: {round(final_silhouette, 3)}")

# ---------------------------
# 5. Visualize the clusters
# ---------------------------
plt.figure(figsize=(7, 5))
colors = ["red", "blue", "green", "purple", "orange"]

for cluster_id in range(k_final):
    cluster_points = data[cluster_labels == cluster_id]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1],
        c=colors[cluster_id], label=f"Cluster {cluster_id}"
    )

plt.xlabel("Annual Income (k)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.tight_layout()
plt.savefig("customer_clusters.png")
print("Saved customer_clusters.png")

# ---------------------------
# 6. Describe each cluster in plain terms
# ---------------------------
print("\n=== Cluster summary ===")
for cluster_id in range(k_final):
    cluster_points = data[cluster_labels == cluster_id]
    avg_income = cluster_points[:, 0].mean()
    avg_spending = cluster_points[:, 1].mean()
    print(f"Cluster {cluster_id}: avg income = {avg_income:.1f}k, "
          f"avg spending score = {avg_spending:.1f}, "
          f"{len(cluster_points)} customers")
