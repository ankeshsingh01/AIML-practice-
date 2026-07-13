# Day 3 - Customer Segmentation with K-Means Clustering

## What kind of problem is this?

This is my first **unsupervised learning** project. Big difference from Day 1 and Day 2:

- **Day 1 (spam classifier)** and **Day 2 (house price)** were both **supervised learning** - I gave the model examples with the "correct answer" already attached (this message IS spam, this house costs THIS much), and the model learned to predict that answer for new data.
- **Day 3 (this one)** has NO correct answers at all. I just gave the model raw customer data (income, spending score) and asked it to find groups/patterns on its own. Nobody told it what the groups should be - it figured that out by itself.

This matters because it changes literally everything about how you evaluate the model. There's no "right answer" to check against, so accuracy/precision/recall/F1 (which I used on Day 1) don't apply here at all. More on that below.

## The dataset

30 customers, 2 features each:
- `annual_income` (in thousands)
- `spending_score` (1 to 100, how much they spend relative to their income)

Small and made-up, but the pattern is realistic - some people earn a lot but spend little, some earn little but spend a lot, etc.

## Step 1: Scaling the data

K-Means groups points based on **distance** (how close points are to each other). If one feature is on a totally different scale than another, it can dominate the distance calculation unfairly. Here both features happen to be roughly similar scale already, but I used `StandardScaler` anyway since it's good practice - it transforms every feature to have mean 0 and standard deviation 1, so no feature accidentally "outweighs" another just because of its units.

## Step 2: The Elbow Method (picking k)

K-Means needs you to say upfront how many clusters (`k`) to look for - it doesn't figure that out on its own. So the question becomes: how do I know the right k?

**Inertia** = sum of squared distances from each point to its cluster's center. Lower inertia = tighter, more compact clusters. But here's the catch: inertia ALWAYS goes down as you add more clusters (in the extreme, if k = number of data points, every point is its own cluster and inertia = 0). So you can't just pick the k with the lowest inertia - you'd always pick the biggest possible k, which is useless.

The Elbow Method: plot inertia against k, and look for the point where the curve stops dropping sharply and starts to flatten out (looks like an elbow/bend in the graph). That bend is roughly where adding more clusters stops giving you much benefit. Ran k = 1 through 8, plotted it, saved as `elbow_plot.png`.

## Step 3: Silhouette Score (a proper number, not just eyeballing a graph)

The elbow method is useful but it's visual and a bit subjective - reading "where the bend is" from a graph isn't precise. So I also calculated **Silhouette Score** for every k from 2 to 8.

What silhouette score actually measures, in plain terms: for every point, it compares (a) how close that point is to others in its OWN cluster, vs (b) how close it is to points in the NEAREST other cluster. If points are much closer to their own cluster than to any other cluster, the score is high (close to +1) - that's good, means clusters are well separated. If clusters overlap or points are just as close to another cluster as their own, the score drops toward 0 or even negative.

Score range: -1 (bad) to +1 (perfect separation).

Results I got:
```
k=2: 0.342
k=3: 0.501
k=4: 0.562
k=5: 0.606   <-- highest
k=6: 0.605
k=7: 0.573
k=8: 0.546
```

k=5 had the highest silhouette score (0.606), and it was basically tied with k=6. This actually matched what the elbow graph was suggesting too, which was reassuring - two different methods agreeing on the same answer gives more confidence than just eyeballing one graph.

## Step 4: Final clustering (k=5) and what the groups actually mean

Ran K-Means with k=5 and looked at the average income/spending per cluster:

| Cluster | Avg Income | Avg Spending Score | Count | What this probably means |
|---|---|---|---|---|
| 0 | 25.1k | 81.4 | 8 | Low income, high spending - lives beyond means / impulsive spenders |
| 1 | 50.8k | 50.4 | 9 | Mid income, mid spending - the "average" customer |
| 2 | 23.6k | 17.0 | 8 | Low income, low spending - careful/frugal |
| 3 | 71.0k | 11.0 | 2 | High income, low spending - saves a lot, doesn't spend much despite having money |
| 4 | 71.0k | 89.7 | 3 | High income, high spending - big spenders, ideal target for premium products |

This is exactly the kind of output a real marketing/business team would use - instead of sending the same offer to every customer, target Cluster 4 with premium products, Cluster 0 with payment plans, Cluster 2/3 with savings-related offers, etc.

## Why accuracy/precision/recall/F1 don't apply here

Kept asking myself this so writing it down clearly: those metrics all require knowing the TRUE label for each data point so you can check if the prediction matches it. In clustering, there IS no true label - "Cluster 0" isn't a real category that existed before I ran the algorithm, it's just a group the algorithm invented. So there's nothing to check the prediction "against." That's exactly why clustering needs its own separate metrics (inertia, silhouette score) that only look at how well-separated and compact the groups are internally, rather than comparing to ground truth.

Quick reference table for myself:

| ML Type | Example from my projects | Metrics used |
|---|---|---|
| Classification | Day 1 - Spam classifier | Accuracy, Precision, Recall, F1 |
| Regression | Day 2 - House price predictor | R² score, MAE |
| Clustering (unsupervised) | Day 3 - Customer segmentation | Inertia, Silhouette Score |

## Stack
Python, Scikit-learn (KMeans, StandardScaler, silhouette_score), Matplotlib, NumPy

## Running it
```bash
pip install scikit-learn matplotlib numpy
python customer_segmentation.py
```

Outputs:
- `elbow_plot.png` - inertia vs k
- `customer_clusters.png` - final 5 clusters, color coded
- Console prints silhouette score for every k, the final chosen k's silhouette score, and a plain-English summary of each cluster

## What I'd try next
- Real dataset (Mall Customer Segmentation on Kaggle) with way more than 30 rows
- Try DBSCAN and compare - it doesn't require picking k upfront, handles weird-shaped clusters better, and can mark outliers as noise instead of forcing them into a cluster
- Add a 3rd feature (like age) and see how/if the clusters change
- Try Hierarchical Clustering and compare its dendrogram to these results
