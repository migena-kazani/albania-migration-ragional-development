# -*- coding: utf-8 -*-
"""
Script 15: Tables 20-21, Figure 4 - Cluster Validation
Produces:
  - Table 20. Cluster Validation Indices, k-means and Hierarchical (Ward) Clustering
  - Table 21. Region-by-Region Comparison of k-means and Hierarchical (Ward)
    Cluster Assignments (k=3)
  - Figure 4. Hierarchical (Ward-linkage) clustering dendrogram
"""
import json
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    cl = json.load(open('../output/clustering.json'))
    X = np.column_stack([cl['mean_netmig'], cl['mean_gdp'], cl['mean_unemp']])
    Xs = StandardScaler().fit_transform(X)

    print("Table 20. Cluster Validation Indices (k=2..5)\n")
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xs)
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(Xs)
        sil_km, db_km = silhouette_score(Xs, km.labels_), davies_bouldin_score(Xs, km.labels_)
        sil_hc, db_hc = silhouette_score(Xs, hc.labels_), davies_bouldin_score(Xs, hc.labels_)
        print(f"k={k}: k-means silhouette={sil_km:.3f}, DB={db_km:.3f} | "
              f"Ward silhouette={sil_hc:.3f}, DB={db_hc:.3f}")

    km3 = KMeans(n_clusters=3, n_init=20, random_state=42).fit(Xs)
    hc3 = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(Xs)
    ari = adjusted_rand_score(km3.labels_, hc3.labels_)

    # align Ward's arbitrary labels to k-means' semantic labels by matching cluster membership
    km_map = {0: "Periphery", 1: "Core", 2: "Coastal/Transitional"}
    # find which ward label corresponds to which km group by majority overlap
    ward_to_km = {}
    for wl in set(hc3.labels_):
        idx = hc3.labels_ == wl
        # majority k-means label among these members
        majority = np.bincount(km3.labels_[idx]).argmax()
        ward_to_km[wl] = majority

    print(f"\nTable 21. Region-by-Region Comparison (Adjusted Rand Index = {ari:.1f})\n")
    for reg, kml, wl in zip(REGIONS, km3.labels_, hc3.labels_):
        print(f"{reg:15s} k-means={km_map[kml]:22s} Ward={km_map[ward_to_km[wl]]}")

    # Figure 4: dendrogram
    Z = linkage(Xs, method='ward')
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    dendrogram(Z, labels=REGIONS, ax=ax, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Distance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../figures/figure4_dendrogram.png', bbox_inches='tight')
    plt.close()
    print("\nFigure 4 saved.")


if __name__ == "__main__":
    main()
