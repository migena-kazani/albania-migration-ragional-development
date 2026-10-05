# -*- coding: utf-8 -*-
"""
Script 16: Table 23, Figure 4 - Clustering Results (k-means, k=3)
"""
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    R = json.load(open('../output/regional_data.json'))

    mean_netmig = np.array([np.mean(np.array(R['net_mig'][reg][2:10]) * 1000) for reg in REGIONS])
    mean_gdp = np.array([np.mean(R['gva'][reg][16:24]) for reg in REGIONS])
    mean_unemp = np.array([np.mean(R['unemp'][reg][0:8]) for reg in REGIONS])

    X = np.column_stack([mean_netmig, mean_gdp, mean_unemp])
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(Xs)

    cluster_names = {0: "Periphery", 1: "Core", 2: "Coastal/Transitional"}
    print("Table 23. Clustering Results (k-means, k=3)\n")
    for reg, lab, nm, gdp, un in zip(REGIONS, km.labels_, mean_netmig, mean_gdp, mean_unemp):
        print(f"{reg:15s} {cluster_names[lab]:22s} netmig={nm:8.0f}  gdp={gdp:10.0f}  unemp={un:5.1f}")

    json.dump({'regions': REGIONS, 'labels': km.labels_.tolist(), 'mean_netmig': mean_netmig.tolist(),
               'mean_gdp': mean_gdp.tolist(), 'mean_unemp': mean_unemp.tolist()},
              open('../output/clustering.json', 'w'))

    # Figure 4: scatter of GDP vs net migration coloured by cluster
    cluster_colors = {0: '#95a5a6', 1: '#c0392b', 2: '#2980b9'}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    for reg, lab, x, y in zip(REGIONS, km.labels_, mean_gdp, mean_netmig):
        ax.scatter(x, y, s=110, color=cluster_colors[lab], edgecolor='white', linewidth=0.8, zorder=3)
        ax.annotate(reg, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    b1, b0 = np.polyfit(mean_gdp, mean_netmig, 1)
    xx = np.linspace(mean_gdp.min() * 0.9, mean_gdp.max() * 1.05, 50)
    ax.plot(xx, b0 + b1 * xx, color='#2c3e50', linestyle='--', linewidth=1.2, zorder=2)
    ax.set_xlabel("Average regional GDP (million ALL, 2016-2023)")
    ax.set_ylabel("Average net migration (persons/year)")
    ax.set_title("Net Migration and Regional GDP: Core-Periphery Structure", fontsize=12, fontweight='bold')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[k],
                           markersize=9, label=v) for k, v in cluster_names.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig('../figures/figure4_gdp_migration_clusters.png', bbox_inches='tight')
    plt.close()
    print("\nFigure 4 saved.")


if __name__ == "__main__":
    main()
