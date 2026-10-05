# -*- coding: utf-8 -*-
"""
Script 19: Table 29, Figure 6 - Regional Economic Inequality Indicators
CV and Theil index of regional GDP, 2000-2023.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    R = json.load(open('../output/regional_data.json'))
    years_gva = R['years_gva']

    cv_by_year, theil_by_year = {}, {}
    for i, y in enumerate(years_gva):
        vals = np.array([R['gva'][reg][i] for reg in REGIONS])
        cv_by_year[y] = vals.std(ddof=1) / vals.mean()
        share = vals / vals.sum()
        n = len(vals)
        theil_by_year[y] = np.sum(share * np.log(share * n))

    print("Table 29. Regional Economic Inequality Indicators, Selected Years (2000-2023)\n")
    print(f"{'Year':6s} {'CV':>8s} {'Theil':>8s}")
    for y in [2000, 2005, 2010, 2014, 2016, 2018, 2020, 2022, 2023]:
        print(f"{y:6d} {cv_by_year[y]:8.3f} {theil_by_year[y]:8.3f}")

    json.dump({'years': years_gva, 'cv': [cv_by_year[y] for y in years_gva],
               'theil': [theil_by_year[y] for y in years_gva]},
              open('../output/inequality.json', 'w'))

    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=200)
    ax1.plot(years_gva, [cv_by_year[y] for y in years_gva], color='#8e44ad', marker='o',
              markersize=3, linewidth=1.8, label='Coefficient of Variation (CV)')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Coefficient of Variation (CV)", color='#8e44ad')
    ax1.tick_params(axis='y', labelcolor='#8e44ad')
    ax2 = ax1.twinx()
    ax2.plot(years_gva, [theil_by_year[y] for y in years_gva], color='#27ae60', marker='s',
              markersize=3, linewidth=1.8, label='Theil Index')
    ax2.set_ylabel("Theil Index", color='#27ae60')
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    ax1.set_title("Rising Economic Inequality Across Regions, 2000\u20132023", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/figure6_inequality_trend.png', bbox_inches='tight')
    plt.close()
    print("\nFigure 6 saved.")


if __name__ == "__main__":
    main()
