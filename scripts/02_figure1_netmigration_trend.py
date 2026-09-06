# -*- coding: utf-8 -*-
"""
Script 02: Figure 1 - Net migration by region, 2016-2023
Produces: Figure 1. Net migration by region, 2016-2023.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']

def make_figure1():
    R = json.load(open('../output/regional_data.json'))
    years_net = R['years_net']
    panel_years = list(range(2016, 2024))

    colors = {'Tiranë': '#c0392b', 'Durrës': '#e67e22', 'Vlorë': '#16a085'}

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    for reg in REGIONS:
        vals = [R['net_mig'][reg][years_net.index(y)] for y in panel_years]
        c = colors.get(reg, '#95a5a6')
        lw = 2.6 if reg in colors else 1.1
        alpha = 1.0 if reg in colors else 0.75
        ax.plot(panel_years, vals, marker='o', markersize=3.5, linewidth=lw, alpha=alpha,
                color=c, label=reg)
    ax.axhline(0, color='#333333', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Year")
    ax.set_ylabel("Net migration (thousand persons)")
    ax.set_title("Net Migration by Region, 2016\u20132023", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig('../figures/figure1_netmigration_trend.png', bbox_inches='tight')
    plt.close()
    print("Figure 1 saved to ../figures/figure1_netmigration_trend.png")

if __name__ == "__main__":
    make_figure1()
