# -*- coding: utf-8 -*-
"""
Script 24: Figure 8 - National Net Migration, Albania 1960-2024
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    N = json.load(open('../output/national_data.json'))
    years, nm = N['years'], N['netmig']

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    ax.plot(years, nm, color='#2980b9', linewidth=1.6)
    ax.axhline(0, color='#333333', linewidth=0.8, linestyle='--')
    ax.fill_between(years, nm, 0, where=[v is not None and v < 0 for v in nm], color='#e74c3c', alpha=0.15)
    ax.fill_between(years, nm, 0, where=[v is not None and v >= 0 for v in nm], color='#2ecc71', alpha=0.15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Net migration (persons)")
    ax.set_title("National Net Migration, Albania 1960\u20132024", fontsize=12, fontweight='bold')
    ax.annotate("1990s transition crisis\nand mass emigration waves", xy=(1991, -63689),
                xytext=(1975, -45000), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#555555'))
    plt.tight_layout()
    plt.savefig('../figures/figure8_national_netmig_timeseries.png', bbox_inches='tight')
    plt.close()
    print("Figure 8 saved.")


if __name__ == "__main__":
    main()
