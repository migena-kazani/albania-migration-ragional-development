# -*- coding: utf-8 -*-
"""
Script 23: Table 34, Figures 7-8 - Beta-Convergence Regression
Produces:
  - Table 34. Beta-Convergence Regression: Annualized GDP Growth (2016-2023) on
    Log Initial GDP (2016) (N=12 Regions)
  - Figure 7. Beta-divergence across Albania's twelve regions
  - Figure 8. Gini, Atkinson and Williamson indices chart
"""
import json
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    R = json.load(open('../output/regional_data.json'))
    years_gva = R['years_gva']
    i16, i23 = years_gva.index(2016), years_gva.index(2023)
    init = np.array([R['gva'][reg][i16] for reg in REGIONS])
    fin = np.array([R['gva'][reg][i23] for reg in REGIONS])
    growth = np.log(fin / init) / 7

    X = sm.add_constant(np.log(init))
    model = sm.OLS(growth, X).fit()

    print("Table 34. Beta-Convergence Regression (N=12)\n")
    print(f"log(Initial GDP) coefficient = {model.params[1]:.4f}, SE = {model.bse[1]:.4f}, "
          f"p = {model.pvalues[1]:.4f}")
    print(f"R2 = {model.rsquared:.3f}, F = {model.fvalue:.2f}, p(F) = {model.f_pvalue:.4f}")
    print("\nPositive, significant coefficient = beta-divergence, not convergence: regions richer "
          "in 2016 grew faster over 2016-2023, opposite to neoclassical growth theory's prediction.")

    # Figure 7: beta-divergence scatter
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    ax.scatter(np.log(init), growth, s=90, color='#c0392b', zorder=3)
    for reg, x, y in zip(REGIONS, np.log(init), growth):
        ax.annotate(reg, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    b1, b0 = np.polyfit(np.log(init), growth, 1)
    xx = np.linspace(np.log(init).min() * 0.98, np.log(init).max() * 1.02, 50)
    ax.plot(xx, b0 + b1 * xx, '--', color='#2c3e50')
    ax.set_xlabel("Log Initial GDP (2016)")
    ax.set_ylabel("Annualized GDP Growth Rate, 2016-2023")
    ax.set_title("Beta-Divergence Across Albanian Regions", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/figure7_beta_convergence.png', bbox_inches='tight')
    plt.close()

    # Figure 8: Gini/Atkinson/Williamson chart
    ineq = json.load(open('../output/inequality_extra.json'))
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    ax.plot(ineq['years'], ineq['gini'], label='Gini', color='#8e44ad', linewidth=1.8)
    ax.plot(ineq['years'], ineq['atkinson'], label='Atkinson (e=1)', color='#e67e22', linewidth=1.8)
    ax.plot(ineq['years'], ineq['williamson'], label='Williamson', color='#16a085', linewidth=1.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Index value")
    ax.set_title("Additional Regional Inequality Indices, 2000-2023", fontsize=12, fontweight='bold')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('../figures/figure8_inequality_indices.png', bbox_inches='tight')
    plt.close()
    print("\nFigures 7 and 8 saved.")

    json.dump({'slope': model.params[1], 'p': model.pvalues[1], 'r2': model.rsquared},
              open('../output/beta_convergence.json', 'w'))


if __name__ == "__main__":
    main()
