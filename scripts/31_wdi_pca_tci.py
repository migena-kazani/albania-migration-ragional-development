# -*- coding: utf-8 -*-
"""
Script 31: Table 44, Figure 10 - Territorial Capital Index (PCA)
Produces:
  - Table 44. Principal Component Analysis Loadings on the First Component
  - Figure 10. Territorial Capital Index, Albania 1996-2024
"""
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WANTED = {
    'GOV_WGI_VA_EST': 'wgi_voice_accountability', 'GOV_WGI_RQ_EST': 'wgi_regulatory_quality',
    'GOV_WGI_RL_EST': 'wgi_rule_of_law', 'GOV_WGI_PV_EST': 'wgi_political_stability',
    'GOV_WGI_GE_EST': 'wgi_govt_effectiveness', 'GOV_WGI_CC_EST': 'wgi_control_corruption',
    'SH.XPD.CHEX.GD.ZS': 'health_exp_pct_gdp', 'SP.DYN.LE00.IN': 'life_expectancy',
    'SE.SEC.ENRR': 'secondary_enroll_gross_pct', 'IT.NET.USER.ZS': 'internet_pct_pop',
    'SE.XPD.TOTL.GD.ZS': 'edu_exp_pct_gdp',
}


def main():
    xl = pd.read_excel('../data/API_ALB_DS2_en_excel_v2_2902.xls', sheet_name='Data', header=3)
    year_cols = [c for c in xl.columns if str(c).isdigit()]

    data = {}
    for code, name in WANTED.items():
        row = xl[xl['Indicator Code'] == code].iloc[0]
        data[name] = {yc: row[yc] for yc in year_cols}

    years_i = [int(y) for y in year_cols]
    dfw = pd.DataFrame({name: [data[name].get(str(y)) for y in years_i] for name in data}, index=years_i)
    dfw = dfw.apply(pd.to_numeric, errors='coerce')
    comp_vars = list(WANTED.values())
    sub = dfw[comp_vars].copy()
    sub_years = sub[sub.isna().sum(axis=1) <= 3].index
    sub = sub.loc[sub_years].interpolate(limit_direction='both').dropna()

    Xs = StandardScaler().fit_transform(sub.values)
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(Xs)
    sign = 1 if pca.components_[0][comp_vars.index('wgi_govt_effectiveness')] > 0 else -1
    loadings = dict(zip(comp_vars, (sign * pca.components_[0]).tolist()))
    tci = (sign * pcs[:, 0])

    print(f"Table 44. PCA Loadings on PC1 (explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance)\n")
    for k, v in loadings.items():
        print(f"  {k}: {v:.4f}")

    json.dump({'loadings': loadings, 'explained_var_pc1': pca.explained_variance_ratio_[0],
               'years': sub.index.tolist(), 'tci': tci.tolist()},
              open('../output/tci_pca.json', 'w'), indent=2)

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    ax.plot(sub.index.tolist(), tci, color='#16a085', marker='o', markersize=3.5, linewidth=1.8)
    ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Year")
    ax.set_ylabel("Territorial Capital Index (PC1, standardized)")
    ax.set_title("Territorial Capital Index, Albania 1996\u20132024", fontsize=11.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/figure10_territorial_capital_index.png', bbox_inches='tight')
    plt.close()
    print("\nFigure 10 saved.")


if __name__ == "__main__":
    main()
