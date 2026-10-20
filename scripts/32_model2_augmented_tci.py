# -*- coding: utf-8 -*-
"""
Script 32: Tables 45–47 - Model 2 Augmented with the Territorial Capital Index
Produces:
  - Table 45. Model 2 Augmented with the Territorial Capital Index (N=25, 1996-2023)
  - Table 46. Multicollinearity Diagnostics: Condition Number and VIF
  - Table 47. Bivariate Correlations of the Territorial Capital Index (N=26)
"""
import json
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats


def main():
    N = json.load(open('../output/national_data.json'))
    tci_data = json.load(open('../output/tci_pca.json'))
    tci_years = tci_data['years']
    tci_map = dict(zip(tci_years, tci_data['tci']))

    years = N['years']
    gdppc_dict = {y: v for y, v in zip(years, N['gdppc_lcu'])}
    unemp_dict = {y: v for y, v in zip(years, N['unemp_nat'])}
    urban_dict = {y: v for y, v in zip(years, N['urban_abs'])}
    netmig_dict = {y: v for y, v in zip(years, N['netmig'])}

    common = [y for y in tci_years if gdppc_dict.get(y) is not None and unemp_dict.get(y) is not None
              and urban_dict.get(y) is not None and netmig_dict.get(y) is not None]

    y_arr = np.array([netmig_dict[yr] for yr in common], dtype=float)
    x1 = np.array([gdppc_dict[yr] for yr in common], dtype=float)
    x2 = np.array([unemp_dict[yr] for yr in common], dtype=float)
    x3 = np.array([urban_dict[yr] for yr in common], dtype=float)
    x4 = np.array([tci_map[yr] for yr in common], dtype=float)

    X_with = sm.add_constant(np.column_stack([x1, x2, x3, x4]))
    X_without = sm.add_constant(np.column_stack([x1, x2, x3]))
    model_with = sm.OLS(y_arr, X_with).fit(cov_type='HC1')
    model_without = sm.OLS(y_arr, X_without).fit(cov_type='HC1')

    print(f"Table 45. Model 2 Augmented with TCI (N={len(common)})\n")
    print("--- With TCI ---")
    print(model_with.summary(xname=['const', 'GDPpc', 'Unemp', 'Urban', 'TCI']))
    print("\n--- Without TCI (for comparison) ---")
    print(model_without.summary(xname=['const', 'GDPpc', 'Unemp', 'Urban']))

    cond_with = np.linalg.cond(X_with)
    cond_without = np.linalg.cond(X_without)
    vif_with = {nm: variance_inflation_factor(X_with, i) for i, nm in
                enumerate(['const', 'gdppc', 'unemp', 'urban', 'tci'])}
    vif_without = {nm: variance_inflation_factor(X_without, i) for i, nm in
                   enumerate(['const', 'gdppc', 'unemp', 'urban'])}

    print(f"\nTable 46. Multicollinearity Diagnostics")
    print(f"Condition number WITH TCI: {cond_with:,.0f}")
    print(f"Condition number WITHOUT TCI: {cond_without:,.0f}")
    print(f"VIF with TCI: {vif_with}")
    print(f"VIF without TCI: {vif_without}")

    r_gdp, p_gdp = stats.pearsonr(np.array([tci_map[y] for y in common]), x1)
    r_urb, p_urb = stats.pearsonr(np.array([tci_map[y] for y in common]), x3)
    print(f"\nTable 47. Bivariate Correlations of TCI (N={len(common)}, restricted to Model-2 sample)")
    print(f"TCI - GDP per Capita: r={r_gdp:.3f}, p={p_gdp:.2e}")
    print(f"TCI - Urban Population: r={r_urb:.3f}, p={p_urb:.2e}")

    # Paper's Table 47 uses the full N=26 TCI series (not restricted to the Model 2 N=25 subsample)
    common_full = [y for y in tci_years if gdppc_dict.get(y) is not None and urban_dict.get(y) is not None]
    x1_full = np.array([gdppc_dict[y] for y in common_full], dtype=float)
    x3_full = np.array([urban_dict[y] for y in common_full], dtype=float)
    tci_full = np.array([tci_map[y] for y in common_full])
    r_gdp_f, p_gdp_f = stats.pearsonr(tci_full, x1_full)
    r_urb_f, p_urb_f = stats.pearsonr(tci_full, x3_full)
    print(f"\n(As reported in the paper, full TCI series, N={len(common_full)}, 1996-2023)")
    print(f"TCI - GDP per Capita: r={r_gdp_f:.3f}, p={p_gdp_f:.2e}")
    print(f"TCI - Urban Population: r={r_urb_f:.3f}, p={p_urb_f:.2e}")


if __name__ == "__main__":
    main()
