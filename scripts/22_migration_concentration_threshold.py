# -*- coding: utf-8 -*-
"""
Script 22: Table 33 - CV of Regional GDP by Migration-Concentration Level (N=8 Years)
Tests part of Combined Hypothesis CH5 (threshold effects on inequality).
"""
import json
import numpy as np
import statsmodels.api as sm
from scipy import stats

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    R = json.load(open('../output/regional_data.json'))
    years_net, years_gva = R['years_net'], R['years_gva']
    panel_years = list(range(2016, 2024))

    cv_by_year, conc_by_year = [], []
    for y in panel_years:
        vals_gdp = np.array([R['gva'][reg][years_gva.index(y)] for reg in REGIONS])
        cv_by_year.append(vals_gdp.std(ddof=1) / vals_gdp.mean())
        vals_mig = np.array([R['net_mig'][reg][years_net.index(y)] for reg in REGIONS])
        positive_mig = vals_mig[vals_mig > 0].sum()
        top_region_mig = vals_mig.max()
        conc_by_year.append(top_region_mig / positive_mig if positive_mig > 0 else np.nan)

    cv_by_year, conc = np.array(cv_by_year), np.array(conc_by_year)

    print("Table 33. CV of Regional GDP by Migration-Concentration Level (N=8 Years)\n")
    for y, c, m in zip(panel_years, cv_by_year, conc):
        print(f"{y}: CV={c:.3f}, concentration={m:.3f}")

    median_conc = np.median(conc)
    low = cv_by_year[conc <= median_conc]
    high = cv_by_year[conc > median_conc]
    tstat, pval = stats.ttest_ind(low, high, equal_var=False)

    X = sm.add_constant(conc)
    m = sm.OLS(cv_by_year, X).fit()

    print(f"\nCV when concentration <= median: mean={low.mean():.3f} (N={len(low)})")
    print(f"CV when concentration > median: mean={high.mean():.3f} (N={len(high)})")
    print(f"Welch t-test: t={tstat:.3f}, p={pval:.4f}")
    print(f"Linear regression CV~concentration: slope={m.params[1]:.3f}, p={m.pvalues[1]:.4f}")
    print("\nNeither test is significant; with only 8 years available, this test is likely "
          "underpowered rather than genuinely disproving a threshold effect.")


if __name__ == "__main__":
    main()
