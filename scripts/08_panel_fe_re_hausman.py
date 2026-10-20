# -*- coding: utf-8 -*-
"""
Script 08: Tables 12–13 - Panel Fixed-Effects, Random-Effects Estimation and the
Hausman Specification Test.
Produces:
  - Table 12. Pooled, Fixed-Effects and Random-Effects Estimates of Net Migration
    on Regional GDP and Unemployment
  - Table 13. Inputs to the Hausman Specification Test (Fixed Effects vs. Random Effects)
"""
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
import statsmodels.api as sm
from scipy.stats import chi2

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    R = json.load(open('../output/regional_data.json'))
    years_net, years_gva, years_unemp = R['years_net'], R['years_gva'], R['years_unemp']
    panel_years = list(range(2016, 2024))

    rows = []
    for reg in REGIONS:
        for y in panel_years:
            nm = R['net_mig'][reg][years_net.index(y)] * 1000
            gdp = R['gva'][reg][years_gva.index(y)] / 1000.0  # billion ALL
            un = R['unemp'][reg][years_unemp.index(y)]
            rows.append(dict(region=reg, year=y, netmig=nm, gdp_bn=gdp, unemp=un))
    df = pd.DataFrame(rows).set_index(['region', 'year'])
    exog = sm.add_constant(df[['gdp_bn', 'unemp']])

    pooled = PooledOLS(df['netmig'], exog).fit(cov_type='clustered', cluster_entity=True)
    fe = PanelOLS(df['netmig'], exog, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    re = RandomEffects(df['netmig'], exog).fit(cov_type='clustered', cluster_entity=True)

    print("=== Table 12. Pooled / FE / RE Estimates ===")
    print("\n--- Pooled OLS (cluster-robust) ---")
    print(pooled.summary)
    print("\n--- Fixed Effects (cluster-robust) ---")
    print(fe.summary)
    print("\n--- Random Effects (cluster-robust) ---")
    print(re.summary)

    # Classic (non-robust) FE and RE for the Hausman test itself
    fe_c = PanelOLS(df['netmig'], exog, entity_effects=True).fit()
    re_c = RandomEffects(df['netmig'], exog).fit()
    b_fe = fe_c.params.drop('const', errors='ignore')
    b_re = re_c.params.drop('const', errors='ignore')
    common = b_fe.index.intersection(b_re.index)
    b_fe, b_re = b_fe[common], b_re[common]
    v_fe = fe_c.cov.loc[common, common]
    v_re = re_c.cov.loc[common, common]
    diff = b_fe - b_re
    var_diff = v_fe - v_re
    stat = float(diff.values @ np.linalg.inv(var_diff.values) @ diff.values.T)
    dof = len(common)
    pval = 1 - chi2.cdf(stat, dof)

    print(f"\n=== Table 13. Hausman Test ===")
    print(f"chi2({dof}) = {stat:.2f}, p = {pval:.6f}")
    print(f"FE coefficients: {fe_c.params.to_dict()}")
    print(f"RE coefficients: {re_c.params.to_dict()}")

    json.dump({
        'fe_params': fe.params.to_dict(), 'fe_se': fe.std_errors.to_dict(),
        're_params': re.params.to_dict(), 're_se': re.std_errors.to_dict(),
        'hausman_stat': stat, 'hausman_dof': dof, 'hausman_p': pval,
    }, open('../output/tables10_11_fe_re_hausman.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
