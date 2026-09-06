# -*- coding: utf-8 -*-
"""
Script 09: Table 13 - Fixed-Effects Sensitivity Analysis: Dropping Influential Regions
"""
import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm


def refit(df, drop_region=None):
    sub = df[df['region'] != drop_region] if drop_region else df
    sub = sub.copy()
    sub['gdp_bn'] = sub['gdp'] / 1000.0
    sub = sub.set_index(['region', 'year'])
    exog = sm.add_constant(sub[['gdp_bn', 'unemp']])
    fe = PanelOLS(sub['netmig'], exog, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    return fe


def main():
    df = pd.read_csv('../output/panel_df.csv')

    print("Table 13. Fixed-Effects Sensitivity Analysis: Dropping Influential Regions\n")
    for drop in [None, 'Tiranë', 'Durrës']:
        fe = refit(df, drop)
        label = f"Excluding {drop}" if drop else "Full sample"
        n = fe.nobs
        print(f"--- {label} (N={n}) ---")
        print(f"GDP coefficient (per bn ALL): {fe.params['gdp_bn']:.2f}, p-value: {fe.pvalues['gdp_bn']:.4f}")
        print(f"Unemployment coefficient: {fe.params['unemp']:.2f}, p-value: {fe.pvalues['unemp']:.4f}\n")


if __name__ == "__main__":
    main()
