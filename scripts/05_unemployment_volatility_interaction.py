# -*- coding: utf-8 -*-
"""
Script 05: Table 8 - Fixed-Effects Model with Unemployment x GDP-Growth-Volatility
Interaction (N=96). Tests Combined Hypothesis CH2 (mediation of unemployment's
effect by the reliability of the growth signal).
"""
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    R = json.load(open('../output/regional_data.json'))
    years_net, years_gva, years_unemp = R['years_net'], R['years_gva'], R['years_unemp']
    panel_years = list(range(2016, 2024))

    # region-specific GDP growth volatility (std of year-on-year growth, 2016-2023)
    volatility = {}
    for reg in REGIONS:
        vals = [R['gva'][reg][years_gva.index(y)] for y in panel_years]
        growth = np.diff(vals) / np.array(vals[:-1])
        volatility[reg] = np.std(growth, ddof=1)

    rows = []
    for reg in REGIONS:
        for y in panel_years:
            nm = R['net_mig'][reg][years_net.index(y)] * 1000
            gdp = R['gva'][reg][years_gva.index(y)] / 1000.0
            un = R['unemp'][reg][years_unemp.index(y)]
            rows.append(dict(region=reg, year=y, netmig=nm, gdp_bn=gdp, unemp=un,
                              volatility=volatility[reg]))

    df = pd.DataFrame(rows)
    df['unemp_vol'] = df['unemp'] * df['volatility']
    dfx = df.set_index(['region', 'year'])

    # volatility itself is time-invariant per region and is absorbed by entity fixed effects
    exog = sm.add_constant(dfx[['gdp_bn', 'unemp', 'unemp_vol']])
    fe = PanelOLS(dfx['netmig'], exog, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

    print("Table 8. Fixed-Effects Model with Unemployment x GDP-Growth-Volatility Interaction (N=96)\n")
    print(fe.summary)
    print("\nRegion-level GDP growth volatility (2016-2023):")
    for r, v in volatility.items():
        print(f"  {r}: {v:.4f}")

    json.dump({'params': fe.params.to_dict(), 'pvalues': fe.pvalues.to_dict(),
               'se': fe.std_errors.to_dict(), 'rsq': fe.rsquared, 'volatility': volatility},
              open('../output/table8_unemp_volatility.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
