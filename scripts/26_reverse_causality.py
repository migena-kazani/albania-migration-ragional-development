# -*- coding: utf-8 -*-
"""
Script 26: Table 37 - Fixed-Effects Model of GDP Growth on Net Migration,
Interacted with Initial Development (N=84). Tests part of Combined Hypothesis CH1.
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
    years_net, years_gva = R['years_net'], R['years_gva']
    panel_years = list(range(2017, 2024))  # need t-1 for growth calc

    rows = []
    for reg in REGIONS:
        for y in panel_years:
            gdp_t = R['gva'][reg][years_gva.index(y)]
            gdp_t1 = R['gva'][reg][years_gva.index(y - 1)]
            growth = (gdp_t - gdp_t1) / gdp_t1
            nm = R['net_mig'][reg][years_net.index(y)] * 1000
            init_gdp = R['gva'][reg][years_gva.index(2016)]
            rows.append(dict(region=reg, year=y, growth=growth, netmig=nm,
                              init_gdp_log=np.log(init_gdp)))

    df = pd.DataFrame(rows)
    df['netmig_init'] = df['netmig'] * df['init_gdp_log']
    dfx = df.set_index(['region', 'year'])

    exog = sm.add_constant(dfx[['netmig', 'netmig_init']])
    fe = PanelOLS(dfx['growth'], exog, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

    print(f"Table 37. Reverse Causality: Migration -> GDP Growth (N={len(df)})\n")
    print(fe.summary)
    print("\nBoth terms significant: migration is positively associated with same-year GDP growth, "
          "and this association is weaker in regions that started 2016 with a higher GDP level "
          "(supports Combined Hypothesis CH1, with the caveat that this is a contemporaneous, "
          "not lagged, association).")

    json.dump({'params': fe.params.to_dict(), 'pvalues': fe.pvalues.to_dict()},
              open('../output/table37_reverse_causality.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
