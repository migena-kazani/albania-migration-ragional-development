# -*- coding: utf-8 -*-
"""
Script 15: Table 22 - Correlations Among Key Regional Variables
Panel-level (N=96) and cross-sectional (N=12) correlations.
"""
import json
import numpy as np
from scipy import stats

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    R = json.load(open('../output/regional_data.json'))
    years_net, years_gva, years_unemp = R['years_net'], R['years_gva'], R['years_unemp']
    panel_years = list(range(2016, 2024))

    netmig_panel, gdp_panel, unemp_panel = [], [], []
    for reg in REGIONS:
        for y in panel_years:
            netmig_panel.append(R['net_mig'][reg][years_net.index(y)] * 1000)
            gdp_panel.append(R['gva'][reg][years_gva.index(y)])
            unemp_panel.append(R['unemp'][reg][years_unemp.index(y)])
    netmig_panel, gdp_panel, unemp_panel = map(np.array, (netmig_panel, gdp_panel, unemp_panel))

    print("Table 22. Correlations Among Key Regional Variables\n")
    print("--- Panel-level (N=96) ---")
    for (n1, a), (n2, b) in [(('NetMig', netmig_panel), ('GDP', gdp_panel)),
                              (('NetMig', netmig_panel), ('Unemp', unemp_panel)),
                              (('GDP', gdp_panel), ('Unemp', unemp_panel))]:
        r, p = stats.pearsonr(a, b)
        print(f"{n1} - {n2}: r={r:.3f}, p={p:.2e}")

    mean_netmig = np.array([np.mean(np.array(R['net_mig'][reg][2:10]) * 1000) for reg in REGIONS])
    mean_gdp = np.array([np.mean(R['gva'][reg][16:24]) for reg in REGIONS])
    pop65 = np.array([R['pop65_abs'][reg] for reg in REGIONS])
    gdppc = np.array([R['gdppc2023'][reg] for reg in REGIONS])

    print("\n--- Cross-sectional (N=12) ---")
    for n1, n2, a, b in [('NetMig', 'Pop65+', mean_netmig, pop65),
                          ('NetMig', 'GDPpc', mean_netmig, gdppc),
                          ('GDP', 'Pop65+', mean_gdp, pop65)]:
        r, p = stats.pearsonr(a, b)
        print(f"{n1} - {n2}: r={r:.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
