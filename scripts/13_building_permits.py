# -*- coding: utf-8 -*-
"""
Script 13: Figure 3, Tables 19–20 - Building Permits as an Infrastructure Investment Proxy
Produces:
  - Figure 3. Building permits issued for new buildings by region, 2020-2024 (INSTAT).
  - Table 19. Fixed-Effects Model with Building Permits (log), 2020-2023 (N=48)
  - Table 20. Non-Fixed-Effects (Pooled) Relationship Between Building Permits and
    Net Migration, 2020-2023 (N=48)
"""
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def extract_permits():
    df = pd.read_excel('../data/lejet-e-ndërtimit-miratuar-për-ndërtesa-të-reja-sipas-qarqeve-2020-2024.xlsx',
                        sheet_name='sheet1', header=None)
    years_bp = [2020, 2021, 2022, 2023, 2024]
    permits = {}
    for i in range(6, 18):
        region = df.iloc[i, 0].strip()
        vals = df.iloc[i, 1:6].tolist()
        permits[region] = dict(zip(years_bp, vals))
    json.dump({'years': years_bp, 'permits': permits}, open('../output/building_permits.json', 'w'),
               ensure_ascii=False)
    return permits


def make_figure2(permits):
    years = [2020, 2021, 2022, 2023, 2024]
    colors = {'Tiranë': '#c0392b', 'Durrës': '#e67e22', 'Vlorë': '#16a085'}
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    for reg in REGIONS:
        vals = [permits[reg][y] for y in years]
        c = colors.get(reg, '#95a5a6')
        lw = 2.6 if reg in colors else 1.1
        ax.plot(years, vals, marker='o', markersize=3.5, linewidth=lw, color=c, label=reg,
                alpha=1.0 if reg in colors else 0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Building permits issued (new buildings)")
    ax.set_title("Building Permits by Region, 2020\u20132024", fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig('../figures/figure3_building_permits.png', bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")


def main():
    permits = extract_permits()
    make_figure2(permits)

    R = json.load(open('../output/regional_data.json'))
    years_net, years_gva, years_unemp = R['years_net'], R['years_gva'], R['years_unemp']
    bp_years = [2020, 2021, 2022, 2023]

    rows = []
    for reg in REGIONS:
        for y in bp_years:
            nm = R['net_mig'][reg][years_net.index(y)] * 1000
            gdp = R['gva'][reg][years_gva.index(y)] / 1000.0
            un = R['unemp'][reg][years_unemp.index(y)]
            bp = permits[reg][y]
            rows.append(dict(region=reg, year=y, netmig=nm, gdp_bn=gdp, unemp=un, permits=bp))

    df = pd.DataFrame(rows)
    df['permits_log'] = np.log1p(df['permits'])
    dfx = df.set_index(['region', 'year'])

    exog = sm.add_constant(dfx[['gdp_bn', 'unemp', 'permits_log']])
    fe = PanelOLS(dfx['netmig'], exog, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    print("\nTable 19. Fixed-Effects Model with Building Permits (log), 2020-2023 (N=48)\n")
    print(fe.summary)

    r_raw, p_raw = stats.pearsonr(df['permits'], df['netmig'])
    X_pooled = sm.add_constant(df['permits'])
    pooled_model = sm.OLS(df['netmig'], X_pooled).fit()
    print(f"\nTable 20. Non-FE (Pooled) Relationship, N={len(df)}")
    print(f"Pearson r = {r_raw:.3f}, p = {p_raw:.6f}")
    print(f"Pooled OLS slope = {pooled_model.params['permits']:.2f}, "
          f"SE = {pooled_model.bse['permits']:.3f}, R2 = {pooled_model.rsquared:.3f}")

    json.dump({'fe_params': fe.params.to_dict(), 'fe_pvalues': fe.pvalues.to_dict(),
               'r_raw': r_raw, 'p_raw': p_raw, 'pooled_slope': pooled_model.params['permits'],
               'pooled_r2': pooled_model.rsquared},
              open('../output/tables15_16_permits.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
