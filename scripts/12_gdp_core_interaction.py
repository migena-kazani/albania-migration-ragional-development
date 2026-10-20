# -*- coding: utf-8 -*-
"""
Script 12: Table 18 - Fixed-Effects Model with GDP x Core Interaction (N=96)
Tests Combined Hypothesis CH1 (whether the GDP effect is nonlinear/stronger in the core region).
"""
import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm


def main():
    df = pd.read_csv('../output/panel_df.csv')
    df['gdp_bn'] = df['gdp'] / 1000.0
    df['core'] = (df['region'] == 'Tiranë').astype(int)
    df['gdp_core'] = df['gdp_bn'] * df['core']
    dfx = df.set_index(['region', 'year'])

    exog = sm.add_constant(dfx[['gdp_bn', 'unemp', 'gdp_core']])
    fe = PanelOLS(dfx['netmig'], exog, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)

    print("Table 18. Fixed-Effects Model with GDP x Core Interaction (N=96)\n")
    print(fe.summary)
    print("\nNeither the GDP main effect nor the GDP x Core interaction is individually significant "
          "once both are included together, even though their combined effect for Tirana is close "
          "to the significant single-slope estimate in Table 12. With only one region in the 'core' "
          "category, this specification cannot statistically separate a genuine threshold/nonlinearity "
          "from a single-region fixed effect.")


if __name__ == "__main__":
    main()
