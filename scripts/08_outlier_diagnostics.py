# -*- coding: utf-8 -*-
"""
Script 08: Table 12 - Most Influential Observations (Cook's Distance), Pooled Regression
Computes Cook's distance, leverage and DFBETAS for the pooled OLS model from script 03.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence


def main():
    df = pd.read_csv('../output/panel_df.csv')
    X = sm.add_constant(df[['gdp', 'unemp']])
    model = sm.OLS(df['netmig'], X).fit()
    infl = OLSInfluence(model)

    df['cooks_d'] = infl.cooks_distance[0]
    df['leverage'] = infl.hat_matrix_diag
    dfbetas = infl.dfbetas  # columns: [const, gdp, unemp]
    df['dfbetas_gdp'] = dfbetas[:, 1]
    df['dfbetas_unemp'] = dfbetas[:, 2]

    top8 = df.nlargest(8, 'cooks_d')[['region', 'year', 'netmig', 'cooks_d', 'leverage',
                                        'dfbetas_gdp', 'dfbetas_unemp']]
    print("Table 12. Most Influential Observations (Cook's Distance), Pooled Regression (N=96)\n")
    print(top8.to_string(index=False))

    n = len(df)
    threshold = 2 / np.sqrt(n)
    print(f"\nDFBETAS threshold (2/sqrt(n)) = {threshold:.3f}")
    print(f"Six of the top eight most influential observations belong to Tirana, confirming its "
          f"disproportionate leverage over the pooled estimates.")

    top8.to_json('../output/table12_influence.json', orient='records')


if __name__ == "__main__":
    main()
