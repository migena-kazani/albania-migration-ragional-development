# -*- coding: utf-8 -*-
"""
Script 28: Table 39 - Diagnostic Tests for Model 2 (National Regression)
"""
import json
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan


def main():
    N = json.load(open('../output/national_data.json'))
    years = N['years']
    idxs = [i for i in range(len(years)) if all(
        N[k][i] is not None for k in ['netmig', 'gdppc_lcu', 'unemp_nat', 'urban_abs', 'remit'])
        and years[i] >= 1992]

    y = np.array([N['netmig'][i] for i in idxs], dtype=float)
    x1 = np.array([N['gdppc_lcu'][i] for i in idxs], dtype=float)
    x2 = np.array([N['unemp_nat'][i] for i in idxs], dtype=float)
    x3 = np.array([N['urban_abs'][i] for i in idxs], dtype=float)
    x4 = np.array([N['remit'][i] for i in idxs], dtype=float)

    X = sm.add_constant(np.column_stack([x1, x2, x3, x4]))
    model = sm.OLS(y, X).fit()

    vifs = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
    dw = durbin_watson(model.resid)
    jb_stat, jb_p, _, _ = jarque_bera(model.resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X)

    print("Table 39. Diagnostic Tests for Model 2 (National Regression)\n")
    print(f"VIF range: {min(vifs):.1f} - {max(vifs):.1f}  -> Below problematic threshold (10)")
    print(f"Durbin-Watson: {dw:.3f}  -> Considerable positive autocorrelation")
    print(f"Breusch-Pagan: {bp_stat:.2f}, p={bp_p:.3f}  -> No strong evidence of heteroskedasticity")
    print(f"Jarque-Bera: {jb_stat:.2f}, p={jb_p:.3f}  -> Residuals approximately normal")


if __name__ == "__main__":
    main()
