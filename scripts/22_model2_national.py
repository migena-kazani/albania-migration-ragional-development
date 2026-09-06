# -*- coding: utf-8 -*-
"""
Script 22: Tables 31-33 - National Model 2
Produces:
  - Table 31. Model Summary (Model 2, National, N=32)
  - Table 32. ANOVA of Model 2
  - Table 33. Coefficients of Model 2
"""
import json
import numpy as np
import statsmodels.api as sm


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

    print(f"Table 31-33. National Model 2 (N={len(idxs)}, {years[idxs[0]]}-{years[idxs[-1]]})\n")
    print(model.summary(xname=['const', 'GDPpc', 'Unemp', 'Urban', 'Remit']))

    json.dump({'years_used': [years[i] for i in idxs], 'params': model.params.tolist(),
               'pvalues': model.pvalues.tolist(), 'se': model.bse.tolist(),
               'r2': model.rsquared, 'r2adj': model.rsquared_adj,
               'f': model.fvalue, 'fp': model.f_pvalue},
              open('../output/model2_national.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
