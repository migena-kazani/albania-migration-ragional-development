# -*- coding: utf-8 -*-
"""
Script 29: Table 42 - Extended National Model (Model 2b, N=27)
Adds FDI and education expenditure to the national model as a robustness test.
"""
import json
import numpy as np
import statsmodels.api as sm


def main():
    N = json.load(open('../output/national_data.json'))
    years = N['years']
    idxs = [i for i in range(len(years)) if all(
        N[k][i] is not None for k in ['netmig', 'gdppc_lcu', 'unemp_nat', 'urban_abs', 'remit', 'fdi', 'edu'])]

    y = np.array([N['netmig'][i] for i in idxs], dtype=float)
    x1 = np.array([N['gdppc_lcu'][i] for i in idxs], dtype=float)
    x2 = np.array([N['unemp_nat'][i] for i in idxs], dtype=float)
    x3 = np.array([N['urban_abs'][i] for i in idxs], dtype=float)
    x4 = np.array([N['remit'][i] for i in idxs], dtype=float)
    x5 = np.array([N['fdi'][i] for i in idxs], dtype=float) / 1e9  # billions
    x6 = np.array([N['edu'][i] for i in idxs], dtype=float)

    X = sm.add_constant(np.column_stack([x1, x2, x3, x4, x5, x6]))
    model = sm.OLS(y, X).fit()

    print(f"Table 42. Extended National Model (Model 2b, N={len(idxs)})\n")
    print(model.summary(xname=['const', 'GDPpc', 'Unemp', 'Urban', 'Remit', 'FDI(bn$)', 'EduExp']))

    json.dump({'params': model.params.tolist(), 'pvalues': model.pvalues.tolist(),
               'r2': model.rsquared, 'r2adj': model.rsquared_adj, 'n': len(idxs)},
              open('../output/table38_model2b.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
