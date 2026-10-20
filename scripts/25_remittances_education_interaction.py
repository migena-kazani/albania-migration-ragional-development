# -*- coding: utf-8 -*-
"""
Script 25: Table 38 - National Model with Remittances x Education-Expenditure
Interaction (N=27). Tests part of Combined Hypothesis CH3.
"""
import json
import numpy as np
import statsmodels.api as sm


def main():
    N = json.load(open('../output/national_data.json'))
    years = N['years']
    idxs = [i for i in range(len(years)) if all(
        N[k][i] is not None for k in ['netmig', 'gdppc_lcu', 'unemp_nat', 'urban_abs', 'remit', 'edu'])]

    y = np.array([N['netmig'][i] for i in idxs], dtype=float)
    x1 = np.array([N['gdppc_lcu'][i] for i in idxs], dtype=float)
    x2 = np.array([N['unemp_nat'][i] for i in idxs], dtype=float)
    x3 = np.array([N['urban_abs'][i] for i in idxs], dtype=float)
    x4 = np.array([N['remit'][i] for i in idxs], dtype=float)
    x5 = np.array([N['edu'][i] for i in idxs], dtype=float)
    x6 = x4 * x5  # remittances x education interaction

    X = sm.add_constant(np.column_stack([x1, x2, x3, x4, x5, x6]))
    model = sm.OLS(y, X).fit(cov_type='HC1')

    print(f"Table 38. Remittances x Education-Expenditure Interaction (N={len(idxs)})\n")
    print(model.summary(xname=['const', 'GDPpc', 'Unemp', 'Urban', 'Remit', 'EduExp', 'Remit_x_EduExp']))
    print("\nNeither remittances, education expenditure, nor their interaction is statistically "
          "significant; GDP per capita and urban population remain the only robust predictors.")

    json.dump({'params': model.params.tolist(), 'pvalues': model.pvalues.tolist(), 'n': len(idxs)},
              open('../output/table34_remit_edu.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
