# -*- coding: utf-8 -*-
"""
Script 27: Tables 39–40 - Augmented Dickey-Fuller Test and Granger Causality
"""
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import warnings
warnings.filterwarnings('ignore')


def adf_report(series, name):
    r = adfuller(series, autolag='AIC')
    verdict = 'STATIONARY' if r[1] < 0.05 else 'NON-STATIONARY (unit root)'
    print(f"{name}: ADF stat={r[0]:.3f}, p={r[1]:.4f} -> {verdict}")
    return r[0], r[1]


def main():
    N = json.load(open('../output/national_data.json'))
    years = N['years']
    idxs = [i for i in range(len(years)) if years[i] >= 1992
            and N['netmig'][i] is not None and N['gdppc_lcu'][i] is not None]
    nm = np.array([N['netmig'][i] for i in idxs], dtype=float)
    gdppc = np.array([N['gdppc_lcu'][i] for i in idxs], dtype=float)

    print("Table 39. Augmented Dickey-Fuller Unit-Root Test Results\n")
    s1, p1 = adf_report(nm, "Net migration, level")
    s2, p2 = adf_report(gdppc, "GDP per capita, level")
    nm_d = np.diff(nm)
    gdp_d = np.diff(np.log(gdppc))
    s3, p3 = adf_report(nm_d, "Net migration, 1st difference")
    s4, p4 = adf_report(gdp_d, "log GDP per capita, 1st difference")

    print("\nTable 40. Granger Causality Tests (Differenced Series, N=32)\n")
    data = pd.DataFrame({'nm': nm_d, 'gdp': gdp_d})

    def report_granger(data, direction_label, maxlag=2):
        print(f"--- {direction_label} ---")
        res = grangercausalitytests(data, maxlag=maxlag)
        for lag in range(1, maxlag + 1):
            f_test = res[lag][0]['ssr_ftest']
            print(f"  Lag {lag}: F={f_test[0]:.2f}, p={f_test[1]:.3f}")
        return res

    report_granger(data[['nm', 'gdp']], "Does GDP growth Granger-cause migration change?")
    print()
    report_granger(data[['gdp', 'nm']], "Does migration change Granger-cause GDP growth?")

    json.dump({'netmig_level_p': p1, 'gdppc_level_p': p2, 'netmig_diff_p': p3, 'gdppc_diff_p': p4},
              open('../output/adf_granger.json', 'w'))


if __name__ == "__main__":
    main()
