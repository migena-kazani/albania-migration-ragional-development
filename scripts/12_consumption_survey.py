# -*- coding: utf-8 -*-
"""
Script 12: Table 17 - Correlation of 2014-2015 Average Household Consumption
Expenditure with Average Net Migration, 2016-2023 (N=12 Regions)
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    df2 = pd.read_excel('../data/tab1-2015.xlsx', sheet_name='Sheet1', header=None)
    cons = {}
    current_region = None
    for i in range(5, 29):
        row = df2.iloc[i]
        if pd.notna(row[0]):
            current_region = row[0].strip()
        year = int(row[1])
        total = row[8]
        cons.setdefault(current_region, {})[year] = total
    json.dump(cons, open('../output/consumption_2015.json', 'w'), ensure_ascii=False)

    R = json.load(open('../output/regional_data.json'))
    years_net = R['years_net']
    mean_netmig = np.array([np.mean(np.array(R['net_mig'][reg][2:10]) * 1000) for reg in REGIONS])
    cons_avg = np.array([np.mean([cons[reg][2014], cons[reg][2015]]) for reg in REGIONS])

    r, p = stats.pearsonr(cons_avg, mean_netmig)
    X = sm.add_constant(cons_avg)
    model = sm.OLS(mean_netmig, X).fit()

    print("Table 17. Correlation of 2014-2015 Consumption Expenditure with Net Migration (N=12)\n")
    print(f"Pearson r = {r:.3f}, p = {p:.4f}")
    print(f"OLS slope = {model.params[1]:.4f}, SE = {model.bse[1]:.4f}, R2 = {model.rsquared:.3f}")
    print("\nThe correlation is positive but falls short of the 5% significance threshold; reported "
          "as descriptive context, not as a formal regressor, given the 1-8 year gap between the two "
          "data periods.")


if __name__ == "__main__":
    main()
