# -*- coding: utf-8 -*-
"""
Script 20: Table 30 - Additional Regional Inequality Indices (Gini, Atkinson, Williamson)
"""
import json
import numpy as np

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def gini(x):
    x = np.sort(np.array(x))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def atkinson(x, eps=1.0):
    x = np.array(x)
    mean = x.mean()
    yi = np.exp(np.mean(np.log(x))) if eps == 1 else (np.mean(x ** (1 - eps))) ** (1 / (1 - eps))
    return 1 - yi / mean


def williamson(x):
    x = np.array(x)
    mean = x.mean()
    var = np.mean((x - mean) ** 2)
    return np.sqrt(var) / mean


def main():
    R = json.load(open('../output/regional_data.json'))
    years_gva = R['years_gva']

    gini_l, atk_l, will_l = [], [], []
    for i, y in enumerate(years_gva):
        vals = np.array([R['gva'][reg][i] for reg in REGIONS])
        gini_l.append(gini(vals))
        atk_l.append(atkinson(vals, 1.0))
        will_l.append(williamson(vals))

    print("Table 30. Additional Regional Inequality Indices, Selected Years\n")
    print(f"{'Year':6s} {'Gini':>8s} {'Atkinson':>10s} {'Williamson':>12s}")
    for y in [2000, 2010, 2016, 2020, 2023]:
        i = years_gva.index(y)
        print(f"{y:6d} {gini_l[i]:8.3f} {atk_l[i]:10.3f} {will_l[i]:12.3f}")

    json.dump({'years': years_gva, 'gini': gini_l, 'atkinson': atk_l, 'williamson': will_l},
              open('../output/inequality_extra.json', 'w'))


if __name__ == "__main__":
    main()
