# -*- coding: utf-8 -*-
"""
Script 05: Table 9 - How the Standard Errors and Significance of Table 8 Change
Under Three Specifications (OLS, HC1, cluster-robust).
Reuses the results computed and saved by script 03.
"""
import json


def fmtp(v):
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def main():
    m6 = json.load(open('../output/model1_table6_full.json'))
    names = ['const', 'gdp', 'unemp']
    labels = {'const': '(Constant)', 'gdp': 'GDP by region', 'unemp': 'Unemployment by region'}

    print("Table 9. SE and significance of Table 8 coefficients under three specifications\n")
    print(f"{'Variable':25s} {'SE(OLS)':>10s} {'Sig(OLS)':>10s} {'SE(HC1)':>10s} {'Sig(HC1)':>10s} "
          f"{'SE(clust)':>10s} {'Sig(clust)':>10s}")
    for nm in names:
        print(f"{labels[nm]:25s} {m6['ols_se'][nm]:10.4f} {fmtp(m6['ols_p'][nm]):>10s} "
              f"{m6['hc1_se'][nm]:10.4f} {fmtp(m6['hc1_p'][nm]):>10s} "
              f"{m6['cluster_se'][nm]:10.4f} {fmtp(m6['cluster_p'][nm]):>10s}")

    print("\nKey finding: Unemployment's significance drops from p<0.001 (OLS classic) and "
          "p<0.001 (HC1) to p=0.074 (cluster-robust, t-distribution with G-1=11 degrees of "
          "freedom, following Cameron & Miller 2015's small-cluster correction), while GDP "
          "remains strongly significant across all three specifications.")


if __name__ == "__main__":
    main()
