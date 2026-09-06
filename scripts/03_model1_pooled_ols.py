# -*- coding: utf-8 -*-
"""
Script 03: Model 1 - Pooled OLS Baseline Regression (Regional Panel)
Produces:
  - Table 4. Model Summary (Model 1, Regional Regression, N=96)
  - Table 5. ANOVA of Model 1 (Sum of Squares in millions)
  - Table 6. Coefficients of Model 1 (with alternative standard errors)
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def build_panel():
    R = json.load(open('../output/regional_data.json'))
    years_net, years_gva, years_unemp = R['years_net'], R['years_gva'], R['years_unemp']
    panel_years = list(range(2016, 2024))
    rows = []
    for reg in REGIONS:
        for y in panel_years:
            nm = R['net_mig'][reg][years_net.index(y)] * 1000
            gdp = R['gva'][reg][years_gva.index(y)]  # million ALL
            un = R['unemp'][reg][years_unemp.index(y)]
            rows.append(dict(region=reg, year=y, netmig=nm, gdp=gdp, unemp=un))
    df = pd.DataFrame(rows)
    df.to_csv('../output/panel_df.csv', index=False)
    return df


def main():
    df = build_panel()
    y = df['netmig'].values
    X = sm.add_constant(df[['gdp', 'unemp']])
    model = sm.OLS(y, X).fit()

    print("=== Table 4/5. Classical OLS summary and ANOVA ===")
    print(model.summary())

    # Standardized (Beta) coefficients
    def zscore(x):
        return (x - x.mean()) / x.std(ddof=1)
    Xz = pd.DataFrame({'gdp': zscore(df['gdp']), 'unemp': zscore(df['unemp'])})
    yz = zscore(df['netmig'])
    modelz = sm.OLS(yz, sm.add_constant(Xz)).fit()

    # HC1-robust
    model_hc1 = sm.OLS(y, X).fit(cov_type='HC1')

    # Cluster-robust with small-cluster t(G-1) correction (Cameron & Miller, 2015)
    def cluster_robust(Xmat, yvec, groups):
        Xm = Xmat.values
        yv = np.asarray(yvec)
        n, k = Xm.shape
        XtX_inv = np.linalg.inv(Xm.T @ Xm)
        beta = XtX_inv @ Xm.T @ yv
        resid = yv - Xm @ beta
        uniq = groups.unique()
        G = len(uniq)
        meat = np.zeros((k, k))
        for g in uniq:
            idx = (groups == g).values
            Xg = Xm[idx]
            ug = resid[idx].reshape(-1, 1)
            score = Xg.T @ ug
            meat += score @ score.T
        corr = (G / (G - 1)) * ((n - 1) / (n - k))
        cov = XtX_inv @ meat @ XtX_inv * corr
        se = np.sqrt(np.diag(cov))
        t = beta / se
        p = 2 * (1 - stats.t.cdf(np.abs(t), G - 1))
        return beta, se, t, p

    beta_cl, se_cl, t_cl, p_cl = cluster_robust(X, pd.Series(y), df['region'])

    print("\n=== Table 6. Coefficients with Beta and alternative SEs ===")
    names = ['const', 'gdp', 'unemp']
    print(f"{'Variable':15s} {'B':>10s} {'Beta':>8s} {'SE(OLS)':>10s} {'SE(HC1)':>10s} {'SE(clust)':>10s} {'p(clust)':>10s}")
    for i, nm in enumerate(names):
        beta_val = modelz.params[nm] if nm != 'const' else float('nan')
        print(f"{nm:15s} {model.params[nm]:10.4f} {beta_val:8.3f} {model.bse[nm]:10.4f} "
              f"{model_hc1.bse[nm]:10.4f} {se_cl[i]:10.4f} {p_cl[i]:10.5f}")

    results = {
        'ols_params': model.params.to_dict(), 'ols_se': model.bse.to_dict(),
        'ols_t': model.tvalues.to_dict(), 'ols_p': model.pvalues.to_dict(),
        'ols_r2': model.rsquared, 'ols_r2adj': model.rsquared_adj,
        'ols_f': model.fvalue, 'ols_fp': model.f_pvalue,
        'beta_params': modelz.params.to_dict(),
        'hc1_se': model_hc1.bse.to_dict(), 'hc1_t': model_hc1.tvalues.to_dict(),
        'hc1_p': model_hc1.pvalues.to_dict(),
        'cluster_se': dict(zip(names, se_cl.tolist())),
        'cluster_t': dict(zip(names, t_cl.tolist())),
        'cluster_p': dict(zip(names, p_cl.tolist())),
    }
    json.dump(results, open('../output/model1_table6_full.json', 'w'), indent=2)
    print("\nSaved to ../output/model1_table6_full.json")


if __name__ == "__main__":
    main()
