# -*- coding: utf-8 -*-
"""
Script 19: Tables 27-28 - Sigma-Convergence Regression
Produces:
  - Table 27. Sigma-Convergence Regression: Coefficient of Variation on a Linear
    Year Trend (N=24, 2000-2023)
  - Table 28. Sigma-Divergence Trend, Before and After 2016
"""
import json
import numpy as np
import statsmodels.api as sm

REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']


def main():
    R = json.load(open('../output/regional_data.json'))
    years_gva = R['years_gva']

    cv_by_year = {}
    for i, y in enumerate(years_gva):
        vals = np.array([R['gva'][reg][i] for reg in REGIONS])
        cv_by_year[y] = vals.std(ddof=1) / vals.mean()

    X = sm.add_constant(np.array(years_gva, dtype=float))
    cv_vals = [cv_by_year[y] for y in years_gva]
    sigma_model = sm.OLS(cv_vals, X).fit()

    print("Table 27. Sigma-Convergence Regression: CV on Linear Year Trend (N=24)\n")
    print(f"Year (trend) coefficient = {sigma_model.params[1]:.4f}, SE = {sigma_model.bse[1]:.4f}, "
          f"p = {sigma_model.pvalues[1]:.2e}")
    print(f"R2 = {sigma_model.rsquared:.3f}, F = {sigma_model.fvalue:.2f}, "
          f"p(F) = {sigma_model.f_pvalue:.2e}")

    # pre/post 2016 comparison
    pre = [y for y in years_gva if y < 2016]
    post = [y for y in years_gva if y >= 2016]

    def trend(years_list):
        yy = np.array(years_list, dtype=float)
        cc = np.array([cv_by_year[y] for y in years_list])
        m = sm.OLS(cc, sm.add_constant(yy)).fit()
        return m

    m_pre, m_post = trend(pre), trend(post)
    yy_all = np.array(years_gva, dtype=float)
    cv_all = np.array(cv_vals)
    post_dummy = (yy_all >= 2016).astype(float)
    X_full = sm.add_constant(np.column_stack([yy_all, post_dummy, yy_all * post_dummy]))
    m_full = sm.OLS(cv_all, X_full).fit()

    print("\nTable 28. Sigma-Divergence Trend, Before and After 2016\n")
    print(f"Pre-2016 (N={len(pre)}): slope={m_pre.params[1]:.5f}, p={m_pre.pvalues[1]:.4f}")
    print(f"Post-2016 (N={len(post)}): slope={m_post.params[1]:.5f}, p={m_post.pvalues[1]:.4f}")
    print(f"Interaction (year x post-2016): coef={m_full.params[3]:.4f}, p={m_full.pvalues[3]:.4f}")

    json.dump({'sigma_slope': sigma_model.params[1], 'sigma_p': sigma_model.pvalues[1],
               'sigma_r2': sigma_model.rsquared,
               'pre_slope': m_pre.params[1], 'pre_p': m_pre.pvalues[1],
               'post_slope': m_post.params[1], 'post_p': m_post.pvalues[1],
               'interaction_coef': m_full.params[3], 'interaction_p': m_full.pvalues[3]},
              open('../output/sigma_convergence.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
