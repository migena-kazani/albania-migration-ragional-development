# -*- coding: utf-8 -*-
"""
Script 33: 95% confidence intervals and standardised effect sizes
(the "CI" columns added to Tables 8, 9, 10, 14, 15, 17, 18, 19, 27, 31, 34, 37, 38,
41, 42, 45 and 48 in the revised manuscript; Table 12 intervals are computed too,
because Table 48 reports them).

Every model is re-estimated from the same data and with the same estimator that the
earlier scripts use; NO coefficient, standard error or p-value is changed. The only
new quantity is  b +/- c * SE,  where the critical value c matches the distribution
that produces the p-value printed in the same table:

    Table 8                 classical OLS                 t(n-k) = t(93)
    Table 9                 cluster-robust pooled OLS     t(G-1) = t(11)
                            (small-sample corrected SE from script 04/05)
    Tables 10, 12, 17, 18,  region-clustered FE/RE/pooled  t(df_resid) as produced by
    19, 41                  linearmodels (no small-sample correction)
    Tables 14, 15           Windmeijer-corrected two-step GMM   normal (z)
    Table 27                maximum-likelihood spatial lag      normal (z)
    Tables 31, 34, 37, 42   classical OLS                 t(n-k)
    Tables 38, 45           heteroskedasticity-robust (HC1) OLS   normal (z)

Standardised coefficients (effect sizes) are  beta = b * SD(x) / SD(y), with overall
standard deviations for pooled / random-effects / national / GMM models and
within-region (demeaned) standard deviations for fixed effects. They are descriptive.

Run after scripts 02-32 (it reads output/*.json, output/panel_df.csv, data/panel_gmm.csv).
Writes output/confidence_intervals.json.
"""
import importlib
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

OUT = {}      # table -> row -> dict(b, se, p, lo, hi)
BETA = {}     # standardised coefficients
REGIONS = ['Berat', 'Dibër', 'Durrës', 'Elbasan', 'Fier', 'Gjirokastër', 'Korçë',
           'Kukës', 'Lezhë', 'Shkodër', 'Tiranë', 'Vlorë']
Z = stats.norm.ppf(0.975)


def put(table, names, b, se, p, crit):
    """Store and print one coefficient table; crit is a scalar critical value."""
    OUT[table] = {}
    print(f"\n{table}")
    print(f"  {'term':<26}{'b':>14}{'SE':>12}{'p':>9}{'95% CI':>34}")
    for i, n in enumerate(names):
        lo, hi = b[i] - crit * se[i], b[i] + crit * se[i]
        OUT[table][n] = dict(b=float(b[i]), se=float(se[i]), p=float(p[i]),
                             lo=float(lo), hi=float(hi))
        print(f"  {n:<26}{b[i]:>14.6g}{se[i]:>12.5g}{p[i]:>9.3f}   ({lo:>12.6g}, {hi:>12.6g})")


def put_lm(table, names, res):
    """linearmodels result: use its own conf_int() (identical to b +/- t * SE)."""
    ci = res.conf_int()
    OUT[table] = {}
    print(f"\n{table}")
    print(f"  {'term':<26}{'b':>14}{'SE':>12}{'p':>9}{'95% CI':>34}")
    for k, n in zip(res.params.index, names):
        d = dict(b=float(res.params[k]), se=float(res.std_errors[k]), p=float(res.pvalues[k]),
                 lo=float(ci.loc[k, 'lower']), hi=float(ci.loc[k, 'upper']))
        OUT[table][n] = d
        print(f"  {n:<26}{d['b']:>14.6g}{d['se']:>12.5g}{d['p']:>9.3f}   ({d['lo']:>12.6g}, {d['hi']:>12.6g})")


def std_beta(b, x, y):
    return float(b * np.std(x, ddof=1) / np.std(y, ddof=1))


def within(v, groups):
    v = pd.Series(np.asarray(v, float))
    return (v - v.groupby(np.asarray(groups)).transform('mean')).values


def main():
    R = json.load(open('../output/regional_data.json'))
    yn, yg, yu = R['years_net'], R['years_gva'], R['years_unemp']
    df = pd.read_csv('../output/panel_df.csv')
    df['gdp_bn'] = df['gdp'] / 1000.0                       # billion ALL (Tables 10, 12, 17-19)
    x = df.set_index(['region', 'year'])
    ex = sm.add_constant(x[['gdp_bn', 'unemp']])

    # ------------------------------------------------------------------ Table 8 / 9
    ols = sm.OLS(df['netmig'], sm.add_constant(df[['gdp', 'unemp']])).fit()
    put('Table 8 (classical OLS, t(93))', ['const', 'GDP (mln ALL)', 'Unemployment'],
        ols.params.values, ols.bse.values, ols.pvalues.values, stats.t.ppf(0.975, ols.df_resid))
    m6 = json.load(open('../output/model1_table6_full.json'))
    names = ['const', 'gdp', 'unemp']
    b = np.array([m6['ols_params'][k] for k in names])
    se = np.array([m6['cluster_se'][k] for k in names])
    pc = 2 * stats.t.sf(np.abs(b / se), 11)
    put('Table 9 (cluster-robust pooled OLS, t(11))', ['const', 'GDP (mln ALL)', 'Unemployment'],
        b, se, pc, stats.t.ppf(0.975, 11))
    BETA['T8_pooled_GDP'] = std_beta(ols.params['gdp'], df['gdp'], df['netmig'])
    BETA['T8_pooled_Unemp'] = std_beta(ols.params['unemp'], df['unemp'], df['netmig'])

    # ------------------------------------------------------------------ Tables 12 / 13
    r = PooledOLS(x['netmig'], ex).fit(cov_type='clustered', cluster_entity=True)
    put_lm('Table 12 pooled OLS (cluster)', ['const', 'GDP (bn ALL)', 'Unemployment'], r)
    fe = PanelOLS(x['netmig'], ex, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    put_lm('Table 12 / 48 FE-region (cluster)', ['const', 'GDP (bn ALL)', 'Unemployment'], fe)
    r = PanelOLS(x['netmig'], ex, entity_effects=True, time_effects=True).fit(
        cov_type='clustered', cluster_entity=True)
    put_lm('Table 12 / 48 FE-region+year (cluster)', ['const', 'GDP (bn ALL)', 'Unemployment'], r)
    re = RandomEffects(x['netmig'], ex).fit(cov_type='clustered', cluster_entity=True)
    put_lm('Table 12 / 48 RE (cluster)', ['const', 'GDP (bn ALL)', 'Unemployment'], re)
    yw = within(df['netmig'], df['region'])
    BETA['T12_FE_GDP'] = std_beta(fe.params['gdp_bn'], within(df['gdp_bn'], df['region']), yw)
    BETA['T12_FE_Unemp'] = std_beta(fe.params['unemp'], within(df['unemp'], df['region']), yw)
    BETA['T12_RE_GDP'] = std_beta(re.params['gdp_bn'], df['gdp_bn'], df['netmig'])
    BETA['T12_RE_Unemp'] = std_beta(re.params['unemp'], df['unemp'], df['netmig'])

    # ------------------------------------------------------------------ Table 10
    vol = {}
    for reg in REGIONS:
        v = [R['gva'][reg][yg.index(y)] for y in range(2016, 2024)]
        vol[reg] = np.std(np.diff(v) / np.array(v[:-1]), ddof=1)
    rows = [dict(region=reg, year=y, netmig=R['net_mig'][reg][yn.index(y)] * 1000,
                 gdp_bn=R['gva'][reg][yg.index(y)] / 1000.0, unemp=R['unemp'][reg][yu.index(y)],
                 volatility=vol[reg]) for reg in REGIONS for y in range(2016, 2024)]
    d10 = pd.DataFrame(rows)
    d10['unemp_vol'] = d10['unemp'] * d10['volatility']
    x10 = d10.set_index(['region', 'year'])
    r = PanelOLS(x10['netmig'], sm.add_constant(x10[['gdp_bn', 'unemp', 'unemp_vol']]),
                 entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    put_lm('Table 10 (FE, cluster)', ['const', 'GDP (bn ALL)', 'Unemployment',
                                       'Unemp x GDP-growth volatility'], r)

    # ------------------------------------------------------------------ Tables 14 / 15 (GMM)
    g = json.load(open('../output/tables14_15_gmm.json'))
    gm = pd.read_csv('../data/panel_gmm.csv')
    for key, tab, nm in [('table14_system_gmm', 'Table 14 (System GMM, z)',
                          ['L1.NetMigration', 'GDP', 'Unemployment', '_con']),
                         ('table15_difference_gmm', 'Table 15 (Difference GMM, z)',
                          ['L1.NetMigration', 'GDP', 'Unemployment'])]:
        rows = g[key]
        put(tab, [r_['variable'] for r_ in rows], np.array([r_['coefficient'] for r_ in rows]),
            np.array([r_['std_err'] for r_ in rows]), np.array([r_['p_value'] for r_ in rows]), Z)
        tag = 'T14' if '14' in tab else 'T15'
        for r_ in rows:
            if r_['variable'] == 'GDP':
                BETA[tag + '_GDP'] = std_beta(r_['coefficient'], gm['GDP'], gm['NetMigration'])
            if r_['variable'] == 'Unemployment':
                BETA[tag + '_Unemp'] = std_beta(r_['coefficient'], gm['Unemployment'], gm['NetMigration'])
    # GMM coefficients are in thousand persons per billion ALL -> x1000 = persons (Table 48 units)
    for key, tab in [('table14_system_gmm', 'Table 48 System GMM GDP (persons per bn ALL)'),
                     ('table15_difference_gmm', 'Table 48 Difference GMM GDP (persons per bn ALL)')]:
        r_ = [q for q in g[key] if q['variable'] == 'GDP'][0]
        put(tab, ['GDP'], np.array([r_['coefficient'] * 1000]), np.array([r_['std_err'] * 1000]),
            np.array([r_['p_value']]), Z)

    # ------------------------------------------------------------------ Table 17
    for drop, tag in [(None, 'full sample'), ('Tiranë', 'excluding Tirana'), ('Durrës', 'excluding Durres')]:
        sub = df[df['region'] != drop] if drop else df
        sx = sub.set_index(['region', 'year'])
        r = PanelOLS(sx['netmig'], sm.add_constant(sx[['gdp_bn', 'unemp']]),
                     entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
        put_lm(f'Table 17 ({tag})', ['const', 'GDP (bn ALL)', 'Unemployment'], r)

    # ------------------------------------------------------------------ Table 18
    d18 = df.copy()
    d18['gdp_core'] = d18['gdp_bn'] * (d18['region'] == 'Tiranë').astype(int)
    x18 = d18.set_index(['region', 'year'])
    r = PanelOLS(x18['netmig'], sm.add_constant(x18[['gdp_bn', 'unemp', 'gdp_core']]),
                 entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    put_lm('Table 18 (FE, cluster)', ['const', 'GDP (bn ALL)', 'Unemployment', 'GDP x Core'], r)

    # ------------------------------------------------------------------ Table 19
    perm = json.load(open('../output/building_permits.json'))
    rows = []
    for reg in REGIONS:
        for y in [2020, 2021, 2022, 2023]:
            pm = perm['permits'][reg].get(str(y), perm['permits'][reg].get(y))
            rows.append(dict(region=reg, year=y, netmig=R['net_mig'][reg][yn.index(y)] * 1000,
                             gdp_bn=R['gva'][reg][yg.index(y)] / 1000.0,
                             unemp=R['unemp'][reg][yu.index(y)], permits=pm))
    d19 = pd.DataFrame(rows)
    d19['permits_log'] = np.log1p(d19['permits'])
    x19 = d19.set_index(['region', 'year'])
    r = PanelOLS(x19['netmig'], sm.add_constant(x19[['gdp_bn', 'unemp', 'permits_log']]),
                 entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    put_lm('Table 19 with permits', ['const', 'GDP (bn ALL)', 'Unemployment', 'Permits (log)'], r)
    r = PanelOLS(x19['netmig'], sm.add_constant(x19[['gdp_bn', 'unemp']]),
                 entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    put_lm('Table 19 without permits', ['const', 'GDP (bn ALL)', 'Unemployment'], r)

    # ------------------------------------------------------------------ Table 27 (spatial lag)
    from libpysal.weights import W
    from spreg import ML_Lag
    sp = importlib.import_module('18_spatial_econometrics')
    w = W(sp.NEIGHBORS)
    w.transform = 'r'
    cl = json.load(open('../output/clustering.json'))
    regs = cl['regions']
    idx = [regs.index(q) for q in w.id_order]
    nm_ = np.array(cl['mean_netmig'])[idx]
    gd_ = np.array(cl['mean_gdp'])[idx]
    un_ = np.array(cl['mean_unemp'])[idx]
    ml = ML_Lag(nm_.reshape(-1, 1), np.column_stack([gd_, un_]), w=w, name_y='netmig',
                name_x=['gdp', 'unemp'])
    b = np.array(ml.betas).flatten()
    se = np.array(ml.std_err).flatten()
    pv = np.array([zz[1] for zz in ml.z_stat])
    put('Table 27 (ML spatial lag, z)', ['const', 'GDP', 'Unemployment', 'rho (W x NetMigration)'],
        b, se, pv, Z)

    # ------------------------------------------------------------------ Table 31 (sigma-convergence)
    cv = np.array([np.std([R['gva'][q][i] for q in REGIONS], ddof=1) /
                   np.mean([R['gva'][q][i] for q in REGIONS]) for i in range(len(yg))])
    m = sm.OLS(cv, sm.add_constant(np.array(yg, float))).fit()
    put('Table 31 (CV on year trend, t(22))', ['const', 'Year (trend)'],
        m.params, m.bse, m.pvalues, stats.t.ppf(0.975, m.df_resid))

    # ------------------------------------------------------------------ Table 34 (beta-convergence)
    init = np.array([R['gva'][q][yg.index(2016)] for q in REGIONS])
    fin = np.array([R['gva'][q][yg.index(2023)] for q in REGIONS])
    m = sm.OLS(np.log(fin / init) / 7, sm.add_constant(np.log(init))).fit()
    put('Table 34 (growth on log initial GDP, t(10))', ['const', 'log(initial GDP)'],
        m.params, m.bse, m.pvalues, stats.t.ppf(0.975, m.df_resid))

    # ------------------------------------------------------------------ national models
    N = json.load(open('../output/national_data.json'))
    yrs = N['years']

    def series(keys, extra=lambda y: True):
        ix = [i for i in range(len(yrs)) if all(N[k][i] is not None for k in keys) and extra(yrs[i])]
        return ix, {k: np.array([N[k][i] for i in ix], float) for k in keys}

    # Table 37 (model 5, N=32)
    ix, v = series(['netmig', 'gdppc_lcu', 'unemp_nat', 'urban_abs', 'remit'], lambda y: y >= 1992)
    X = sm.add_constant(np.column_stack([v['gdppc_lcu'], v['unemp_nat'], v['urban_abs'], v['remit']]))
    m = sm.OLS(v['netmig'], X).fit()
    put('Table 37 (national model 5, t(27))', ['const', 'GDP per capita', 'Unemployment', 'Urban population',
                                               'Remittances'],
        m.params, m.bse, m.pvalues, stats.t.ppf(0.975, m.df_resid))
    for j, k in enumerate(['gdppc_lcu', 'unemp_nat', 'urban_abs', 'remit']):
        BETA['T37_' + k] = std_beta(m.params[j + 1], v[k], v['netmig'])

    # Tables 38 and 42 (N=27)
    keys = ['netmig', 'gdppc_lcu', 'unemp_nat', 'urban_abs', 'remit', 'fdi', 'edu']
    ix, v = series(keys)
    v['fdi'] = v['fdi'] / 1e9
    X42 = sm.add_constant(np.column_stack([v[k] for k in keys[1:]]))
    m = sm.OLS(v['netmig'], X42).fit()
    put('Table 42 (extended national model, t(20))',
        ['const', 'GDP per capita', 'Unemployment', 'Urban population', 'Remittances', 'FDI (bn USD)',
         'Education expenditure'], m.params, m.bse, m.pvalues, stats.t.ppf(0.975, m.df_resid))
    ix38, w38 = series(['netmig', 'gdppc_lcu', 'unemp_nat', 'urban_abs', 'remit', 'edu'])
    X38 = sm.add_constant(np.column_stack([w38['gdppc_lcu'], w38['unemp_nat'], w38['urban_abs'],
                                           w38['remit'], w38['edu'], w38['remit'] * w38['edu']]))
    m = sm.OLS(w38['netmig'], X38).fit(cov_type='HC1')
    put('Table 38 (remittances x education, HC1, z)',
        ['const', 'GDPpc', 'Unemp', 'Urban', 'Remit', 'EduExp', 'Remit x EduExp'],
        m.params, m.bse, m.pvalues, Z)

    # ------------------------------------------------------------------ Table 41
    rows = []
    for reg in REGIONS:
        for y in range(2017, 2024):
            g0 = R['gva'][reg][yg.index(y - 1)]
            rows.append(dict(region=reg, year=y, growth=(R['gva'][reg][yg.index(y)] - g0) / g0,
                             netmig=R['net_mig'][reg][yn.index(y)] * 1000,
                             lg=np.log(R['gva'][reg][yg.index(2016)])))
    d41 = pd.DataFrame(rows)
    d41['inter'] = d41['netmig'] * d41['lg']
    x41 = d41.set_index(['region', 'year'])
    r = PanelOLS(x41['growth'], sm.add_constant(x41[['netmig', 'inter']]),
                 entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
    put_lm('Table 41 (FE, cluster)', ['const', 'Net migration', 'Net migration x log(initial GDP)'], r)

    # ------------------------------------------------------------------ Table 45 (TCI)
    tci = json.load(open('../output/tci_pca.json'))
    tm = dict(zip(tci['years'], tci['tci']))
    D = {k: dict(zip(yrs, N[k])) for k in ['gdppc_lcu', 'unemp_nat', 'urban_abs', 'netmig']}
    common = [yy for yy in tci['years'] if all(D[k].get(yy) is not None for k in D)]
    y = np.array([D['netmig'][yy] for yy in common], float)
    cols = [np.array([D[k][yy] for yy in common], float) for k in ['gdppc_lcu', 'unemp_nat', 'urban_abs']]
    Xw = sm.add_constant(np.column_stack(cols + [np.array([tm[yy] for yy in common], float)]))
    Xo = sm.add_constant(np.column_stack(cols))
    m = sm.OLS(y, Xw).fit(cov_type='HC1')
    put('Table 45 with TCI (HC1, z)', ['const', 'GDPpc', 'Unemp', 'Urban', 'TCI'], m.params, m.bse, m.pvalues, Z)
    m = sm.OLS(y, Xo).fit(cov_type='HC1')
    put('Table 45 without TCI (HC1, z)', ['const', 'GDPpc', 'Unemp', 'Urban'], m.params, m.bse, m.pvalues, Z)

    print("\n\nStandardised coefficients (beta = b * SD(x) / SD(y)):")
    for k, v_ in BETA.items():
        print(f"  {k:<18}{v_:>9.3f}")
    json.dump({'ci': OUT, 'standardised_beta': BETA}, open('../output/confidence_intervals.json', 'w'), indent=1)
    print("\nSaved ../output/confidence_intervals.json")


if __name__ == "__main__":
    main()
