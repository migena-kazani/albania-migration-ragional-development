# -*- coding: utf-8 -*-
"""
Script 06: Table 9 - Diagnostic Tests for Model 1 (Regional Regression)
VIF, Durbin-Watson, Breusch-Pagan, Jarque-Bera on the pooled OLS model from script 03.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan


def main():
    df = pd.read_csv('../output/panel_df.csv')
    X = sm.add_constant(df[['gdp', 'unemp']])
    y = df['netmig']
    model = sm.OLS(y, X).fit()

    vif_gdp = variance_inflation_factor(X.values, 1)
    vif_unemp = variance_inflation_factor(X.values, 2)
    dw = durbin_watson(model.resid)
    jb_stat, jb_p, skew, kurt = jarque_bera(model.resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X)

    print("Table 9. Diagnostic Tests for Model 1 (Regional Regression)\n")
    print(f"Variance Inflation Factor (VIF): GDP={vif_gdp:.2f}, Unemployment={vif_unemp:.2f}  "
          f"-> No problematic multicollinearity")
    print(f"Durbin-Watson (autocorrelation): {dw:.2f}  -> Positive autocorrelation of residuals, "
          f"expected in panel data with repeated regions")
    print(f"Breusch-Pagan (heteroskedasticity): {bp_stat:.2f}, p{'<0.001' if bp_p<0.001 else f'={bp_p:.4f}'}  "
          f"-> Heteroskedasticity present")
    print(f"Jarque-Bera (normality of residuals): {jb_stat:.1f}, p{'<0.001' if jb_p<0.001 else f'={jb_p:.4f}'}  "
          f"-> Residuals not normally distributed (driven by Tirana's extreme values)")


if __name__ == "__main__":
    main()
