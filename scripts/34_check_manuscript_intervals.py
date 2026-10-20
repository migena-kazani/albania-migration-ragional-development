# -*- coding: utf-8 -*-
"""
Script 34: check the 95% confidence intervals printed in the manuscript tables
against the values recomputed by script 33.

EXPECTED lists every interval shown in the revised manuscript (Tables 8, 9, 10, 14, 15,
17, 18, 19, 27, 31, 34, 37, 38, 41, 42, 45 and the GDP column of Table 48), exactly as
printed. For each one the script reports

    exact       the recomputed bound rounds to the printed value
    last digit  it differs by one unit in the last printed digit; this happens where a
                manuscript interval was computed from the already-rounded coefficient
                and standard error shown in the table, instead of from the unrounded
                estimates that script 33 uses
    DIFF        a larger difference - these cells should be updated to the script value

Run after script 33.
"""
import json

EXPECTED = [
    ('Table 8 (classical OLS, t(93))', 'const', '-4336.5', '-2443.3', 'Table 8 row 1'),
    ('Table 9 (cluster-robust pooled OLS, t(11))', 'const', '-4972.4', '-1807.4', 'Table 9 row 1'),
    ('Table 8 (classical OLS, t(93))', 'GDP (mln ALL)', '0.0137', '0.0181', 'Table 8 row 2'),
    ('Table 9 (cluster-robust pooled OLS, t(11))', 'GDP (mln ALL)', '0.0148', '0.017', 'Table 9 row 2'),
    ('Table 8 (classical OLS, t(93))', 'Unemployment', '39.9', '190.06', 'Table 8 row 3'),
    ('Table 9 (cluster-robust pooled OLS, t(11))', 'Unemployment', '-12.96', '242.92', 'Table 9 row 3'),
    ('Table 10 (FE, cluster)', 'GDP (bn ALL)', '-17.39', '-10.87', 'Table 10 row 1'),
    ('Table 10 (FE, cluster)', 'Unemployment', '-49.05', '649.64', 'Table 10 row 2'),
    ('Table 10 (FE, cluster)', 'Unemp x GDP-growth volatility', '-7897.8', '-409', 'Table 10 row 3'),
    ('Table 14 (System GMM, z)', 'L1.NetMigration', '0.3768', '0.5282', 'Table 14 row 1'),
    ('Table 14 (System GMM, z)', 'GDP', '0.0067', '0.0095', 'Table 14 row 2'),
    ('Table 14 (System GMM, z)', 'Unemployment', '-0.0379', '0.0361', 'Table 14 row 3'),
    ('Table 14 (System GMM, z)', '_con', '-1.6958', '-0.5238', 'Table 14 row 4'),
    ('Table 15 (Difference GMM, z)', 'L1.NetMigration', '0.1217', '0.8183', 'Table 15 row 1'),
    ('Table 15 (Difference GMM, z)', 'GDP', '-0.0294', '-0.009', 'Table 15 row 2'),
    ('Table 15 (Difference GMM, z)', 'Unemployment', '-0.238', '-0.0698', 'Table 15 row 3'),
    ('Table 17 (full sample)', 'GDP (bn ALL)', '-21.24', '-12.64', 'Table 17 row 1'),
    ('Table 17 (full sample)', 'Unemployment', '-111.27', '17.31', 'Table 17 row 1'),
    ('Table 17 (excluding Tirana)', 'GDP (bn ALL)', '-29.13', '21.18', 'Table 17 row 2'),
    ('Table 17 (excluding Tirana)', 'Unemployment', '-102.34', '23.76', 'Table 17 row 2'),
    ('Table 17 (excluding Durres)', 'GDP (bn ALL)', '-21.54', '-10.77', 'Table 17 row 3'),
    ('Table 17 (excluding Durres)', 'Unemployment', '-114.92', '18.69', 'Table 17 row 3'),
    ('Table 18 (FE, cluster)', 'GDP (bn ALL)', '-28.29', '21.99', 'Table 18 row 1'),
    ('Table 18 (FE, cluster)', 'Unemployment', '-93.6', '33.76', 'Table 18 row 2'),
    ('Table 18 (FE, cluster)', 'GDP x Core', '-39.8', '9.13', 'Table 18 row 3'),
    ('Table 19 with permits', 'GDP (bn ALL)', '-3.07', '0.14', 'Table 19 row 1'),
    ('Table 19 with permits', 'Unemployment', '-47.78', '49.82', 'Table 19 row 2'),
    ('Table 19 with permits', 'Permits (log)', '-772.8', '-73.1', 'Table 19 row 3'),
    ('Table 19 without permits', 'GDP (bn ALL)', '-4.34', '0.39', 'Table 19 row 1'),
    ('Table 19 without permits', 'Unemployment', '-36.04', '69.53', 'Table 19 row 2'),
    ('Table 27 (ML spatial lag, z)', 'const', '-5323.2', '-1997.6', 'Table 27 row 1'),
    ('Table 27 (ML spatial lag, z)', 'GDP', '0.0141', '0.0203', 'Table 27 row 2'),
    ('Table 27 (ML spatial lag, z)', 'Unemployment', '-17.7', '266.3', 'Table 27 row 3'),
    ('Table 27 (ML spatial lag, z)', 'rho (W x NetMigration)', '-0.32', '0.4', 'Table 27 row 4'),
    ('Table 31 (CV on year trend, t(22))', 'Year (trend)', '0.0071', '0.0129', 'Table 31 row 1'),
    ('Table 31 (CV on year trend, t(22))', 'const', '-24.84', '-12.87', 'Table 31 row 2'),
    ('Table 34 (growth on log initial GDP, t(10))', 'log(initial GDP)', '0.0045', '0.0201', 'Table 34 row 1'),
    ('Table 34 (growth on log initial GDP, t(10))', 'const', '-0.1701', '0.0069', 'Table 34 row 2'),
    ('Table 37 (national model 5, t(27))', 'const', '-238954', '-93816', 'Table 37 row 1'),
    ('Table 37 (national model 5, t(27))', 'GDP per capita', '0.017', '0.0662', 'Table 37 row 2'),
    ('Table 37 (national model 5, t(27))', 'Unemployment', '-680.3', '1284.5', 'Table 37 row 3'),
    ('Table 37 (national model 5, t(27))', 'Urban population', '0.0309', '0.1249', 'Table 37 row 4'),
    ('Table 37 (national model 5, t(27))', 'Remittances', '-1752.9', '1234.5', 'Table 37 row 5'),
    ('Table 38 (remittances x education, HC1, z)', 'GDPpc', '0.0118', '0.0575', 'Table 38 row 1'),
    ('Table 38 (remittances x education, HC1, z)', 'Unemp', '-1201.9', '651.4', 'Table 38 row 2'),
    ('Table 38 (remittances x education, HC1, z)', 'Urban', '0.0888', '0.1461', 'Table 38 row 3'),
    ('Table 38 (remittances x education, HC1, z)', 'Remit', '-8879.3', '11525.8', 'Table 38 row 4'),
    ('Table 38 (remittances x education, HC1, z)', 'EduExp', '-28421.5', '50967.9', 'Table 38 row 5'),
    ('Table 38 (remittances x education, HC1, z)', 'Remit x EduExp', '-3319.9', '2552.2', 'Table 38 row 6'),
    ('Table 41 (FE, cluster)', 'Net migration', '0.000014', '0.000181', 'Table 41 row 1'),
    ('Table 41 (FE, cluster)', 'Net migration x log(initial GDP)', '-0.0000136', '-0.000001', 'Table 41 row 2'),
    ('Table 42 (extended national model, t(20))', 'const', '-320073', '-145147', 'Table 42 row 1'),
    ('Table 42 (extended national model, t(20))', 'GDP per capita', '-0.0183', '0.0769', 'Table 42 row 2'),
    ('Table 42 (extended national model, t(20))', 'Unemployment', '-1303.2', '604.6', 'Table 42 row 3'),
    ('Table 42 (extended national model, t(20))', 'Urban population', '0.0657', '0.1655', 'Table 42 row 4'),
    ('Table 42 (extended national model, t(20))', 'Remittances', '-1933.3', '2101.3', 'Table 42 row 5'),
    ('Table 42 (extended national model, t(20))', 'FDI (bn USD)', '-24976.9', '31009.1', 'Table 42 row 6'),
    ('Table 42 (extended national model, t(20))', 'Education expenditure', '-4008.9', '17772.7', 'Table 42 row 7'),
    ('Table 45 with TCI (HC1, z)', 'GDPpc', '-0.0113', '0.0946', 'Table 45 row 1'),
    ('Table 45 with TCI (HC1, z)', 'Unemp', '-1186.1', '952.1', 'Table 45 row 2'),
    ('Table 45 with TCI (HC1, z)', 'Urban', '0.0257', '0.1526', 'Table 45 row 3'),
    ('Table 45 with TCI (HC1, z)', 'TCI', '-4942.1', '5236.5', 'Table 45 row 4'),
    ('Table 45 without TCI (HC1, z)', 'GDPpc', '0.0212', '0.0645', 'Table 45 row 1'),
    ('Table 45 without TCI (HC1, z)', 'Unemp', '-1162.4', '937.5', 'Table 45 row 2'),
    ('Table 45 without TCI (HC1, z)', 'Urban', '0.039', '0.1433', 'Table 45 row 3'),
    ('Table 12 pooled OLS (cluster)', 'GDP (bn ALL)', '15.02', '16.77', 'Table 48 row 1'),
    ('Table 12 / 48 FE-region (cluster)', 'GDP (bn ALL)', '-21.24', '-12.64', 'Table 48 row 2'),
    ('Table 12 / 48 FE-region+year (cluster)', 'GDP (bn ALL)', '-24.64', '-21.11', 'Table 48 row 3'),
    ('Table 12 / 48 RE (cluster)', 'GDP (bn ALL)', '9.07', '11.22', 'Table 48 row 4'),
    ('Table 17 (excluding Tirana)', 'GDP (bn ALL)', '-29.13', '21.18', 'Table 48 row 5'),
    ('Table 19 with permits', 'GDP (bn ALL)', '-3.07', '0.14', 'Table 48 row 6'),
    ('Table 48 System GMM GDP (persons per bn ALL)', 'GDP', '6.73', '9.47', 'Table 48 row 7'),
    ('Table 48 Difference GMM GDP (persons per bn ALL)', 'GDP', '-29.39', '-9.01', 'Table 48 row 8'),
    ('Table 27 (ML spatial lag, z)', 'rho (W x NetMigration)', '-0.32', '0.4', 'Table 48 row 9'),
    ('Table 37 (national model 5, t(27))', 'GDP per capita', '0.017', '0.0662', 'Table 48 row 10'),
    ('Table 42 (extended national model, t(20))', 'GDP per capita', '-0.0183', '0.0769', 'Table 48 row 11')
]


def decimals(s):
    return len(s.split('.')[1]) if '.' in s else 0


def main():
    calc = json.load(open('../output/confidence_intervals.json'))['ci']
    counts = {'exact': 0, 'last digit': 0, 'DIFF': 0}
    rows = []
    for table, term, lo, hi, where in EXPECTED:
        c = calc[table][term]
        status = 'exact'
        detail = []
        for label, printed, value in (('lo', lo, c['lo']), ('hi', hi, c['hi'])):
            d = decimals(printed)
            diff_units = abs(value - float(printed)) / 10 ** (-d)
            if diff_units > 1.5:
                status = 'DIFF'
            elif diff_units > 0.5 + 1e-9 and status != 'DIFF':
                status = 'last digit'
            if diff_units > 0.5 + 1e-9:
                detail.append(f"{label}: printed {printed}, script {value:.{d + 1}f}")
        counts[status] += 1
        rows.append((status, where, table, term, lo, hi, '; '.join(detail)))
    for status, where, table, term, lo, hi, detail in rows:
        if status != 'exact':
            print(f"[{status:10s}] {where:<18} {term:<32} ({lo}, {hi})   {detail}")
    print(f"\nChecked {len(rows)} intervals: exact {counts['exact']}, "
          f"last-digit rounding {counts['last digit']}, larger difference {counts['DIFF']}")


if __name__ == "__main__":
    main()
