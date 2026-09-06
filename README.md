# Internal Migration and Regional Development in Albania (2000–2023)
### Replication Package

This repository contains the data and Python analysis code supporting the paper
**"Internal Migration and Regional Development in Albania: A Panel Data Analysis of
Regional Inequalities (2000–2023)"** (IJITIS, Submission ID 610), final revised version.

All 30 scripts are numbered in the exact order the corresponding tables and figures
appear in the final manuscript (Tables 1–43, Figures 1–9). Running them in order
reproduces every quantitative result in the paper from the original source data.

## Repository structure

```
.
├── data/
│   ├── Databaza_Scopus.xlsx                Regional (INSTAT) and national (World Bank)
│   │                                        source data, and raw SPSS output
│   ├── API_ALB_DS2_en_excel_v2_2902.xls    Complete World Bank World Development
│   │                                        Indicators series for Albania (1960–2025)
│   ├── lejet-e-ndërtimit-...-2020-2024.xlsx INSTAT building permits by prefecture
│   └── tab1-2015.xlsx                      INSTAT household consumption survey, 2014–2015
├── scripts/                                30 numbered Python scripts (see table below)
├── output/                                 JSON/CSV intermediate results (regenerated
│                                            automatically when scripts are run)
├── figures/                                9 PNG figures (regenerated automatically)
├── requirements.txt
├── LICENSE
└── README.md
```

## Script-to-table/figure mapping

| Script | Produces |
|---|---|
| 01_extract_data.py | Tables 1–3 |
| 02_figure1_netmigration_trend.py | Figure 1 |
| 03_model1_pooled_ols.py | Tables 4–6 |
| 04_model1_robust_se_table.py | Table 7 |
| 05_unemployment_volatility_interaction.py | Table 8 |
| 06_model1_diagnostics.py | Table 9 |
| 07_panel_fe_re_hausman.py | Tables 10–11 |
| 08_outlier_diagnostics.py | Table 12 |
| 09_sensitivity_analysis.py | Table 13 |
| 10_gdp_core_interaction.py | Table 14 |
| 11_building_permits.py | Figure 2, Tables 15–16 |
| 12_consumption_survey.py | Table 17 |
| 13_correlation_matrix.py | Table 18 |
| 14_clustering.py | Table 19, Figure 3 |
| 15_cluster_validation.py | Tables 20–21, Figure 4 |
| 16_spatial_econometrics.py | Tables 22–24 |
| 17_inequality_indicators.py | Table 25, Figure 5 |
| 18_additional_inequality_indices.py | Table 26 |
| 19_sigma_convergence.py | Tables 27–28 |
| 20_migration_concentration_threshold.py | Table 29 |
| 21_beta_convergence.py | Table 30, Figures 6–7 |
| 22_model2_national.py | Tables 31–33 |
| 23_remittances_education_interaction.py | Table 34 |
| 24_figure8_national_timeseries.py | Figure 8 |
| 25_adf_granger_causality.py | Tables 35–36 |
| 26_reverse_causality.py | Table 37 |
| 27_model2b_extended.py | Table 38 |
| 28_model2_diagnostics.py | Table 39 |
| 29_wdi_pca_tci.py | Table 40, Figure 9 |
| 30_model2_augmented_tci.py | Tables 41–43 |

## How to reproduce the analysis

1. Install Python 3.10+ and the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the scripts **in numerical order** from inside the `scripts/` folder — several
   scripts read intermediate files produced by earlier ones:
   ```bash
   cd scripts
   for f in $(ls *.py | sort -V); do python "$f"; done
   ```
   Or run them individually in order (01, 02, 03, ... 30).
3. Results are written to `output/` (JSON/CSV) and `figures/` (PNG). Each script also
   prints its main results to the console.

All 30 scripts were tested end-to-end on a clean environment and run without errors.

## Data sources

- **INSTAT** (Institute of Statistics of Albania) — regional net migration, GDP (Gross
  Value Added), unemployment, population, building permits, and household consumption.
  https://www.instat.gov.al
- **World Bank, World Development Indicators** — national-level net migration, GDP per
  capita, unemployment, urban population, remittances, FDI, education expenditure, and
  the Worldwide Governance Indicators used to build the Territorial Capital Index.
  https://databank.worldbank.org/source/world-development-indicators

## Key packages and methods

- pandas, numpy, scipy — data handling and statistics
- statsmodels — OLS, diagnostic tests (VIF, Durbin–Watson, Breusch–Pagan, Jarque–Bera,
  ADF), Granger causality
- linearmodels — panel Fixed-Effects / Random-Effects estimation, Hausman test
- scikit-learn — k-means and hierarchical clustering, silhouette / Davies–Bouldin
  validation, Principal Component Analysis
- libpysal, esda, spreg — spatial weights matrix, Global Moran's I, Maximum-Likelihood
  Spatial Lag Model

## Citation

If you use this data or code, please cite:

> Kazani, M. (2026). Internal Migration and Regional Development in Albania: A Panel
> Data Analysis of Regional Inequalities (2000–2023). *International Journal of
> Innovative Technology and Interdisciplinary Sciences*. [DOI to be added upon publication]

## License

Code: MIT License (see `LICENSE`). Data: redistributed from public INSTAT and World
Bank sources; refer to those sources' own terms of use for the underlying data.

## Contact

Migena Kazani — Department of Statistics and Applied Informatics, Faculty of Economy,
University of Tirana. migena.musallari@unitir.edu.al
