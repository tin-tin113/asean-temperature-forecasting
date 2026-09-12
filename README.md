# Forecasting Monthly Temperature Change in the ASEAN Region

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Design Science](https://img.shields.io/badge/Framework-Design%20Science%20Research-green.svg)](#-overview)
[![Dataset: FAOSTAT](https://img.shields.io/badge/Data-FAOSTAT%20ET-orange.svg)](http://www.fao.org/faostat/en/#data/ET)

Official open-source repository and reproducibility artifact for the research paper:  
**"Forecasting Monthly Temperature Change in the ASEAN Region: A Data-Driven Framework for Climate Risk Preparedness and Resilience"**

---

## 📌 Overview

Regional climate adaptation planning in the ASEAN region requires comparable, country-specific monthly temperature-change forecasts for near-term risk preparedness. This study implements a quantitative, data-driven **Design Science Research (DSR)** framework to develop, tune, and benchmark monthly temperature anomaly forecasting models across all **10 ASEAN member states**:
* **Equatorial / Maritime:** Brunei Darussalam, Indonesia, Malaysia, the Philippines, Singapore
* **Mainland / Monsoonal:** Cambodia, Lao PDR, Myanmar, Thailand, Viet Nam

Using historical **United Nations FAOSTAT Temperature Change on Land** monthly series (1961–2023, 756 observations per country), the framework systematically evaluates four candidate time-series and machine learning architectures under a standardized chronological pipeline:
1. **Holt-Winters Triple Exponential Smoothing** (Classical statistical baseline)
2. **SARIMA** (Seasonal Autoregressive Integrated Moving Average in state-space form)
3. **Prophet** (Decomposable additive time-series model with changepoint selection)
4. **XGBoost** (Gradient-boosted decision trees with autoregressive lag and rolling features)

---

## 🔬 Key Methodological Highlights

* **Calibrated Spatial Proxy Imputation:** Singapore's historical missing values (358 months in training, 2 in test) are reconstructed using a calibrated Simple Linear Regression (SLR) spatial proxy based on contiguous neighbor Malaysia ($\hat{Y}_{\text{Singapore}, t} = -0.1701 + 1.3203 \cdot X_{\text{Malaysia}, t}$, $R^2 = 0.6111$, $p < 0.0001$, $\text{VIF} = 1.00$).
* **Zero-Leakage Masking Protocol:** Test-period imputations (2 months) are strictly masked out during accuracy evaluations ($n = 118$ for Singapore; $n = 120$ for all other countries). Models are evaluated exclusively against genuine empirical measurements.
* **Rolling-Origin Cross-Validation:** Hyperparameters are optimized per country using rolling validation strictly inside the 1961–2013 training history (636 months) to avoid lookahead bias.
* **10-Year Fixed-Origin Holdout Benchmark:** Final forecast evaluation is executed as a single unassisted 120-month (2014–2023) multi-step projection from December 2013, evaluated across MAE, RMSE, $\text{MAPE}_{\varepsilon}$ ($\varepsilon = 0.1$), sMAPE, and MASE.

---

## 📂 Repository Structure

```text
├── asean_temperature_forecasting.py       # Master forecasting & benchmarking pipeline
├── ASEAN_Temperature(Pre_Process.xlsx     # Input dataset (10 ASEAN member states, 1961-2023)
├── requirements.txt                       # Python environment dependencies
├── LICENSE                                # MIT Open-Source License
├── README.md                              # Repository overview & quickstart guide
│
├── datasets/                              # Processed & imputed data artifacts
│   ├── ASEAN_Temperature_Imputed_SLR.xlsx # Cleaned panel with SLR spatial proxy for Singapore
│   ├── asean_temperature_monthly_panel_imputed.csv
│   ├── cleaned_dataset_with_imputation_flags.csv
│   └── imputation_report.csv              # Audit log for missingness reconstruction
│
├── results/                               # Benchmark logs & experimental tables (cited in paper)
│   ├── tuned_avg_metrics_per_model.csv    # Table III: Aggregate regional performance rankings
│   ├── tuned_best_model_per_country.csv   # Table IV: Winning model assignments per country
│   ├── tuned_selected_hyperparameters.csv # Winning hyperparameter configurations per country
│   ├── tuned_validation_grid_results.csv  # Full cross-validation grid search evaluation logs
│   ├── tuned_per_country_results.csv      # Complete 4-model evaluation across all 10 countries
│   ├── tuned_forecasts_long.csv           # 120-month test forecasts (Figure 2 data)
│   ├── final_test_near_zero_diagnostic.csv# Near-zero anomaly metric stability diagnostic
│   ├── sensitivity_avg_metrics_excluding_singapore.csv
│   └── sensitivity_best_model_per_country_excluding_singapore.csv
│
└── docs/                                  # Extended academic & methodology documentation
    ├── MODELS_AND_METRICS_EXPLAINED.md    # In-depth guide to algorithms, equations, & metrics
    └── STUDY_OVERVIEW_AND_CONCEPTS.md     # Complete study framework, DSR design, & defense notes
```

---

## 📚 In-Depth Documentation

For comprehensive mathematical formulations, model derivations, and research background, consult the dedicated guides in the [`docs/`](docs/) directory:
* 📘 **[Models & Metrics Explained](docs/MODELS_AND_METRICS_EXPLAINED.md)** — Architectural breakdowns of Holt-Winters, SARIMA, Prophet, and XGBoost; exact mathematical formulas for MAE, RMSE, $\text{MAPE}_{\varepsilon}$, sMAPE, and MASE; and analysis of metric stability for near-zero temperature anomalies.
* 📗 **[Study Overview & Concepts](docs/STUDY_OVERVIEW_AND_CONCEPTS.md)** — Design Science Research (DSR) methodology, ASEAN geographic and climatic classifications, Singapore SLR spatial proxy derivation, and defense Q&A guide.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/tin-tin113/asean-temperature-forecasting.git
cd asean-temperature-forecasting
```

### 2. Install Dependencies
Ensure you have Python 3.10 or higher installed:
```bash
pip install -r requirements.txt
```

### 3. Reproduce the Paper Results
To execute the full hyperparameter grid search, model refitting, and holdout evaluation as presented in the paper:
```bash
python asean_temperature_forecasting.py --src "ASEAN_Temperature(Pre_Process.xlsx" --grid balanced --selection-metric MAE --validation-mode rolling
```

*(Optional fast dry-run for environment and pipeline testing):*
```bash
python asean_temperature_forecasting.py --src "ASEAN_Temperature(Pre_Process.xlsx" --grid tiny --selection-metric MAE
```

---

## 📊 Published Benchmark Results (2014–2023 Holdout)

### Table III: Aggregate Forecast Error Across ASEAN
| Rank | Model Family | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) | MASE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Prophet** | **0.5051** | **0.6248** | 60.96 | **47.96** | **0.9727** |
| 2 | SARIMA | 0.5221 | 0.6398 | 57.40 | 51.67 | 0.9745 |
| 3 | XGBoost | 0.6285 | 0.7481 | 66.57 | 66.17 | 1.2292 |
| 4 | Holt-Winters | 0.6490 | 0.7659 | 58.03 | 76.35 | 1.1356 |

> **Key Insight:** Prophet achieves the lowest regional MAE and RMSE with the narrowest penalty spread ($\Delta = 0.12^\circ\text{C}$), demonstrating resilience against acute temperature extremes. Although Holt-Winters achieves a deceptively low MAPE ($58.03\%$) due to near-zero anomaly denominators, symmetric sMAPE ($76.35\%$) and MAE reveal its true error magnitude.

### Table IV: Best Model Assigned Per Country
| Country | Assigned Architecture | Selected Hyperparameters | MAE (°C) | RMSE (°C) | sMAPE (%) | MASE |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Brunei Darussalam** | SARIMA | $(0,1,1)(1,1,1)_{12}$ | 0.2582 | 0.3418 | 26.79 | **0.6394** |
| **Cambodia** | Prophet | $\text{cps}=0.01, \text{sps}=1.0$ | 0.4985 | 0.6324 | 54.62 | **0.8367** |
| **Indonesia** | Holt-Winters | Additive trend & seasonality | 0.2259 | 0.2985 | 23.54 | **0.7751** |
| **Lao PDR** | Prophet | $\text{cps}=0.01, \text{sps}=10.0$ | 0.7143 | 0.9056 | 64.80 | **0.8733** |
| **Malaysia** | SARIMA | $(1,1,2)(2,1,1)_{12}$ | 0.2614 | 0.3350 | 24.35 | **0.7641** |
| **Myanmar** | Prophet | $\text{cps}=0.1, \text{sps}=0.1$ | 0.5520 | 0.6584 | 45.76 | 1.0329 |
| **Philippines** | Holt-Winters | Additive trend, un-damped | 0.2844 | 0.3537 | 23.50 | **0.7576** |
| **Singapore** | XGBoost | $\text{lr}=0.05, \text{depth}=2$ | 0.4933 | 0.6308 | 36.75 | **0.9321** |
| **Thailand** | XGBoost | $\text{lr}=0.03, \text{depth}=3$ | 0.5782 | 0.7288 | 60.08 | **0.8780** |
| **Viet Nam** | Prophet | $\text{cps}=0.01, \text{sps}=10.0$ | 0.6402 | 0.7884 | 67.01 | **0.8346** |

> **Operational Significance:** 9 out of 10 countries achieve $\text{MASE} < 1.0$, demonstrating that tailored time-series modeling delivers genuine predictive value over in-sample seasonal persistence. Myanmar ($\text{MASE} = 1.0329$) defines the boundary where univariate models encounter predictability limits and motivate external ocean-atmosphere covariates (ENSO, SST).

---

## 📖 Citation

If you utilize this pipeline, dataset, or methodology in your research, please cite:

### IEEE Style
```text
M. C. Diasanta, K. O. J. Crisostomo, R. Lucero, and E. J. P. Bibangco, "Forecasting Monthly Temperature Change in the ASEAN Region: A Data-Driven Framework for Climate Risk Preparedness and Resilience," in Proc. 2026 11th Int. Conf. Inf. Technol. Digit. Appl. (ICITDA), Lombok, Indonesia, Nov. 2026.
```

### BibTeX
```bibtex
@inproceedings{diasanta2026asean,
  author    = {Diasanta, Martin C. and Crisostomo, Kymer Oriel J. and Lucero, Raffy and Bibangco, El Jireh P.},
  title     = {Forecasting Monthly Temperature Change in the ASEAN Region: A Data-Driven Framework for Climate Risk Preparedness and Resilience},
  booktitle = {Proceedings of the 2026 11th International Conference on Information Technology and Digital Applications (ICITDA)},
  address   = {Lombok, Indonesia},
  month     = {November},
  year      = {2026}
}
```

---

## 👥 Authors & Affiliations
* **Martin C. Diasanta** (`martin.diasanta@chmsu.edu.ph`)
* **Kymer Oriel J. Crisostomo** (`kymeroriel.crisostomo@chmsu.edu.ph`)
* **Raffy Lucero** (`raffy.lucero@chmsu.edu.ph`)
* **El Jireh P. Bibangco** (`ej.bibangco@chmsu.edu.ph`)

*Department of Information Technology, College of Computer Studies, Carlos Hilado Memorial State University, Talisay City, Negros Occidental, Philippines.*

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
