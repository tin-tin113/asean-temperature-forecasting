# Forecasting Monthly Temperature Change in the ASEAN Region

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Design Science](https://img.shields.io/badge/Framework-Design%20Science%20Research-green.svg)](#methodology)

Official open-source repository and reproducibility artifact for the research paper:  
**"Forecasting Monthly Temperature Change in the ASEAN Region: A Data-Driven Framework for Climate Risk Preparedness and Regional Adaptation"**

---

## 📌 Overview

Regional climate adaptation planning in the ASEAN region requires comparable, country-specific monthly temperature-change forecasts for near-term risk preparedness. This study implements a quantitative, data-driven, **Design Science Research (DSR)** framework to develop, tune, and benchmark monthly temperature anomaly forecasting models for all **10 ASEAN member states**:
* Brunei Darussalam
* Cambodia
* Indonesia
* Lao PDR
* Malaysia
* Myanmar
* Philippines
* Singapore
* Thailand
* Viet Nam

Using historical **FAOSTAT Temperature Change on Land** monthly series (1961–2023, 756 observations per country), the framework systematically evaluates four candidate model families under a standardized chronological pipeline:
1. **Holt-Winters Triple Exponential Smoothing** (Classical statistical baseline)
2. **SARIMA** (Seasonal Autoregressive Integrated Moving Average in state-space form)
3. **Prophet** (Decomposable additive time-series model)
4. **XGBoost** (Gradient-boosted decision trees with lag/rolling features)

---

## 🔬 Key Methodological Innovations

* **Leak-Free Regional Ridge Imputer:** Singapore's historical missing values (358 months in training period, 2 in test period) are estimated using a regional Ridge Regression model trained strictly on pre-2014 observed data from Malaysia, Indonesia, Brunei Darussalam, trend, and harmonic seasonal indicators.
* **Zero Test-Leakage Masking:** Imputed test-period values are excluded from final accuracy metric calculations; models are strictly evaluated against actual empirical observations.
* **Chronological Rolling-Window Tuning:** Hyperparameters are optimized per country using rolling-window cross-validation on the 1961–2013 training history (636 months).
* **10-Year Holdout Testing:** Final forecast evaluation is conducted on the 2014–2023 test period (120 months) across MAE, RMSE, $\epsilon$-adjusted MAPE, sMAPE, and MASE.

---

## 📂 Repository Structure

```text
├── asean_temperature_forecasting.py       # Main forecasting & benchmarking pipeline
├── ASEAN_Temperature(Pre_Process.xlsx     # Starting raw dataset (10 ASEAN countries, 1961-2023)
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore configuration
├── LICENSE                                # MIT License
│
├── Final Paper/
│   └── m29805-diasanta paper.pdf          # Final paper preprint (IEEE format)
│
└── Results / Benchmark CSVs/
    ├── tuned_avg_metrics_per_model.csv    # Table II: Aggregate regional rankings
    ├── tuned_best_model_per_country.csv   # Table III: Best model per country
    ├── tuned_selected_hyperparameters.csv # Optimal hyperparameters per country/model
    ├── tuned_per_country_results.csv      # Full 4-model evaluation across 10 countries
    ├── tuned_forecasts_long.csv           # 120-month test forecasts (Figure 2 data)
    ├── cleaned_dataset_with_imputation_flags.csv  # 7,560-row preprocessed dataset
    ├── imputation_report.csv              # Audit log for Singapore imputation
    ├── sensitivity_avg_metrics_excluding_singapore.csv # Robustness test summary
    └── sensitivity_best_model_per_country_excluding_singapore.csv
```

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

*(Optional fast dry-run for code testing):*
```bash
python asean_temperature_forecasting.py --src "ASEAN_Temperature(Pre_Process.xlsx" --grid tiny --selection-metric MAE
```

---

## 📊 Key Results (Published Paper)

### Table II: Aggregate Forecast Error Across ASEAN (2014–2023 Test Period)
| Rank | Model | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) | MASE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Prophet** | **0.5008** | **0.6198** | 60.74 | **47.54** | **0.9823** |
| 2 | SARIMA | 0.5175 | 0.6346 | 57.20 | 51.22 | 0.9835 |
| 3 | XGBoost | 0.6293 | 0.7492 | 66.56 | 66.24 | 1.2454 |
| 4 | Holt-Winters | 0.6482 | 0.7648 | 57.96 | 76.30 | 1.1490 |

### Table III: Best Model Assigned Per Country
| Country | Best Model | MAE (°C) | RMSE (°C) | sMAPE (%) | MASE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Brunei Darussalam** | SARIMA | 0.2582 | 0.3418 | 26.79 | 0.6394 |
| **Cambodia** | Prophet | 0.4985 | 0.6324 | 54.62 | 0.8367 |
| **Indonesia** | Holt-Winters | 0.2259 | 0.2985 | 23.54 | 0.7751 |
| **Lao PDR** | Prophet | 0.7143 | 0.9056 | 64.80 | 0.8733 |
| **Malaysia** | SARIMA | 0.2614 | 0.3350 | 24.35 | 0.7641 |
| **Myanmar** | Prophet | 0.5520 | 0.6584 | 45.76 | 1.0329 |
| **Philippines** | Holt-Winters | 0.2844 | 0.3537 | 23.50 | 0.7576 |
| **Singapore** | XGBoost | 0.5013 | 0.6414 | 37.51 | 1.0941 |
| **Thailand** | XGBoost | 0.5782 | 0.7288 | 60.08 | 0.8780 |
| **Viet Nam** | Prophet | 0.6402 | 0.7884 | 67.01 | 0.8346 |

---

## 📖 Citation

If you utilize this pipeline, dataset, or methodology in your research, please cite our paper:

### IEEE Style
```text
M. C. Diasanta, K. O. J. Crisostomo, R. Lucero, and E. J. P. Bibangco, "Forecasting Monthly Temperature Change in the ASEAN Region: A Data-Driven Framework for Climate Risk Preparedness and Regional Adaptation," College of Computer Studies, Carlos Hilado Memorial State University, Talisay City, Philippines, 2026.
```

### BibTeX
```bibtex
@article{diasanta2026forecasting,
  title={Forecasting Monthly Temperature Change in the ASEAN Region: A Data-Driven Framework for Climate Risk Preparedness and Regional Adaptation},
  author={Diasanta, Martin C. and Crisostomo, Kymer Oriel J. and Lucero, Raffy and Bibangco, El Jireh P.},
  journal={College of Computer Studies, Carlos Hilado Memorial State University},
  year={2026}
}
```

---

## 👥 Authors & Affiliation
* **Martin C. Diasanta** (`martin.diasanta@chmsu.edu.ph`)
* **Kymer Oriel J. Crisostomo** (`kymeroriel.crisostomo@chmsu.edu.ph`)
* **Raffy Lucero** (`raffy.lucero@chmsu.edu.ph`)
* **El Jireh P. Bibangco** (`ej.bibangco@chmsu.edu.ph`)

*College of Computer Studies, Carlos Hilado Memorial State University, Talisay City, Negros Occidental, Philippines.*

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).
