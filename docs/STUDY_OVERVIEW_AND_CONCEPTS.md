# Comprehensive Study Guide: ASEAN Monthly Temperature Change Forecasting (1961–2023)

> **Paper Title:** *Forecasting Monthly Temperature Change in the ASEAN Region: A Data-Driven Framework for Climate Risk Preparedness and Regional Adaptation*  
> **Authors:** Martin C. Diasanta, Kymer Oriel J. Crisostomo, Raffy Lucero, El Jireh P. Bibangco  
> **Affiliation:** College of Computer Studies, Carlos Hilado Memorial State University, Talisay City, Negros Occidental, Philippines  
> **Target Conference:** The 11th International Conference on Information Technology and Digital Applications (ICITDA 2026)  
> **Code & Data Artifact:** Open-source reproducible Python pipeline (`asean_temperature_forecasting.py`)

---

## 🧭 Table of Contents
1. [The 60-Second Elevator Pitch](#1-the-60-second-elevator-pitch)
2. [Background & The Real-World Problem](#2-background--the-real-world-problem)
3. [The 4 Formal Objectives of the Study](#3-the-4-formal-objectives-of-the-study)
4. [Core Concepts Explained in Plain English](#4-core-concepts-explained-in-plain-english)
5. [The Dataset & Time Splits](#5-the-dataset--time-splits)
6. [Methodological Innovation: Zero-Leakage Singapore Imputation](#6-methodological-innovation-zero-leakage-singapore-imputation)
7. [The 4 Forecasting Models (How They Work & Why Chosen)](#7-the-4-forecasting-models-how-they-work--why-chosen)
8. [The 5 Evaluation Metrics (What They Mean & Why They Disagree)](#8-the-5-evaluation-metrics-what-they-mean--why-they-disagree)
9. [Key Empirical Results (The Heart of the Study)](#9-key-empirical-results-the-heart-of-the-study)
10. [Geographic Climatology: Why Different Countries Need Different Models](#10-geographic-climatology-why-different-countries-need-different-models)
11. [Policy & Adaptation Insights (How Governments Can Use This)](#11-policy--adaptation-insights-how-governments-can-use-this)
12. [Limitations & Future Roadmap](#12-limitations--future-roadmap)
13. [Panel Defense Cheat Sheet (Frequently Asked Questions)](#13-panel-defense-cheat-sheet-frequently-asked-questions)

---

## 1. The 60-Second Elevator Pitch

> *"Climate change is accelerating across Southeast Asia, but national adaptation offices lack a fair, standardized way to compare temperature forecasts across member states. In this study, we built an end-to-end, leak-free benchmarking framework across all 10 ASEAN countries spanning 63 years (1961–2023). We tuned and compared four major forecasting model families—Holt-Winters, SARIMA, Prophet, and XGBoost—on a 10-year holdout test set (2014–2023).  
> 
> Our central finding is that **there is no single universally superior model for ASEAN**. While Prophet achieved the best regional average error, it was optimal in only 4 out of 10 countries. Island/maritime countries perform best with classical statistical models, while mainland and complex urban zones require non-linear models. We also discovered that for 8 countries, models beat simple seasonal repetition, but in high-noise microclimates (Singapore and Myanmar), univariate models reach their predictability limit, proving the urgent need for satellite and ocean covariates."*

---

## 2. Background & The Real-World Problem

### Why Southeast Asia (ASEAN)?
* Southeast Asia is recognized as one of the world's most climate-vulnerable regions.
* It faces intensifying heatwaves, erratic monsoon seasons, prolonged droughts, and agricultural disruptions.
* Sectors affected include:
  * **Agriculture & Rice Yields:** Sensitive to subtle monthly warming anomalies during flowering cycles.
  * **Water Security:** Reservoir allocation depends on seasonal heating and evaporation.
  * **Energy Grids:** Cooling and air-conditioning demand spikes during high heat anomalies.
  * **Disaster Preparedness:** Early contingency buffering against extreme heatwave surges.

### The Scientific Research Gap
Prior research had two major flaws:
1. **Coarse Regional Averages:** Many global climate studies lump Southeast Asia into a single regional average. This obscures national realities (e.g., equatorial Indonesia behaves completely differently from mountainous, continental Lao PDR).
2. **Disconnected Single-Country Studies:** Studies that did examine individual countries used completely different datasets, different time periods, different preprocessing tricks, and different error metrics. You could never fairly compare if Model A in Thailand was better than Model B in the Philippines.

### The Solution: A Design Science Research (DSR) Framework
Instead of inventing a brand-new algorithm from scratch, this study adopts a **Design Science Research** approach: designing, building, and benchmarking a **standardized, leak-free forecasting pipeline** applied under the exact same validation rules across all 10 member states.

---

## 3. The 4 Formal Objectives of the Study

As written in Section I of the manuscript, the study set out to accomplish four concrete objectives:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                THE 4 FORMAL OBJECTIVES                  │
                  └─────────────────────────────────────────────────────────┘
                                               │
         ┌─────────────────────┬───────────────┴───────────────┬─────────────────────┐
         ▼                     ▼                               ▼                     ▼
   [Objective 1]         [Objective 2]                   [Objective 3]         [Objective 4]
Characterize 63-year    Standardized Benchmark          Country-Specific       Adaptation Planning
temperature patterns    of 4 Model Families             Model Assignment       Decision Framework
(1961–2023 FAOSTAT)     (Holt-Winters, SARIMA,          (Best model for each   (MAE for baseline,
across all 10 states.   Prophet, XGBoost).              of the 10 countries).  RMSE-MAE for shocks,
                                                                               MASE for gating).
```

1. **Characterize Long-Term Patterns:** Examine 63 years (756 monthly observations per country) of FAOSTAT monthly temperature change records across all 10 ASEAN countries.
2. **Benchmark 4 Model Families Fairly:** Tune and evaluate Holt-Winters, SARIMA, Prophet, and XGBoost using identical rolling-window cross-validation on pre-2014 data.
3. **Identify the Best Model per Country:** Determine the specific optimal model for each of the 10 member states rather than forcing an arbitrary one-size-fits-all model.
4. **Translate Findings into Adaptation Policy:** Formulate a concrete decision-support guide for national climate planners and disaster agencies.

---

## 4. Core Concepts Explained in Plain English

### Concept A: What is a "Temperature Anomaly" (Temperature Change)?
* **It is NOT absolute temperature:** The data does not say "Manila was 32.5°C in May."
* **It is a Deviation from Normal:** It measures how much warmer or cooler a specific month was compared to a historical baseline (1951–1980 climatological normal for that month).
* *Example:* If Bangkok's historical average for April is 30.0°C, and in April 2023 it was 31.2°C, the **anomaly is $+1.2^\circ\text{C}$**.
* **Why this matters mathematically:** Anomaly values fluctuate around **$0.0^\circ\text{C}$** (often crossing from negative to positive). This causes specific problems for percentage metrics like MAPE (explained in [Section 8](#8-the-5-evaluation-metrics-what-they-mean--why-they-disagree)).

### Concept B: What is "Data Leakage" and Why is it Fatal in Science?
* **Data Leakage** occurs when information from the future (the test set) accidentally contaminates the past (the training set or preprocessing steps).
* If a model "peeks" into test data during preprocessing or tuning, its test scores look artificially amazing, but it will fail catastrophically in the real world.
* **Your Paper's Core Strength:** Absolute zero leakage. Preprocessing, imputer fitting, and hyperparameter tuning were restricted strictly to the pre-2014 training data.

### Concept C: Design Science Research (DSR)
* Rather than just describing a phenomenon ("temperatures are rising"), DSR focuses on **building a functional artifact** (a validated, automated, reproducible computational pipeline) that solves an organizational problem (climate adaptation planning).

---

## 5. The Dataset & Time Splits

### The Dataset
* **Source:** United Nations Food and Agriculture Organization (**UN FAOSTAT**) *Temperature Change on Land* database.
* **Coverage:** 10 ASEAN Countries:
  * *Maritime/Equatorial:* Brunei Darussalam, Indonesia, Malaysia, Philippines, Singapore.
  * *Continental/Monsoon:* Cambodia, Lao PDR, Myanmar, Thailand, Viet Nam.
* **Span:** January 1961 to December 2023 (**63 full years = 756 consecutive monthly observations per country**).
* **Total Records:** $756 \times 10 = 7,560$ monthly observations.

### Chronological Time Splits

```
1961                                                        2014                   2023
├─────────────────────────────────────────────────────────────┼──────────────────────┤
│               TRAINING PERIOD (636 months)                  │  TEST PERIOD (120 m) │
│  • Used to train models                                     │  • Strictly unseen   │
│  • Used for internal rolling-window tuning                  │  • Final 10-year     │
│  • Used to calibrate Singapore SLR proxy                    │    holdout benchmark │
```

* **Training Set:** January 1961 – December 2013 (**636 months / 53 years**).
* **Holdout Test Set:** January 2014 – December 2023 (**120 months / 10 full years**).
* **Internal Hyperparameter Tuning:** Rolling-window walk-forward cross-validation on pre-2014 data:
  * *Fold 1:* Train 1961–2003 $\rightarrow$ Validate on 2004–2008 (60 months).
  * *Fold 2:* Train 1961–2008 $\rightarrow$ Validate on 2009–2013 (60 months).
  * The best hyperparameter configuration chosen had the lowest average Validation MAE across folds.

---

## 6. Methodological Innovation: Zero-Leakage Singapore Imputation

### The Challenge
* 9 of the 10 ASEAN countries had 100% complete records from 1961 to 2023.
* **Singapore had 360 missing monthly records** (358 missing months in the 1961–2013 training period and 2 isolated missing months in the 2014–2023 test period).

### Why Standard Imputation Fails
* Filling missing values with monthly medians or linear interpolation destroys multi-decade warming trends and flattens extreme climate cycles.

### Your Solution: Calibrated Simple Linear Regression (SLR) Spatial Proxy
* **Collinearity Diagnostics:** Multi-country candidate regression models exhibited severe multicollinearity among regional predictors (Malaysia $\text{VIF} = 11.81$, Brunei $\text{VIF} = 7.95$, mutual $r > 0.90$).
* **Parsimony and Contiguity:** In accordance with the principle of parsimony and to eliminate variance inflation (reducing VIF to 1.00), Malaysia was adopted as the single explanatory variable due to its immediate geographical contiguity across the Johor Strait (< 1 km), shared monsoonal regime, and 100% complete historical record:
  $$\hat{Y}_{\text{Singapore}, t} = -0.1701 + 1.3203 \cdot X_{\text{Malaysia}, t}$$
  ($R^2 = 0.6111, F = 619.2, p < 0.0001, t = 24.88, N = 396$).
* **Physical Significance:** The calibrated slope ($\beta_1 = 1.3203$) physically captures Singapore's amplified urban microclimate and heat-island dynamics relative to Malaysia's nationally aggregated baseline.
* **Strict Leakage Prevention Guarantee:**
  1. The SLR spatial proxy was calibrated across all $N = 396$ observed overlapping historical months (May 1978–December 2023).
  2. For the 2 test months in Singapore (April 2015 and January 2023) that were imputed to allow autoregressive lag continuity, **those 2 months were strictly masked out during final test metric calculation**. All evaluation metrics (MAE, RMSE, sMAPE, MASE) were computed strictly against genuine empirical measurements.
  3. **Sensitivity Analysis:** Running the pipeline with and without Singapore yields identical model selection for all 9 other countries, proving that the findings are robust and uncompromised.

---

## 7. The 4 Forecasting Models (How They Work & Why Chosen)

| Model Family | Paradigm | Core Mechanism | Why Included in the Study |
| :--- | :--- | :--- | :--- |
| **Holt-Winters** | Classical Statistical | Triple Exponential Smoothing: captures level ($\ell_t$), trend ($b_t$), and 12-month additive seasonal factors ($s_t$). | The standard, highly interpretable baseline for seasonal time series. |
| **SARIMA** | Classical Statistical / State-Space | Box-Jenkins Seasonal Autoregressive Integrated Moving Average: $(p,d,q) \times (P,D,Q)_{12}$ implemented via state-space Kalman filtering. | Mathematically rigorous model for autocorrelation and seasonal differencing. |
| **Prophet** | Decomposable Additive (Modern) | Decomposes series into piecewise linear/logistic trend, Fourier series annual seasonality, and changepoint detection: $y(t) = g(t) + s(t) + \epsilon_t$. | Specifically engineered by Meta for automated trend shifts and robust handling of seasonal variations. |
| **XGBoost** | Machine Learning (Non-Linear) | Gradient Boosted Decision Trees trained on autoregressive lag features ($L \in \{1..12, 24\}$) and rolling statistics ($w=3, 6, 12$). Multi-step forecasts are produced recursively. | Evaluates whether non-linear decision trees with recursive feedback can outperform classical linear/additive models. |

---

## 8. The 5 Evaluation Metrics (What They Mean & Why They Disagree)

A core focus of your paper's revision is that **no single metric tells the whole story**. You evaluated 5 complementary metrics:

### 1. MAE (Mean Absolute Error) — *Primary Metric*
* **Formula:** $\text{MAE} = \frac{1}{H} \sum |y_t - \hat{y}_t|$
* **Plain English:** The average forecast error in actual degrees Celsius (°C).
* **Intuition:** If MAE is $0.25^\circ\text{C}$, your forecast is off by an average of a quarter of a degree. It treats all errors proportionally (linear penalty).

### 2. RMSE (Root Mean Squared Error) — *Shock Sensitivity*
* **Formula:** $\text{RMSE} = \sqrt{\frac{1}{H} \sum (y_t - \hat{y}_t)^2}$
* **Plain English:** Measures error in °C, but squares each mistake before averaging.
* **Intuition:** A $2.0^\circ\text{C}$ miss is penalized 4 times more than a $1.0^\circ\text{C}$ miss!
* **The Penalty Spread ($\Delta = \text{RMSE} - \text{MAE}$):**
  * If $\Delta$ is small ($\approx 0.07^\circ\text{C}$), errors are stable and consistent.
  * If $\Delta$ is large ($> 0.18^\circ\text{C}$), the model occasionally suffers massive forecast blunders (e.g., completely missing an El Niño heatwave spike).

### 3. MAPE vs. sMAPE — *The Zero-Crossing Trap*
* **Standard MAPE Formula:** $\frac{100\%}{H} \sum \left|\frac{y_t - \hat{y}_t}{y_t}\right|$
* **The Problem:** In temperature anomalies, actual values hover near zero (e.g., $y_t = +0.02^\circ\text{C}$). If the model predicts $+0.22^\circ\text{C}$, the error is only $0.20^\circ\text{C}$, but MAPE calculates:
  $$\frac{|0.02 - 0.22|}{0.02} \times 100\% = 1,000\% \text{ error!}$$
  This division-by-zero distortion causes standard MAPE to blow up artificially.
* **The Discrepancy in Table II:** Holt-Winters had the lowest MAPE ($58.03\%$) despite having the **worst** MAE ($0.6490^\circ\text{C}$), RMSE, and sMAPE. This occurred because Holt-Winters made slightly smaller errors in a few near-zero months, artificially lowering the denominator explosion.
* **The Solution: Symmetric MAPE (sMAPE):** Bounds the percentage by dividing by $(|y_t| + |\hat{y}_t|)/2$, capping individual errors at 200%. Under sMAPE, Prophet clearly wins ($47.96\%$) and Holt-Winters drops to last place ($76.35\%$).

### 4. MASE (Mean Absolute Scaled Error) — *The Operational Reality Check*
* **Formula:** $\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{seasonal naive in-sample}}}$
* **Plain English:** Compares your model against the dumbest possible seasonal benchmark: **"Assume this month's temperature anomaly will be identical to the same month last year"** ($y_t = y_{t-12}$).
* **Interpretation:**
  * **$\text{MASE} < 1.0$ (WIN):** The model is smarter than repeating last year's calendar month. Genuine predictive skill achieved.
  * **$\text{MASE} = 1.0$ (TIE):** The model is no better than repeating last year.
  * **$\text{MASE} > 1.0$ (FAIL):** The model is actively worse than just looking at last year's calendar month!

---

## 9. Key Empirical Results (The Heart of the Study)

### Table II: Regional Aggregate Performance (120-Month Test Period across all 10 Countries)

| Rank | Model | MAE (°C) | RMSE (°C) | MAPE (%) | sMAPE (%) | MASE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Prophet** | **0.5051** | **0.6248** | 60.96 | **47.96** | **0.9727** |
| 🥈 2 | SARIMA | 0.5221 | 0.6398 | 57.40 | 51.67 | 0.9745 |
| 🥉 3 | XGBoost | 0.6285 | 0.7481 | 66.57 | 66.17 | 1.2292 |
| 4 | Holt-Winters | 0.6490 | 0.7659 | **58.03\*** | 76.35 | 1.1356 |

*\*Note: Holt-Winters' MAPE score is an artifact of near-zero denominators (see Section 8).*

#### Key Takeaways from Table II:
1. **Prophet wins in regional aggregate:** It beats SARIMA by 3.3% in MAE, XGBoost by 19.6%, and Holt-Winters by 22.2%.
2. **MAE and RMSE agree:** Prophet achieves both the lowest typical error (MAE) and fewest catastrophic outlier errors (RMSE).
3. **Aggregate MASE is only marginally below 1.0:** Prophet ($0.9727$) and SARIMA ($0.9745$) beat seasonal naive by less than 2% when averaged across the whole region.

---

### Table III: Country-by-Country Best Model Assignment

| Country | Optimal Model | MAE (°C) | RMSE (°C) | sMAPE (%) | MASE | Status vs. Naive Benchmark |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Brunei Darussalam** | **SARIMA** | **0.2582** | 0.3418 | 26.79 | **0.6394** | 🟢 **Beats Naive by 36.1%** |
| **Cambodia** | **Prophet** | 0.4985 | 0.6324 | 54.62 | **0.8367** | 🟢 Beats Naive by 16.3% |
| **Indonesia** | **Holt-Winters** | **0.2259** | 0.2985 | 23.54 | **0.7751** | 🟢 Beats Naive by 22.5% |
| **Lao PDR** | **Prophet** | 0.7143 | 0.9056 | 64.80 | **0.8733** | 🟢 Beats Naive by 12.7% |
| **Malaysia** | **SARIMA** | **0.2614** | 0.3350 | 24.35 | **0.7641** | 🟢 Beats Naive by 23.6% |
| **Myanmar** | **Prophet** | 0.5520 | 0.6584 | 45.76 | **1.0329** | 🔴 **Fails to Beat Naive ($\text{MASE} > 1.0$)** |
| **Philippines** | **Holt-Winters** | **0.2844** | 0.3537 | 23.50 | **0.7576** | 🟢 Beats Naive by 24.2% |
| **Singapore** | **XGBoost** | 0.4933 | 0.6308 | 36.75 | **0.9321** | 🔴 **Fails to Beat Naive ($\text{MASE} > 1.0$)** |
| **Thailand** | **XGBoost** | 0.5782 | 0.7288 | 60.08 | **0.8780** | 🟢 Beats Naive by 12.2% |
| **Viet Nam** | **Prophet** | 0.6402 | 0.7884 | 67.01 | **0.8346** | 🟢 Beats Naive by 16.5% |

---

## 10. Geographic Climatology: Why Different Countries Need Different Models

The study reveals two breakthrough insights regarding geography and predictability:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               GEOGRAPHIC ARCHETYPES                                    │
├───────────────────────────────────┬────────────────────────────────────────────────────┤
│ 🌊 MARITIME EQUATORIAL ARCHETYPE  │ 🏔️ CONTINENTAL MONSOON ARCHETYPE                   │
│ (Indonesia, Philippines,          │ (Cambodia, Lao PDR, Myanmar, Viet Nam)             │
│  Malaysia, Brunei)                │                                                    │
│ • High ocean thermal inertia      │ • Large continental seasonal swings                │
│ • Narrow error range (MAE < 0.29) │ • Higher error range (MAE 0.50 – 0.71)             │
│ • Low shock gap (Δ ≈ 0.07°C)      │ • Pronounced shock gap (Δ ≈ 0.15 – 0.19°C)         │
│ • WINNERS: Holt-Winters & SARIMA  │ • WINNER: Prophet (Flexible trend & Fourier terms) │
└───────────────────────────────────┴────────────────────────────────────────────────────┘
```

### Breakthrough Insight 1: "No Universally Superior Model"
* Prophet won regional average rankings, but it was optimal in **only 4 out of 10 countries**!
* In maritime nations (Indonesia, Philippines), simple Holt-Winters was the champion ($\text{MAE} < 0.29^\circ\text{C}$), beating Prophet.
* In complex urban and inland zones (Singapore, Thailand), XGBoost won.
* **Core Takeaway:** Imposing a single regional model is suboptimal. Regional rankings $\neq$ national suitability.

### Breakthrough Insight 2: "Nine of Ten Beat Seasonal Persistence; Myanmar Sole Outlier"
* In **9 countries**, models achieved $\text{MASE} < 1.0$, producing high predictive skill over seasonal persistence (Brunei improved by 36.1% over naive!).
* Under the calibrated SLR spatial proxy, **Singapore's tuned XGBoost achieved $\text{MASE} = 0.9321$**, successfully beating seasonal persistence by 6.8%. However, its margin of gain is narrower than maritime peers due to intense Urban Heat Island (UHI) microclimate drift.
* In contrast, **Myanmar ($\text{MASE} = 1.0329$) is the sole country** where the candidate model failed to outperform seasonal persistence.
* **Why?** Severe monsoonal turbulence, complex continental terrain, and Bay of Bengal tropical storm cycles create high year-to-year volatility, showing where univariate modeling hits its absolute limits and external ocean-atmosphere drivers (ENSO/SST) are required.

---

## 11. Policy & Adaptation Insights (How Governments Can Use This)

Your paper outlines a practical **3-Tier Decision Support Matrix** for national climate offices:

```
┌────────────────────────────────────────────────────────────────────────┐
│                  ADAPTATION DECISION SUPPORT MATRIX                    │
├─────────────────┬─────────────────┬────────────────────────────────────┤
│ Metric          │ Role in Policy  │ Practical Planning Action          │
├─────────────────┼─────────────────┼────────────────────────────────────┤
│ 1. MAE          │ Baseline        │ Routine municipal budgets, water   │
│                 │ Expected Error  │ reservoir schedules, agricultural  │
│                 │                 │ planting calendars, energy grid    │
│                 │                 │ baseload cooling capacity.         │
├─────────────────┼─────────────────┼────────────────────────────────────┤
│ 2. RMSE-MAE (Δ) │ Disaster Buffer │ Contingency hedging against acute  │
│                 │ / Shock Penalty │ heatwave spikes, emergency civil   │
│                 │                 │ defense water reserves, healthcare │
│                 │                 │ heat-stress alerts (e.g. Lao PDR). │
├─────────────────┼─────────────────┼────────────────────────────────────┤
│ 3. MASE         │ Reliability     │ If MASE < 1: Deploy model into     │
│                 │ Gatekeeper      │ national policy immediately.       │
│                 │                 │ If MASE > 1 (Singapore/Myanmar):   │
│                 │                 │ Treat as advisory only; mandate    │
│                 │                 │ satellite & ENSO radar inputs.     │
└─────────────────┴─────────────────┴────────────────────────────────────┘
```

---

## 12. Limitations & Future Roadmap

To demonstrate academic humility and rigor in your paper and defense:
1. **Univariate Limitation:** Models relied solely on historical temperature anomaly series without external physical climate drivers.
2. **Missing Covariates:** Future iterations should include **El Niño Southern Oscillation (ENSO / ONI index)**, **Sea Surface Temperatures (SST)**, and **monsoon rainfall data**.
3. **Spatial Resolution:** Country-level spatial aggregation averages out internal regional microclimates (e.g., Northern vs. Southern Viet Nam). Future work should test gridded ERA5 satellite reanalysis.
4. **Hybrid Models:** Testing SARIMA-XGBoost or Prophet-LSTM hybrid architectures.

---

## 13. Panel Defense Cheat Sheet (Frequently Asked Questions)

Print or review this section before any presentation or defense:

### Q1: "Why did you use standard models instead of creating a new deep learning algorithm?"
> *"Our contribution is not algorithmic invention, but an empirical and architectural framework. In regional climate adaptation, planners lack a standardized, comparable benchmark across ASEAN member states. By testing established models under identical chronological validation and holdout protocols, we proved that regional rankings do not translate into universal national suitability—an actionable finding that would have been obscured by an overfitted deep neural network."*

### Q2: "If Prophet had the lowest regional error, why not just mandate Prophet across all of ASEAN?"
> *"Because a one-size-fits-all approach degrades forecast quality in 60% of ASEAN member states. While Prophet led the regional average, it was optimal in only 4 out of 10 countries. In maritime countries like Indonesia and the Philippines, simple Holt-Winters exponential smoothing outperformed Prophet with errors below 0.28°C. Country-specific model selection is scientifically necessary."*

### Q3: "How did you guarantee zero data leakage when imputing Singapore's missing data?"
> *"We reconstructed Singapore's missing baseline using a calibrated Simple Linear Regression spatial proxy with contiguous neighbor Malaysia ($\hat{Y}_{\text{Singapore}} = -0.1701 + 1.3203 \cdot X_{\text{Malaysia}}$, $R^2 = 0.6111$, $VIF = 1.00$). To strictly guarantee zero evaluation leakage, the two test-period imputed months (April 2015 and January 2023) were completely masked out during final metric calculation (MAE, RMSE, sMAPE, MASE), ensuring 100% empirical evaluation against genuine observations."*

### Q4: "Why did Holt-Winters win lowest MAPE if it had the worst MAE and RMSE?"
> *"Because our target variable is temperature anomaly, which hovers around 0.0°C. Standard MAPE divides by the actual observation; when an anomaly is 0.05°C, even a minor miss produces an artificial 300% error explosion. Holt-Winters benefited from slightly smaller errors during a few near-zero months. When evaluating symmetric sMAPE—which bounds denominator skewness—alongside MAE and RMSE, Prophet is proven to be the true superior model."*

### Q5: "Why did Singapore and Myanmar have MASE scores above 1.0?"
> *"MASE measures whether a model outperforms a seasonal persistence baseline. A score above 1.0 means the model did not beat simply repeating last year's anomaly for that calendar month. Singapore's rapid urban microclimate changes (Urban Heat Island effect) and Myanmar's severe monsoonal fluctuations create high non-linear volatility. This establishes a key scientific boundary: univariate time-series reach a predictability ceiling in noisy microclimates, proving the urgent operational need for exogenous satellite and ENSO covariates."*

### Q6: "Why does the code use SARIMAX if the paper discusses SARIMA?"
> *"In Python's statsmodels library, the `SARIMAX` class is the standard state-space implementation for seasonal autoregressive integrated moving average modeling. Because we do not pass exogenous covariates (`exog=None`), the model is mathematically identical to pure SARIMA, while gaining numerical stability through Kalman filtering."*

---

## 📁 Key File Index in This Repository

* [`asean_temperature_forecasting.py`](file:///c:/Users/Administrator/Desktop/ITEQMT/Project/asean_temperature_forecasting.py): Complete executable pipeline (SLR spatial proxy, grid search, holdout evaluation).
* [`datasets/`](file:///c:/Users/Administrator/Desktop/ITEQMT/Project/datasets): Imputed panel datasets and missing-value audit reports.
* [`results/`](file:///c:/Users/Administrator/Desktop/ITEQMT/Project/results): Benchmark CSV outputs, hyperparameter search logs, and model rankings.
* [`docs/MODELS_AND_METRICS_EXPLAINED.md`](file:///c:/Users/Administrator/Desktop/ITEQMT/Project/docs/MODELS_AND_METRICS_EXPLAINED.md): Mathematical formulations, architectural explanations, and metric behaviors.
* [`requirements.txt`](file:///c:/Users/Administrator/Desktop/ITEQMT/Project/requirements.txt): Pinned software environment dependencies.
* [`README.md`](file:///c:/Users/Administrator/Desktop/ITEQMT/Project/README.md): Quickstart instructions, reproducibility commands, and published paper tables.
