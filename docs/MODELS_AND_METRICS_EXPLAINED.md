# Complete Guide: Models & Metrics Used in the ASEAN Temperature Study

> **Project:** Forecasting Monthly Temperature Change in the ASEAN Region (1961–2023)  
> **Paper:** *A Data-Driven Framework for Climate Risk Preparedness and Regional Adaptation* (ICITDA 2026)  
> **Target Audience:** Anyone wanting to understand the models, metrics, formulas, reasons for selection, and practical examples in simple, everyday English.

---

## 🧭 Table of Contents
1. [Introduction: The Core Challenge](#1-introduction-the-core-challenge)
2. [The 4 Forecasting Models (+ 1 Imputation Model)](#2-the-4-forecasting-models--1-imputation-model)
   - [Model 1: Holt-Winters Triple Exponential Smoothing](#model-1-holt-winters-triple-exponential-smoothing)
   - [Model 2: SARIMA (Seasonal ARIMA in State-Space Form)](#model-2-sarima-seasonal-arima-in-state-space-form)
   - [Model 3: Prophet (Additive Decomposition by Meta)](#model-3-prophet-additive-decomposition-by-meta)
   - [Model 4: XGBoost (Extreme Gradient Boosted Trees with Recursive Lags)](#model-4-xgboost-extreme-gradient-boosted-trees-with-recursive-lags)
   - [Supplementary: Calibrated Spatial Proxy Imputation (Singapore)](#supplementary-architecture-calibrated-spatial-proxy-imputation-singapore)
   - [Model Comparison Cheat-Sheet](#model-comparison-cheat-sheet)
3. [The 5 Evaluation Metrics](#3-the-5-evaluation-metrics)
   - [Metric 1: MAE (Mean Absolute Error)](#metric-1-mae-mean-absolute-error)
   - [Metric 2: RMSE (Root Mean Squared Error) & The Shock Penalty Spread](#metric-2-rmse-root-mean-squared-error--the-shock-penalty-spread)
   - [Metric 3: MAPE (Mean Absolute Percentage Error) & The Zero Trap](#metric-3-mape-mean-absolute-percentage-error--the-zero-trap)
   - [Metric 4: sMAPE (Symmetric Mean Absolute Percentage Error)](#metric-4-smape-symmetric-mean-absolute-percentage-error)
   - [Metric 5: MASE (Mean Absolute Scaled Error) — The Reality Check](#metric-5-mase-mean-absolute-scaled-error--the-reality-check)
   - [Metric Comparison Cheat-Sheet](#metric-comparison-cheat-sheet)
4. [Step-by-Step Practical Numerical Walkthrough](#4-step-by-step-practical-numerical-walkthrough)
5. [Summary: How to Answer in Your Defense](#5-summary-how-to-answer-in-your-defense)

---

## 1. Introduction: The Core Challenge

Before diving into models and metrics, remember what we are predicting:
* We are predicting **monthly temperature anomalies (change relative to 1951–1980 baseline)** in degrees Celsius (°C).
* An anomaly is **NOT** the absolute air temperature (like $32^\circ\text{C}$). It is the deviation:
  * $+0.80^\circ\text{C}$ means $0.8^\circ\text{C}$ warmer than historical normal for that month.
  * $-0.20^\circ\text{C}$ means $0.2^\circ\text{C}$ cooler than historical normal.
  * $0.00^\circ\text{C}$ means exactly normal.
* Because values constantly cross and hover near **$0.00^\circ\text{C}$**, standard forecasting formulas can behave strangely. Choosing the right models and metrics is the core scientific foundation of this study.

---

## 2. The 4 Forecasting Models (+ 1 Imputation Model)

```
                       MODELS EVALUATED IN THE STUDY
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
[CLASSICAL STATISTICAL]                                  [MODERN / MACHINE LEARNING]
  • Holt-Winters (Smoothing)                               • Prophet (Additive Decomposition)
  • SARIMA (Autoregressive Moving Average)                 • XGBoost (Gradient Boosted Trees)
```

---

### Model 1: Holt-Winters Triple Exponential Smoothing

#### What is it in simple terms?
Imagine predicting tomorrow's weather by taking recent weather and weighting newer months more heavily than older months, while also tracking an overall upward trend and repeating a 12-month calendar cycle. That is Holt-Winters.

#### How it works:
It splits the time series into three separate components:
1. **Level ($\ell_t$):** The current baseline temperature.
2. **Trend ($b_t$):** Is the temperature gradually climbing year by year?
3. **Seasonality ($s_t$):** What is the recurring 12-month rhythm (e.g., April is usually hot, December is usually mild)?
* It updates these components using exponential decay weights ($\alpha, \beta, \gamma$ between 0 and 1). Recent months get the highest weight, and older months fade exponentially.

#### Why was it used in the study?
* It is the **gold-standard classical statistical baseline**.
* It is computationally lightning fast, fully transparent, and has no complex black-box layers.
* If a modern deep learning or machine learning model cannot beat simple Holt-Winters, the complex model is useless.

#### Real-World Example:
* Suppose for the **Philippines**, the baseline level is $+0.50^\circ\text{C}$.
* The estimated warming trend is $+0.01^\circ\text{C}$ per year.
* The seasonal factor for May is $+0.25^\circ\text{C}$.
* Holt-Winters forecast for May = Level ($0.50$) + Trend ($0.01$) + Seasonal ($0.25$) = **$+0.76^\circ\text{C}$**.

#### How did it perform in ASEAN?
* **Won 2 countries: Indonesia ($\text{MAE} = 0.2259^\circ\text{C}$) and Philippines ($\text{MAE} = 0.2844^\circ\text{C}$)**.
* In maritime islands, ocean water stabilizes the climate, creating steady, rhythmic seasonal patterns that Holt-Winters captures with high accuracy.

---

### Model 2: SARIMA (Seasonal ARIMA in State-Space Form)

#### What is it in simple terms?
SARIMA stands for **Seasonal Autoregressive Integrated Moving Average**.  
Instead of just smoothing numbers, SARIMA uses mathematical memory:
* *"If the temperature was high last month, it is likely still high this month"* (**Autoregression, AR**).
* *"If a sudden heat shock occurred last month, part of that shock lingers into this month"* (**Moving Average, MA**).
* *"Subtract this May from last May so we only model the net change"* (**Seasonal Differencing, I**).

#### The Parameters $(p, d, q) \times (P, D, Q)_{12}$:
* $p, P$: Non-seasonal and seasonal autoregressive lags (looking back 1 month, or 12 months).
* $d, D$: Differencing steps needed to make the data stationary (removing trends).
* $q, Q$: Moving average error terms (reacting to past forecast errors).
* **State-Space formulation:** In our code, we implemented it via `statsmodels.tsa.statespace.sarimax.SARIMAX(exog=None)`. The state-space form uses the **Kalman filter**, which handles numerical stability much better than classical maximum likelihood.

#### Why was it used in the study?
* It is the most respected theoretical statistical benchmark in time-series econometrics and meteorology.
* It explicitly models autocorrelation (when consecutive months are correlated).

#### Real-World Example:
* Suppose in **Brunei**, May temperature anomaly is modeled as:
  $$\hat{y}_t = 0.6 \times y_{t-1} + 0.3 \times y_{t-12} + \epsilon_t$$
* If last month was $+0.40^\circ\text{C}$ and last year's May was $+0.30^\circ\text{C}$:
  $$\hat{y}_t = (0.6 \times 0.40) + (0.3 \times 0.30) = 0.24 + 0.09 = \mathbf{+0.33^\circ\text{C}}$$

#### How did it perform in ASEAN?
* **Won 2 countries: Brunei ($\text{MAE} = 0.2582^\circ\text{C}$) and Malaysia ($\text{MAE} = 0.2614^\circ\text{C}$)**.
* **Brunei SARIMA was the single most successful model in the entire study relative to naive baseline**, achieving $\text{MASE} = 0.6394$ (a **36.1% error reduction** over seasonal naive).

---

### Model 3: Prophet (Additive Decomposition by Meta)

#### What is it in simple terms?
Prophet is an open-source forecasting library created by Meta (Facebook). Instead of looking at step-by-step lags, Prophet views time series like a curve-fitting problem:
$$y(t) = \text{Trend}(t) + \text{Seasonality}(t) + \text{Holidays}(t) + \text{Error}(t)$$
* **Trend with Changepoints:** Prophet draws a flexible trendline that can change direction at specific dates (changepoints). If warming accelerated after 1998, Prophet bends its trendline to follow it.
* **Fourier Seasonality:** Instead of crude monthly buckets, Prophet uses smooth combinations of sine and cosine waves to model the calendar year.

#### Why was it used in the study?
* Climate change is non-linear: temperatures don't just increase at a constant flat rate; warming often accelerates in spurts.
* Prophet excels at handling trend shifts while remaining robust to missing data and seasonal irregularities.

#### Real-World Example:
* In **Lao PDR**, temperatures experienced decadal warming shifts after the 1990s.
* Prophet fits a piecewise curve:
  * 1961–1990: Trend warming rate was $+0.008^\circ\text{C}$/year.
  * 1991–2023: Changepoint detected $\rightarrow$ trend rate increased to $+0.022^\circ\text{C}$/year.
* For any future month, Prophet computes the current trend position on the curve, adds the smooth Fourier wave for that month, and gives the forecast.

#### How did it perform in ASEAN?
* **Ranked #1 in Regional Aggregate:** Across all 10 countries averaged together, Prophet had the lowest $\text{MAE} = 0.5051^\circ\text{C}$ and lowest $\text{RMSE} = 0.6248^\circ\text{C}$.
* **Won 4 countries:** All in continental Indochina (**Cambodia, Lao PDR, Myanmar, Viet Nam**). Its flexible trend and Fourier curves easily adapted to intense continental monsoon swings.

---

### Model 4: XGBoost (Extreme Gradient Boosted Trees with Recursive Lags)

#### What is it in simple terms?
XGBoost is an elite machine learning algorithm built from hundreds of **decision trees**.  
Think of playing a game of "20 Questions":
* *Is lag-1 > 0.4°C?* $\rightarrow$ YES.
* *Is it currently May or June?* $\rightarrow$ YES.
* *Is the 6-month moving average > 0.6°C?* $\rightarrow$ NO.
* $\rightarrow$ *Prediction: +0.52°C.*
* Every tree tries to correct the errors made by the previous trees (boosting).

#### How it forecasts 10 years into the future (Recursive Forecasting):
* XGBoost cannot naturally look 10 years ahead on its own because it needs input features (lags from last month).
* To forecast 120 months (2014–2023), we implemented **recursive multi-step forecasting**:
  * Step 1 (Jan 2014): Uses actual lags from Dec 2013 $\rightarrow$ predicts Jan 2014.
  * Step 2 (Feb 2014): Feeds the *predicted* Jan 2014 back in as a lag feature $\rightarrow$ predicts Feb 2014.
  * Repeats this loop all the way to month 120 (Dec 2023).

#### Why was it used in the study?
* To test whether modern non-linear machine learning decision trees can outperform classical linear and additive statistical models.
* It can uncover complex, non-linear relationships (e.g., if temperatures spike only when three specific conditions occur simultaneously).

#### How did it perform in ASEAN?
* **Won 2 countries: Singapore ($\text{MAE} = 0.4933^\circ\text{C}$) and Thailand ($\text{MAE} = 0.5782^\circ\text{C}$)**.
* Excelled in environments with rapid urbanization (Singapore's Urban Heat Island) and complex mountain/plain topography (Thailand), where linear formulas struggle.

---

### Supplementary Architecture: Calibrated Spatial Proxy Imputation (Singapore)

#### What is it?
Singapore had 360 missing monthly records in the historical FAOSTAT archive (358 in the 1961–2013 training period, and 2 in the 2014–2023 holdout period). We deployed a calibrated **Simple Linear Regression (SLR) spatial proxy** using contiguous neighbor Malaysia as the sole predictor to reconstruct these missing baseline records without data leakage.

#### Why Simple Linear Regression with Malaysia?
1. **Physical Contiguity:** Malaysia shares an immediate land-sea border with Singapore (< 1 km across the Johor Strait) and exhibits identical equatorial monsoonal regimes (Northeast and Southwest monsoons).
2. **Collinearity Prevention (Occam's Razor):** In exploratory modeling, candidate multi-country predictors (Malaysia, Indonesia, Brunei) showed severe multicollinearity ($\text{VIF}_{\text{Malaysia}} = 11.81, \text{VIF}_{\text{Brunei}} = 7.95, r > 0.90$). Selecting Malaysia as the single predictor establishes an uncompromised $\text{VIF} = 1.00$.
3. **High Empirical Fit:** The calibrated model explains $61.1\%$ of monthly anomaly variance across all $N = 396$ overlapping observed months (May 1978–December 2023):
   $$\hat{Y}_{\text{Singapore}, t} = -0.1701 + 1.3203 \cdot X_{\text{Malaysia}, t} \quad (R^2 = 0.6111, F = 619.2, p < 0.0001, t = 24.88)$$
4. **Physical Microclimate Scaling:** The slope $\beta_1 = 1.3203 > 1.0$ physically captures Singapore's amplified urban microclimate and Urban Heat Island (UHI) dynamics relative to Malaysia's nationally aggregated baseline.

#### Strict Scientific Safeguards Enforced:
* **Zero Leakage:** Imputed values for the 2 missing months in the 2014–2023 holdout period were **strictly masked out** during accuracy evaluations ($n = 118$ for Singapore). All metrics (MAE, RMSE, sMAPE, MASE) were computed purely against genuine empirical measurements.
* **Harmonized Benchmark Window:** Imputation enables all 10 ASEAN countries to share a strictly unified 63-year timeline (1961–2023).

---

### Model Comparison Cheat-Sheet

| Model | Type | Best at Handling | Weakness | Won in ASEAN? |
| :--- | :--- | :--- | :--- | :---: |
| **Holt-Winters** | Classical Smoothing | Smooth, steady seasonal cycles | Cannot adapt to abrupt regime changes | **2 countries** (IDN, PHL) |
| **SARIMA** | Classical State-Space | Autocorrelation, seasonal differencing | Rigid linear assumptions | **2 countries** (BRN, MYS) |
| **Prophet** | Decomposable Additive | Changing trend slopes, seasonal Fourier waves | Vulnerable to acute extreme shocks | **4 countries** (KHM, LAO, MMR, VNM) |
| **XGBoost** | Machine Learning Trees | Complex, non-linear interactions & urban shifts | Error compounding in recursive lags | **2 countries** (SGP, THA) |

---

## 3. The 5 Evaluation Metrics

To evaluate model accuracy over the 120-month unseen test period (2014–2023), your study used **5 distinct metrics**. Each metric answers a completely different scientific question:

```
                  THE 5 METRICS & THEIR PURPOSE
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  [ERROR SCALE]           [ERROR NATURE]          [BENCHMARK SKILL]
  • MAE (Typical °C error) • MAPE (Percentage)     • MASE (vs. Naive Baseline)
  • RMSE (Outlier shock)   • sMAPE (Symmetric %)
```

---

### Metric 1: MAE (Mean Absolute Error)

#### The Formula:
$$\text{MAE} = \frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|$$

#### Plain English Explanation:
* **The average size of the mistakes in degrees Celsius (°C).**
* You subtract the predicted temperature from the actual temperature, discard any negative signs (absolute value), and take the average.
* **Linear Penalty:** An error of $1.0^\circ\text{C}$ is treated as exactly twice as bad as an error of $0.5^\circ\text{C}$.

#### Why was it used in the study?
* It is the **primary evaluation and model selection metric**.
* It provides direct, physically interpretable numbers for human planners: *"On average, this model is off by 0.25°C."*

#### Numerical Example:
Suppose for 3 test months, actual and predicted anomalies are:
* Month 1: Actual $= +0.50^\circ\text{C}$, Predicted $= +0.70^\circ\text{C} \implies \text{Error} = |0.50 - 0.70| = 0.20^\circ\text{C}$
* Month 2: Actual $= +0.80^\circ\text{C}$, Predicted $= +0.50^\circ\text{C} \implies \text{Error} = |0.80 - 0.50| = 0.30^\circ\text{C}$
* Month 3: Actual $= +0.10^\circ\text{C}$, Predicted $= +0.20^\circ\text{C} \implies \text{Error} = |0.10 - 0.20| = 0.10^\circ\text{C}$
$$\text{MAE} = \frac{0.20 + 0.30 + 0.10}{3} = \mathbf{0.20^\circ\text{C}}$$

---

### Metric 2: RMSE (Root Mean Squared Error) & The Shock Penalty Spread

#### The Formula:
$$\text{RMSE} = \sqrt{\frac{1}{H} \sum_{t=1}^H (y_t - \hat{y}_t)^2}$$

#### Plain English Explanation:
* Also measures error in °C, but **squares each mistake before averaging, then takes the square root**.
* **Quadratic Penalty:** It punishes large blunders far more harshly than small mistakes.
  * An error of $0.2^\circ\text{C} \rightarrow (0.2)^2 = 0.04$
  * An error of $2.0^\circ\text{C} \rightarrow (2.0)^2 = 4.00$ (100 times bigger penalty for a 10-times bigger error!)

#### Why was it used in the study?
* In climate adaptation, a model that is off by $0.2^\circ\text{C}$ every month is manageable. But a model that is off by $2.5^\circ\text{C}$ during a deadly heatwave can cause agricultural collapse or power grid blackouts. RMSE catches those deadly blunders.

#### The "Penalty Spread" ($\Delta = \text{RMSE} - \text{MAE}$):
* Because $\text{RMSE} \ge \text{MAE}$ always, the difference between them reveals model stability:
  * **Small spread ($\Delta \approx 0.07^\circ\text{C}$):** Errors are consistent, steady, and predictable (e.g., Maritime Indonesia).
  * **Large spread ($\Delta \ge 0.18^\circ\text{C}$):** The model occasionally suffers massive forecast blowouts during climate shocks (e.g., Lao PDR during El Niño).

#### Numerical Example:
Using the same 3 months as before:
* Month 1: Error $= 0.20 \implies 0.20^2 = 0.04$
* Month 2: Error $= 0.30 \implies 0.30^2 = 0.09$
* Month 3: Error $= 0.10 \implies 0.10^2 = 0.01$
$$\text{RMSE} = \sqrt{\frac{0.04 + 0.09 + 0.01}{3}} = \sqrt{\frac{0.14}{3}} = \sqrt{0.0467} = \mathbf{0.216^\circ\text{C}}$$

---

### Metric 3: MAPE (Mean Absolute Percentage Error) & The Zero Trap

#### The Formula:
$$\text{MAPE} = \frac{100\%}{H} \sum_{t=1}^H \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$

#### Plain English Explanation:
* Expresses forecast error as a percentage of the actual value.
* Popular in business forecasting (e.g., *"We were off by 5% of total sales"*).

#### Why it BREAKS DOWN for Temperature Anomalies (The Zero-Crossing Trap):
* In temperature anomalies, actual values hover close to zero (e.g., $y_t = +0.02^\circ\text{C}$).
* Look what happens when you divide by a number near zero:
  * Actual $= +0.02^\circ\text{C}$, Predicted $= +0.22^\circ\text{C}$ (a tiny error of just $0.20^\circ\text{C}$).
  $$\text{Percentage Error} = \left| \frac{0.02 - 0.22}{0.02} \right| \times 100\% = \frac{0.20}{0.02} \times 100\% = \mathbf{1,000\%!}$$
* A tiny fraction of a degree causes a **division-by-zero mathematical explosion**.

#### The Big Mystery in Table II Explained:
* In Table II of your paper, **Holt-Winters had the lowest MAPE (58.03%)**, but had the **worst MAE ($0.6490^\circ\text{C}$)** and **worst RMSE ($0.7659^\circ\text{C}$)**!
* **Why?** Holt-Winters happened to be slightly closer to actual values during a few specific near-zero months, which artificially avoided denominator explosions. It did **not** mean Holt-Winters was the better model.
* To prevent total division by zero, the study added a tiny $\epsilon = 0.1$ threshold, but standard MAPE remained distorted.

---

### Metric 4: sMAPE (Symmetric Mean Absolute Percentage Error)

#### The Formula:
$$\text{sMAPE} = \frac{100\%}{H} \sum_{t=1}^H \frac{2 |y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t| + \epsilon}$$

#### Plain English Explanation:
* Solves the zero-denominator explosion of standard MAPE.
* Instead of dividing only by actual $y_t$, it divides by the **average of actual and predicted** $(|y_t| + |\hat{y}_t|)/2$.
* This guarantees that no single month's error can ever exceed **200%**.

#### Why was it used in the study?
* It provides a fair, scale-free percentage benchmark that does not blow up when anomalies cross zero.
* **The Result:** Under sMAPE, the false distortion of Holt-Winters vanishes:
  * **Prophet clearly wins sMAPE at 47.96%**.
  * Holt-Winters drops to last place at **76.35%** (matching its poor MAE and RMSE).

---

### Metric 5: MASE (Mean Absolute Scaled Error) — The Reality Check

#### The Formula:
$$\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{seasonal naive in-sample}}} = \frac{\frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|}{\frac{1}{T-12} \sum_{t=13}^T |y_t - y_{t-12}|}$$

#### Plain English Explanation:
* MASE compares your smart model against the simplest, dumbest possible baseline rule:  
  **"Seasonal Naive: Predict that this month will have the exact same temperature anomaly as this exact month last year"** ($y_t = y_{t-12}$).
* If your complex AI cannot beat simply repeating last year's calendar month, your model is practically useless!

#### How to Read MASE:
* **$\text{MASE} < 1.0$ (SUCCESS):** Your model is better than repeating last year.  
  * *Example:* Brunei SARIMA had $\text{MASE} = 0.6394 \implies$ **36.1% error reduction** compared to seasonal naive!
* **$\text{MASE} = 1.0$ (TIE):** Your model is exactly as good as repeating last year.
* **$\text{MASE} > 1.0$ (FAILURE):** Your model is **worse** than repeating last year.

#### The Critical Insight in Your Paper ("Best $\neq$ Good"):
* In **9 countries**, the best model achieved $\text{MASE} < 1.0$ (proven predictive skill). Under our calibrated SLR proxy, Singapore's tuned XGBoost achieved $\text{MASE} = 0.9321$ (a 6.8% error reduction over seasonal naive).
* **Myanmar ($\text{MASE} = 1.0329$) is the sole country** in ASEAN where the candidate model failed to beat seasonal persistence.
* **Why?** For Myanmar, complex continental monsoon transitions and Bay of Bengal storm turbulence hit a univariate predictability boundary. In Singapore, while the model beats seasonal naive, its margin is narrower than maritime peers due to Urban Heat Island (UHI) volatility.

---

### Metric Comparison Cheat-Sheet

| Metric | Full Name | Unit | What it Punishes | Why Needed in Study |
| :--- | :--- | :--- | :--- | :--- |
| **MAE** | Mean Absolute Error | °C | Constant linear penalty | Primary scale-accurate operational metric for planning |
| **RMSE** | Root Mean Squared Error | °C | Large outlier blunders heavily | Flags vulnerability to sudden heatwaves/climate shocks |
| **MAPE** | Mean Absolute Percentage Error | % | Near-zero denominators | Included for standard reporting, but proven to be mathematically flawed for anomalies |
| **sMAPE** | Symmetric MAPE | % | Percentage error bounded at 200% | Resolves MAPE's zero-crossing distortion |
| **MASE** | Mean Absolute Scaled Error | Ratio | Inability to beat repeating last year | Operational gatekeeper: checks if the model has genuine predictive value |

---

## 4. Step-by-Step Practical Numerical Walkthrough

Let's calculate all 5 metrics together for a realistic 3-month test period:

### Given Data:
* In-sample seasonal naive average error = $0.40^\circ\text{C}$
* **Month 1 (Jan):** Actual $= +0.50^\circ\text{C}$, Predicted $= +0.70^\circ\text{C}$
* **Month 2 (Feb):** Actual $= +0.10^\circ\text{C}$, Predicted $= +0.20^\circ\text{C}$ *(near-zero month)*
* **Month 3 (Mar):** Actual $= +0.80^\circ\text{C}$, Predicted $= +0.40^\circ\text{C}$ *(large miss)*

---

### Step 1: Calculate Absolute Errors & Squared Errors

| Month | Actual ($y_t$) | Pred ($\hat{y}_t$) | Absolute Error $|y - \hat{y}|$ | Squared Error $(y - \hat{y})^2$ |
| :---: | :---: | :---: | :---: | :---: |
| Jan | $+0.50$ | $+0.70$ | $0.20$ | $0.0400$ |
| Feb | $+0.10$ | $+0.20$ | $0.10$ | $0.0100$ |
| Mar | $+0.80$ | $+0.40$ | $0.40$ | $0.1600$ |
| **Sum** | | | **$0.70$** | **$0.2100$** |

* **MAE:**
  $$\text{MAE} = \frac{0.70}{3} = \mathbf{0.233^\circ\text{C}}$$
* **RMSE:**
  $$\text{RMSE} = \sqrt{\frac{0.2100}{3}} = \sqrt{0.0700} = \mathbf{0.265^\circ\text{C}}$$
* **Shock Spread ($\Delta$):**
  $$\Delta = \text{RMSE} - \text{MAE} = 0.265 - 0.233 = \mathbf{0.032^\circ\text{C}} \quad (\text{Very stable!})$$

---

### Step 2: Calculate MAPE vs. sMAPE

* **Month 1 (Jan):**
  * $\text{MAPE}_1 = |0.20 / 0.50| \times 100\% = 40.0\%$
  * $\text{sMAPE}_1 = \frac{2 \times 0.20}{0.50 + 0.70} \times 100\% = \frac{0.40}{1.20} \times 100\% = 33.3\%$

* **Month 2 (Feb - Near Zero):**
  * $\text{MAPE}_2 = |0.10 / 0.10| \times 100\% = \mathbf{100.0\%}$ *(See how high this jumps despite the error being only $0.1^\circ\text{C}$!)*
  * $\text{sMAPE}_2 = \frac{2 \times 0.10}{0.10 + 0.20} \times 100\% = \frac{0.20}{0.30} \times 100\% = \mathbf{66.7\%}$ *(Much more reasonable)*

* **Month 3 (Mar):**
  * $\text{MAPE}_3 = |0.40 / 0.80| \times 100\% = 50.0\%$
  * $\text{sMAPE}_3 = \frac{2 \times 0.40}{0.80 + 0.40} \times 100\% = \frac{0.80}{1.20} \times 100\% = 66.7\%$

* **Average MAPE:**
  $$\text{MAPE} = \frac{40.0 + 100.0 + 50.0}{3} = \mathbf{63.3\%}$$
* **Average sMAPE:**
  $$\text{sMAPE} = \frac{33.3 + 66.7 + 66.7}{3} = \mathbf{55.6\%}$$

---

### Step 3: Calculate MASE

$$\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}} = \frac{0.233}{0.400} = \mathbf{0.583}$$

* **Conclusion:** Because $0.583 < 1.0$, this model is **41.7% more accurate** than simply repeating last year's calendar month. It passes the operational deployment gate!

---

## 5. Summary: How to Answer in Your Defense

If a panelist asks you:

1. **"Why did you test these 4 specific models?"**
   > *"We chose them to represent four distinct modeling paradigms: Holt-Winters (classical exponential smoothing), SARIMA (rigorous state-space econometrics), Prophet (modern additive curve decomposition with changepoints), and XGBoost (non-linear recursive tree-based machine learning). This benchmark allowed us to evaluate whether complex modern architectures genuinely outperform simple classical baselines across different Southeast Asian topographies."*

2. **"Why did you use 5 different metrics instead of just MAE?"**
   > *"Because each metric evaluates a different operational dimension. MAE provides the expected error scale in degrees Celsius for municipal budgeting. RMSE flags vulnerability to extreme heatwave blunders. sMAPE gives a percentage error that doesn't blow up near zero. And MASE acts as an operational reality check, proving whether our models actually beat simple seasonal persistence."*

3. **"Why did Holt-Winters have the best MAPE if it was the worst model?"**
   > *"That is a known mathematical flaw of standard MAPE when applied to temperature anomalies. Because anomaly values hover near zero degrees Celsius, tiny deviations produce artificial percentage explosions. Holt-Winters simply had smaller errors during a few near-zero months. When evaluating symmetric sMAPE alongside MAE, RMSE, and MASE, Prophet is confirmed as the true superior regional model."*

4. **"What does MASE tell us about Myanmar, and how did Singapore perform?"**
   > *"MASE acts as an operational reality check. Nine of our ten tuned national models beat seasonal persistence (MASE < 1.0). For Singapore, our calibrated SLR spatial proxy enabled tuned XGBoost to achieve MASE = 0.9321 (a 6.8% gain over seasonal persistence), though its margin is narrower than maritime neighbors due to intense Urban Heat Island dynamics. Myanmar (MASE = 1.0329) is the sole country where candidate models hit an absolute predictability boundary, proving that univariate modeling alone cannot navigate continental monsoonal turbulence without external ocean-atmosphere covariates like ENSO."*
