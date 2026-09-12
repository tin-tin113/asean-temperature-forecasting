#!/usr/bin/env python3
"""
ASEAN Monthly Temperature Change Forecasting Framework & Benchmarking Pipeline.

Evaluates four candidate time-series and machine learning architectures (Holt-Winters,
SARIMA, Prophet, XGBoost) across the 10 ASEAN member states using United Nations FAOSTAT
Temperature Change on Land monthly data (1961-2023).

Workflow:
1. Panel Preprocessing & Historical Missingness Reconstruction (Singapore SLR Spatial Proxy).
2. Country-Specific Hyperparameter Optimization via Rolling-Origin Cross-Validation (1961-2013).
3. Out-of-Sample Holdout Evaluation over 120 Months (2014-2023).
4. Multi-Metric Performance Profiling (MAE, RMSE, MAPE_eps, sMAPE, MASE).
5. Regional Model Assignment and Sensitivity Analysis.

Usage:
    python asean_temperature_forecasting.py --src "ASEAN_Temperature(Pre_Process.xlsx" --grid balanced --selection-metric MAE --validation-mode rolling
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")

# Suppress verbose logging from Prophet and CmdStan backends
for _name in ["prophet", "cmdstanpy"]:
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.ERROR)
    _logger.propagate = False

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    Prophet = None
    HAS_PROPHET = False


TRAIN_END = pd.Timestamp("2013-12-01")
TEST_START = pd.Timestamp("2014-01-01")
TEST_END = pd.Timestamp("2023-12-01")

# Rolling-origin cross-validation folds strictly within training horizon (1961-2013)
FOLDS_HOLDOUT = [
    (pd.Timestamp("2008-12-01"), pd.Timestamp("2009-01-01"), pd.Timestamp("2013-12-01")),
]
FOLDS_ROLLING = [
    (pd.Timestamp("2003-12-01"), pd.Timestamp("2004-01-01"), pd.Timestamp("2008-12-01")),
    (pd.Timestamp("2008-12-01"), pd.Timestamp("2009-01-01"), pd.Timestamp("2013-12-01")),
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_MAP = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def find_excel_source(cli_src: Optional[str], script_dir: Path) -> Path:
    """Find the Excel source file."""
    if cli_src:
        p = Path(cli_src)
        if p.exists():
            return p
        # If user passes only a filename, also check script directory.
        p2 = script_dir / cli_src
        if p2.exists():
            return p2
        raise FileNotFoundError(f"Could not find source file: {cli_src}")

    candidates = (
        list(script_dir.glob("*.xlsx"))
        + list(script_dir.glob("*.xls"))
        + list((script_dir / "datasets").glob("*.xlsx"))
    )
    preferred_tokens = ("asean", "temperature", "pre", "process", "model", "ready", "filtered")
    preferred = [p for p in candidates if any(tok in p.name.lower() for tok in preferred_tokens)]
    if preferred:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        "No Excel source found. Pass --src, for example: --src \"ASEAN_Temperature(Pre_Process.xlsx\""
    )


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def params_to_json(params: Dict[str, Any]) -> str:
    return json.dumps(to_jsonable(params), sort_keys=True)


def get_country_name(countries: Sequence[str], target: str) -> Optional[str]:
    target_lower = target.lower()
    for c in countries:
        if c.lower() == target_lower:
            return c
    for c in countries:
        if target_lower in c.lower():
            return c
    return None


def time_month_fill(s: pd.Series) -> pd.Series:
    """Time-aware fallback fill that preserves seasonality better than global median."""
    s = s.copy().astype(float)
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("time_month_fill requires a DatetimeIndex")

    out = s.interpolate(method="time", limit_direction="both")

    if out.isna().any():
        month_medians = s.groupby(s.index.month).median()
        for idx in out.index[out.isna()]:
            med = month_medians.get(idx.month, np.nan)
            if pd.notna(med):
                out.loc[idx] = med

    if out.isna().any():
        out = out.fillna(s.median())
    if out.isna().any():
        out = out.ffill().bfill()
    return out.astype(float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_masked_arrays(actual: Sequence[float], pred: Sequence[float], mask: Optional[Sequence[bool]] = None) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    if mask is None:
        m = np.ones(len(a), dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
    m = m & np.isfinite(a) & np.isfinite(p)
    return a[m], p[m]


def mase_scale(train: pd.Series, seasonal_period: int = 12) -> float:
    y = np.asarray(train.dropna().values, dtype=float)
    if len(y) > seasonal_period:
        denom = np.mean(np.abs(y[seasonal_period:] - y[:-seasonal_period]))
    elif len(y) > 1:
        denom = np.mean(np.abs(np.diff(y)))
    else:
        denom = np.nan
    if not np.isfinite(denom) or denom <= 1e-12:
        denom = np.nan
    return float(denom)


def compute_metrics(
    actual: Sequence[float],
    pred: Sequence[float],
    train_for_mase: Optional[pd.Series] = None,
    mask: Optional[Sequence[bool]] = None,
    eps: float = 0.1,
) -> Dict[str, float]:
    """Compute metrics using only mask=True observations.

    MAPE uses an epsilon denominator by default to reduce extreme division by
    values very close to zero. The paper should state that MAPE is interpreted
    cautiously for anomaly data.
    """
    a, p = metric_masked_arrays(actual, pred, mask)
    n = int(len(a))
    if n == 0:
        return {
            "n_eval": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "sMAPE": np.nan,
            "MASE": np.nan,
            "near_zero_actuals": 0,
        }

    ae = np.abs(a - p)
    mae = float(np.mean(ae))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))

    denom = np.where(np.abs(a) < eps, eps, np.abs(a))
    mape = float(np.mean(ae / denom) * 100.0)

    smape_denom = np.maximum((np.abs(a) + np.abs(p)) / 2.0, eps)
    smape = float(np.mean(ae / smape_denom) * 100.0)

    if train_for_mase is not None:
        scale = mase_scale(train_for_mase, seasonal_period=12)
        mase = float(mae / scale) if np.isfinite(scale) and scale > 0 else np.nan
    else:
        mase = np.nan

    return {
        "n_eval": n,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "near_zero_actuals": int(np.sum(np.abs(a) < eps)),
    }


# ---------------------------------------------------------------------------
# Data loading and corrected imputation
# ---------------------------------------------------------------------------

def load_and_prepare_data(src: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Optional[str]]:
    """Load data, create monthly panel, and impute missing values.

    Returns:
        wide_imputed: complete monthly country panel used for model fitting.
        observed_mask: True where the source Value was observed, False where imputed.
        long_clean: long-format data with imputation flags.
        countries: country names.
        singapore_name: detected Singapore column name, if present.
    """
    raw = pd.read_excel(src)
    required = {"Area", "Months", "Year", "Value"}
    missing_cols = sorted(required - set(raw.columns))
    if missing_cols:
        raise ValueError(f"Source is missing required columns: {missing_cols}")

    raw = raw.dropna(subset=["Area", "Months", "Year"]).copy()
    raw["Area"] = raw["Area"].astype(str).str.strip()
    raw["Months"] = raw["Months"].astype(str).str.strip()
    raw["Year"] = raw["Year"].astype(int)
    raw["MonthNum"] = raw["Months"].str.lower().map(MONTH_MAP)
    raw = raw.dropna(subset=["MonthNum"]).copy()
    raw["MonthNum"] = raw["MonthNum"].astype(int)
    raw["Date"] = pd.to_datetime(dict(year=raw["Year"], month=raw["MonthNum"], day=1))
    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")

    dup = (
        raw.groupby(["Area", "Date"]).size().reset_index(name="count")
        .query("count > 1")
    )
    dup.to_csv(out_dir / "duplicate_country_month_records.csv", index=False)

    wide = raw.pivot_table(index="Date", columns="Area", values="Value", aggfunc="mean")
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    wide = wide.reindex(full_index)
    wide.index.name = "Date"

    observed_mask = wide.notna().copy()
    countries = list(wide.columns)
    singapore_name = get_country_name(countries, "Singapore")

    missing_report_rows = []
    for c in countries:
        total_missing = int(wide[c].isna().sum())
        train_missing = int(wide.loc[:TRAIN_END, c].isna().sum())
        test_missing = int(wide.loc[TEST_START:TEST_END, c].isna().sum())
        missing_report_rows.append({
            "Country": c,
            "Rows": int(wide[c].shape[0]),
            "Observed": int(wide[c].notna().sum()),
            "Missing_Total": total_missing,
            "Missing_Train_Through_2013": train_missing,
            "Missing_Test_2014_2023": test_missing,
            "Missing_Rate": total_missing / max(int(wide[c].shape[0]), 1),
        })
    missing_report = pd.DataFrame(missing_report_rows)
    missing_report.to_csv(out_dir / "missing_values_report.csv", index=False)

    wide_imputed = wide.copy()

    # Interpolate non-Singapore series if sporadic missing months exist
    for c in countries:
        if c == singapore_name:
            continue
        if wide_imputed[c].isna().any():
            wide_imputed[c] = time_month_fill(wide_imputed[c])

    imputation_rows = []

    # Calibrated Simple Linear Regression (SLR) spatial proxy for Singapore using Malaysia
    # Calibrated on overlapping observed records (1978-2023): Singapore = -0.1701 + 1.3203 * Malaysia
    malaysia_name = get_country_name(countries, "Malaysia")
    if singapore_name is not None and wide[singapore_name].isna().any() and malaysia_name is not None:
        sg = wide[singapore_name].copy()
        my = wide_imputed[malaysia_name].copy()
        sg_missing = sg.isna()
        overlap_mask = sg.notna() & my.notna()

        if int(overlap_mask.sum()) >= 24:
            slr = LinearRegression(fit_intercept=True)
            slr.fit(my.loc[overlap_mask].values.reshape(-1, 1), sg.loc[overlap_mask])
            pred_all = pd.Series(slr.predict(my.values.reshape(-1, 1)), index=wide.index)
            wide_imputed.loc[sg_missing, singapore_name] = pred_all.loc[sg_missing]
            method = f"slr_malaysia (y = {slr.intercept_:.4f} + {slr.coef_[0]:.4f} * Malaysia)"
        else:
            wide_imputed[singapore_name] = time_month_fill(sg)
            method = "time_month_fallback_insufficient_overlap"

        # Fallback interpolation for residual unobserved periods
        if wide_imputed[singapore_name].isna().any():
            fallback = time_month_fill(wide_imputed[singapore_name])
            wide_imputed[singapore_name] = wide_imputed[singapore_name].fillna(fallback)
            method = method + "+fallback"

        for dt in wide.index[sg_missing]:
            imputation_rows.append({
                "Country": singapore_name,
                "Date": dt,
                "Year": int(dt.year),
                "Month": int(dt.month),
                "Original_Value": np.nan,
                "Imputed_Value": safe_float(wide_imputed.loc[dt, singapore_name]),
                "Imputation_Method": method,
                "In_Final_Test_Period": bool(TEST_START <= dt <= TEST_END),
            })

    # Record imputation audit entries for any remaining series gaps
    for c in countries:
        if wide_imputed[c].isna().any():
            missing_dates = wide_imputed.index[wide_imputed[c].isna()]
            fallback = time_month_fill(wide_imputed[c])
            wide_imputed[c] = wide_imputed[c].fillna(fallback)
            for dt in missing_dates:
                imputation_rows.append({
                    "Country": c,
                    "Date": dt,
                    "Year": int(dt.year),
                    "Month": int(dt.month),
                    "Original_Value": np.nan,
                    "Imputed_Value": safe_float(wide_imputed.loc[dt, c]),
                    "Imputation_Method": "final_time_month_fallback",
                    "In_Final_Test_Period": bool(TEST_START <= dt <= TEST_END),
                })

    imputation_report = pd.DataFrame(imputation_rows)
    imputation_report.to_csv(out_dir / "imputation_report.csv", index=False)

    long_rows = []
    for c in countries:
        for dt in wide.index:
            long_rows.append({
                "Country": c,
                "Date": dt,
                "Year": int(dt.year),
                "Month": int(dt.month),
                "Value_Source": safe_float(wide.loc[dt, c]),
                "Value_Modeling": safe_float(wide_imputed.loc[dt, c]),
                "WasObservedInSource": bool(observed_mask.loc[dt, c]),
                "WasImputed": bool(not observed_mask.loc[dt, c]),
                "InFinalTrain": bool(dt <= TRAIN_END),
                "InFinalTest": bool(TEST_START <= dt <= TEST_END),
            })
    long_clean = pd.DataFrame(long_rows)
    long_clean.to_csv(out_dir / "cleaned_dataset_with_imputation_flags.csv", index=False)

    return wide_imputed, observed_mask, long_clean, countries, singapore_name


# ---------------------------------------------------------------------------
# Model grids
# ---------------------------------------------------------------------------

def holt_winters_grid(grid: str) -> List[Dict[str, Any]]:
    base = []
    if grid == "tiny":
        combos = [
            ("add", False, "add", False),
            (None, False, "add", False),
        ]
    elif grid == "quick":
        combos = [
            ("add", False, "add", False),
            ("add", True, "add", False),
            (None, False, "add", False),
            ("add", False, "add", True),
            ("add", True, "add", True),
            (None, False, "add", True),
        ]
    else:  # balanced
        combos = []
        for trend in [None, "add"]:
            for damped in [False, True]:
                if trend is None and damped:
                    continue
                for seasonal in ["add"]:
                    for remove_bias in [False, True]:
                        combos.append((trend, damped, seasonal, remove_bias))
        # Non-seasonal exponential smoothing baselines
        combos.append(("add", False, None, False))
        combos.append((None, False, None, False))

    for trend, damped, seasonal, remove_bias in combos:
        base.append({
            "trend": trend,
            "damped_trend": bool(damped),
            "seasonal": seasonal,
            "seasonal_periods": 12 if seasonal is not None else None,
            "remove_bias": bool(remove_bias),
        })
    return base


def sarima_grid(grid: str) -> List[Dict[str, Any]]:
    if grid == "tiny":
        nonseasonal = [(1, 0, 1), (0, 1, 1), (1, 1, 0)]
        seasonal = [(1, 1, 1, 12), (0, 1, 1, 12)]
    elif grid == "quick":
        nonseasonal = [
            (0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 1),
            (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1),
        ]
        seasonal = [(0, 1, 1, 12), (1, 1, 0, 12), (1, 1, 1, 12)]
    else:  # balanced
        nonseasonal = [
            (0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 1), (1, 0, 2),
            (2, 0, 2), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1),
            (1, 1, 2), (2, 1, 2),
        ]
        seasonal = [
            (0, 1, 1, 12), (1, 1, 0, 12), (1, 1, 1, 12),
            (0, 0, 1, 12), (1, 0, 1, 12), (2, 1, 1, 12),
            (1, 1, 2, 12),
        ]
    return [{"order": o, "seasonal_order": so} for o in nonseasonal for so in seasonal]


def prophet_grid(grid: str) -> List[Dict[str, Any]]:
    if grid == "tiny":
        cps = [0.05]
        sps = [10.0]
    elif grid == "quick":
        cps = [0.01, 0.05, 0.1]
        sps = [1.0, 10.0]
    else:
        cps = [0.001, 0.01, 0.05, 0.1, 0.2]
        sps = [0.1, 1.0, 10.0]
    return [
        {
            "changepoint_prior_scale": float(cp),
            "seasonality_prior_scale": float(sp),
            "seasonality_mode": "additive",
            "yearly_seasonality": True,
        }
        for cp in cps for sp in sps
    ]


def xgboost_grid(grid: str) -> List[Dict[str, Any]]:
    if grid == "tiny":
        return [
            {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "lag_set": [1, 2, 3, 6, 12],
                "rolling_windows": [3, 12],
            },
            {
                "n_estimators": 400,
                "learning_rate": 0.03,
                "max_depth": 4,
                "subsample": 0.85,
                "colsample_bytree": 0.9,
                "lag_set": [1, 2, 3, 6, 12, 24],
                "rolling_windows": [3, 6, 12],
            },
        ]

    if grid == "quick":
        n_estimators = [300, 500]
        learning_rates = [0.03, 0.05]
        max_depths = [3, 4]
        lag_sets = [[1, 2, 3, 6, 12], [1, 2, 3, 6, 12, 24]]
        rolling_sets = [[3, 12]]
    else:
        n_estimators = [300, 500]
        learning_rates = [0.02, 0.03, 0.05]
        max_depths = [2, 3, 4]
        lag_sets = [[1, 2, 3, 6, 12], [1, 2, 3, 6, 12, 24]]
        rolling_sets = [[3, 12], [3, 6, 12]]

    params = []
    for ne in n_estimators:
        for lr in learning_rates:
            for md in max_depths:
                for lags in lag_sets:
                    for rolls in rolling_sets:
                        params.append({
                            "n_estimators": int(ne),
                            "learning_rate": float(lr),
                            "max_depth": int(md),
                            "subsample": 0.9,
                            "colsample_bytree": 0.9,
                            "lag_set": list(lags),
                            "rolling_windows": list(rolls),
                        })
    return params


def make_model_grids(grid: str) -> Dict[str, List[Dict[str, Any]]]:
    grids = {
        "Holt-Winters": holt_winters_grid(grid),
        "SARIMA": sarima_grid(grid),
        "XGBoost": xgboost_grid(grid),
    }
    if HAS_PROPHET:
        grids["Prophet"] = prophet_grid(grid)
    else:
        print("WARNING: Prophet is not installed. Prophet will be skipped.")
    return grids


# ---------------------------------------------------------------------------
# Forecasting functions
# ---------------------------------------------------------------------------

def forecast_holt_winters(train: pd.Series, future_index: pd.DatetimeIndex, params: Dict[str, Any]) -> np.ndarray:
    seasonal_periods = params.get("seasonal_periods", 12)
    model = ExponentialSmoothing(
        train.astype(float),
        trend=params.get("trend"),
        damped_trend=bool(params.get("damped_trend", False)),
        seasonal=params.get("seasonal"),
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True, remove_bias=bool(params.get("remove_bias", False)))
    pred = fit.forecast(len(future_index))
    return np.asarray(pred, dtype=float)


def forecast_sarima(train: pd.Series, future_index: pd.DatetimeIndex, params: Dict[str, Any], maxiter: int) -> np.ndarray:
    # Fit seasonal ARIMA via state-space representation (no exogenous regressors)
    model = SARIMAX(
        train.astype(float),
        order=tuple(params["order"]),
        seasonal_order=tuple(params["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=maxiter)
    pred = res.get_forecast(steps=len(future_index)).predicted_mean
    return np.asarray(pred, dtype=float)


def forecast_prophet(train: pd.Series, future_index: pd.DatetimeIndex, params: Dict[str, Any]) -> np.ndarray:
    if not HAS_PROPHET:
        raise RuntimeError("Prophet is not installed")
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(
        yearly_seasonality=bool(params.get("yearly_seasonality", True)),
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=float(params.get("changepoint_prior_scale", 0.05)),
        seasonality_prior_scale=float(params.get("seasonality_prior_scale", 10.0)),
        seasonality_mode=str(params.get("seasonality_mode", "additive")),
    )
    model.fit(dfp)
    future = pd.DataFrame({"ds": future_index})
    fc = model.predict(future)
    return np.asarray(fc["yhat"].values, dtype=float)


def xgb_feature_row(history: pd.Series, date: pd.Timestamp, params: Dict[str, Any]) -> Dict[str, float]:
    lags = list(params.get("lag_set", [1, 2, 3, 6, 12]))
    rolls = list(params.get("rolling_windows", [3, 12]))
    row: Dict[str, float] = {}

    for lag in lags:
        row[f"lag_{lag}"] = float(history.iloc[-lag]) if len(history) >= lag else np.nan
    for w in rolls:
        row[f"roll_mean_{w}"] = float(history.iloc[-w:].mean()) if len(history) >= w else np.nan
        row[f"roll_std_{w}"] = float(history.iloc[-w:].std(ddof=0)) if len(history) >= w else 0.0

    row["month"] = float(date.month)
    row["month_sin"] = float(np.sin(2 * np.pi * date.month / 12.0))
    row["month_cos"] = float(np.cos(2 * np.pi * date.month / 12.0))
    row["year_trend"] = float(date.year + (date.month - 1) / 12.0 - 1961.0)
    return row


def xgb_supervised_frame(series: pd.Series, params: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    max_lag = max(list(params.get("lag_set", [1, 2, 3, 6, 12])) + list(params.get("rolling_windows", [3, 12])))
    rows = []
    ys = []
    idxs = []
    for i in range(max_lag, len(series)):
        date = series.index[i]
        history = series.iloc[:i]
        row = xgb_feature_row(history, date, params)
        if all(np.isfinite(v) for v in row.values()):
            rows.append(row)
            ys.append(float(series.iloc[i]))
            idxs.append(date)
    X = pd.DataFrame(rows, index=idxs)
    y = pd.Series(ys, index=idxs, name="y")
    return X, y


def forecast_xgboost(train: pd.Series, future_index: pd.DatetimeIndex, params: Dict[str, Any], random_state: int) -> np.ndarray:
    X_train, y_train = xgb_supervised_frame(train.astype(float), params)
    if len(X_train) < 24:
        raise ValueError("Not enough training rows for XGBoost after lag creation")

    model = XGBRegressor(
        n_estimators=int(params.get("n_estimators", 400)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 3)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        objective="reg:squarederror",
        random_state=random_state,
        verbosity=0,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    history = train.astype(float).copy()
    preds = []
    feature_cols = list(X_train.columns)
    for dt in future_index:
        row = xgb_feature_row(history, pd.Timestamp(dt), params)
        X_one = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(X_one)[0])
        preds.append(pred)
        # Recursive multi-step projection: append prediction to history for subsequent lag construction
        history.loc[pd.Timestamp(dt)] = pred
    return np.asarray(preds, dtype=float)


def forecast_model(
    model_name: str,
    train: pd.Series,
    future_index: pd.DatetimeIndex,
    params: Dict[str, Any],
    maxiter: int,
    random_state: int,
) -> np.ndarray:
    if model_name == "Holt-Winters":
        return forecast_holt_winters(train, future_index, params)
    if model_name == "SARIMA":
        return forecast_sarima(train, future_index, params, maxiter=maxiter)
    if model_name == "Prophet":
        return forecast_prophet(train, future_index, params)
    if model_name == "XGBoost":
        return forecast_xgboost(train, future_index, params, random_state=random_state)
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Tuning and final evaluation
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    ok: bool
    metrics: Dict[str, float]
    error: str = ""


def evaluate_candidate(
    series: pd.Series,
    observed: pd.Series,
    model_name: str,
    params: Dict[str, Any],
    folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    maxiter: int,
    random_state: int,
) -> CandidateResult:
    actual_all: List[float] = []
    pred_all: List[float] = []
    mask_all: List[bool] = []
    mase_values: List[float] = []
    near_zero_total = 0

    try:
        for train_end, val_start, val_end in folds:
            train = series.loc[:train_end].astype(float)
            val = series.loc[val_start:val_end].astype(float)
            if len(train) < 48 or len(val) == 0:
                continue
            pred = forecast_model(model_name, train, val.index, params, maxiter=maxiter, random_state=random_state)
            val_mask = observed.loc[val.index].astype(bool).values
            fold_metrics = compute_metrics(val.values, pred, train_for_mase=train, mask=val_mask)
            if fold_metrics["n_eval"] > 0 and np.isfinite(fold_metrics["MASE"]):
                mase_values.append(float(fold_metrics["MASE"]))
            near_zero_total += int(fold_metrics.get("near_zero_actuals", 0))
            actual_all.extend(list(val.values))
            pred_all.extend(list(pred))
            mask_all.extend(list(val_mask))

        metrics = compute_metrics(actual_all, pred_all, train_for_mase=None, mask=mask_all)
        metrics["MASE"] = float(np.mean(mase_values)) if len(mase_values) else np.nan
        metrics["near_zero_actuals"] = near_zero_total
        if metrics["n_eval"] == 0 or not np.isfinite(metrics["MAE"]):
            return CandidateResult(False, metrics, "No valid validation observations")
        return CandidateResult(True, metrics, "")
    except Exception as e:
        return CandidateResult(False, {}, str(e))


def select_best(grid_rows: List[Dict[str, Any]], selection_metric: str) -> Optional[Dict[str, Any]]:
    ok_rows = [r for r in grid_rows if r.get("Status") == "OK" and np.isfinite(safe_float(r.get(selection_metric)))]
    if not ok_rows:
        return None

    # Rank candidates by primary metric with sequential tie-breakers: RMSE, MAE, MASE
    def key(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (
            safe_float(r.get(selection_metric)),
            safe_float(r.get("RMSE")),
            safe_float(r.get("MAE")),
            safe_float(r.get("MASE")),
        )
    return sorted(ok_rows, key=key)[0]


def run_experiment(args: argparse.Namespace) -> None:
    t0 = time.time()
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (script_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    src = find_excel_source(args.src, script_dir)
    print(f"Using Excel source: {src}")
    print(f"Output directory: {out_dir}")
    print(f"Grid size: {args.grid}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Validation mode: {args.validation_mode}")
    print("MAPE uses epsilon denominator eps=0.1; MAE/RMSE are primary metrics.")

    wide, observed_mask, long_clean, countries, singapore_name = load_and_prepare_data(src, out_dir)
    print(f"Countries ({len(countries)}): {countries}")
    if singapore_name:
        sg_missing_total = int((~observed_mask[singapore_name]).sum())
        sg_missing_test = int((~observed_mask.loc[TEST_START:TEST_END, singapore_name]).sum())
        print(f"Singapore detected as '{singapore_name}'. Missing total={sg_missing_total}, missing final-test actuals={sg_missing_test}")
        print("Singapore missing values were imputed for modeling, but imputed final-test actuals are excluded from metrics.")

    folds = FOLDS_ROLLING if args.validation_mode == "rolling" else FOLDS_HOLDOUT
    model_grids = make_model_grids(args.grid)

    validation_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    forecast_rows: List[Dict[str, Any]] = []

    for country in countries:
        series = wide[country].astype(float).asfreq("MS")
        observed = observed_mask[country].reindex(series.index).fillna(False).astype(bool)

        final_train = series.loc[:TRAIN_END]
        final_test = series.loc[TEST_START:TEST_END]
        final_test_observed = observed.loc[final_test.index]

        print(
            f"\n--- {country} "
            f"(final_train={len(final_train)}, test={len(final_test)}, "
            f"test_observed_for_metrics={int(final_test_observed.sum())}) ---"
        )

        for model_name, candidates in model_grids.items():
            print(f"    tuning {model_name}: {len(candidates)} candidate(s)")
            model_grid_rows: List[Dict[str, Any]] = []
            m_start = time.time()

            for i, params in enumerate(candidates, start=1):
                result = evaluate_candidate(
                    series=series,
                    observed=observed,
                    model_name=model_name,
                    params=params,
                    folds=folds,
                    maxiter=args.maxiter,
                    random_state=args.random_state,
                )

                row = {
                    "Country": country,
                    "Model": model_name,
                    "Candidate_ID": i,
                    "Params": params_to_json(params),
                    "Status": "OK" if result.ok else "FAILED",
                    "Error": result.error,
                    "Validation_Mode": args.validation_mode,
                }
                if result.ok:
                    row.update(result.metrics)
                else:
                    row.update({
                        "n_eval": 0,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "MAPE": np.nan,
                        "sMAPE": np.nan,
                        "MASE": np.nan,
                        "near_zero_actuals": np.nan,
                    })
                validation_rows.append(row)
                model_grid_rows.append(row)

            best = select_best(model_grid_rows, args.selection_metric)
            if best is None:
                print(f"    {model_name:<13} no valid candidate; skipped")
                continue

            best_params = json.loads(best["Params"])
            selected_row = {
                "Country": country,
                "Model": model_name,
                "Selected_Params": best["Params"],
                "Selection_Metric": args.selection_metric,
                "Validation_n_eval": best.get("n_eval"),
                "Validation_MAE": best.get("MAE"),
                "Validation_RMSE": best.get("RMSE"),
                "Validation_MAPE": best.get("MAPE"),
                "Validation_sMAPE": best.get("sMAPE"),
                "Validation_MASE": best.get("MASE"),
            }
            selected_rows.append(selected_row)

            try:
                pred = forecast_model(
                    model_name=model_name,
                    train=final_train,
                    future_index=final_test.index,
                    params=best_params,
                    maxiter=args.maxiter,
                    random_state=args.random_state,
                )
                final_metrics = compute_metrics(
                    final_test.values,
                    pred,
                    train_for_mase=final_train,
                    mask=final_test_observed.values,
                )
                final_row = {
                    "Country": country,
                    "Model": model_name,
                    "Selected_Params": best["Params"],
                    **final_metrics,
                }
                final_rows.append(final_row)

                for dt, actual, yhat, was_obs in zip(final_test.index, final_test.values, pred, final_test_observed.values):
                    forecast_rows.append({
                        "Country": country,
                        "Model": model_name,
                        "Date": dt,
                        "Actual_Modeling_Value": safe_float(actual),
                        "Forecast": safe_float(yhat),
                        "ActualObservedInSource": bool(was_obs),
                        "UsedInMetrics": bool(was_obs),
                        "Error": safe_float(actual - yhat) if bool(was_obs) else np.nan,
                        "Absolute_Error": safe_float(abs(actual - yhat)) if bool(was_obs) else np.nan,
                    })

                print(
                    f"    final {model_name:<13} "
                    f"MAE={final_metrics['MAE']:.4f} RMSE={final_metrics['RMSE']:.4f} "
                    f"MAPE={final_metrics['MAPE']:.2f}% sMAPE={final_metrics['sMAPE']:.2f}% "
                    f"MASE={final_metrics['MASE']:.3f} "
                    f"({time.time() - m_start:.1f}s)"
                )
            except Exception as e:
                print(f"    final {model_name:<13} FAILED: {e}")
                final_rows.append({
                    "Country": country,
                    "Model": model_name,
                    "Selected_Params": best["Params"],
                    "n_eval": 0,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE": np.nan,
                    "sMAPE": np.nan,
                    "MASE": np.nan,
                    "near_zero_actuals": np.nan,
                    "Final_Error": str(e),
                })

    validation_df = pd.DataFrame(validation_rows)
    selected_df = pd.DataFrame(selected_rows)
    final_df = pd.DataFrame(final_rows)
    forecasts_df = pd.DataFrame(forecast_rows)

    # Country-specific model assignment based on holdout MAE with secondary tie-breakers
    valid_final = final_df.dropna(subset=["MAE", "RMSE"]).copy()
    best_per_country = (
        valid_final.sort_values(["Country", "MAE", "RMSE", "MASE"], na_position="last")
        .groupby("Country", as_index=False)
        .first()
    )

    avg_metrics = (
        valid_final.groupby("Model", as_index=False)[["MAE", "RMSE", "MAPE", "sMAPE", "MASE"]]
        .mean()
        .sort_values(["MAE", "RMSE"])
    )

    # Sensitivity analysis: regional model aggregates evaluated without Singapore
    if singapore_name:
        no_sg = valid_final[valid_final["Country"] != singapore_name].copy()
        avg_no_sg = (
            no_sg.groupby("Model", as_index=False)[["MAE", "RMSE", "MAPE", "sMAPE", "MASE"]]
            .mean()
            .sort_values(["MAE", "RMSE"])
        )
        best_no_sg = best_per_country[best_per_country["Country"] != singapore_name].copy()
    else:
        avg_no_sg = pd.DataFrame()
        best_no_sg = pd.DataFrame()

    # Diagnostic audit: identify near-zero test actuals to evaluate percentage metric stability
    nz_rows = []
    for c in countries:
        s_test = wide.loc[TEST_START:TEST_END, c]
        obs_test = observed_mask.loc[TEST_START:TEST_END, c].astype(bool)
        observed_values = s_test[obs_test]
        nz = observed_values[np.abs(observed_values) < 0.1]
        nz_rows.append({
            "Country": c,
            "Observed_Test_Months": int(obs_test.sum()),
            "Near_Zero_Test_Months_abs_lt_0_1": int(len(nz)),
            "Min_Abs_Observed_Test_Value": float(np.abs(observed_values).min()) if len(observed_values) else np.nan,
        })
    near_zero_df = pd.DataFrame(nz_rows)

    validation_df.to_csv(out_dir / "tuned_validation_grid_results.csv", index=False)
    selected_df.to_csv(out_dir / "tuned_selected_hyperparameters.csv", index=False)
    final_df.to_csv(out_dir / "tuned_per_country_results.csv", index=False)
    best_per_country.to_csv(out_dir / "tuned_best_model_per_country.csv", index=False)
    avg_metrics.to_csv(out_dir / "tuned_avg_metrics_per_model.csv", index=False)
    forecasts_df.to_csv(out_dir / "tuned_forecasts_long.csv", index=False)
    near_zero_df.to_csv(out_dir / "final_test_near_zero_diagnostic.csv", index=False)
    avg_no_sg.to_csv(out_dir / "sensitivity_avg_metrics_excluding_singapore.csv", index=False)
    best_no_sg.to_csv(out_dir / "sensitivity_best_model_per_country_excluding_singapore.csv", index=False)

    print("\n========== TUNED AVERAGES ==========")
    if not avg_metrics.empty:
        print(avg_metrics.round(4).to_string(index=False))
    else:
        print("No valid final metrics.")

    print("\n========== TUNED BEST PER COUNTRY ==========")
    if not best_per_country.empty:
        cols = ["Country", "Model", "MAE", "RMSE", "MAPE", "sMAPE", "MASE", "n_eval"]
        cols = [c for c in cols if c in best_per_country.columns]
        print(best_per_country[cols].round(4).to_string(index=False))
    else:
        print("No valid best-model table.")

    if singapore_name and not avg_no_sg.empty:
        print("\n========== SENSITIVITY: AVERAGES EXCLUDING SINGAPORE ==========")
        print(avg_no_sg.round(4).to_string(index=False))

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print("\nSaved files:")
    for filename in [
        "missing_values_report.csv",
        "imputation_report.csv",
        "cleaned_dataset_with_imputation_flags.csv",
        "final_test_near_zero_diagnostic.csv",
        "tuned_validation_grid_results.csv",
        "tuned_selected_hyperparameters.csv",
        "tuned_per_country_results.csv",
        "tuned_best_model_per_country.csv",
        "tuned_avg_metrics_per_model.csv",
        "tuned_forecasts_long.csv",
        "sensitivity_avg_metrics_excluding_singapore.csv",
        "sensitivity_best_model_per_country_excluding_singapore.csv",
    ]:
        print(f"  {out_dir / filename}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASEAN Monthly Temperature Change Forecasting and Benchmarking Pipeline.")
    parser.add_argument("--src", default=None, help="Path to Excel dataset. If omitted, script searches beside this .py file.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to the 'results' folder beside this .py file.")
    parser.add_argument("--grid", choices=["tiny", "quick", "balanced"], default="quick", help="Grid size.")
    parser.add_argument("--selection-metric", choices=["MAE", "RMSE", "sMAPE", "MASE"], default="MAE", help="Metric used during validation grid search.")
    parser.add_argument("--validation-mode", choices=["holdout", "rolling"], default="rolling", help="Internal validation design before final test.")
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations for SARIMA fitting.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for XGBoost.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run_experiment(parse_args())
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
