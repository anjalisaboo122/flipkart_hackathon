"""
Prophet-based spatiotemporal violation predictor.

For each of the top N hotspot zones:
  - Train Prophet on daily violation counts (Nov 2023 – Apr 2024)
  - Validate on May 2024 (held-out month) → compute MAPE
  - Forecast next FORECAST_DAYS from today using learned weekly seasonality

Prediction strategy for dates beyond training data (we are in 2026):
  We extract Prophet's weekly seasonal component (which day-of-week is high/low)
  and combine it with each zone's historical mean to produce calibrated forecasts.
  Rankings between zones remain meaningful even if absolute counts drift slightly.
"""

import logging
import warnings
import numpy as np
import pandas as pd

from config import TOP_ZONES_PROPHET, FORECAST_DAYS

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")   # suppress Prophet/Stan output


TRAIN_END = pd.to_datetime("2024-02-28")
VAL_START = pd.to_datetime("2024-03-01")
VAL_END   = pd.to_datetime("2024-04-08")


def train_prophet_models(daily_series: dict) -> dict:
    """
    Train one Prophet model per zone.
    Returns {zone_id: fitted Prophet model}
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.error("prophet not installed — run: pip install prophet")
        return {}

    models = {}
    zones  = list(daily_series.keys())
    logger.info("Training Prophet on %d zones...", len(zones))

    for i, zid in enumerate(zones):
        ts    = daily_series[zid]
        train = ts[pd.to_datetime(ts["ds"]) <= TRAIN_END].copy()

        if len(train) < 30:
            logger.debug("Zone %s skipped — only %d training days", zid, len(train))
            continue

        try:
            m = Prophet(
                yearly_seasonality  = False,
                weekly_seasonality  = True,
                daily_seasonality   = False,
                seasonality_mode    = "multiplicative",
                changepoint_prior_scale = 0.05,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(train)
            models[zid] = m
        except Exception as e:
            logger.warning("Prophet fit failed for zone %s: %s", zid, e)

    logger.info("Trained %d Prophet models", len(models))
    return models


def validate_may(models: dict, daily_series: dict) -> pd.DataFrame:
    """
    Predict May 2024 for each trained zone and compute MAPE vs actual.
    Returns DataFrame with columns: zone_id, mape, mae, n_days, mean_actual, mean_predicted
    """
    results = []

    for zid, model in models.items():
        ts  = daily_series.get(zid)
        if ts is None:
            continue

        actual = ts[(pd.to_datetime(ts["ds"]) >= VAL_START) & (pd.to_datetime(ts["ds"]) <= VAL_END)].copy()
        if len(actual) < 5:
            continue

        try:
            # Make future dataframe that covers May 2024
            last_train = TRAIN_END
            periods    = (VAL_END - last_train).days + 5
            future     = model.make_future_dataframe(periods=periods, freq="D")
            forecast   = model.predict(future)

            may_fc = forecast[
                (forecast["ds"] >= VAL_START) & (forecast["ds"] <= VAL_END)
            ][["ds", "yhat"]].copy()
            may_fc["yhat"] = may_fc["yhat"].clip(lower=0)

            merged = actual.merge(may_fc, on="ds")
            if len(merged) < 5:
                continue

            y_true = merged["y"].values
            y_pred = merged["yhat"].values
            valid_mask = y_true > 0
            mape = float(np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask]))) * 100
            mae    = float(np.mean(np.abs(y_true - y_pred)))

            results.append({
                "zone_id":        zid,
                "mape":           round(mape, 1),
                "mae":            round(mae, 2),
                "n_days":         len(merged),
                "mean_actual":    round(float(y_true.mean()), 1),
                "mean_predicted": round(float(y_pred.mean()), 1),
            })
        except Exception as e:
            logger.warning("Validation failed for zone %s: %s", zid, e)

    return pd.DataFrame(results).sort_values("mape") if results else pd.DataFrame()


def get_validation_chart_data(models: dict, daily_series: dict, zone_id: str) -> pd.DataFrame:
    """Actual vs predicted daily counts for May 2024 for a single zone."""
    model = models.get(zone_id)
    ts    = daily_series.get(zone_id)
    if model is None or ts is None:
        return pd.DataFrame()

    actual = ts[(pd.to_datetime(ts["ds"]) >= VAL_START) & (pd.to_datetime(ts["ds"]) <= VAL_END)].copy()
    periods = (VAL_END - TRAIN_END).days + 5
    future  = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)

    may_fc = forecast[
        (forecast["ds"] >= VAL_START) & (forecast["ds"] <= VAL_END)
    ][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    may_fc["yhat"]       = may_fc["yhat"].clip(lower=0)
    may_fc["yhat_lower"] = may_fc["yhat_lower"].clip(lower=0)

    return actual.merge(may_fc, on="ds")


def get_zone_predictions(models: dict, daily_series: dict,
                          hour_dist: dict, zones_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a risk forecast for each zone for today and the next FORECAST_DAYS.

    Strategy:
      1. Forecast daily violation counts using Prophet (extrapolated to today).
      2. Calibrate: scale by (historical_mean / training_period_mean) so that
         2-year extrapolation drift doesn't distort relative rankings.
      3. Distribute daily count to hourly using historical hourly patterns.
      4. Return: zone_id, date, hour, predicted_violations, day_rank

    Returns a flat DataFrame — one row per (zone, date, hour).
    """
    today  = pd.Timestamp.now().normalize()
    dates  = pd.date_range(today, periods=FORECAST_DAYS, freq="D")

    rows = []
    for zid, model in models.items():
        ts = daily_series.get(zid)
        if ts is None:
            continue

        train_mean = ts[pd.to_datetime(ts["ds"]) <= TRAIN_END]["y"].mean()
        if train_mean <= 0:
            continue

        try:
            # Extend future dataframe to today
            last_train_date = TRAIN_END
            extra_periods   = (today - last_train_date).days + FORECAST_DAYS + 5
            future   = model.make_future_dataframe(periods=extra_periods, freq="D")
            forecast = model.predict(future)

            # Extract predictions for our target dates
            fc = forecast[forecast["ds"].isin(dates)][["ds", "yhat"]].copy()
            fc["yhat"] = fc["yhat"].clip(lower=0)

            # Calibrate to historical mean (corrects for long-range trend drift)
            fc_mean = fc["yhat"].mean()
            scale   = train_mean / (fc_mean + 1e-9)
            fc["yhat_cal"] = (fc["yhat"] * scale).clip(lower=0)

            hdist = hour_dist.get(zid, np.full(24, 1 / 24))

            for _, row_fc in fc.iterrows():
                for h in range(24):
                    rows.append({
                        "h3_index":           zid,
                        "date":               row_fc["ds"].date(),
                        "hour":               h,
                        "predicted_daily":    round(float(row_fc["yhat_cal"]), 2),
                        "predicted_hourly":   round(float(row_fc["yhat_cal"]) * hdist[h], 3),
                    })

        except Exception as e:
            logger.warning("Forecast failed for zone %s: %s", zid, e)

    if not rows:
        return pd.DataFrame()

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.merge(
        zones_df[["h3_index", "lat", "lng", "violation_count",
                  "top_violation", "top_vehicle", "peak_hour_rate"]],
        on="h3_index", how="left"
    )
    return pred_df


def get_current_hour_risk(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract predicted risk for the current hour from the forecast dataframe.
    Returns one row per zone sorted by predicted_hourly descending.
    """
    if pred_df.empty or "date" not in pred_df.columns:
        return pd.DataFrame()

    now   = pd.Timestamp.now()
    today = now.date()
    hour  = now.hour

    current = pred_df[(pred_df["date"] == today) & (pred_df["hour"] == hour)].copy()

    if current.empty:
        # Fall back to today any hour
        current = pred_df[pred_df["date"] == today].copy()
        if current.empty:
            # Fall back to first available date
            current = pred_df.copy()

        current = current.groupby("h3_index", as_index=False)["predicted_hourly"].mean()
        meta_cols = [c for c in ["h3_index", "lat", "lng", "violation_count",
                                  "top_violation", "top_vehicle", "predicted_daily"]
                     if c in pred_df.columns]
        current = current.merge(
            pred_df[meta_cols].drop_duplicates("h3_index"),
            on="h3_index", how="left"
        )

    return current.sort_values("predicted_hourly", ascending=False).reset_index(drop=True)
