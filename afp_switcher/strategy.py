from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    mom_short_days: int = 63
    mom_long_days: int = 252
    ewma_halflife: int = 63
    vol_penalty: float = 1.0
    switch_cost_bps: float = 10.0
    min_days_between_switches: int = 20


def _ewma_vol(returns: pd.DataFrame, halflife: int) -> pd.Series:
    # annualized-ish proxy: daily vol * sqrt(252)
    vol = returns.ewm(halflife=halflife, adjust=False).std().iloc[-1] * np.sqrt(252.0)
    return vol


def _momentum(prices: pd.DataFrame, days: int) -> pd.Series:
    if len(prices) <= days:
        raise ValueError(f"Not enough data for momentum window: {days} days.")
    mom = prices.iloc[-1] / prices.iloc[-(days + 1)] - 1.0
    return mom


def score_funds(prices: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    rets = prices.pct_change().dropna(how="all")
    mom_s = _momentum(prices, cfg.mom_short_days)
    mom_l = _momentum(prices, cfg.mom_long_days)
    vol = _ewma_vol(rets, cfg.ewma_halflife).replace(0.0, np.nan)

    # Combine momentums; penalize volatility.
    raw = 0.6 * mom_s + 0.4 * mom_l
    score = raw - cfg.vol_penalty * vol.fillna(vol.max())
    return score.sort_values(ascending=False)


def recommend_fund(
    prices: pd.DataFrame,
    *,
    current_fund: str | None,
    days_since_last_switch: int | None,
    cfg: StrategyConfig,
) -> tuple[str, pd.Series]:
    scores = score_funds(prices, cfg)
    best = str(scores.index[0])

    if current_fund is None:
        return best, scores

    if days_since_last_switch is not None and days_since_last_switch < cfg.min_days_between_switches:
        return current_fund, scores

    if best != current_fund:
        # Simple switch-cost gate: only switch if advantage beats cost.
        cost = cfg.switch_cost_bps / 10_000.0
        if (scores[best] - scores[current_fund]) < cost:
            return current_fund, scores

    return best, scores


def backtest(
    prices: pd.DataFrame,
    *,
    start: str | None,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    px = prices.copy()
    if start is not None:
        px = px.loc[pd.to_datetime(start) :]
    px = px.dropna(how="any")
    if len(px) < (cfg.mom_long_days + 2):
        raise ValueError("Not enough history for backtest with current config.")

    rets = px.pct_change().dropna()
    dates = rets.index

    current = None
    last_switch_idx = -10**9

    rows: list[dict] = []
    for i, dt in enumerate(dates):
        hist_px = px.loc[:dt]
        if len(hist_px) < (cfg.mom_long_days + 2):
            continue

        days_since = i - last_switch_idx if current is not None else None
        rec, scores = recommend_fund(
            hist_px,
            current_fund=current,
            days_since_last_switch=days_since,
            cfg=cfg,
        )
        if current is None:
            current = rec
            last_switch_idx = i
        elif rec != current:
            current = rec
            last_switch_idx = i

        day_ret = float(rets.loc[dt, current])
        rows.append(
            {
                "date": dt,
                "fund": current,
                "daily_return": day_ret,
                "equity": np.nan,
                "best_fund_today": str(scores.index[0]),
            }
        )

    out = pd.DataFrame(rows).set_index("date")
    if out.empty:
        raise ValueError("Backtest produced no rows. Check input dates and config.")
    out["equity"] = (1.0 + out["daily_return"]).cumprod()
    return out

