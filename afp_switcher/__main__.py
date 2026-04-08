from __future__ import annotations

import argparse
import json

from .io import read_prices_csv
from .strategy import StrategyConfig, backtest, recommend_fund


def _cfg_from_args(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        mom_short_days=args.mom_short_days,
        mom_long_days=args.mom_long_days,
        ewma_halflife=args.ewma_halflife,
        vol_penalty=args.vol_penalty,
        switch_cost_bps=args.switch_cost_bps,
        min_days_between_switches=args.min_days_between_switches,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="afp_switcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--prices", required=True, help="CSV with date,A,B,C,D,E price series.")
    common.add_argument("--mom-short-days", type=int, default=63)
    common.add_argument("--mom-long-days", type=int, default=252)
    common.add_argument("--ewma-halflife", type=int, default=63)
    common.add_argument("--vol-penalty", type=float, default=1.0)
    common.add_argument("--switch-cost-bps", type=float, default=10.0)
    common.add_argument("--min-days-between-switches", type=int, default=20)

    r = sub.add_parser("recommend", parents=[common])
    r.add_argument("--current-fund", default=None, choices=[None, "A", "B", "C", "D", "E"])
    r.add_argument("--days-since-last-switch", type=int, default=None)

    b = sub.add_parser("backtest", parents=[common])
    b.add_argument("--start", default=None, help="YYYY-MM-DD")
    b.add_argument("--out", default=None, help="Optional output CSV path for backtest.")

    return p


def cmd_recommend(args: argparse.Namespace) -> int:
    prices = read_prices_csv(args.prices)
    cfg = _cfg_from_args(args)
    rec, scores = recommend_fund(
        prices,
        current_fund=args.current_fund,
        days_since_last_switch=args.days_since_last_switch,
        cfg=cfg,
    )
    payload = {
        "recommended_fund": rec,
        "asof": str(prices.index.max().date()),
        "scores": {k: float(v) for k, v in scores.items()},
        "config": cfg.__dict__,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    prices = read_prices_csv(args.prices)
    cfg = _cfg_from_args(args)
    bt = backtest(prices, start=args.start, cfg=cfg)
    summary = {
        "start": str(bt.index.min().date()),
        "end": str(bt.index.max().date()),
        "final_equity": float(bt["equity"].iloc[-1]),
        "max_drawdown": float((bt["equity"] / bt["equity"].cummax() - 1.0).min()),
        "switches": int((bt["fund"] != bt["fund"].shift(1)).sum()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        bt.to_csv(args.out)
    return 0


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    if args.cmd == "recommend":
        return cmd_recommend(args)
    if args.cmd == "backtest":
        return cmd_backtest(args)
    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

