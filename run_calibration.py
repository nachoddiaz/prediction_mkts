#!/usr/bin/env python3
"""
run_calibration.py
───────────────────
Interactive CLI script to run and visualize calibration of GLFT and
Cartea-Jaimungal model parameters.

Allows selection between:
  1. Synthetic data (300+ samples, guarantees mathematical convergence).
  2. Real data from the local DuckDB database (runs against existing ticks/features).
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import strategies.market_making.params as params_module
from storage.reader import MarketDataReader
from strategies.market_making.params import CJCalibrator, GLFTCalibrator

# Terminal color codes
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


def header(title: str) -> None:
    width = 65
    print(f"\n{CYAN}{'═' * width}{RESET}")
    print(f"  {BOLD}{title.upper()}{RESET}")
    print(f"{CYAN}{'═' * width}{RESET}")


def subheader(title: str) -> None:
    width = 65
    print(f"\n  {BOLD}{MAGENTA}{title}{RESET}")
    print(f"  {DIM}{'─' * (width - 4)}{RESET}")


def print_row(label: str, value: object, unit: str = "") -> None:
    print(f"  {DIM}▶{RESET} {label:<35} {GREEN}{value}{RESET} {DIM}{unit}{RESET}")


def make_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic tick and feature data with known parameter priors."""
    n = 350
    rng = np.random.default_rng(42)
    base_time = datetime.now(UTC) - timedelta(hours=5)

    # 1. Ticks
    increments = rng.normal(0.0, 0.005, n)
    mids = np.cumsum(increments) + 0.50
    mids = np.clip(mids, 0.01, 0.99)
    spreads = np.abs(rng.normal(0.02, 0.005, n))

    timestamps = [base_time + timedelta(seconds=i * 10) for i in range(n)]

    ticks_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "tick_type": ["quote"] * n,
            "yes_bid": mids - spreads / 2,
            "yes_ask": mids + spreads / 2,
            "mid": mids,
            "spread": spreads,
            "volume": [0.0] * n,
            "side": [None] * n,
        }
    )

    # 2. Features
    # Generate OBI signal with mean-reversion drift
    obi = rng.uniform(-0.8, 0.8, n)
    tau = np.linspace(0.1, 0.0001, n)

    features_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "obi": obi,
            "quoted_spread": spreads,
            "relative_spread": spreads / mids,
            "belief_vol": [0.15] * n,
            "ewma_vol": [0.12] * n,
            "tau_years": tau,
            "mu_hat": obi * 0.05 + rng.normal(0, 0.01, n),
        }
    )

    return ticks_df, features_df


def run_synthetic_calibration() -> None:
    header("Mode 1: Calibration using Synthetic Data")
    print("  Generating 350 data points with known priors...")

    ticks_df, features_df = make_synthetic_data()

    # Fit GLFT
    subheader("1. Fitting GLFT Parameters (MLE Optimizer)")
    glft_cal = GLFTCalibrator()
    glft_res = glft_cal.fit(ticks_df)

    print_row(
        "Kappa Price (decay/dollar)", f"{glft_res.kappa_p:.4f}", "(higher means deeper order book)"
    )
    print_row("Kappa Logit (decay/logit)", f"{glft_res.kappa_x:.4f}", "(model parameter)")
    print_row("Baseline Arrival Rate (A)", f"{glft_res.A:.4f}", "fills/sec at zero spread")
    print_row("MLE Log-Likelihood", f"{glft_res.log_likelihood:.2f}")

    # Fit CJ
    subheader("2. Fitting Cartea-Jaimungal Parameters (Ridge + AR1)")
    cj_cal = CJCalibrator()
    cj_res = cj_cal.fit(ticks_df, features_df)

    print_row("Phi (mean-reversion speed)", f"{cj_res.phi:.4f}")
    print_row("Eta (drift volatility)", f"{cj_res.eta:.6f}")
    print_row("Rho (price-signal correlation)", f"{cj_res.rho:+.4f}", "[-0.99, +0.99]")
    print_row("Weight OBI (w_obi)", f"{cj_res.w_obi:+.6f}")
    print_row("Signal R²", f"{cj_res.signal_r2:.4f}")
    print_row("AR(1) Alpha", f"{cj_res.ar1_alpha:.4f}")


def run_database_calibration() -> None:
    header("Mode 2: Calibration using DuckDB Database Data")

    db_path = "./data/duckdb/markets.duckdb"
    if not os.path.exists(db_path):
        print(f"  {YELLOW}Error: Database not found at {db_path}. Please run ingest first.{RESET}")
        return

    # Temporarily override min tick restrictions for demo
    print(
        f"  {DIM}Overriding minimum tick counts"
        f" (MIN_TICKS_GLFT=5, MIN_TICKS_CJ=5) for demo...{RESET}"
    )
    params_module.MIN_TICKS_GLFT = 5
    params_module.MIN_TICKS_CJ = 5

    with MarketDataReader(db_path) as reader:
        markets_df = reader.markets()
        if markets_df.empty:
            print("  No markets found in database.")
            return

        print("\n  Available Markets in Local Database:")
        markets_list = markets_df.to_dict("records")

        # Count ticks and features per market, richest first
        for m in markets_list:
            mid = m["market_id"]
            m["_ticks"] = reader.count_ticks(mid)
            feat_df = reader.features(mid)
            m["_features"] = len(feat_df) if feat_df is not None else 0
        markets_list.sort(key=lambda m: m["_ticks"], reverse=True)

        for i, m in enumerate(markets_list, 1):
            print(
                f"    [{i}] {m['market_id']:<32}"
                f" | Ticks: {GREEN}{m['_ticks']:>5}{RESET}"
                f" | Features: {YELLOW}{m['_features']:>5}{RESET}"
                f" | {m['status']:<8}"
            )

        print(f"\n  Select a market number (1-{len(markets_list)}) [default 1]: ", end="")
        try:
            choice = input().strip()
            choice_idx = int(choice) - 1 if choice else 0
            if choice_idx < 0 or choice_idx >= len(markets_list):
                choice_idx = 0
        except ValueError:
            choice_idx = 0

        selected_market = markets_list[choice_idx]["market_id"]
        print(f"\n  Selected: {BOLD}{selected_market}{RESET}")

        # Load ticks & features
        ticks_df = reader.ticks(selected_market)
        features_df = reader.features(selected_market)

        print(f"  Loaded {len(ticks_df)} ticks and {len(features_df)} features from DuckDB.")

        if len(ticks_df) < 5 or len(features_df) < 5:
            print(
                f"  {YELLOW}Error: Selected market has too few rows"
                f" ({len(ticks_df)} ticks) to fit limits.{RESET}"
            )
            return

        # Fit GLFT
        try:
            subheader("1. Fitting GLFT Parameters")
            glft_res = GLFTCalibrator().fit(ticks_df)
            print_row("Kappa Price (decay/dollar)", f"{glft_res.kappa_p:.4f}")
            print_row("Kappa Logit (decay/logit)", f"{glft_res.kappa_x:.4f}")
            print_row("Baseline Arrival Rate (A)", f"{glft_res.A:.4f}")
            print_row("MLE Log-Likelihood", f"{glft_res.log_likelihood:.2f}")
        except Exception as e:
            print(f"  {YELLOW}GLFT fit failed: {e}{RESET}")

        # Fit CJ
        try:
            subheader("2. Fitting Cartea-Jaimungal Parameters")
            cj_res = CJCalibrator().fit(ticks_df, features_df)
            print_row("Phi (mean-reversion speed)", f"{cj_res.phi:.4f}")
            print_row("Eta (drift volatility)", f"{cj_res.eta:.6f}")
            print_row("Rho (price-signal correlation)", f"{cj_res.rho:+.4f}")
            print_row("Weight OBI (w_obi)", f"{cj_res.w_obi:+.6f}")
            print_row("Signal R²", f"{cj_res.signal_r2:.4f}")
            print_row("AR(1) Alpha", f"{cj_res.ar1_alpha:.4f}")
        except Exception as e:
            print(f"  {YELLOW}CJ fit failed: {e}{RESET}")


def main() -> None:
    print(f"\n{BOLD}{CYAN}=== Prediction Market Model Parameter Calibration CLI ==={RESET}")
    print("This tool fits GLFT and Cartea-Jaimungal parameters.")
    print("\nOptions:")
    print("  [1] Run with Synthetic Data (guarantees optimization convergence)")
    print("  [2] Run with Local DuckDB Data (fits existing database records)")
    print("\nChoose an option [1]: ", end="")

    choice = input().strip()
    if choice == "2":
        run_database_calibration()
    else:
        run_synthetic_calibration()
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Exiting.")
