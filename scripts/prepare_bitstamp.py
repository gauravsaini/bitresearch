"""
Data preparation for BTC/USD trading signal generation.
Adapts bitstamp OHLCV data into token format for WebGPU training.

Usage:
    python prepare_bitstamp.py                  # full dataset
    python prepare_bitstamp.py --sample 500000  # use first N rows
"""

import os
import sys
import json
import argparse
import struct

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PUBLIC_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "public", "data"
)
BITSTAMP_DATA_PATH = "/home/gsai/bitstamp-btcusd-minute-data/data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
VAL_RATIO = 0.1

# Discretization bins for each feature
N_BINS = 32  # bins per feature
VOCAB_SIZE = 8192  # must match model config

# ---------------------------------------------------------------------------
# Technical Indicators (pure pandas/numpy)
# ---------------------------------------------------------------------------


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def compute_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return sma + (std * num_std), sma, sma - (std * num_std)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_features(df):
    """Extract and normalize features from OHLCV data."""
    feats = {}

    # Returns
    feats["returns"] = df["close"].pct_change()
    feats["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Price range
    feats["hl_range"] = (df["high"] - df["low"]) / df["close"]
    feats["oc_range"] = (df["close"] - df["open"]) / df["open"]

    # Moving average crosses
    sma7 = df["close"].rolling(7).mean()
    sma25 = df["close"].rolling(25).mean()
    sma99 = df["close"].rolling(99).mean()
    feats["sma7_25"] = sma7 / sma25 - 1
    feats["sma25_99"] = sma25 / sma99 - 1

    # RSI (normalized to -1..1)
    feats["rsi"] = (compute_rsi(df["close"]) - 50) / 50

    # MACD (normalized)
    macd, macd_sig = compute_macd(df["close"])
    feats["macd"] = macd / df["close"]
    feats["macd_signal"] = macd_sig / df["close"]
    feats["macd_hist"] = (macd - macd_sig) / df["close"]

    # Bollinger position
    bb_upper, bb_mid, bb_lower = compute_bollinger(df["close"])
    feats["bb_pos"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    feats["bb_width"] = (bb_upper - bb_lower) / bb_mid

    # Volume
    vol_sma = df["volume"].rolling(20).mean()
    feats["vol_ratio"] = df["volume"] / (vol_sma + 1e-8)

    return pd.DataFrame(feats)


# ---------------------------------------------------------------------------
# Discretization: convert continuous features to token IDs
# ---------------------------------------------------------------------------


def discretize_features(features_df, n_bins=N_BINS):
    """
    Convert continuous features into discretized token IDs.
    Each timestep becomes a single token ID encoding all features.
    Token = sum of (feature_bin * n_bins^feature_idx) with quantization.
    """
    # Drop NaN rows
    valid = features_df.dropna()
    print(f"After dropping NaN: {len(valid)} rows")

    # Clip and bin each feature
    col_names = list(valid.columns)
    n_features = len(col_names)
    max_vocab = n_bins**n_features

    if max_vocab > VOCAB_SIZE:
        # Need to reduce - use hashing approach
        print(
            f"Using hash-based tokenization ({n_features} features, {n_bins} bins each)"
        )
        tokens = np.zeros(len(valid), dtype=np.int32)

        for i, col in enumerate(col_names):
            vals = valid[col].values
            # Clip to [-3, 3] range (roughly 99.7% of normal distribution)
            vals = np.clip(vals, -3, 3)
            # Normalize to [0, 1]
            vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            # Bin
            bins = (vals_norm * (n_bins - 1)).astype(np.int32)
            # Hash into vocab space
            tokens = (tokens * 31 + bins) % (
                VOCAB_SIZE - 3
            )  # reserve 0,1,2 for signals

        tokens = tokens + 3  # offset to avoid 0,1,2 (reserved for hold,buy,sell)
    else:
        # Exact encoding (only for small feature counts)
        tokens = np.zeros(len(valid), dtype=np.int32)
        for i, col in enumerate(col_names):
            vals = valid[col].values
            vals = np.clip(vals, -3, 3)
            vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            bins = (vals_norm * (n_bins - 1)).astype(np.int32)
            tokens += bins * (n_bins**i)

    valid = valid.copy()
    valid["token"] = tokens
    return valid


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------


def generate_signals(df):
    """Generate buy/sell/hold labels based on future returns."""
    # Look ahead 5 minutes
    future_return = df["close"].shift(-5) / df["close"] - 1

    # Thresholds
    buy_thresh = 0.002  # 0.2% gain
    sell_thresh = -0.002  # 0.2% loss

    signals = np.zeros(len(df), dtype=np.int32)
    signals[future_return > buy_thresh] = 1  # buy
    signals[future_return < sell_thresh] = 2  # sell
    # 0 = hold (default)

    return signals


# ---------------------------------------------------------------------------
# Save as binary tokens
# ---------------------------------------------------------------------------


def save_tokens(tokens, signals, prefix=""):
    """
    Save token sequence as binary Int32Array.
    Format: interleaved [token, signal, token, signal, ...]
    The model predicts the next signal given token context.
    """
    os.makedirs(PUBLIC_DATA_DIR, exist_ok=True)

    # Interleave: input is tokens, target is next signal
    # Sequence: T0, S0, T1, S1, T2, S2, ...
    # Model learns: given T0..Tn predict S0..Sn
    interleaved = np.zeros(len(tokens) * 2, dtype=np.int32)
    interleaved[0::2] = tokens
    interleaved[1::2] = signals

    bin_path = os.path.join(PUBLIC_DATA_DIR, f"{prefix}tokens.bin")
    meta_path = os.path.join(PUBLIC_DATA_DIR, f"{prefix}tokens_meta.json")

    # Write binary
    interleaved.tofile(bin_path)

    # Write metadata
    meta = {
        "numTokens": len(interleaved),
        "vocabSize": VOCAB_SIZE,
        "nFeatures": len(tokens),
        "format": "interleaved_token_signal",
        "signalEncoding": {"0": "hold", "1": "buy", "2": "sell"},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(interleaved)} tokens to {bin_path}")
    print(
        f"Signal distribution: hold={np.sum(signals == 0)}, buy={np.sum(signals == 1)}, sell={np.sum(signals == 2)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Use first N rows")
    args = parser.parse_args()

    # Load
    print(f"Loading {BITSTAMP_DATA_PATH}")
    df = pd.read_csv(BITSTAMP_DATA_PATH, compression="gzip")
    print(f"Loaded {len(df)} records")

    if args.sample:
        df = df.head(args.sample)
        print(f"Using sample of {args.sample}")

    # Extract features
    print("Extracting features...")
    features = extract_features(df)

    # Discretize
    print("Discretizing features...")
    disc = discretize_features(features)
    tokens = disc["token"].values

    # Generate signals (aligned with discretized data)
    aligned_df = df.loc[disc.index]
    signals = generate_signals(aligned_df)

    # Train/val split
    split = int(len(tokens) * (1 - VAL_RATIO))
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    train_signals, val_signals = signals[:split], signals[split:]

    print(f"Train: {len(train_tokens)} tokens")
    print(f"Val: {len(val_tokens)} tokens")

    # Save
    save_tokens(train_tokens, train_signals, prefix="")
    save_tokens(val_tokens, val_signals, prefix="val_")

    print("Done!")


if __name__ == "__main__":
    main()
