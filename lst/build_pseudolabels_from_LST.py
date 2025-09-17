import os, sys

# PATH CONFIGURATION
# Get the project root (parent directory of the 'lst' folder)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add root to sys.path if it's not already there, so we can import "utils"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import numpy as np
import pandas as pd
# Import helper functions from utils.py
from utils import load_lst_lat_lon, normalize_lon, rolling_feats, parse_bbox


def main():
    # ARGUMENT PARSING
    ap = argparse.ArgumentParser()
    # Directory with S3_SL_2_LST____ files (input scenes)
    ap.add_argument("--lst-sen3", required=True, help="Folder S3?_SL_2_LST____...SEN3 (scene for train)")
    # Output file with samples + labels
    ap.add_argument("--out", default="Dataset/lst/train_samples.parquet", help="Parquet with features+label")
    # Optional bounding box (for cropping)
    ap.add_argument("--bbox", default="", help="min_lat,min_lon,max_lat,max_lon (optional)")
    # Kernel size for local statistics
    ap.add_argument("--kernel", type=int, default=31, help="window size for local statistics")

    # Manual thresholds for pseudo-labels
    ap.add_argument("--z-pos", type=float, default=3.5, help="zmad >= z_pos => positive candidate")
    ap.add_argument("--k-pos", type=float, default=330.0, help="LST_K >= k_pos (K) => positive candidate")
    ap.add_argument("--z-neg", type=float, default=1.0, help="zmad <= z_neg => negative candidate")
    ap.add_argument("--k-neg", type=float, default=310.0, help="LST_K <= k_neg (K) => negative candidate")

    # If no positives result from the above thresholds, AUTO fallback can be enabled
    ap.add_argument("--auto", action="store_true",
                    help="If no positives with given thresholds, use quantile-based thresholds.")
    ap.add_argument("--pos-q", type=float, default=0.99, help="Quantile for positives (0–1) in AUTO")
    ap.add_argument("--neg-q", type=float, default=0.40, help="Quantile for negatives (0–1) in AUTO")
    ap.add_argument("--zmin",  type=float, default=1.25, help="Minimum ZMAD in AUTO")

    # Sampling / balancing control
    ap.add_argument("--neg-per-pos", type=int, default=3, help="negative/positive ratio to keep")
    ap.add_argument("--pos-max", type=int, default=200_000, help="limit #positives kept (memory)")
    ap.add_argument("--stride", type=int, default=1, help="subsample grid (e.g., 2 => every other pixel)")

    # Possibility to specify explicit files (if only SEN3 folder is not given)
    ap.add_argument("--lst-nc", default="", help="Explicit LST file (e.g. LST_in.nc)")
    ap.add_argument("--lat-nc", default="", help="Explicit latitude file (e.g. geodetic_in.nc)")
    ap.add_argument("--lon-nc", default="", help="Explicit longitude file (e.g. geodetic_in.nc)")
    ap.add_argument("--lst-var", default="", help="LST variable name (e.g. LST or LST_in)")
    ap.add_argument("--lat-var", default="", help="Latitude variable name (e.g. latitude_in)")
    ap.add_argument("--lon-var", default="", help="Longitude variable name (e.g. longitude_in)")
    args = ap.parse_args()

    # === 1) Load LST + coordinates
    LST, lat, lon = load_lst_lat_lon(
        lst_sen3_dir=args.lst_sen3,
        lst_nc=args.lst_nc, lat_nc=args.lat_nc, lon_nc=args.lon_nc,
        lst_var=args.lst_var, lat_var=args.lat_var, lon_var=args.lon_var
    )
    # Names of lat and lon dimensions
    ydim, xdim = LST.dims[-2], LST.dims[-1]

    # 2) Apply BBOX (optional)
    if args.bbox:
        # Parse min/max coordinates
        min_lat, min_lon, max_lat, max_lon = parse_bbox(args.bbox)
        # Build a selection mask for the given area
        m = (lat >= min_lat) & (lat <= max_lat) & (normalize_lon(lon) >= min_lon) & (normalize_lon(lon) <= max_lon)
        # Apply mask to data
        LST, lat, lon = LST.where(m), lat.where(m), lon.where(m)

    # Sub-sampling (stride)
    if args.stride > 1:
        s = int(args.stride)
        LST = LST.isel({ydim: slice(None, None, s), xdim: slice(None, None, s)})
        lat = lat.isel({ydim: slice(None, None, s), xdim: slice(None, None, s)})
        lon = lon.isel({ydim: slice(None, None, s), xdim: slice(None, None, s)})

    # 4) Compute local features (rolling)
    # Compute median, MAD, mean, std and zmad in a local window
    med, mad, mean, std, zmad = rolling_feats(LST, ydim, xdim, k=args.kernel)

    # 5) Vectorize into Pandas DataFrame
    df = pd.DataFrame({
        "latitude":  lat.values.ravel(),
        "longitude": normalize_lon(lon.values.ravel()),
        "LST_K":     LST.values.ravel(),
        "med_k":     med.values.ravel(),
        "mad_k":     mad.values.ravel(),
        "mean_k":    mean.values.ravel(),
        "std_k":     std.values.ravel(),
        "zmad_k":    zmad.values.ravel(),
    }).dropna()  # drop NaN values

    # If no data after processing, exit
    if df.empty:
        raise SystemExit("No valid observation after pre-processing (BBOX/stride?).")

    # Generate pseudo-labels (manual)
    pos_mask = (df["zmad_k"] >= args.z_pos) & (df["LST_K"] >= args.k_pos)  # positives = hot anomalies
    neg_mask = (df["zmad_k"] <= args.z_neg) & (df["LST_K"] <= args.k_neg)  # negatives = cold and stable zones
    pos_df = df[pos_mask]
    neg_df = df[neg_mask]

    # AUTO fallback if no positives
    if len(pos_df) == 0 and args.auto:
        # Automatically compute thresholds from quantiles
        k_pos_auto = float(df["LST_K"].quantile(args.pos_q))
        z_pos_auto = max(float(df["zmad_k"].quantile(args.pos_q)), args.zmin)
        k_neg_auto = float(df["LST_K"].quantile(args.neg_q))

        # Regenerate sets
        pos_df = df[(df["LST_K"] >= k_pos_auto) | (df["zmad_k"] >= z_pos_auto)]
        neg_df = df[(df["LST_K"] <= k_neg_auto) & (df["zmad_k"] <= 0.5)]

        print(f"[AUTO] k_pos={k_pos_auto:.2f}, z_pos={z_pos_auto:.2f}, k_neg={k_neg_auto:.2f} "
              f"-> pos={len(pos_df)}, neg={len(neg_df)}")

    # If still no positives, stop
    if len(pos_df) == 0:
        raise SystemExit("No positive samples resulted with the given thresholds. Use --auto or relax --k-pos/--z-pos.")
    # If no negatives, take bottom quantile as fallback
    if len(neg_df) == 0:
        k_neg_auto = float(df["LST_K"].quantile(0.4))
        neg_df = df[(df["LST_K"] <= k_neg_auto) & (df["zmad_k"] <= 0.5)]
        print(f"[FALLBACK] generated negatives with k_neg_auto={k_neg_auto:.2f} -> neg={len(neg_df)}")
        if len(neg_df) == 0:
            raise SystemExit("Failed to obtain negatives even with fallback.")

    # 7) Balance dataset
    # Limit number of positives (max pos_max)
    if len(pos_df) > args.pos_max:
        pos_df = pos_df.sample(args.pos_max, random_state=42)
    # Keep a reasonable number of negatives (negative/positive ratio)
    neg_keep = min(len(neg_df), args.neg_per_pos * len(pos_df) + 50_000)
    if neg_keep < len(neg_df):
        neg_df = neg_df.sample(neg_keep, random_state=42)

    # Add label column: 1 for positives, 0 for negatives
    pos_df = pos_df.copy(); pos_df["label"] = 1
    neg_df = neg_df.copy(); neg_df["label"] = 0

    # Concatenate and shuffle dataset
    train_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=42)

    # 8) Save to Parquet file
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_parquet(args.out, index=False)
    print(f">> Written {args.out} with {len(train_df):,} samples (pos={len(pos_df):,}, neg={len(neg_df):,}).")


if __name__ == "__main__":
    main()
