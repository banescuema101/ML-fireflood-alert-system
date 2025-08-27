# lst/predict_pixels_from_LST_model.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

import argparse, os, joblib
import pandas as pd
from utils import load_lst_lat_lon, normalize_lon, rolling_feats, parse_bbox

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lst-sen3", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="predictions/lst_pred_ro.parquet")
    ap.add_argument("--bbox", default="")
    ap.add_argument("--kernel", type=int, default=31)
    ap.add_argument("--proba-th", type=float, default=0.5)
    ap.add_argument("--lst-nc", default="")
    ap.add_argument("--lat-nc", default="")
    ap.add_argument("--lon-nc", default="")
    ap.add_argument("--lst-var", default="")
    ap.add_argument("--lat-var", default="")
    ap.add_argument("--lon-var", default="")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    FEATURES = bundle["features"]; model = bundle["model"]

    LST, lat, lon = load_lst_lat_lon(
        lst_sen3_dir=args.lst_sen3,
        lst_nc=args.lst_nc, lat_nc=args.lat_nc, lon_nc=args.lon_nc,
        lst_var=args.lst_var, lat_var=args.lat_var, lon_var=args.lon_var
    )
    ydim, xdim = LST.dims[-2], LST.dims[-1]

    if args.bbox:
        min_lat, min_lon, max_lat, max_lon = parse_bbox(args.bbox)
        m = (lat>=min_lat) & (lat<=max_lat) & (normalize_lon(lon)>=min_lon) & (normalize_lon(lon)<=max_lon)
        LST, lat, lon = LST.where(m), lat.where(m), lon.where(m)

    med, mad, mean, std, zmad = rolling_feats(LST, ydim, xdim, k=args.kernel)

    df = pd.DataFrame({
        "latitude":  lat.values.ravel(),
        "longitude": normalize_lon(lon.values.ravel()),
        "LST_K":     LST.values.ravel(),
        "med_k":     med.values.ravel(),
        "mad_k":     mad.values.ravel(),
        "mean_k":    mean.values.ravel(),
        "std_k":     std.values.ravel(),
        "zmad_k":    zmad.values.ravel(),
    }).dropna()

    if args.sample>0 and len(df)>args.sample:
        df = df.sample(args.sample, random_state=42)

    prob = model.predict_proba(df[FEATURES])[:,1]
    df["proba_fire"] = prob
    df["pred"] = (prob >= args.proba_th).astype("uint8")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.endswith(".csv"):
        df[["latitude","longitude","proba_fire","pred"]].to_csv(args.out, index=False)
    else:
        df[["latitude","longitude","proba_fire","pred"]].to_parquet(args.out, index=False)

    print(f">> Scris {args.out} cu {len(df):,} pixeli.")

if __name__ == "__main__":
    main()
