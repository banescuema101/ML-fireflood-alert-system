# make_map_webgl.py
import argparse, pandas as pd, numpy as np, pydeck as pdk
from utils import parse_bbox, normalize_lon

RO_BBOX = (43.6, 20.2, 48.3, 29.9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV sau PARQUET cu latitude, longitude, proba_fire, pred")
    ap.add_argument("--html", default="fire_map_webgl.html")
    ap.add_argument("--bbox", default="", help="min_lat,min_lon,max_lat,max_lon")
    ap.add_argument("--lat-col", default="latitude")
    ap.add_argument("--lon-col", default="longitude")
    ap.add_argument("--radius", type=float, default=120.0)
    args = ap.parse_args()

    if args.csv.lower().endswith(".parquet"):
        df = pd.read_parquet(args.csv, columns=[args.lat_col, args.lon_col, "proba_fire", "pred"])
    else:
        df = pd.read_csv(args.csv, usecols=[args.lat_col, args.lon_col, "proba_fire", "pred"])

    df[args.lon_col] = normalize_lon(df[args.lon_col].astype(float))
    df = df[df[args.lat_col].between(-90,90) & df[args.lon_col].between(-180,180)]

    if args.bbox:
        min_lat, min_lon, max_lat, max_lon = parse_bbox(args.bbox)
    else:
        min_lat, min_lon, max_lat, max_lon = RO_BBOX
    df = df[df[args.lat_col].between(min_lat,max_lat) & df[args.lon_col].between(min_lon,max_lon)]

    if df.empty:
        raise SystemExit("Nu sunt puncte în BBOX.")

    # culoare: roșu pt pred=1, verde pt pred=0; alpha ~ proba
    df["_r"] = np.where(df["pred"]==1, 220, 30)
    df["_g"] = np.where(df["pred"]==1, 30, 180)
    df["_b"] = 40
    df["_a"] = (50 + 205*df["proba_fire"].clip(0,1)).astype(int)

    view = pdk.ViewState(
        latitude=float(df[args.lat_col].median()),
        longitude=float(df[args.lon_col].median()),
        zoom=5.8, min_zoom=4, max_zoom=12
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=[args.lon_col, args.lat_col],
        get_fill_color=["_r","_g","_b","_a"],
        get_radius=args.radius,
        radius_min_pixels=1, radius_max_pixels=10,
        pickable=True, auto_highlight=True, stroked=False
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html":"<b>p_fire</b>={proba_fire:.3f}<br/><b>pred</b>={pred}",
                 "style":{"backgroundColor":"white","color":"black"}}
    )
    deck.to_html(args.html, css_background_color="#ffffff")
    print(f">> Harta WebGL salvată: {args.html}")

if __name__ == "__main__":
    main()
