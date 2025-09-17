# Script pentru a genera pseudo-labels pe baza datelor LST
# (Land Surface Temperature)
# Se lucreaza doar cu LST, fara alte surse
# (ex. FRP). Include fallback AUTO pentru praguri.

import os, sys

# CONFIGURARE PATH
# Obtinem root-ul proiectului (directorul parinte al folderului 'lst')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Adaugam root-ul in sys.path daca nu e deja acolo, pentru a putea importa "utils"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import numpy as np
import pandas as pd
# Importam functii ajutatoare din utils.py
from utils import load_lst_lat_lon, normalize_lon, rolling_feats, parse_bbox


def main():
    # PARSARE ARGUMENTE
    ap = argparse.ArgumentParser()
    # Directorul cu fisiere S3_SL_2_LST____ (scenele de intrare)
    ap.add_argument("--lst-sen3", required=True, help="Folder S3?_SL_2_LST____...SEN3 (scena pentru train)")
    # Fisierul de iesire cu mostre + labels
    ap.add_argument("--out", default="Dataset/lst/train_samples.parquet", help="Parquet cu features+label")
    # Bounding box optional (pentru decupare)
    ap.add_argument("--bbox", default="", help="min_lat,min_lon,max_lat,max_lon (optional)")
    # Dimensiunea ferestrei pentru statistici locale
    ap.add_argument("--kernel", type=int, default=31, help="fereastra pt statistici locale")

    # Praguri manuale pentru pseudo-labels
    ap.add_argument("--z-pos", type=float, default=3.5, help="zmad >= z_pos => candidat pozitiv")
    ap.add_argument("--k-pos", type=float, default=330.0, help="LST_K >= k_pos (K) => candidat pozitiv")
    ap.add_argument("--z-neg", type=float, default=1.0, help="zmad <= z_neg => candidat negativ")
    ap.add_argument("--k-neg", type=float, default=310.0, help="LST_K <= k_neg (K) => candidat negativ")

    # Daca nu se obtin pozitivi cu pragurile de mai sus, se poate activa fallback AUTO
    ap.add_argument("--auto", action="store_true",
                    help="Daca nu ies pozitivi cu pragurile date, foloseste praguri din cuantile.")
    ap.add_argument("--pos-q", type=float, default=0.99, help="Cuantila pentru pozitivi (0–1) in AUTO")
    ap.add_argument("--neg-q", type=float, default=0.40, help="Cuantila pentru negativi (0–1) in AUTO")
    ap.add_argument("--zmin",  type=float, default=1.25, help="ZMAD minim in AUTO")

    # Control sampling / balansare
    ap.add_argument("--neg-per-pos", type=int, default=3, help="raport negativi/pozitivi pastrati")
    ap.add_argument("--pos-max", type=int, default=200_000, help="limiteaza #pozitive pastrate (memorie)")
    ap.add_argument("--stride", type=int, default=1, help="sub-esantioneaza grila (ex. 2 => fiecare al doilea pixel)")

    # Posibilitatea de a specifica fisiere explicite (daca nu e dat doar folderul SEN3)
    ap.add_argument("--lst-nc", default="", help="Fisier LST explicit (ex. LST_in.nc)")
    ap.add_argument("--lat-nc", default="", help="Fisier latitude explicit (ex. geodetic_in.nc)")
    ap.add_argument("--lon-nc", default="", help="Fisier longitude explicit (ex. geodetic_in.nc)")
    ap.add_argument("--lst-var", default="", help="Nume variabila LST (ex. LST sau LST_in)")
    ap.add_argument("--lat-var", default="", help="Nume variabila lat (ex. latitude_in)")
    ap.add_argument("--lon-var", default="", help="Nume variabila lon (ex. longitude_in)")
    args = ap.parse_args()

    # === 1) incarcare LST + coordonate
    LST, lat, lon = load_lst_lat_lon(
        lst_sen3_dir=args.lst_sen3,
        lst_nc=args.lst_nc, lat_nc=args.lat_nc, lon_nc=args.lon_nc,
        lst_var=args.lst_var, lat_var=args.lat_var, lon_var=args.lon_var
    )
    # Numele dimensiunilor de lat si lon
    ydim, xdim = LST.dims[-2], LST.dims[-1]

    # 2) Aplicare BBOX (optional)
    if args.bbox:
        # Parsam coordonatele min/max
        min_lat, min_lon, max_lat, max_lon = parse_bbox(args.bbox)
        # Construim o masca de selectie pentru zona data
        m = (lat >= min_lat) & (lat <= max_lat) & (normalize_lon(lon) >= min_lon) & (normalize_lon(lon) <= max_lon)
        # Aplicam masca asupra datelor
        LST, lat, lon = LST.where(m), lat.where(m), lon.where(m)

    # Sub-esantionare (stride)
    if args.stride > 1:
        s = int(args.stride)
        LST = LST.isel({ydim: slice(None, None, s), xdim: slice(None, None, s)})
        lat = lat.isel({ydim: slice(None, None, s), xdim: slice(None, None, s)})
        lon = lon.isel({ydim: slice(None, None, s), xdim: slice(None, None, s)})

    # 4) Calcul features locale (rolling)
    # Calculam mediana, MAD, media, deviatia standard si zmad intr-o fereastra locala
    med, mad, mean, std, zmad = rolling_feats(LST, ydim, xdim, k=args.kernel)

    # 5) Vectorizare in DataFrame Pandas
    df = pd.DataFrame({
        "latitude":  lat.values.ravel(),
        "longitude": normalize_lon(lon.values.ravel()),
        "LST_K":     LST.values.ravel(),
        "med_k":     med.values.ravel(),
        "mad_k":     mad.values.ravel(),
        "mean_k":    mean.values.ravel(),
        "std_k":     std.values.ravel(),
        "zmad_k":    zmad.values.ravel(),
    }).dropna()  # elimina valorile NaN

    # Daca dupa procesare nu avem date, iesim
    if df.empty:
        raise SystemExit("Nicio observatie valida dupa preprocesare (BBOX/stride?).")

    # Generare de pseudo-labels (manual)
    pos_mask = (df["zmad_k"] >= args.z_pos) & (df["LST_K"] >= args.k_pos)  # pozitivi = anomalii fierbinti
    neg_mask = (df["zmad_k"] <= args.z_neg) & (df["LST_K"] <= args.k_neg)  # negativi = zone reci si stabile
    pos_df = df[pos_mask]
    neg_df = df[neg_mask]

    # Fallback AUTO daca nu exista pozitivi
    if len(pos_df) == 0 and args.auto:
        # Practic, se calculeaza pragurile automat din cuantile
        k_pos_auto = float(df["LST_K"].quantile(args.pos_q))
        z_pos_auto = max(float(df["zmad_k"].quantile(args.pos_q)), args.zmin)
        k_neg_auto = float(df["LST_K"].quantile(args.neg_q))

        # Regeneram seturile
        pos_df = df[(df["LST_K"] >= k_pos_auto) | (df["zmad_k"] >= z_pos_auto)]
        neg_df = df[(df["LST_K"] <= k_neg_auto) & (df["zmad_k"] <= 0.5)]

        print(f"[AUTO] k_pos={k_pos_auto:.2f}, z_pos={z_pos_auto:.2f}, k_neg={k_neg_auto:.2f} "
              f"-> pos={len(pos_df)}, neg={len(neg_df)}")

    # Daca nici acum nu avem pozitivi, oprim executia
    if len(pos_df) == 0:
        raise SystemExit("Nu au rezultat pozitive cu pragurile date. Foloseste --auto sau relaxeaza --k-pos/--z-pos.")
    # Daca nu avem negative, luam implicit bottom quantile
    if len(neg_df) == 0:
        k_neg_auto = float(df["LST_K"].quantile(0.4))
        neg_df = df[(df["LST_K"] <= k_neg_auto) & (df["zmad_k"] <= 0.5)]
        print(f"[FALLBACK] generat negative cu k_neg_auto={k_neg_auto:.2f} -> neg={len(neg_df)}")
        if len(neg_df) == 0:
            raise SystemExit("Nu am reusit sa obtin negative nici cu fallback.")

    # 7) Balansare set
    # Limitam numarul de pozitivi (maxim pos_max)
    if len(pos_df) > args.pos_max:
        pos_df = pos_df.sample(args.pos_max, random_state=42)
    # Pastram un numar rezonabil de negativi (raport negativi/pozitivi)
    neg_keep = min(len(neg_df), args.neg_per_pos * len(pos_df) + 50_000)
    if neg_keep < len(neg_df):
        neg_df = neg_df.sample(neg_keep, random_state=42)

    # Adaugam coloana de label: 1 pentru pozitivi, 0 pentru negativi
    pos_df = pos_df.copy(); pos_df["label"] = 1
    neg_df = neg_df.copy(); neg_df["label"] = 0

    # Concatenam si amestecam setul de date
    train_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=42)

    # 8) Salvare in fisier Parquet
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_parquet(args.out, index=False)
    print(f">> Scris {args.out} cu {len(train_df):,} mostre (pos={len(pos_df):,}, neg={len(neg_df):,}).")


if __name__ == "__main__":
    main()