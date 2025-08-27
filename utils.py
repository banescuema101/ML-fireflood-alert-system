# utils.py
import os, re, glob
import numpy as np
import xarray as xr

def normalize_lon(a):
    return ((a + 180.0) % 360.0) - 180.0

def parse_bbox(b):
    a = [float(x) for x in b.split(",")]
    if len(a) != 4:
        raise ValueError("BBOX invalid. Folosește 'min_lat,min_lon,max_lat,max_lon'")
    return a

def find_nc_with_var(folder, patterns, prefer_name_substr=None):
    candidates = sorted(glob.glob(os.path.join(folder, "*.nc")))
    if prefer_name_substr:
        pref = [p for p in candidates if re.search(prefer_name_substr, os.path.basename(p), re.I)]
        candidates = pref + [p for p in candidates if p not in pref]
    for path in candidates:
        try:
            ds = xr.open_dataset(path)
        except Exception:
            continue
        names = list(ds.data_vars) + list(ds.variables)
        for v in names:
            for rgx in patterns:
                if re.fullmatch(rgx, v, flags=re.I):
                    return path, v
    return None, None

# utils.py — înlocuiește funcțiile load_lst_lat_lon și rolling_feats cu cele de mai jos

def load_lst_lat_lon(
    lst_sen3_dir: str,
    lst_nc: str = "", lat_nc: str = "", lon_nc: str = "",
    lst_var: str = "", lat_var: str = "", lon_var: str = ""
):
    """
    Încarcă LST + latitude + longitude ca DataArray-uri aliniate.
    Handle:
      - dimensiune 'time' (luăm prima felie)
      - mască 'exception' (fallback dacă golește tot)
      - normalizare lon în [-180,180]
    """
    import numpy as np

    # ---- LST
    if lst_nc:
        dsL = xr.open_dataset(lst_nc)
        if not lst_var:
            lst_var = next((v for v in list(dsL.data_vars)+list(dsL.variables) if v.lower().startswith("lst")), None)
        if not lst_var or lst_var not in dsL.variables:
            raise SystemExit("Nu am găsit variabila LST în --lst-nc. Setează --lst-var.")
    else:
        path, var = find_nc_with_var(lst_sen3_dir, [r"LST", r"LST_?in"], prefer_name_substr="LST")
        if not path:
            raise SystemExit("Nu am găsit fișier/variabilă LST în scena LST.")
        dsL, lst_var = xr.open_dataset(path), var

    LST = dsL[lst_var]

    # ia prima felie pe orice dim 'time'-like
    for tdim in ("time", "scanline_time", "time_utc"):
        if tdim in LST.dims:
            LST = LST.isel({tdim: 0})
            break

    # ---- LAT
    if lat_nc:
        dsLat = xr.open_dataset(lat_nc)
        if not lat_var:
            lat_var = next((v for v in list(dsLat.data_vars)+list(dsLat.variables) if re.search(r"lat", v, re.I)), None)
    else:
        path_lat, var_lat = find_nc_with_var(lst_sen3_dir, [r"latitude", r"lat"], prefer_name_substr="lat")
        dsLat, lat_var = (xr.open_dataset(path_lat), var_lat) if path_lat else (dsL, "latitude")

    # ---- LON
    if lon_nc:
        dsLon = xr.open_dataset(lon_nc)
        if not lon_var:
            lon_var = next((v for v in list(dsLon.data_vars)+list(dsLon.variables) if re.search(r"lon|long", v, re.I)), None)
    else:
        path_lon, var_lon = find_nc_with_var(lst_sen3_dir, [r"longitude", r"lon|long"], prefer_name_substr="lon")
        dsLon, lon_var = (xr.open_dataset(path_lon), var_lon) if path_lon else (dsL, "longitude")

    lat = dsLat[lat_var] if lat_var in dsLat.variables else dsL[lat_var]
    lon = dsLon[lon_var] if lon_var in dsLon.variables else dsL[lon_var]

    # aliniază formele prin broadcast
    LST_m, lat_m, lon_m = xr.broadcast(LST, lat, lon)

    # aplică mască 'exception' dacă există, dar cu fallback dacă golește tot
    LST_masked = LST_m
    if "exception" in dsL.variables:
        try:
            exc = dsL["exception"]
            for tdim in ("time", "scanline_time", "time_utc"):
                if tdim in exc.dims and tdim in LST.dims:
                    exc = exc.isel({tdim: 0})
                    break
            exc_b = exc.broadcast_like(LST_m)
            LST_masked = LST_m.where(exc_b == 0)
        except Exception:
            pass

    # dacă masca a lăsat doar NaN, folosește nemascat (altfel nu mai avem nimic)
    if not np.isfinite(LST_masked.values).any():
        LST_masked = LST_m

    # normalizează lon în [-180,180]
    lon_m = ((lon_m + 180.0) % 360.0) - 180.0

    return LST_masked, lat_m, lon_m


def rolling_feats(A, ydim, xdim, k=31):
    """
    Rolling local stats (mean/std) NaN-aware, rapid (SciPy).
    median/MAD sunt aproximări robuste din mean/std ca să fie foarte rapide.
    Ajustează automat k dacă gridul e mic.
    """
    import xarray as xr, numpy as np
    try:
        import scipy.ndimage as ndi
    except Exception:
        # fallback la varianta xarray (mai lentă)
        k = int(k);  k = k if (k % 2 == 1) else (k+1)
        # mic safety: dacă fereastra e prea mare pt grilă, reduce-o
        h, w = A.shape[-2], A.shape[-1]
        k = max(3, min(k, h - (1 - h % 2), w - (1 - w % 2)))
        med = A.rolling({ydim:k, xdim:k}, center=True).median()
        mad = (A - med).abs().rolling({ydim:k, xdim:k}, center=True).median()
        mean = A.rolling({ydim:k, xdim:k}, center=True).mean()
        std  = A.rolling({ydim:k, xdim:k}, center=True).std()
        zmad = 0.6745 * (A - med) / (mad + 1e-6)
        return med, mad, mean, std, zmad

    # SciPy branch (rapid, NaN-aware)
    k = int(k);  k = k if (k % 2 == 1) else (k+1)
    h, w = A.shape[-2], A.shape[-1]
    # ajustează k dacă gridul e mic
    k = max(3, min(k, h - (1 - h % 2), w - (1 - w % 2)))

    arr = np.asarray(A.values, dtype=np.float32)
    mask = np.isfinite(arr).astype(np.float32)
    arr_z = np.where(mask, arr, 0.0)

    # medii NaN-aware: sum/count
    sum_  = ndi.uniform_filter(arr_z, size=(k, k), mode="nearest")
    cnt_  = ndi.uniform_filter(mask,  size=(k, k), mode="nearest")
    mean  = sum_ / (cnt_ + 1e-6)

    # var NaN-aware
    sumsq = ndi.uniform_filter(arr_z * arr_z, size=(k, k), mode="nearest")
    var   = np.maximum(sumsq / (cnt_ + 1e-6) - mean * mean, 1e-6)
    std   = np.sqrt(var, dtype=np.float32)

    # aproximări robuste
    med = mean
    mad = 1.4826 * std
    zmad = 0.6745 * (arr - med) / (mad + 1e-6)

    return (
        xr.DataArray(med, dims=A.dims, coords=A.coords),
        xr.DataArray(mad, dims=A.dims, coords=A.coords),
        xr.DataArray(mean, dims=A.dims, coords=A.coords),
        xr.DataArray(std,  dims=A.dims, coords=A.coords),
        xr.DataArray(zmad, dims=A.dims, coords=A.coords),
    )