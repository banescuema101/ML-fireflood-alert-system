import pandas as pd
import geopandas as gpd
import folium
import os

PRED_FILE = "Dataset/predictions/lst_pred_ro.parquet"
COUNTY_FILE = "data/gadm41_ROU_1.geojson"
OUT_HTML = "fire_map_by_county.html"
PIXEL_MAPS_DIR = "pixel_maps"

os.makedirs(PIXEL_MAPS_DIR, exist_ok=True)

# 1) Load data
df = pd.read_parquet(PRED_FILE)
gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
counties = gpd.read_file(COUNTY_FILE).to_crs("EPSG:4326")
name_col = "NAME_1"

# 2) Spatial join: pixel -> county
joined = gpd.sjoin(gdf_points, counties, how="inner", predicate="within")

# 3) Aggregate statistics per county
stats = joined.groupby(name_col).agg(
    total_pixels=("pred", "size"),
    fires=("pred", "sum"),
    mean_proba=("proba_fire", "mean")
).reset_index()
stats["fire_rate"] = stats["fires"] / stats["total_pixels"]

# 4) Merge with polygons
counties_stats = counties.merge(stats, on=name_col, how="left").fillna(0)

# 5) Main map
m = folium.Map(location=[45.8, 24.9], zoom_start=7, tiles="cartodbpositron")

choropleth = folium.Choropleth(
    geo_data=counties_stats,
    data=counties_stats,
    columns=[name_col, "fire_rate"],
    key_on=f"feature.properties.{name_col}",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.8,
    legend_name="Fire pixel percentage"
).add_to(m)

# 6) For each county, create a separate pixel map
for county_name in counties_stats[name_col]:
    subset = joined[joined[name_col] == county_name]

    if subset.empty:
        continue

    # Small map for county
    m_county = folium.Map(
        location=[subset.geometry.y.mean(), subset.geometry.x.mean()],
        zoom_start=8, tiles="cartodbpositron"
    )

    for _, row in subset.iterrows():
        color = "red" if row["pred"] == 1 else "green"
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m_county)

    # Save county-level map
    county_file = os.path.join(PIXEL_MAPS_DIR, f"map_{county_name}.html")
    m_county.save(county_file)

    # Add link to main map (popup on county)
    folium.Marker(
        location=[subset.geometry.y.mean(), subset.geometry.x.mean()],
        popup=f"<a href='{county_file}' target='_blank'>View pixels {county_name}</a>"
    ).add_to(m)

# 7) Save main map
m.save(OUT_HTML)
print(f">> Main map saved to {OUT_HTML}")
print(f">> Pixel-level maps are in folder {PIXEL_MAPS_DIR}")
