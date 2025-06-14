# Required for tile download
import math
import requests
from PIL import Image
from io import BytesIO
import os
import osmnx as ox
import pyproj
from shapely.geometry import Point
from shapely.ops import transform


def fetch_spatial_features(lat, lng, row_num, radius=1500):
    center = Point(lng, lat)

    tags = {
        'amenity': ['police', 'parking', 'bus_station'],
        'highway': True,
        'landuse': ['residential', 'industrial', 'forest'],
        'natural': ['water'],
        'barrier': ['fence', 'wall'],
        'building': True,
        'shop': True,
        'construction': True,
        'surface': True
    }

    gdf = ox.features.features_from_point((lat, lng), tags=tags, dist=radius)
    gdf = ox.projection.project_gdf(gdf)

    project_fn = pyproj.Transformer.from_crs("epsg:4326", gdf.crs, always_xy=True).transform
    center_proj = transform(project_fn, center)
    gdf['dist_m'] = gdf.geometry.distance(center_proj)

    result = {
        'row': row_num,
        'lat': lat,
        'lng': lng,
        'building_count': gdf[gdf['building'].notnull()].shape[0] if 'building' in gdf.columns else 0,
        'road_count': gdf[gdf['highway'].notnull()].shape[0] if 'highway' in gdf.columns else 0,
        'residential_count': gdf[gdf['landuse'] == 'residential'].shape[0] if 'landuse' in gdf.columns else 0,
        'industrial_count': gdf[gdf['landuse'] == 'industrial'].shape[0] if 'landuse' in gdf.columns else 0,
        'forest_count': gdf[gdf['landuse'] == 'forest'].shape[0] if 'landuse' in gdf.columns else 0,
        'waterbody_count': gdf[gdf['natural'] == 'water'].shape[0] if 'natural' in gdf.columns else 0,
        'fence_count': gdf[gdf['barrier'] == 'fence'].shape[0] if 'barrier' in gdf.columns else 0,
        'wall_count': gdf[gdf['barrier'] == 'wall'].shape[0] if 'barrier' in gdf.columns else 0,
        'parking_count': gdf[gdf['amenity'] == 'parking'].shape[0] if 'amenity' in gdf.columns else 0,
        'bus_stop_count': gdf[gdf['amenity'] == 'bus_station'].shape[0] if 'amenity' in gdf.columns else 0,
        'shop_count': gdf[gdf['shop'].notnull()].shape[0] if 'shop' in gdf.columns else 0,
        'surface_count': gdf[gdf['surface'].notnull()].shape[0] if 'surface' in gdf.columns else 0
    }

    if 'amenity' in gdf.columns:
        police = gdf[gdf['amenity'] == 'police'].nsmallest(1, 'dist_m')
        result['nearest_police_m'] = float(police['dist_m'].iloc[0]) if not police.empty else None
    else:
        result['nearest_police_m'] = None

    return result


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_tile(x, y, z, server="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"):
    url = server.format(x=x, y=y, z=z)
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print("❌ Tile fetch failed:", url)
        return None

def fetch_and_save_tile(lat, lon, zoom, filename):
    # if os.path.exists(filename):
    #     print(f"✅ Tile already exists: {filename}")
    #     return True
    x, y = deg2num(lat, lon, zoom)
    tile = fetch_tile(x, y, zoom)
    if tile:
        tile.save(filename)
        print(f"✅ Tile saved: {filename}")
        return True
    else:
        print(f"⚠️ Failed to save tile: {filename}")
        return False


from tensorflow.keras.models import load_model

model = load_model("model_wts/classModel.h5")

import numpy as np
from tensorflow.keras.preprocessing import image

# def preprocess_image(img_path, target_size=(128, 128)):
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def feature_vector_from_dict(row_dict):
    keys = [
        'building_count', 'road_count', 'residential_count', 'industrial_count', 'forest_count',
        'waterbody_count', 'fence_count', 'wall_count', 'parking_count', 'bus_stop_count',
        'shop_count', 'surface_count', 'nearest_police_m'
    ]
    return np.array([[row_dict.get(k, 0) if row_dict.get(k) is not None else 0 for k in keys]])

def predict_from_gps(lat, lon, row_num, zoom=17):
    filename = f"Testing/test_row_{row_num}_tile.jpg"
    tile_success = fetch_and_save_tile(lat, lon, zoom, filename)

    if not tile_success:
        print("❌ Could not fetch tile.")
        return None

    features = fetch_spatial_features(lat, lon, row_num)
    img_input = preprocess_image(filename)
    feature_input = feature_vector_from_dict(features)

    # Model prediction
    prediction = model.predict([img_input, feature_input])
    print(f"✅ Prediction for GPS ({lat}, {lon}):", "Illegal" if prediction[0][0] > 0.5 else "Legal", f"(score: {prediction[0][0]:.4f})")
    return prediction[0][0]

