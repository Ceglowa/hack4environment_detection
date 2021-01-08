import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import OPTICS
import numpy as np
from sklearn.metrics import pairwise_distances
import pickle as pkl
import sklearn
from geopy import distance

from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist


with open("jsondata.json") as f:
    js = json.load(f)
    coords = [feat['geometry']['coordinates'] for feat in js['features']]
    ble = [Point(c) for c in coords]
    coords_np = np.array([c[::-1] for c in coords])
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df = gpd.GeoDataFrame(gpd.GeoSeries(ble), columns=['geometry'], crs="EPSG:4326")
    # print(pd.DataFrame(coords).describe())
    # world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # df = df.to_crs(world.crs)
    # df.to_file("data.shp")
    # base = world.plot(color='white', edgecolor='black')
    # df.plot(ax=base, marker='o', color='red', markersize=5)
    # plt.show()
    result = gpd.sjoin(df, world, how='left')
    netherlads = result[result.name == "Netherlands"]
    
    
    # sc_dist = pairwise_distances(coords_np, metric="haversine", n_jobs=-1)
    # sc_dist = pairwise_distances(coords_np, metric=lambda u, v: geodist(u, v).meters, n_jobs=-1)
    # sc_dist = cdist(coords_np, coords_np, metric=lambda u, v: geodist(u, v).meters)
    # sc_dist = pairwise_distances(coords_np, metric="euclidean", n_jobs=-1)
    # with open("distances.pkl.gz", "wb") as f:
    #     pkl.dump(sc_dist, f)
    # clustering = OPTICS(metric='precomputed', n_jobs=15).fit(sc_dist)
    # with open("clustering.pkl.gz", "wb") as f:
    #     pkl.dump(clustering, f)