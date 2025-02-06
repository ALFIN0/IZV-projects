#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as cx
import sklearn.cluster
import numpy as np
import datetime
from matplotlib.colors import Normalize
from matplotlib import cm

def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """Convert dataframe to geopandas.GeoDataFrame with appropriate coding

    Args:
    df (pd.DataFrame) -- input dataframe
    """

    # transformation DataFrame to GeoDataFrame with CRS = ESPG:5514
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.d, df.e), crs="EPSG:5514")
    # drop rows without complete coordinations
    gdf.drop(df.index[(np.isnan(df['d'])) | (np.isnan(df['e']))], inplace = True)

    return gdf


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """Plot graph with accidents with alcohol or drugs influence in years 2018-2021
    
    Args:
    gdf (geopandas.GeoDataFrame) -- input
    fig_location (str, optional) -- output file location (default None)
    show_figure (bool, optional) -- show figure to output (default False)
    """

    # copy of dataframe with reduced rows to rows with MSK region
    newGDF = gdf.copy()
    newGDF.drop(newGDF.index[(newGDF['region'] != 'MSK')], inplace = True)
    
    # set year column for separate axes plotting
    newGDF['p2a'] = pd.to_datetime(newGDF['p2a'])
    newGDF['year'] = newGDF.apply(lambda x: x.p2a.year, axis=1)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))

    year = 2018
    for ax in axes.flatten():
        # plot alcohol or drugs influence accidents for year in MSK region 
        newGDF[(newGDF.year == datetime.datetime(year,1,1).year) & (newGDF.p11 >= 3)].plot(ax=ax, 
            color='red', markersize=1)
        # add geo base map background
        cx.add_basemap(ax, crs=newGDF.crs.to_string(), source=cx.providers.Stamen.TonerLite,
            alpha=0.9)
        ax.set_title("MSK kraj (" + str(year) + ")")
        # hide x and y axis
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.setp(ax.spines.values(), color="White")
        year += 1
    
    if fig_location is not None:
        fig.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """Plot graph with location of accidents in region as clusters
    
    Args:
    gdf (geopandas.GeoDataFrame) -- input
    fig_location (str, optional) -- output file location (default None)
    show_figure (bool, optional) -- show figure to output (default False)
    """
    
    # copy of dataframe with reduced rows to rows with MSK region
    newGDF = gdf.copy()
    newGDF.drop(newGDF.index[(newGDF['region'] != 'MSK')], inplace = True)
    newGDF.drop(newGDF.index[(newGDF.p36 < 1) | (newGDF.p36 > 3)], inplace = True)

    gdf_c = newGDF.to_crs("EPSG:5514")
    gdf_c["area"] = gdf_c.area
    gdf_c = gdf_c.set_geometry(gdf_c.centroid).to_crs(epsg=3857)
    
    # create GeoDataFrame with clusters with KMeans
    coords = np.dstack([gdf_c.geometry.x, gdf_c.geometry.y]).reshape(-1, 2)
    db = sklearn.cluster.MiniBatchKMeans(n_clusters=50, n_init='auto').fit(coords)
    gdf3 = gdf_c.copy()
    gdf3["cluster"] = db.labels_
    
    # GeoDataFrame join points to cluster with count of accidents
    gdf4 = gdf3.dissolve(by="cluster", aggfunc={"p1": "count", "area": "sum"}).rename(columns=dict(p1="cnt"))
    # join GDF with all points with GDF containing counts based on cluster
    gdf5 = geopandas.sjoin(gdf3, gdf4, how='left')

    fig = plt.figure(figsize=(20, 18))
    ax = plt.gca()
    gdf5.plot(ax=ax, alpha=0.7, markersize=2, column="cnt")

    # add geo base map background
    cx.add_basemap(ax, crs=gdf_c.crs.to_string(), source=cx.providers.Stamen.TonerLite, 
        alpha=0.6)

    ax.set_title("Nehody v MSK kraji na sinicích 1., 2. a 3. třídy")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.setp(ax.spines.values(), color="White")

    # parameters of color bar
    mn = gdf5.cnt.min()
    mx = gdf5.cnt.max()
    norm = Normalize(vmin=mn, vmax=mx)
    n_cmap = cm.ScalarMappable(norm=norm, cmap="viridis")
    n_cmap.set_array([])

    #color bar setting
    cbar_ax = fig.add_axes([0.02, -0.04, 0.96, 0.02])
    cbar = plt.colorbar(n_cmap, ax=ax, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Počet nehod v úseku')

    if fig_location is not None:
        #color bar setting
        cbar_ax = fig.add_axes([0.02, -0.04, 0.96, 0.02])
        cbar = plt.colorbar(n_cmap, ax=ax, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Počet nehod v úseku')
        fig.savefig(fig_location, bbox_inches='tight')
    if show_figure:
        #color bar setting
        cbar = plt.colorbar(n_cmap, ax=ax, orientation='horizontal')
        cbar.set_label('Počet nehod v úseku')
        plt.show()

if __name__ == "__main__":
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)

