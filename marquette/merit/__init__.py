import logging

import geopandas as gpd

log = logging.getLogger(__name__)


def _plot_gdf(gdf: gpd.GeoDataFrame) -> None:
    """
    A function to find the correct flowline of all MERIT basins using glob

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The geodataframe you want to plot

    Returns
    -------
    None

    Raises
    ------
    None
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax)
    ax.set_title("Polyline Plot")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()
