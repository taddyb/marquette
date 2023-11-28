"""It looks like there were some changes with merit basins. Keeping v1 here just in case"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

from marquette.merit._graph import (
    data_to_csv,
    get_edge_counts,
    segments_to_edges,
    Segment,
)

log = logging.getLogger(__name__)


def _plot_gdf(gdf: gpd.GeoDataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax)
    ax.set_title("Polyline Plot")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


def create_graph(cfg):
    flowline_file = cfg.save_paths.flow_lines
    polyline_gdf = gpd.read_file(flowline_file)

    # Convert multiple columns to int type
    for col in [
        "COMID",
        "NextDownID",
        "up1",
        "up2",
        "up3",
        "up4",
        "maxup",
        "order_",
    ]:
        polyline_gdf[col] = polyline_gdf[col].astype(int)

    crs = polyline_gdf.crs

    # Generate segments using list comprehension
    segments = [Segment(row, row.geometry, crs) for _, row in polyline_gdf.iterrows()]

    dx = cfg.dx  # Unit: Meters
    buffer = cfg.buffer * dx  # Unit: Meters
    sorted_segments = sorted(
        segments, key=lambda segment: segment.uparea, reverse=False
    )
    segment_das = {
        segment.id: segment.uparea for segment in segments
    }  # Simplified with dict comprehension
    edge_counts = get_edge_counts(sorted_segments, dx, buffer)
    edges_ = [
        edge
        for segment in tqdm(sorted_segments, desc="Processing segments")
        for edge in segments_to_edges(
            segment, edge_counts, segment_das
        )  # returns many edges
    ]
    edges = data_to_csv(edges_)
    edges.to_csv(cfg.csv.edges, index=False)

    return edges


def map_streamflow_to_river_graph(cfg: DictConfig, edges: pd.DataFrame) -> None:
    huc_to_merit_TM = pd.read_csv(cfg.save_paths.huc_to_merit_tm)
    merit_to_edge_TM = _create_TM(cfg, edges, huc_to_merit_TM)
    streamflow_predictions = pd.read_csv(cfg.save_paths.streamflow)


def _create_TM(cfg: DictConfig, edges, huc_to_merit_TM) -> pd.DataFrame:
    tm = Path(cfg.save_paths.merit_to_river_graph_tm)
    if tm.exists():
        return pd.read_csv(tm)
    else:
        merit_basins = huc_to_merit_TM.columns[huc_to_merit_TM.columns != 'HUC10'].values
        sorted_edges = edges.sort_values(by='merit_basin', ascending=False)
        river_graph_ids = sorted_edges["id"].values
        river_graph_ids.sort()
        df = pd.DataFrame(index=merit_basins, columns=river_graph_ids)
        df["Merit_Basins"] = merit_basins
        df = df.set_index("Merit_Basins")
        for idx, id in enumerate(tqdm(merit_basins, desc="creating TM")):
            merit_reaches = edges[edges["merit_basin"] == int(id)]
            total_area = merit_reaches.iloc[-1].uparea
            previous_upstream_area = 0
            for j, reach in merit_reaches.iterrows():
                data = np.zeros([merit_basins.shape[0]])
                data[idx] = (reach.uparea - previous_upstream_area) / total_area
                previous_upstream_area = reach.uparea
                df[reach.id] = data
        df.to_csv(tm)
        log.info(f"Wrote output to {tm}")
    return df
