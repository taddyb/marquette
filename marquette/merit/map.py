"""It looks like there were some changes with merit basins. Keeping v1 here just in case"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from scipy.sparse import csr_matrix
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
    """
    Ways to check if this is right:
      - sort each column and index row
      - subtract the HUC cols from flow from the HUC index of hm_TM
        (f_cols.astype("int") - huc_to_merit_TM["HUC10"]).sum()
      - subtract the MERIT cols from hm_TM from the MERIT index of mrg_TM
        (hm_cols.astype("int") - merit_to_river_graph_TM["Merit_Basins"]).sum()
    :param cfg:
    :param edges:
    :return:
    """
    huc_to_merit_TM = pd.read_csv(cfg.save_paths.huc_to_merit_tm)
    hm_TM = csr_matrix(huc_to_merit_TM.drop("HUC10", axis=1).values)
    merit_to_river_graph_TM = _create_TM(cfg, edges, huc_to_merit_TM)
    try:
        mrg_TM = csr_matrix(merit_to_river_graph_TM.drop("Merit_Basins", axis=1).values)
    except KeyError:
        mrg_TM = csr_matrix(merit_to_river_graph_TM.values)
    streamflow_predictions_interpolated = _read_interpolated(cfg)
    streamflow_predictions_interpolated['dates'] = pd.to_datetime(streamflow_predictions_interpolated['dates'])
    grouped_predictions = streamflow_predictions_interpolated.groupby(streamflow_predictions_interpolated['dates'].dt.year)
    save_path = Path(cfg.csv.mapped_streamflow_dir)
    for year, group in tqdm(grouped_predictions, desc="writing each year's data"):
        flow = csr_matrix(
            group.drop("dates", axis=1)
            .sort_index(axis=1)
            .values
        )
        mapped_flow_merit = flow.dot(hm_TM)
        mapped_flow_river_graph = mapped_flow_merit.dot(mrg_TM)
        mapped_flow_array = mapped_flow_river_graph.toarray()
        df = pd.DataFrame(
            mapped_flow_array,
            index=group["dates"],
            columns=merit_to_river_graph_TM.drop("Merit_Basins", axis=1).columns,
            dtype="float32"
        )
        df.to_csv(save_path / f"{year}_{cfg.basin}_mapped_streamflow.csv.gz", compression='gzip')
    log.info("Wrote adjusted flow to disk")


def _create_TM(
    cfg: DictConfig, edges: pd.DataFrame, huc_to_merit_TM: pd.DataFrame
) -> pd.DataFrame:
    """
    Creating a TM that maps MERIT basins to their reaches. Flow predictions are distributed
    based on reach length/ total merit reach length
    :param cfg:
    :param edges:
    :param huc_to_merit_TM:
    :return:
    """
    tm = Path(cfg.save_paths.merit_to_river_graph_tm)
    if tm.exists():
        return pd.read_csv(tm)
    else:
        merit_basins = huc_to_merit_TM.columns[
            huc_to_merit_TM.columns != "HUC10"
        ].values
        sorted_edges = edges.sort_values(by="merit_basin", ascending=False)
        river_graph_ids = sorted_edges["id"].values
        river_graph_ids.sort()
        df = pd.DataFrame(index=merit_basins, columns=river_graph_ids)
        df["Merit_Basins"] = merit_basins
        df = df.set_index("Merit_Basins")
        for idx, id in enumerate(tqdm(merit_basins, desc="creating TM")):
            merit_reaches = edges[edges["merit_basin"] == int(id)]
            total_length = sum(
                [merit_reaches.iloc[i].len for i in range(merit_reaches.shape[0])]
            )
            for j, reach in merit_reaches.iterrows():
                data = np.zeros([merit_basins.shape[0]])
                data[idx] = reach.len / total_length
                df[reach.id] = data
        df.to_csv(tm)
        log.info(f"Wrote output to {tm}")
        return df


def _read_interpolated(cfg) -> pd.DataFrame:
    streamflow_interpolated = Path(cfg.save_paths.streamflow_interpolated)
    if streamflow_interpolated.exists():
        return pd.read_csv(streamflow_interpolated)
    else:
        streamflow = Path(cfg.save_paths.streamflow)
        df = pd.read_csv(streamflow)
        if cfg.units.lower() == "mm/day":
            df_temp = df.copy()
            _attr = pd.read_csv(cfg.save_paths.attributes)
            attr = _attr[["gage_ID", "area"]]
            for idx, row in enumerate(tqdm(attr, desc="converting to m3/s")):
                flow = df[row["gage_ID"]].values
                area = row["area"]
                m3s = flow * area * 1000 / 86400
                df_temp[row["gage_ID"]] = m3s
            df = df_temp.copy()
            df.to_csv(streamflow.parent / f"{cfg.name}_streamflow_m3s.csv")
        df = df.resample("H").asfreq()
        df = df.interpolate(method="linear")
        df_reset = df.reset_index()
        df_reset.to_csv(streamflow_interpolated)
    return df_reset


def _apply_tau(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    DATE_FORMAT = "%m/%d/%Y %H:%M"
    try:
        df["dates"] = pd.to_datetime(df["dates"], format=DATE_FORMAT)
    except KeyError as e:
        log.info("no date column. Adding one")
        date_range = pd.date_range(
            start="01/01/1980 00:00", end="12/31/2019 23:00", freq="H"
        )
        date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]
        df["dates"] = date_range
        df["dates"] = pd.to_datetime(df["dates"], format=DATE_FORMAT)
    df["dates"] = df["dates"] - pd.Timedelta(hours=cfg.tau)
    return df
