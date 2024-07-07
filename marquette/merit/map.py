"""It looks like there were some changes with merit basins. Keeping v1 here just in case"""

import logging
import multiprocessing
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from tqdm import tqdm

from marquette.merit._graph import (Segment, data_to_csv, get_edge_counts,
                                    segments_to_edges)

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
        for segment in tqdm(
            sorted_segments,
            desc="Processing segments",
            ncols=140,
            ascii=True,
        )
        for edge in segments_to_edges(
            segment, edge_counts, segment_das
        )  # returns many edges
    ]
    edges = data_to_csv(edges_)
    edges.to_csv(cfg.csv.edges, index=False, compression="gzip")

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
    huc_to_merit_TM = pd.read_csv(cfg.save_paths.huc_to_merit_tm, compression="gzip")
    huc_10_list = huc_to_merit_TM["HUC10"].values
    hm_TM = csr_matrix(huc_to_merit_TM.drop("HUC10", axis=1).values)
    merit_to_river_graph_TM = _create_TM(cfg, edges, huc_to_merit_TM)
    if "Merit_Basins" in merit_to_river_graph_TM.columns:
        columns = merit_to_river_graph_TM.drop("Merit_Basins", axis=1).columns
        mrg_TM = csr_matrix(merit_to_river_graph_TM.drop("Merit_Basins", axis=1).values)
    else:
        columns = merit_to_river_graph_TM.columns
        mrg_TM = csr_matrix(merit_to_river_graph_TM.values)
    log.info("Reading streamflow predictions")
    streamflow_predictions = _read_flow(cfg, huc_10_list)
    streamflow_predictions["dates"] = pd.to_datetime(streamflow_predictions["dates"])
    grouped_predictions = streamflow_predictions.groupby(
        streamflow_predictions["dates"].dt.year
    )
    args_iter = (
        (cfg, group, hm_TM.copy(), mrg_TM.copy(), columns)
        for group in grouped_predictions
    )
    with multiprocessing.Pool(cfg.num_cores) as pool:
        log.info("Writing Process to disk")
        pool.map(_write_to_disk, args_iter)


def _write_to_disk(args):
    cfg, grouped_predictions, hm_TM, mrg_TM, columns = args
    save_path = Path(cfg.csv.mapped_streamflow_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    year = grouped_predictions[0]
    group = grouped_predictions[1]
    log.info(f"Writing data for year {year}")
    interpolated_flow = _interpolate(cfg, group, year)
    flow = csr_matrix(interpolated_flow.drop("dates", axis=1).sort_index(axis=1).values)
    mapped_flow_merit = flow.dot(hm_TM)
    mapped_flow_river_graph = mapped_flow_merit.dot(mrg_TM)
    mapped_flow_array = mapped_flow_river_graph.toarray()
    log.info(f"Writing {year} to disk")
    df = pd.DataFrame(
        mapped_flow_array,
        index=interpolated_flow["dates"],
        columns=columns,
        dtype="float32",
    )
    df.to_csv(
        save_path / f"{year}_{cfg.basin}_mapped_streamflow.csv.gz",
        compression="gzip",
    )
    log.info(f"Done with {year}")


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
        return pd.read_csv(tm, compression="gzip")
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
        for idx, id in enumerate(
            tqdm(
                merit_basins,
                desc="creating TM",
                ncols=140,
                ascii=True,
            )
        ):
            merit_reaches = edges[edges["merit_basin"] == int(id)]
            if merit_reaches.shape[0] == 0:
                log.error(f"Missing row for {id}")
            total_length = sum(
                [merit_reaches.iloc[i].len for i in range(merit_reaches.shape[0])]
            )
            for j, reach in merit_reaches.iterrows():
                data = np.zeros([merit_basins.shape[0]])
                data[idx] = reach.len / total_length
                df[reach.id] = data
        # df = df.reset_index()
        log.info("Writing TM")
        df.to_csv(tm, compression="gzip", index=False)
        return df


def _read_flow(cfg, huc_10_list: np.ndarray) -> pd.DataFrame:
    streamflow_output = Path(cfg.save_paths.streamflow_output)
    if streamflow_output.exists():
        df = pd.read_csv(streamflow_output, compression="gzip")
    else:
        df = _create_streamflow(cfg, huc_10_list)
        streamflow_output.parent.mkdir(exist_ok=True)
        df.to_csv(streamflow_output, index=False, compression="gzip")
    return df


def _interpolate(cfg: DictConfig, df: pd.DataFrame, year: int) -> pd.DataFrame:
    streamflow_interpolated = Path(cfg.save_paths.streamflow_interpolated.format(year))
    if streamflow_interpolated.exists():
        return pd.read_csv(streamflow_interpolated, compression="gzip")
    else:
        df["dates"] = pd.to_datetime(df["dates"])
        df.set_index("dates", inplace=True)
        df = df.resample("H").asfreq()
        df = df.interpolate(method="linear")
        df_reset = df.reset_index()
        df_reset.to_csv(streamflow_interpolated, index=False, compression="gzip")
        return df_reset


def _create_streamflow(cfg: DictConfig, huc_10_list: np.ndarray) -> pd.DataFrame:
    """
    extracting streamflow from many files based on HUC IDs
    :param cfg:
    :return:
    """

    def extract_numbers(filename):
        """
        Extracts the first set of numbers from the filename and returns them as an integer.
        Assumes the filename contains numbers in the format 'xxxx_yyyy'.
        """
        import re

        match = re.search(r"(\d+)_(\d+)", str(filename))
        if match:
            return tuple(map(int, match.groups()))
        return (0, 0)  # Default value if no numbers are found

    attrs_df = pd.read_csv(cfg.save_paths.attributes)
    huc10_ids = attrs_df["gage_ID"].values.astype("str")
    bins_size = 1000
    bins = [huc10_ids[i : i + bins_size] for i in range(0, len(huc10_ids), bins_size)]
    basin_hucs = huc_10_list
    basin_indexes = _sort_into_bins(basin_hucs, bins)
    streamflow_data = []
    columns = []
    folder = Path(cfg.save_paths.streamflow_files)
    file_paths = [file for file in folder.glob("*") if file.is_file()]
    file_paths.sort(key=extract_numbers)
    iterable = basin_indexes.keys()
    pbar = tqdm(
        iterable,
        ncols=140,
        ascii=True,
    )
    for i, key in enumerate(pbar):
        pbar.set_description("Processing Qr files")
        values = basin_indexes[key]
        if values:
            file = file_paths[i]
            df = pd.read_csv(file, dtype=np.float32, header=None)
            for val in values:
                id = list(val.keys())[0]
                columns.append(id)
                row = attrs_df[attrs_df["gage_ID"] == id]
                try:
                    attr_idx = row.index[0]
                    try:
                        row_idx = attr_idx - (
                            key * 1000
                        )  # taking only the back three numbers
                        _streamflow = df.iloc[row_idx].values
                    except IndexError:
                        #  TODO SOLVE THIS for UPPER COLORADO
                        log.error("here")
                    if cfg.units.lower() == "mm/day":
                        # converting from mm/day to m3/s
                        area = row["area"].values[0]
                        _streamflow = _streamflow * area * 1000 / 86400
                    streamflow_data.append(_streamflow)
                except IndexError:
                    #  TODO Get the HUC values that are missing. Adding temporary fixes
                    #  Using the previous HUC's prediction
                    log.info(
                        f"HUC10 {id} is missing from the attributes file. Replacing with previous HUC prediction"
                    )
                    streamflow_data.append(_streamflow)

    output = np.column_stack(streamflow_data)
    date_range = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="D")
    output_df = pd.DataFrame(output, columns=columns)
    output_df["dates"] = date_range
    return output_df


def _sort_into_bins(ids: np.ndarray, bins: List[np.ndarray]):
    """
    :param ids: a list of HUC10 IDS
    :return:
    """

    def find_list_of_str(target: int, sorted_lists: List[np.ndarray]):
        left, right = 0, len(sorted_lists) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_list = sorted_lists[mid]
            if mid_list.size > 0:
                first_element = int(mid_list[0])
                last_element = int(mid_list[-1])
                if target < first_element:
                    right = mid - 1
                elif target > last_element:
                    left = mid + 1
                else:
                    return mid
            else:
                left += 1
        return None

    keys = list(range(0, 16, 1))
    grouped_values = {key: [] for key in keys}
    for idx, value in enumerate(ids):
        id = int(ids[idx])
        _key = find_list_of_str(id, bins)
        grouped_values[_key].append({id: idx})

    return grouped_values
