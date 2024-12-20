import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask.dataframe.io.io import from_pandas
from omegaconf import DictConfig
from tqdm import tqdm

from marquette.merit_s3._connectivity_matrix import (create_gage_connectivity,
                                                  map_gages_to_zone,
                                                  new_zone_connectivity)
from marquette.merit_s3._edge_calculations import (
    calculate_num_edges, create_segment,
    many_segment_to_edge_partition, singular_segment_to_edge_partition
)

log = logging.getLogger(__name__)



def create_edges(cfg: DictConfig) -> xr.Dataset:
    try:
        root = xr.open_datatree(f"{cfg.create_edges.edges}", engine="zarr")
        edges = root[str(cfg.zone)]
        log.info("Edge data already exists on s3")
    except FileNotFoundError:
        log.info("Edge data does not exist. Creating connections")
        polyline_gdf = gpd.read_file(cfg.create_edges.flowlines)
        dx = cfg.create_edges.dx  # Unit: Meters
        buffer = cfg.create_edges.buffer * dx  # Unit: Meters
        for col in [
            "COMID",
            "NextDownID",
            "up1",
            "up2",
            "up3",
            "up4",
            "maxup",
            "order",
        ]:
            polyline_gdf[col] = polyline_gdf[col].astype(int)
        computed_series = polyline_gdf.apply(
            lambda df: create_segment(df, polyline_gdf.crs, dx, buffer), axis=1
        )
        segments_dict = computed_series.to_dict()
        segment_das = {
            segment["id"]: segment["uparea"] for segment in segments_dict.values()
        }
        sorted_keys = sorted(
            segments_dict, key=lambda key: segments_dict[key]["uparea"]
        )
        num_edges_dict = {
            _segment["id"]: calculate_num_edges(_segment["len"], dx, buffer)
            for _, _segment in tqdm(
                segments_dict.items(),
                desc="Processing Number of Edges",
                ncols=140,
                ascii=True,
            )
        }
        one_edge_segment = {
            seg_id: edge_info
            for seg_id, edge_info in tqdm(
                num_edges_dict.items(),
                desc="Filtering Segments == 1",
                ncols=140,
                ascii=True,
            )
            if edge_info[0] == 1
        }
        many_edge_segment = {
            seg_id: edge_info
            for seg_id, edge_info in tqdm(
                num_edges_dict.items(),
                desc="Filtering Segments > 1",
                ncols=140,
                ascii=True,
            )
            if edge_info[0] > 1
        }
        segments_with_more_than_one_edge = {}
        segments_with_one_edge = {}
        for i, segment in segments_dict.items():
            segment_id = segment["id"]
            segment["index"] = i

            if segment_id in many_edge_segment:
                segments_with_more_than_one_edge[segment_id] = segment
            elif segment_id in one_edge_segment:
                segments_with_one_edge[segment_id] = segment
            else:
                print(f"MISSING ID: {segment_id}")

        df_one = pd.DataFrame.from_dict(segments_with_one_edge, orient="index")
        df_many = pd.DataFrame.from_dict(
            segments_with_more_than_one_edge, orient="index"
        )
        ddf_one = from_pandas(df_one, npartitions=64)
        ddf_many = from_pandas(df_many, npartitions=64)

        meta = pd.DataFrame(
            {
                "id": pd.Series(dtype="str"),
                "merit_basin": pd.Series(dtype="int"),
                "segment_sorting_index": pd.Series(dtype="int"),
                "order": pd.Series(dtype="int"),
                "len": pd.Series(dtype="float"),
                "len_dir": pd.Series(dtype="float"),
                "ds": pd.Series(dtype="str"),
                "up": pd.Series(dtype="object"),  # List or array
                "up_merit": pd.Series(dtype="object"),  # List or array
                "slope": pd.Series(dtype="float"),
                "sinuosity": pd.Series(dtype="float"),
                "stream_drop": pd.Series(dtype="float"),
                "uparea": pd.Series(dtype="float"),
                "coords": pd.Series(dtype="str"),
                "crs": pd.Series(dtype="object"),  # CRS object
            }
        )

        edges_results_one = ddf_one.map_partitions(
            singular_segment_to_edge_partition,
            edge_info=one_edge_segment,
            num_edge_dict=num_edges_dict,
            segment_das=segment_das,
            meta=meta,
        )
        edges_results_many = ddf_many.map_partitions(
            many_segment_to_edge_partition,
            edge_info=many_edge_segment,
            num_edge_dict=num_edges_dict,
            segment_das=segment_das,
            meta=meta,
        )
        edges_results_one_df = edges_results_one.compute()
        edges_results_many_df = edges_results_many.compute()
        merged_df = pd.concat([edges_results_one_df, edges_results_many_df])
        for col in ["id", "ds", "up", "coords", "up_merit", "crs"]:
            merged_df[col] = merged_df[col].astype(dtype=str)
        for col in ["merit_basin", "segment_sorting_index", "order"]:
            merged_df[col] = merged_df[col].astype(dtype=np.int32)
        for col in ["len", "len_dir", "slope", "sinuosity", "stream_drop", "uparea"]:
            merged_df[col] = merged_df[col].astype(dtype=np.float32)

        idx = np.argsort(merged_df["uparea"])
        sorted_df = merged_df.iloc[idx]
        merit_basins = sorted_df["merit_basin"]
        
        edges = xr.Dataset(
            {var: (["comid"], sorted_df[var]) for var in sorted_df.columns if var != "crs"},
            coords={"comid": merit_basins}
        )
        edges.attrs['crs'] = sorted_df["crs"].unique()[0]
        dt = xr.DataTree(name="root")
        dt[str(cfg.zone)] = edges
        dt.to_zarr(
            store=cfg.create_edges.edges,
            mode='w', 
            consolidated=True
        )
    return edges


def create_N(cfg: DictConfig, edges: xr.Dataset) -> None:
    try:
        gage_coo_root = xr.open_datatree(cfg.create_N.gage_coo_indices, engine="zarr")
        zone_root = gage_coo_root[cfg.zone]
        log.info("Connectivity Matrix pulled from s3")
        if cfg.create_N.run_whole_zone:
            if "full_zone" not in zone_root:
                new_zone_connectivity(cfg, edges, zone_root)   
                log.info("Full zone sparse Matrix created")
        
    except FileNotFoundError:
        gage_coo_root = xr.DataTree(name="root")
        gage_coo_root[str(cfg.zone)] = xr.DataTree(name=str(cfg.zone))
        zone_root = xr.DataTree(name=str(cfg.zone))
        if cfg.create_N.run_whole_zone:
            new_zone_connectivity(cfg, edges, zone_root)   
            log.info("Full zone sparse Matrix created")
        zone_csv_path = Path(cfg.create_N.zone_obs_dataset)
        if zone_csv_path.exists():
            zone_csv = pd.read_csv(zone_csv_path)
        else:
            zone_csv = map_gages_to_zone(cfg, edges)
        if zone_csv is not False:
            create_gage_connectivity(cfg, edges, zone_root, zone_csv)
            log.info("All sparse gage matrices are created")
        gage_coo_root.to_zarr(
            store=cfg.cfg.create_N.gage_coo_indices,
            mode='w', 
            consolidated=True
        )
