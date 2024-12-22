import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.dataframe.io.io import from_pandas
from omegaconf import DictConfig
from tqdm import tqdm

from marquette.merit._connectivity_matrix import (create_gage_connectivity,
                                                  map_gages_to_zone,
                                                  new_zone_connectivity)
from marquette.merit._edge_calculations import (
    calculate_num_edges, create_segment, find_flowlines,
    many_segment_to_edge_partition, singular_segment_to_edge_partition,
    sort_xarray_dataarray)
from marquette.merit._map_lake_points import _map_lake_points
from marquette.merit._streamflow_conversion_functions import (
    calculate_huc10_flow_from_individual_files, calculate_merit_flow,
    separate_basins)
from marquette.merit._TM_calculations import (  # create_sparse_MERIT_FLOW_TM,
    create_HUC_MERIT_TM, create_MERIT_FLOW_TM, join_geospatial_data)

log = logging.getLogger(__name__)


def write_streamflow(cfg: DictConfig, edges: xr.Dataset) -> None:
    """
    Process and write streamflow data to a Zarr store.
    """
    streamflow_path = Path(cfg.create_streamflow.data_store)
    version = cfg.create_streamflow.version.lower()
    if streamflow_path.exists() is False:
        if version in ["dpl_v1", "dpl_v2", "dpl_v2.5", "dpl_v3-pre"]:
            """Expecting to read from individual files"""
            separate_basins(cfg)
            calculate_huc10_flow_from_individual_files(cfg)
        elif version == "dpl_v3":
            calculate_huc10_flow_from_individual_files(cfg)
        elif "merit" in version:
            calculate_merit_flow(cfg, edges)
        else:
            raise KeyError(f"streamflow version: {version}" "not supported")
    else:
        log.info("Streamflow data already exists")


def create_edges(cfg: DictConfig) -> xr.Dataset:
    try:
        dt = xr.open_datatree(filename_or_obj=f"{cfg.create_edges.edges}", engine="zarr")
    except FileNotFoundError:
        dt = xr.DataTree(name="root")
    try:
        edges = dt[str(cfg.zone)]
        log.info("Edge data already exists")
    except KeyError:
        flowline_file = find_flowlines(cfg)
        polyline_gdf = gpd.read_file(flowline_file)
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
        sorted_keys_array = np.array(sorted_keys)
        
        edges: xr.Dataset = xr.Dataset()
        edges.attrs['crs'] = merged_df["crs"].unique()[0]
        for var_name in merged_df.columns:
            if var_name != "crs":
                sorted_data = sort_xarray_dataarray(
                    merged_df[var_name].value_counts(),
                    sorted_keys_array,
                    merged_df["segment_sorting_index"].values,
                    merit_basins
                )
                edges[var_name] = sorted_data
        
        dt[str(cfg.zone)] = edges
        dt.to_zarr(
            store=cfg.create_edges.edges,
            mode='a', 
            consolidated=True
        )
    return edges


def create_N(cfg: DictConfig, edges: xr.Dataset) -> None:
    gage_coo_root = zarr.open_group(Path(cfg.create_N.gage_coo_indices), mode="a")
    zone_root = gage_coo_root.require_group(cfg.zone)
    if cfg.create_N.run_whole_zone:
        if "full_zone" in zone_root:
            log.info("Full zone already exists")
        else:
            full_zone_root = zone_root.require_group("full_zone")
            new_zone_connectivity(cfg, edges, full_zone_root)
        log.info("Full zone sparse Matrix created")
    else:
        zone_csv_path = Path(cfg.create_N.zone_obs_dataset)
        if zone_csv_path.exists():
            zone_csv = pd.read_csv(zone_csv_path)
        else:
            zone_csv = map_gages_to_zone(cfg, edges)
        if zone_csv is not False:
            create_gage_connectivity(cfg, edges, zone_root, zone_csv)
            log.info("All sparse gage matrices are created")


def create_TMs(cfg: DictConfig, edges: xr.Dataset) -> None:
    if "HUC" in cfg.create_TMs:
        huc_to_merit_path = Path(cfg.create_TMs.HUC.TM)
        if huc_to_merit_path.exists():
            log.info("HUC -> MERIT data already exists in zarr format")
        else:
            log.info("Creating HUC10 -> MERIT TM")
            overlayed_merit_basins = join_geospatial_data(cfg)
            create_HUC_MERIT_TM(cfg, edges, overlayed_merit_basins)
    merit_to_river_graph_path = Path(cfg.create_TMs.MERIT.TM)
    if merit_to_river_graph_path.exists():
        log.info("MERIT -> FLOWLINE data already exists in zarr format")
    else:
        log.info("Creating MERIT -> FLOWLINE TM")
        # if cfg.create_TMs.MERIT.save_sparse:
        #     create_sparse_MERIT_FLOW_TM(cfg, edges)
        # else:
        create_MERIT_FLOW_TM(cfg, edges)


def map_lake_points(cfg: DictConfig, edges: xr.Dataset) -> None:
    """Maps HydroLAKES pour points to the corresponding edge

    Parameters
    ----------
    cfg: DictConfig
        The configuration object
    edges: xr.Dataset
        The zarr group containing the edges
    """
    if "hylak_id" in edges:
        log.info("HydroLakes Intersection already exists in Zarr format")
    else:
        log.info("Mapping HydroLakes Pour Points to Edges")
        _map_lake_points(cfg, edges)
        

def run_extensions(cfg: DictConfig, edges: xr.Dataset) -> None:
    """
    The function for running post-processing data extensions

    :param cfg: Configuration object.
    :type cfg: DictConfig
    :return: None
    """
    edges = edges.compute()
    dt = xr.open_datatree(cfg.create_edges.edges, engine="zarr").compute()
    if "soils_data" in cfg.extensions:
        from marquette.merit.extensions import soils_data

        log.info("Adding soils information to your MERIT River Graph")
        if "ksat" in edges:
            log.info("soils information already exists in zarr format")
        else:
            soils_data(cfg, edges, dt)
    if "pet_forcing" in cfg.extensions:
        from marquette.merit.extensions import pet_forcing

        log.info("Adding PET forcing to your MERIT River Graph")
        if "pet" in edges:
            log.info("PET forcing already exists in zarr format")
        else:
            pet_forcing(cfg, edges, dt)
    if "temp_mean" in cfg.extensions:
        from marquette.merit.extensions import temp_forcing

        log.info("Adding temp_mean forcing to your MERIT River Graph")
        if "temp_mean" in edges:
            log.info("Temp_mean forcing already exists in zarr format")
        else:
            temp_forcing(cfg, edges, dt)
    if "global_dhbv_static_inputs" in cfg.extensions:
        from marquette.merit.extensions import global_dhbv_static_inputs

        log.info("Adding global dHBV static input data to your MERIT River Graph")
        if "aridity" in edges:
            log.info("global_dhbv_static_inputs already exists in zarr format")
        else:
            global_dhbv_static_inputs(cfg, edges, dt)

    if "incremental_drainage_area" in cfg.extensions:
        from marquette.merit.extensions import \
            calculate_incremental_drainage_area

        log.info("Adding edge/catchment area input data to your MERIT River Graph")
        if "incremental_drainage_area" in edges:
            log.info("incremental_drainage_area already exists in zarr format")
        else:
            calculate_incremental_drainage_area(cfg, edges, dt)

    if "q_prime_sum" in cfg.extensions:
        from marquette.merit.extensions import calculate_q_prime_summation

        log.info("Adding q_prime_sum to your MERIT River Graph")
        if "summed_q_prime" in edges:
            log.info("q_prime_sum already exists in zarr format")
        else:
            calculate_q_prime_summation(cfg, edges, dt)
            
    if "log_uparea" in cfg.extensions:
        from marquette.merit.extensions import log_uparea

        log.info("Adding log_uparea to your MERIT River Graph")
        if "log_uparea" in edges:
            log.info("log_uparea already exists in zarr format")
        else:
            log_uparea(cfg, edges, dt)
            

    if "upstream_basin_avg_mean_p" in cfg.extensions:
        from marquette.merit.extensions import calculate_mean_p_summation

        log.info("Adding q_prime_sum to your MERIT River Graph")
        if "upstream_basin_avg_mean_p" in edges:
            log.info("upstream_basin_avg_mean_p already exists in zarr format")
        else:
            calculate_mean_p_summation(cfg, edges, dt)
            
    # if "q_prime_sum_stats" in cfg.extensions:
    #     from marquette.merit.extensions import calculate_q_prime_sum_stats

    #     log.info("Adding q_prime_sum statistics to your MERIT River Graph")
    #     if "summed_q_prime_median" in edges:
    #         log.info("q_prime_sum statistics already exists in zarr format")
    #     else:
    #         calculate_q_prime_sum_stats(cfg, edges)
            
    if "lstm_stats" in cfg.extensions:
        from marquette.merit.extensions import format_lstm_forcings

        log.info("Adding lstm statistics from global LSTM to your MERIT River Graph")
        if "precip_comid" in edges:
            log.info("q_prime_sum statistics already exists in zarr format")
        else:
            format_lstm_forcings(cfg, edges, dt)
