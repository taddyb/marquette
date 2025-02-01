from collections import defaultdict
import logging
from pathlib import Path

import ast
import binsparse
import cupy as cp
from cupyx.scipy import sparse as cp_sparse
import dask.dataframe as dd
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
import zarr
from omegaconf import DictConfig
from tqdm import tqdm, trange
import xarray as xr

log = logging.getLogger(__name__)


def soils_data(cfg: DictConfig, edges: zarr.Group) -> None:
    flowline_file = (
        Path(cfg.data_path)
        / f"raw/routing_soil_properties/riv_pfaf_{cfg.zone}_buff_split_soil_properties.shp"
    )
    polyline_gdf = gpd.read_file(flowline_file)
    gdf = pd.DataFrame(polyline_gdf.drop(columns="geometry"))
    df = pl.from_pandas(gdf)
    df = df.with_columns(pl.col("COMID").cast(pl.Int64))  # convert COMID to int64
    attributes = [
        "Ks_05_M_25",
        "N_05_M_250",
        "SF_05_M_25",
        "AF_05_M_25",
        "OM_05_M_25",
        "Cl_05_Mn",
        "Sd_05_Mn",
        "St_05_Mn",
    ]
    names = [
        "ksat",
        "N",
        "sat-field",
        "alpha",
        "ormc",
        "clay_mean_05",
        "sand_mean_05",
        "silt_mean_05",
    ]
    df_filled = df.with_columns(
        pl.when(pl.col(["Ks_05_M_25", "SF_05_M_25"]).is_null())
        .then(pl.col(["Ks_05_M_25", "SF_05_M_25"]).fill_null(strategy="max"))
        .otherwise(pl.col(["Ks_05_M_25", "SF_05_M_25"]))
    )
    df_filled = df_filled.with_columns(
        pl.when(pl.col(["N_05_M_250", "AF_05_M_25"]).is_null())
        .then(pl.col(["N_05_M_250", "AF_05_M_25"]).fill_null(strategy="min"))
        .otherwise(pl.col(["N_05_M_250", "AF_05_M_25"]))
    )
    df_filled = df_filled.with_columns(
        pl.when(pl.col(["OM_05_M_25", "Cl_05_Mn", "Sd_05_Mn", "St_05_Mn"]).is_null())
        .then(
            pl.col(["OM_05_M_25", "Cl_05_Mn", "Sd_05_Mn", "St_05_Mn"]).fill_null(
                strategy="forward"
            )
        )
        .otherwise(pl.col(["OM_05_M_25", "Cl_05_Mn", "Sd_05_Mn", "St_05_Mn"]))
    )
    graph_cols = ["COMID", "up1", "NextDownID"]
    df_cols = graph_cols + attributes
    _df = df_filled.select(pl.col(df_cols))
    edges_df = pl.DataFrame({"COMID": edges.merit_basin[:]})
    joined_df = edges_df.join(_df, on="COMID", how="left", join_nulls=True)
    for i in range(len(names)):
        edges.array(
            name=names[i],
            data=joined_df.select(pl.col(attributes[i])).to_numpy().squeeze(),
        )


def pet_forcing(cfg: DictConfig, edges: zarr.Group) -> None:
    pet_file_path = Path(f"/projects/mhpi/data/global/zarr_sub_zone/{cfg.zone}")
    num_timesteps = pd.date_range(
        start=cfg.create_streamflow.start_date,
        end=cfg.create_streamflow.end_date,
        freq="d",
    ).shape[0]
    if pet_file_path.exists() is False:
        raise FileNotFoundError("PET forcing data not found")
    edge_merit_basins: np.ndarray = edges.merit_basin[:]
    pet_edge_data = []
    pet_comid_data = []
    mapping = np.empty_like(edge_merit_basins, dtype=int)
    pet_data = zarr.open_group(pet_file_path, mode="r")
    for key in pet_data.keys():
        pet_zone_data =pet_data[key]
        comids = pet_zone_data.COMID[:]
        pet = pet_zone_data.PET[:]
        pet_comid_data.append(comids)
        pet_edge_data.append(pet)
    pet_comid_arr = np.concatenate(pet_comid_data)
    pet_arr = np.concatenate(pet_edge_data)

    if pet_arr.shape[0] != len(np.unique(edge_merit_basins)):
        raise ValueError(
            "PET forcing data is not consistent. Check the number of comids in the data and the edge_merit_basins array."
        )
    if pet_arr.shape[1] != num_timesteps:
        raise ValueError(
            "PET forcing data is not consistent. Check the number of timesteps in the data and the num_timesteps variable."
        )

    for i, id in enumerate(tqdm(pet_comid_arr, desc="\rProcessing PET data")):
        idx = np.where(edge_merit_basins == id)[0]
        mapping[idx] = i
    mapped_attr = pet_arr[mapping]
    edges.array(name="pet", data=mapped_attr)


def temp_forcing(cfg: DictConfig, edges: zarr.Group) -> None:
    temp_file_path = Path(f"/projects/mhpi/data/global/zarr_sub_zone/{cfg.zone}")
    num_timesteps = pd.date_range(
        start=cfg.create_streamflow.start_date,
        end=cfg.create_streamflow.end_date,
        freq="d",
    ).shape[0]
    if temp_file_path.exists() is False:
        raise FileNotFoundError("Temp forcing data not found")
    edge_merit_basins: np.ndarray = edges.merit_basin[:]
    temp_edge_data = []
    temp_comid_data = []
    mapping = np.empty_like(edge_merit_basins, dtype=int)
    temp_data = zarr.open_group(temp_file_path, mode="r")
    for key in temp_data.keys():
        temp_zone_data =temp_data[key]
        comids = temp_zone_data.COMID[:]
        temp_mean = temp_zone_data.Temp[:]
        temp_comid_data.append(comids)
        temp_edge_data.append(temp_mean)
    temp_comid_arr = np.concatenate(temp_comid_data)
    temp_arr = np.concatenate(temp_edge_data)

    if temp_arr.shape[0] != len(np.unique(edge_merit_basins)):
        raise ValueError(
            "Temp forcing data is not consistent. Check the number of comids in the data and the edge_merit_basins array."
        )
    if temp_arr.shape[1] != num_timesteps:
        raise ValueError(
            "Temp forcing data is not consistent. Check the number of timesteps in the data and the num_timesteps variable."
        )

    for i, id in enumerate(tqdm(temp_comid_arr, desc="\rProcessing temp data")):
        idx = np.where(edge_merit_basins == id)[0]
        mapping[idx] = i
    mapped_attr = temp_arr[mapping]
    edges.array(name="temp_mean", data=mapped_attr)


def global_dhbv_static_inputs(cfg: DictConfig, edges: zarr.Group) -> None:
    """
    Pulling Data from the global_dhbv_static_inputs data and storing it in a zarr store
    All attrs are as follows:
    attributeLst = ['area','ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
       'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
       'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
       'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
       'meanelevation', 'meanslope', 'permafrost', 'permeability',
       'seasonality_P', 'seasonality_PET', 'snow_fraction',
       'snowfall_fraction']
    """
    file_path = Path(f"/projects/mhpi/data/global/zarr_sub_zone/{cfg.zone}")
    if file_path.exists() is False:
        raise FileNotFoundError("global_dhbv_static_inputs data not found")
    edge_merit_basins: np.ndarray = edges.merit_basin[:]
    comid_data = []
    aridity_data = []
    porosity_data = []
    mean_p_data = []
    mean_elevation_data = []
    glaciers_data = []
    ndvi_data = []
    meanTa_data = []
    seasonality_P_data = []
    permeability_data = []

    mapping = np.empty_like(edge_merit_basins, dtype=int)
    root = zarr.open_group(file_path, mode="r")
    for key in root.keys():
        pet_zone_data = root[key]
        comids = pet_zone_data.COMID[:]
        aridity = pet_zone_data["attrs"]["aridity"][:]
        porosity = pet_zone_data["attrs"]["Porosity"][:]
        mean_p = pet_zone_data["attrs"]["meanP"][:]
        mean_elevation = pet_zone_data["attrs"]["meanelevation"][:]
        glaciers = pet_zone_data["attrs"]["glaciers"][:]
        ndvi = pet_zone_data["attrs"]["NDVI"][:]
        meanTa = pet_zone_data["attrs"]["meanTa"][:]
        seasonality_P = pet_zone_data["attrs"]["seasonality_P"][:]
        permeability = pet_zone_data["attrs"]["permeability"][:]

        comid_data.append(comids)
        aridity_data.append(aridity)
        porosity_data.append(porosity)
        mean_p_data.append(mean_p)
        mean_elevation_data.append(mean_elevation)
        glaciers_data.append(glaciers)
        ndvi_data.append(ndvi)
        meanTa_data.append(meanTa)
        seasonality_P_data.append(seasonality_P)
        permeability_data.append(permeability)

    comid_arr = np.concatenate(comid_data)
    aridity_arr = np.concatenate(aridity_data)
    porosity_arr = np.concatenate(porosity_data)
    mean_p_arr = np.concatenate(mean_p_data)
    mean_elevation_arr = np.concatenate(mean_elevation_data)
    glacier_arr = np.concatenate(glaciers_data)
    ndvi_arr = np.concatenate(ndvi_data)
    meanTa_arr = np.concatenate(meanTa_data)
    seasonality_P_arr = np.concatenate(seasonality_P_data)
    permeability_arr = np.concatenate(permeability_data)

    if comid_arr.shape[0] != len(np.unique(edge_merit_basins)):
        raise ValueError(
            "data is not consistent. Check the number of comids in the data and the edge_merit_basins array."
        )

    for i, id in enumerate(tqdm(comid_arr, desc="\rProcessing data")):
        idx = np.where(edge_merit_basins == id)[0]
        mapping[idx] = i
    edges.array(name="aridity", data=aridity_arr[mapping])
    edges.array(name="porosity", data=porosity_arr[mapping])
    edges.array(name="mean_p", data=mean_p_arr[mapping])
    edges.array(name="mean_elevation", data=mean_elevation_arr[mapping])
    edges.array(name="glacier", data=glacier_arr[mapping])
    edges.array(name="NDVI", data=ndvi_arr[mapping])
    edges.array(name="meanTa", data=meanTa_arr[mapping])
    edges.array(name="seasonality_P", data=seasonality_P_arr[mapping])
    edges.array(name="permeability", data=permeability_arr[mapping])


def calculate_incremental_drainage_area(cfg: DictConfig, edges: zarr.Group) -> None:
    """
    Runs a Polars query to calculate the incremental drainage area for each edge in the MERIT dataset
    """
    basin_file = (
        Path(cfg.data_path)
        / f"raw/basins/cat_pfaf_{cfg.zone}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
    )
    if basin_file.exists() is False:
        raise FileNotFoundError("Basin file not found")
    gdf = gpd.read_file(basin_file)
    _df = pd.DataFrame(gdf.drop(columns="geometry"))
    df = pl.from_pandas(_df)
    edges_df = pl.DataFrame(
        {
            "COMID": edges.merit_basin[:],
            "id": edges.id[:],
            "order": np.arange(edges.id.shape[0]),
        }
    )

    result = (
        df.lazy()
        .join(other=edges_df.lazy(), left_on="COMID", right_on="COMID", how="left")
        .group_by(
            by="COMID",
        )
        .agg(
            [
                pl.map_groups(
                    exprs=["unitarea", pl.first("unitarea")],
                    function=lambda list_of_series: list_of_series[1]
                    / list_of_series[0].shape[0],
                ).alias("incremental_drainage_area")
            ]
        )
        .join(other=edges_df.lazy(), left_on="COMID", right_on="COMID", how="left")
        .sort(by="order")
        .collect()
    )
    edges.array(
        name="incremental_drainage_area",
        data=result.select(pl.col("incremental_drainage_area")).to_numpy().squeeze(),
    )


def calculate_q_prime_summation(cfg: DictConfig, edges: zarr.Group) -> None:
    """Creates Q` summed data for all edges in a given MERIT zone

    Parameters:
    ----------
    cfg: DictConfig
        The configuration object.
    edges: zarr.Group
        The edges group in the MERIT zone
    """
    n = 10  # number of splits (used for reducing memory load)
    cp.cuda.runtime.setDevice(7)  # manually setting the device to 2

    streamflow_group = Path(
        f"/projects/mhpi/data/MERIT/streamflow/zarr/{cfg.create_streamflow.version}/{cfg.zone}"
    )
    if streamflow_group.exists() is False:
        raise FileNotFoundError("streamflow_group data not found")
    streamflow_zarr: zarr.Group = zarr.open_group(streamflow_group, mode="r")
    
    mapping_matrix = cfg.create_TMs.MERIT.TM
    zarr_group = zarr.open_group(
        Path(mapping_matrix),
        mode="r",
    )
    sparse_zone = binsparse.read(zarr_group["TM"])
    edge_mapped_streamflow = (streamflow_zarr.streamflow[:] @ sparse_zone).astype(np.float32)

    zone_uparea = edges.uparea[:]
    coo = zarr.open_group(cfg.create_N.gage_coo_indices)[str(cfg.zone)].full_zone
    sorted_indices = np.argsort(zone_uparea[coo.pairs[:, 0].astype(int)])
    sorted_pairs = coo.pairs[sorted_indices]
    
    non_nan_pairs = sorted_pairs[~np.isnan(sorted_pairs[:, 1])].astype(np.int32)
    df = pd.DataFrame(data={"source": non_nan_pairs[:, 1], "target": non_nan_pairs[:, 0]})

    q_prime_matrix = np.eye(edge_mapped_streamflow.shape[1], edge_mapped_streamflow.shape[1], dtype=np.float32)
    
    G = nx.from_pandas_edgelist(
        df=df,
        create_using=nx.DiGraph(),
    )
    
    all_descendants = defaultdict(set)
    processed = np.zeros(np.max(non_nan_pairs) + 1, dtype=bool)

    for idx in tqdm(non_nan_pairs[:, 1], desc="processing connections from pairs"):
        if processed[idx]:
            continue
        descendants = list(nx.dfs_preorder_nodes(G, source=idx))
        for i in range(len(descendants)):
            _idx = descendants[i]
            all_descendants[_idx].add(_idx)
            all_descendants[_idx].update(descendants[:i])
            processed[_idx] = True
            
    rows = []
    cols = []
    for idx, descendents in tqdm(all_descendants.items(), desc="creating sparse q_prime matrix"):
        rows.extend([idx] * len(descendents))
        cols.extend(list(descendents))

    print("Writing sparse matrix")
    rows = cp.array(rows)
    cols = cp.array(cols)
    sparse_q_prime_matrix = cp_sparse.csr_matrix(
        (cp.ones(len(rows)), (rows, cols)), 
        shape=q_prime_matrix.shape,
        ).T
    # for idx, descendents in tqdm(all_descendants.items(), desc="creating q_prime matrix"):
    #     rows = np.array(idx)
    #     cols = np.array(descendents)
    #     q_prime_matrix[rows, cols] = 1 
    
    # print("Writing q_prime_matrix to sparse")
    # sparse_q_prime_matrix = sparse.csr_matrix(q_prime_matrix.T)
    # q_prime_matrix = cp.array(q_prime_matrix.T, dtype = cp.int8)\
    print("Performing matrix multiplication")
    # q_prime_np = cp.asnumpy(cp.array(edge_mapped_streamflow, dtype=cp.float32) @ sparse_q_prime_matrix)
    base_size = edge_mapped_streamflow.shape[0] // n
    q_prime_np = np.zeros_like(edge_mapped_streamflow)
    
    for i in trange(n):
        start = i * base_size
        end = start + base_size if i < n-1 else edge_mapped_streamflow.shape[0]
        q_prime_np[start:end] = cp.asnumpy(
            cp.array(edge_mapped_streamflow[start:end], dtype=cp.float32) @ sparse_q_prime_matrix
        )

    print("Saving GPU Memory to CPU; freeing GPU Memory")
    edges.array(
        name="summed_q_prime_v2",
        data=q_prime_np,
        dtype=np.float32
    )
    del sparse_q_prime_matrix
    cp.get_default_memory_pool().free_all_blocks()
    
def map_last_edge_to_comid(arr):
    unique_elements = np.unique(arr)
    mapping = {val: np.where(arr == val)[0][-1] for val in unique_elements}
    last_indices = np.array([mapping[val] for val in unique_elements])

    return unique_elements, last_indices
    
    
def calculate_mean_p_summation(cfg: DictConfig, edges: zarr.Group) -> None:
    """Creates Q` summed data for all edges in a given MERIT zone

    Parameters:
    ----------
    cfg: DictConfig
        The configuration object.
    edges: zarr.Group
        The edges group in the MERIT zone
    """
    cp.cuda.runtime.setDevice(6)  # manually setting the device to 2

    edge_comids = edges.merit_basin[:]
    edge_comids_cp = cp.array(edges.merit_basin[:])
    ordered_merit_basin, indices = map_last_edge_to_comid(edge_comids)
    ordered_merit_basin_cp = cp.array(ordered_merit_basin)
    indices_cp = cp.array(indices)
    mean_p_data = cp.array(edges.mean_p[:])  # type: ignore  # UNIT: mm/year

    # Generating a networkx DiGraph object
    df_path = Path(f"{cfg.create_edges.edges}").parent / f"{cfg.zone}_graph_df.csv"
    if df_path.exists():
        df = pd.read_csv(df_path)
    else:
        raise FileNotFoundError("edges graph data not found. Have you calculated summed_q_prime yet?")
    G = nx.from_pandas_edgelist(
        df=df,
        create_using=nx.DiGraph(),
    )
    mean_p_sum = cp.zeros([indices.shape[0]])
    idx_counts = cp.zeros([indices.shape[0]])

    for j, index in enumerate(
        tqdm(
            indices,
            desc="calculating mean p sum data",
            ascii=True,
            ncols=140,
        )
    ):
        try:
            graph = nx.descendants(G, source=index, backend="cugraph")
            graph.add(index)  # Adding the idx to ensure it's counted
            downstream_idx = np.array(list(graph))  # type: ignore
            
            unique_merit_basin = cp.unique(edge_comids_cp[downstream_idx])  # type: ignore
            positions = cp.searchsorted(ordered_merit_basin_cp, unique_merit_basin)
            # downstream_comid_idx = indices_cp[positions]
            
            mean_p_sum[positions] += mean_p_data[index]  # type: ignore
            idx_counts[positions] += 1
        except nx.exception.NetworkXError:
            # This means there is no downstream connectivity from this basin. It's one-node graph            
            mean_p_sum[j] = mean_p_data[index]
            idx_counts[j] += 1

    print("Saving GPU Memory to CPU; freeing GPU Memory")
    upstream_basin_avg_mean_p = mean_p_sum / idx_counts
    upstream_basin_avg_mean_p_np = cp.asnumpy(upstream_basin_avg_mean_p)
    del upstream_basin_avg_mean_p
    del mean_p_sum
    del idx_counts
    cp.get_default_memory_pool().free_all_blocks()

    edges.array(
        name="upstream_basin_avg_mean_p",
        data=upstream_basin_avg_mean_p_np,
    )
    
    
def calculate_q_prime_sum_stats(cfg: DictConfig, edges: zarr.Group) -> None:
    """Creates Q` summed data for all edges in a given MERIT zone

    Parameters:
    ----------
    cfg: DictConfig
        The configuration object.
    edges: zarr.Group
        The edges group in the MERIT zone
    """
    try:
        summed_q_prime: np.ndarray = edges.summed_q_prime[:]
    except AttributeError:
        raise AttributeError("summed_q_prime data not found")
    edges.array(
        name="summed_q_prime_median",
        data=np.median(summed_q_prime, axis=0),
    )    
    edges.array(
        name="summed_q_prime_std",
        data=np.std(summed_q_prime, axis=0),
    )  
    edges.array(
        name="summed_q_prime_p90",
        data=np.percentile(summed_q_prime, 90, axis=0),
    )  
    edges.array(
        name="summed_q_prime_p10",
        data=np.percentile(summed_q_prime, 10, axis=0),
    )         
    
def format_lstm_forcings(cfg: DictConfig, edges: zarr.Group) -> None:
    forcings_store = zarr.open(Path("/projects/mhpi/data/global/zarr_sub_zone") / f"{cfg.zone}")

    edge_comids = np.unique(edges.merit_basin[:])  # already sorted
    log.info(msg="Reading Zarr Store")
    zone_keys = [
        key for key in forcings_store.keys() if str(cfg.zone) in key
    ]
    zone_comids = []
    zone_precip = []
    zone_pet = []
    # zone_temp = []
    zone_ndvi = []
    zone_aridity = []
    for key in zone_keys:
        zone_comids.append(forcings_store[key].COMID[:])
        zone_precip.append(forcings_store[key].P[:])
        zone_pet.append(forcings_store[key].PET[:])
        # zone_temp.append(streamflow_predictions_root[key].Temp[:])
        zone_ndvi.append(forcings_store[key]["attrs"]["NDVI"])
        zone_aridity.append(forcings_store[key]["attrs"]["aridity"])

    streamflow_comids = np.concatenate(zone_comids).astype(int)
    file_precip = np.transpose(np.concatenate(zone_precip))
    file_pet = np.transpose(np.concatenate(zone_pet))
    # file_temp = np.transpose(np.concatenate(zone_temp))
    file_ndvi = np.concatenate(zone_ndvi)
    file_aridity = np.concatenate(zone_aridity)
    del zone_comids
    del zone_precip
    del zone_pet
    # del zone_temp
    del zone_ndvi
    del zone_aridity
    
    log.info("Mapping to zone COMIDs")
    precip_full_zone = np.zeros((file_precip.shape[0], edge_comids.shape[0]))
    pet_full_zone = np.zeros((file_precip.shape[0], edge_comids.shape[0]))
    ndvi_full_zone = np.zeros((edge_comids.shape[0]))
    aridity_full_zone = np.zeros((edge_comids.shape[0]))
    
    
    indices = np.searchsorted(edge_comids, streamflow_comids)
    precip_full_zone[:, indices] = file_precip
    pet_full_zone[:, indices] = file_pet
    ndvi_full_zone[indices] = file_ndvi
    aridity_full_zone[indices] = file_aridity
    
    log.info("Writing outputs to zarr")
    edges.array(
        name="precip_comid",
        data=precip_full_zone,
    )    
    edges.array(
        name="pet_comid",
        data=pet_full_zone,
    )  
    edges.array(
        name="ndvi_comid",
        data=ndvi_full_zone,
    )  
    edges.array(
        name="aridity_comid",
        data=aridity_full_zone,
    )        
    

