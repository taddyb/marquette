import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


def soils_data(cfg: DictConfig, edges: zarr.Group) -> None:
    """Creates soils data attributes for the edges based on a provided shapefile

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    edges : zarr.Group
        The zarr group object containing the edge data.

    Raises
    ------
    FileNotFoundError
        If the soils data file is not found.
    """
    flowline_file = (
        Path(cfg.data_path) / f"raw/routing_soil_properties/riv_pfaf_{cfg.zone}_buff_split_soil_properties.shp"
    )
    if flowline_file.exists() is False:
        raise FileNotFoundError("Soils data not found")
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
        .then(pl.col(["OM_05_M_25", "Cl_05_Mn", "Sd_05_Mn", "St_05_Mn"]).fill_null(strategy="forward"))
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
    """Creates PET data attributes for the edges based on a provided shapefile

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    edges : zarr.Group
        The zarr group object containing the edge data.

    Raises
    ------
    FileNotFoundError
        If the PET data file is not found.
    ValueError
        There is a mismatch in the number of comids in the data and the edge_merit_basins array.
    ValueError
        There is a mismatch in the number of timesteps in the data and the num_timesteps variable.
    """
    pet_file_path = Path(f"/projects/mhpi/data/global/zarr_sub_zone/{cfg.zone}")
    num_timesteps = pd.date_range(
        start=cfg.create_streamflow.start_date,
        end=cfg.create_streamflow.end_date,
        freq="d",
    ).shape[0]
    if pet_file_path.exists() is False:
        raise FileNotFoundError("PET forcing data not found")
    edge_merit_basins: np.ndarray = edges.merit_basin[:] # type: ignore
    pet_edge_data = []
    pet_comid_data = []
    mapping = np.empty_like(edge_merit_basins, dtype=int)
    files = pet_file_path.glob("*")
    for file in files:
        pet_zone_data = zarr.open_group(file, mode="r")
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


def global_dhbv_static_inputs(cfg: DictConfig, edges: zarr.Group) -> None:
    """
    Pulling Data from the global_dhbv_static_inputs data and storing it in a zarr store

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    edges : zarr.Group
        The zarr group object containing the edge data.

    Raises
    ------
    FileNotFoundError
        If the global_dhbv_static_inputs data is not found.
    ValueError
        If the data is not consistent. Check the number of comids in the data and the edge_merit_basins array.
    """
#     Attributes in the provided file are as follows:
#     ['area','ETPOT_Hargr', 'FW', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
#    'HWSD_silt', 'NDVI', 'Porosity', 'SoilGrids1km_clay',
#    'SoilGrids1km_sand', 'SoilGrids1km_silt', 'T_clay', 'T_gravel',
#    'T_sand', 'T_silt', 'aridity', 'glaciers', 'meanP', 'meanTa',
#    'meanelevation', 'meanslope', 'permafrost', 'permeability',
#    'seasonality_P', 'seasonality_PET', 'snow_fraction',
#    'snowfall_fraction']
    file_path = Path(f"/projects/mhpi/data/global/zarr_sub_zone/{cfg.zone}")
    if file_path.exists() is False:
        raise FileNotFoundError("global_dhbv_static_inputs data not found")
    edge_merit_basins: np.ndarray = edges.merit_basin[:] # type: ignore
    comid_data = []
    aridity_data = []
    porosity_data = []
    mean_p_data = []
    mean_elevation_data = []
    glaciers_data = []

    mapping = np.empty_like(edge_merit_basins, dtype=int)
    files = file_path.glob("*")
    for file in files:
        pet_zone_data = zarr.open_group(file, mode="r")
        comids = pet_zone_data.COMID[:]
        aridity = pet_zone_data["attrs"]["aridity"][:]
        porosity = pet_zone_data["attrs"]["Porosity"][:]
        mean_p = pet_zone_data["attrs"]["meanP"][:]
        mean_elevation = pet_zone_data["attrs"]["meanelevation"][:]
        glaciers = pet_zone_data["attrs"]["glaciers"][:]

        comid_data.append(comids)
        aridity_data.append(aridity)
        porosity_data.append(porosity)
        mean_p_data.append(mean_p)
        mean_elevation_data.append(mean_elevation)
        glaciers_data.append(glaciers)

    comid_arr = np.concatenate(comid_data)
    aridity_arr = np.concatenate(aridity_data)
    porosity_arr = np.concatenate(porosity_data)
    mean_p_arr = np.concatenate(mean_p_data)
    mean_elevation_arr = np.concatenate(mean_elevation_data)
    glacier_arr = np.concatenate(glaciers_data)

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


def calculate_incremental_drainage_area(cfg: DictConfig, edges: zarr.Group) -> None:
    """
    Runs a Polars query to calculate the incremental drainage area for each edge in the MERIT dataset

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    edges : zarr.Group
        The zarr group object containing the edge data.

    Raises
    ------
    FileNotFoundError
        If the basin file is not found.
    """
    basin_file = Path(cfg.data_path) / f"raw/basins/cat_pfaf_{cfg.zone}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
    if basin_file.exists() is False:
        raise FileNotFoundError("Basin file not found")
    gdf = gpd.read_file(basin_file)
    _df = pd.DataFrame(gdf.drop(columns="geometry"))
    df = pl.from_pandas(_df)
    edges_df = pl.DataFrame(
        {
            "COMID": edges.merit_basin[:],
            "id": edges.id[:],
            "order": np.arange(edges.id.shape[0]), # type: ignore
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
                    function=lambda list_of_series: list_of_series[1] / list_of_series[0].shape[0],
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
