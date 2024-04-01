import logging
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)


class Direction(Enum):
    up = "up"
    down = "down"


def traverse(gdf: gpd.GeoDataFrame, comid: float, attribute: str, direction: Direction):
    if direction == Direction.up:
        # Using only the up1 dir as we know it always exists
        _traverse = "up1"
    elif direction == Direction.down:
        _traverse = "NextDownID"
    else:
        raise ValueError("Invalid direction. Look at the Direction Enum and ")
    if comid == 0:
        return np.nan
    else:
        fill_value = gdf[attribute][gdf.COMID == comid].values
        if np.isnan(fill_value)[0]:
            next_id = gdf[_traverse][gdf.COMID == comid].values[0]
            return traverse(gdf, next_id, attribute, direction)
        else:
            return fill_value


def spatial_nan_filter(
    attribute: str,
    attr_values: np.ndarray,
    mapping: np.ndarray,
    polyline_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Filling NaN values based on the predictions up or downstream
    Using the min value of the attribute is no Up or Downstream values are available
    """
    _attr_mapped = attr_values[mapping]
    mask = np.isnan(_attr_mapped)
    if mask.sum() == 0:
        return _attr_mapped
    else:
        nan_idx = mapping[mask]
        nan_fill = np.empty_like(nan_idx, dtype=float)
        for i, idx in enumerate(
            tqdm(nan_idx, desc=f"\rFilling NaN values for {attribute}")
        ):
            downstream_comid = polyline_gdf.NextDownID[idx]
            fill_value = traverse(
                polyline_gdf, downstream_comid, attribute, Direction.down
            )
            if np.isnan(fill_value):
                upstream_comid = polyline_gdf.up1[idx]
                fill_value = traverse(
                    polyline_gdf, upstream_comid, attribute, Direction.up
                )
                if np.isnan(fill_value):
                    fill_value = np.nanmin(polyline_gdf[attribute])
            nan_fill[i] = fill_value
        _attr_mapped[mask] = nan_fill
        return _attr_mapped


def soils_data(cfg: DictConfig, edges: zarr.Group) -> None:
    soils_data_path = Path(cfg.data_path) / f"extensions/soils_data/{cfg.zone}"
    if soils_data_path.exists():
        log.info("Soils data already exists in zarr format")
    else:
        root = zarr.group(soils_data_path)
        flowline_file = (
            Path(cfg.data_path)
            / f"raw/routing_soil_properties/riv_pfaf_{cfg.zone}_buff_split_soil_properties.shp"
        )
        polyline_gdf = gpd.read_file(flowline_file)
        edge_merit_basins: np.ndarray = edges.merit_basin[:]
        gdf_ids = np.array(polyline_gdf.COMID, dtype=int)
        mapping = np.empty_like(edge_merit_basins, dtype=int)
        for i, id in enumerate(tqdm(gdf_ids, desc="\rProcessing soil data")):
            idx = np.where(edge_merit_basins == id)[0]
            mapping[idx] = i
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
        for i, attr in enumerate(attributes):
            attr_values = polyline_gdf[attr].values
            _attr_nan_filter = spatial_nan_filter(
                attr, attr_values, mapping, polyline_gdf
            )
            root.array(name=names[i], data=_attr_nan_filter[mapping])


def pet_forcing(cfg: DictConfig, edges: zarr.Group) -> None:
    pet_zarr_data_path = Path(cfg.data_path) / f"extensions/pet_forcing/{cfg.zone}"
    if pet_zarr_data_path.exists():
        log.info("PET forcing data already exists in zarr format")
    else:
        root = zarr.group(store=pet_zarr_data_path)
        pet_file_path = Path(
            f"/projects/mhpi/hjj5218/data/global/zarr_sub_zone/{cfg.zone}"
        )
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
        root.array(name="pet", data=mapped_attr)
        root.array(name="comid", data=pet_comid_arr)
        
        

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
    zarr_data_path = Path(cfg.data_path) / f"extensions/global_dhbv_static_inputs/{cfg.zone}"
    if zarr_data_path.exists():
        log.info("global_dhbv_static_inputs already exists in zarr format")
    else:
        root = zarr.group(store=zarr_data_path)
        file_path = Path(
            f"/projects/mhpi/hjj5218/data/global/zarr_sub_zone/{cfg.zone}"
        )
        num_timesteps = pd.date_range(
            start=cfg.create_streamflow.start_date,
            end=cfg.create_streamflow.end_date,
            freq="d",
        ).shape[0]
        if file_path.exists() is False:
            raise FileNotFoundError("global_dhbv_static_inputs data not found")
        edge_merit_basins: np.ndarray = edges.merit_basin[:]
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
            aridity = pet_zone_data.attrs.aridity[:]
            porosity = pet_zone_data.attrs.Porosity[:]
            mean_p = pet_zone_data.attrs.meanP[:]
            mean_elevation = pet_zone_data.attrs.meanelevation[:]
            glaciers = pet_zone_data.attrs.glaciers[:]
            
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
        root.array(name="aridity", data=aridity_arr[mapping])
        root.array(name="comid", data=comid_arr[mapping])
        root.array(name="porosity", data=porosity_arr[mapping])
        root.array(name="mean_p", data=mean_p_arr[mapping])
        root.array(name="mean_elevation", data=mean_elevation_arr[mapping])
        root.array(name="glacier", data=glacier_arr[mapping])
