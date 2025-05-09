import logging
import time

import hydra
import zarr
from omegaconf import DictConfig

log = logging.getLogger(name=__name__)


@hydra.main(
    version_base="1.3",
    config_path="conf/",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """
    Main function for running experiments.

    :param cfg: Configuration object.
    :type cfg: DictConfig
    :return: None
    """
    if cfg.name.lower() == "hydrofabric":
        raise ImportError("Hydrofabric functionality not yet supported")
    elif cfg.name.lower() == "merit":
        from marquette.merit.create import (create_edges, create_N, create_TMs,
                                            map_lake_points, write_streamflow)

        start = time.perf_counter()
        log.info(f"Creating MERIT {cfg.zone} River Graph")
        edges = create_edges(cfg)

        log.info(f"Creating MERIT {cfg.zone} Connectivity Matrix (N) for gages")
        create_N(cfg, edges)

        log.info(f"Mapping {cfg.zone} Streamflow to TMs")
        create_TMs(cfg, edges)
        
        # log.info("Mapping Lake Pour Points to Edges")
        # map_lake_points(cfg, edges)

        log.info("Converting Streamflow to zarr")
        write_streamflow(cfg, edges)

        # log.info("Running Data Post-Processing Extensions")
        # run_extensions(cfg, edges)

        end = time.perf_counter()
        log.info(f"Extracting data took : {(end - start):.6f} seconds")
    else:
        log.error(f"incorrect name specified: {cfg.name}")


def run_extensions(cfg: DictConfig, edges: zarr.Group) -> None:
    """
    The function for running post-processing data extensions

    :param cfg: Configuration object.
    :type cfg: DictConfig
    :return: None
    """
    if "soils_data" in cfg.extensions:
        from marquette.merit.extensions import soils_data

        log.info("Adding soils information to your MERIT River Graph")
        if "ksat" in edges:
            log.info("soils information already exists in zarr format")
        else:
            soils_data(cfg, edges)
    if "log_uparea" in cfg.extensions:
        from marquette.merit.extensions import log_uparea

        log.info("Adding log_uparea to your MERIT River Graph")
        if "log_uparea" in edges:
            log.info("log_uparea already exists in zarr format")
        else:
            log_uparea(cfg, edges)
    if "pet_forcing" in cfg.extensions:
        from marquette.merit.extensions import pet_forcing

        log.info("Adding PET forcing to your MERIT River Graph")
        if "pet" in edges:
            log.info("PET forcing already exists in zarr format")
        else:
            pet_forcing(cfg, edges)
    if "temp_mean" in cfg.extensions:
        from marquette.merit.extensions import temp_forcing

        log.info("Adding temp_mean forcing to your MERIT River Graph")
        if "temp_mean" in edges:
            log.info("Temp_mean forcing already exists in zarr format")
        else:
            temp_forcing(cfg, edges)
    if "global_dhbv_static_inputs" in cfg.extensions:
        from marquette.merit.extensions import global_dhbv_static_inputs

        log.info("Adding global dHBV static input data to your MERIT River Graph")
        if "fw" in edges:
            log.info("global_dhbv_static_inputs already exists in zarr format")
        else:
            global_dhbv_static_inputs(cfg, edges)

    if "incremental_drainage_area" in cfg.extensions:
        from marquette.merit.extensions import \
            calculate_incremental_drainage_area

        log.info("Adding edge/catchment area input data to your MERIT River Graph")
        if "incremental_drainage_area" in edges:
            log.info("incremental_drainage_area already exists in zarr format")
        else:
            calculate_incremental_drainage_area(cfg, edges)

    if "q_prime_sum" in cfg.extensions:
        from marquette.merit.extensions import calculate_q_prime_summation

        log.info("Adding q_prime_sum to your MERIT River Graph")
        if "summed_q_prime_v2" in edges:
            log.info("q_prime_sum already exists in zarr format")
        else:
            calculate_q_prime_summation(cfg, edges)
            

    if "upstream_basin_avg_mean_p" in cfg.extensions:
        from marquette.merit.extensions import calculate_mean_p_summation

        log.info("Adding q_prime_sum to your MERIT River Graph")
        if "upstream_basin_avg_mean_p" in edges:
            log.info("upstream_basin_avg_mean_p already exists in zarr format")
        else:
            calculate_mean_p_summation(cfg, edges)
            
    if "q_prime_sum_stats" in cfg.extensions:
        from marquette.merit.extensions import calculate_q_prime_sum_stats

        log.info("Adding q_prime_sum statistics to your MERIT River Graph")
        if "summed_q_prime_median" in edges:
            log.info("q_prime_sum statistics already exists in zarr format")
        else:
            calculate_q_prime_sum_stats(cfg, edges)
            
    if "lstm_stats" in cfg.extensions:
        from marquette.merit.extensions import format_lstm_forcings

        log.info("Adding lstm statistics from global LSTM to your MERIT River Graph")
        if "precip_comid" in edges:
            log.info("q_prime_sum statistics already exists in zarr format")
        else:
            format_lstm_forcings(cfg, edges)
        
    if "hf_width" in cfg.extensions:
        from marquette.merit.extensions import calculate_hf_width

        log.info("Adding hf_width to your MERIT River Graph")
        if "hf_v22_ch_slope" in edges:
            log.info("hf_width already exists in zarr format")
        else:
            calculate_hf_width(cfg, edges)
           
    if "stream_geo_attr" in cfg.extensions:
        from marquette.merit.extensions import calculate_stream_geo_attr

        log.info("Adding stream_geo_attr to your MERIT River Graph")
        if "stream_geos_width" in edges:
            log.info("stream_geo_attr already exists in zarr format")
        else:
            calculate_stream_geo_attr(cfg, edges) 
        
    if "contains_lake" in cfg.extensions:
        from marquette.merit.extensions import calculate_if_lake

        log.info("Adding contains_lake to your MERIT River Graph")
        if "contains_lake" in edges:
            log.info("contains_lake already exists in zarr format")
        else:
            calculate_if_lake(cfg, edges) 
            
    if "hydrofabric_v22" in cfg.extensions:
        from marquette.merit.extensions import map_hydrofabric_v22

        log.info("Adding hydrofabric_v22 attributes to your MERIT River Graph")
        if "hydrofabric_v22_ch_slp" in edges:
            log.info("hydrofabric_v22 attributes already exists in zarr format")
        else:
            map_hydrofabric_v22(cfg, edges) 

    if "chi" in cfg.extensions:
        from marquette.merit.extensions import map_chi

        log.info("Adding chi to your MERIT River Graph")
        if "chi_max" in edges:
            log.info("chi already exists in zarr format")
        else:
            map_chi(cfg, edges) 


if __name__ == "__main__":
    main()
