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
        
        log.info("Mapping Lake Pour Points to Edges")
        map_lake_points(cfg, edges)

        log.info("Converting Streamflow to zarr")
        write_streamflow(cfg, edges)

        log.info("Running Data Post-Processing Extensions")
        run_extensions(cfg, edges)

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
    if "pet_forcing" in cfg.extensions:
        from marquette.merit.extensions import pet_forcing

        log.info("Adding PET forcing to your MERIT River Graph")
        if "pet" in edges:
            log.info("PET forcing already exists in zarr format")
        else:
            pet_forcing(cfg, edges)
    if "global_dhbv_static_inputs" in cfg.extensions:
        from marquette.merit.extensions import global_dhbv_static_inputs

        log.info("Adding global dHBV static input data to your MERIT River Graph")
        if "aridity" in edges:
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
        if f"summed_q_prime_{cfg.create_streamflow.version}" in edges:
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
            format_lstm_forcings(cfg, edges)


if __name__ == "__main__":
    main()
