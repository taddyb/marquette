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
    elif cfg.name.lower() == "merit_s3":
        from marquette.merit_s3.create import (create_edges, create_N, create_TMs, write_streamflow)
        
        start = time.perf_counter()
        log.info(f"Creating MERIT S3 {cfg.zone} River Graph")
        edges = create_edges(cfg)

        log.info(f"Creating MERIT S3 {cfg.zone} Connectivity Matrix (N) for gages")
        create_N(cfg, edges)

        log.info(f"Mapping {cfg.zone} Streamflow to S3 TMs")
        create_TMs(cfg, edges)

        log.info("Converting Streamflow to S3 DataTree")
        write_streamflow(cfg, edges)

        end = time.perf_counter()
        log.info(f"Extracting data took : {(end - start):.6f} seconds")
        
    elif cfg.name.lower() == "merit":
        from marquette.merit.create import (create_edges, create_N, create_TMs,
                                            map_lake_points, write_streamflow, run_extensions)

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


if __name__ == "__main__":
    main()  # type: ignore
