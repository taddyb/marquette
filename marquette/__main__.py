import logging
import time

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


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
                                            write_streamflow)

        start = time.perf_counter()
        log.info(f"Creating MERIT {cfg.zone} River Graph")
        edges = create_edges(cfg)

        log.info(f"Creating MERIT {cfg.zone} Connectivity Matrix (N) for gages")
        create_N(cfg, edges)

        log.info("Converting Streamflow to zarr")
        write_streamflow(cfg, edges)

        log.info(f"Mapping {cfg.zone} Streamflow to Nodes/Edges")
        create_TMs(cfg, edges)

        log.info("Running Data Post-Processing Extensions")
        run_extensions(cfg, edges)

        end = time.perf_counter()
        log.info(f"Extracting data took : {(end - start):.6f} seconds")
    else:
        log.error(f"incorrect name specified: {cfg.name}")


def run_extensions(cfg, edges):
    """
    The function for running post-processing data extensions

    :param cfg: Configuration object.
    :type cfg: DictConfig
    :return: None
    """
    if "soils_data" in cfg.extensions:
        from marquette.merit.extensions import soils_data

        log.info("Adding soils information to your MERIT River Graph")
        soils_data(cfg, edges)


if __name__ == "__main__":
    main()
