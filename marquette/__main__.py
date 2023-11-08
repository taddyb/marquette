import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
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
        extract_hydrofabric(cfg)
    elif cfg.name.lower() == "merit":
        extract_merit(cfg)


def extract_hydrofabric(cfg: DictConfig) -> None:
    from marquette.hydrofabric.map import create_graph, create_network
    log.info(f"Creating River Graph")
    edges = create_graph(cfg)
    log.info(f"Connecting Noes/Edges")
    create_network(cfg, edges)
    log.info(f"Done!")

def extract_merit(cfg: DictConfig) -> None:
    from marquette.merit.map import create_graph, create_network
    log.info(f"Creating River Graph")
    edges = create_graph(cfg)
    log.info(f"Connecting Noes/Edges")
    create_network(cfg, edges)
    log.info(f"Done!")

if __name__ == "__main__":
    main()
