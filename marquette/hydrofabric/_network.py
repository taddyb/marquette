import logging
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch

from marquette.hydrofabric._graph import HydroFabricEdge

log = logging.getLogger(__name__)


def create_edges(reach, basin_segments, gdf, node_len, dx, buffer):
    crs = gdf.crs
    upstream_node, length = calc_upstream_metrics(reach)
    num_edges = 2
    edge_length = node_len / num_edges
    while edge_length > (dx + buffer):
        num_edges = num_edges + 1
        edge_length = node_len / num_edges
    for i in range(num_edges):
        if i == 0:
            edge = HydroFabricEdge(
                reach,
                crs,
                from_node=upstream_node,
                to_node=f"{reach.COMID}_{i + 1}",
                id=f"{reach.COMID}_{i}",
            )
        else:
            if i < (num_edges - 1):
                ds = f"{reach.COMID}_{i + 1}"
            else:
                ds = f"{reach.ToNode}",
            edge = HydroFabricEdge(
                reach,
                crs,
                from_node=f"{reach.COMID}_{i + 1}",
                to_node=ds,
                id=f"{reach.COMID}_{i}",
            )
        # TODO make sure we calculate each edge's attributes
        edge.len = edge_length
        basin_segments.append(edge)


def create_edge(reach, basin_segments, gdf):
    edge = HydroFabricEdge(reach, gdf.crs, from_node=reach.FromNode, to_node=reach.ToNode, id=reach.COMID)
    basin_segments.append(edge)


def calc_upstream_metrics(reach):
    upstream_node = reach.FromNode
    if isinstance(upstream_node, pd.Series) is False:
        upstream_node = [upstream_node]
        length = [reach.LENGTHKM * 1000]  # Unit: Meters
    else:
        upstream_node = [node for node in upstream_node]
        length = [length * 1000 for length in reach.LENGTHKM]
    return upstream_node, length


def generate_sub_basin(
    cfg: DictConfig, reach: object, gdf: gpd.GeoDataFrame, basin_segments: List
) -> List:
    dx = cfg.dx  # Unit: Meters
    buffer = cfg.buffer * dx  # Unit: Meters
    upstream_node, length = calc_upstream_metrics(reach)
    for i in range(len(upstream_node)):
        node = upstream_node[i]
        node_len = length[i]
        if node_len < buffer:
            upstream_edge = gdf[gdf['ToNode'] == node]
            if upstream_edge.empty:
                log.debug("No upstream nodes for this small tributary. Let's forget it exists.")
                pass
            else:
                log.debug("skipping this basin and passing connectivity to the upstream node")
                _upstream_node, _ = calc_upstream_metrics(reach)
                if len(_upstream_node) == 1:
                    log.debug("merging some lines together")
                else:
                    for j in range(len(_upstream_node)):
                        upstream_edge.ToNode = reach.ToNode
                        generate_sub_basin(cfg, upstream_edge, gdf, basin_segments)
        else:
            if node_len > (dx + buffer):
                create_edges(reach, basin_segments, gdf, node_len, dx, buffer)
                log.debug("Splitting large reach into smaller pieces")
            else:
                create_edge(reach, basin_segments, gdf)
            upstream_edge = gdf[gdf['ToNode'] == node]
            if upstream_edge is not None:
                generate_sub_basin(cfg, upstream_edge, gdf, basin_segments)
