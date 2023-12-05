import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from marquette.hydrofabric._graph import (
    data_to_csv,
    get_edge_counts,
    segments_to_edges,
    HydroFabricSegment,
)
from marquette.hydrofabric._network import generate_sub_basin

log = logging.getLogger(__name__)


def create_graph(cfg):
    flowline_file = cfg.file_paths.flow_lines
    polyline_gdf = gpd.read_file(flowline_file)

    # Making sure connectivity is an int
    for col in [
        "COMID",
        "Divergence",
        "FromNode",
        "ToNode",
        "toCOMID",
        "REACHCODE",
        "streamleve",
        "StreamOrde",
    ]:
        polyline_gdf[col] = polyline_gdf[col].astype(int)

    sorted_gdf = polyline_gdf.sort_values(by="TotDASqKM", ascending=True)
    starting_point = sorted_gdf.iloc[-1]
    edges = []
    generate_sub_basin(cfg, starting_point, polyline_gdf, basin_segments=edges)
    plot_edges_vs_gdf(edges, sorted_gdf)
    edges = data_to_csv(edges)
    # edges.to_csv(cfg.csv.edges, index=False)

    return edges


def plot_edges_vs_gdf(edges, gdf):
    line_strings = []
    for line in edges:
        if isinstance(line.geometry, pd.Series):
            for geometry in line.geometry:
                line_strings.append(geometry)
        else:
            line_strings.append(line.geometry)
    fig, ax = plt.subplots()
    gdf.plot(ax=ax, color='blue')
    file_path = '/data/tkb5476/projects/marquette/data/plots/sorted_gdf.png'
    plt.savefig(file_path, bbox_inches='tight')
    fig, ax = plt.subplots()
    line_series = gpd.GeoSeries(line_strings)
    line_series.plot(ax=ax, color='red')
    file_path = '/data/tkb5476/projects/marquette/data/plots/created_edges.png'
    plt.savefig(file_path, bbox_inches='tight')

def create_network(cfg, edges):
    graph_map = {row[0]: i for i, row in enumerate(edges.itertuples(index=False))}
    upstream_lsts = edges["up"].to_numpy()
    max_len = max(len(lst) for lst in upstream_lsts)
    connectivity = np.array(
        [lst + [None] * (max_len - len(lst)) for lst in upstream_lsts],
        dtype=object,
    )

    network = torch.zeros(
        [connectivity.shape[0], connectivity.shape[0]],
    )
    for i, row in enumerate(connectivity):
        values = [graph_map[key] if key in graph_map.keys() else None for key in row]
        if any(value is not None for value in values):
            for j, value in enumerate(values):
                if value is not None:
                    network[i, value] = 1

    # Outputting results to csv for debugging. The index of 0 is the COMID field
    network_df = pd.DataFrame(
        network.cpu().numpy(),
        index=edges.iloc[:, 0].to_numpy(),
        columns=edges.iloc[:, 0].to_numpy(),
    )
    network_df.to_csv(cfg.csv.network)
    return network
