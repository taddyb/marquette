import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from omegaconf import DictConfig
from shapely.geometry import LineString, MultiLineString, Point
from tqdm import tqdm

log = logging.getLogger(__name__)


class Edge:
    """A class to represent an edge in a river network."""

    def __init__(self, segment, up=None, ds=None, edge_id=None):
        self.id = edge_id
        self.merit_basin = segment.id
        self.order = segment.order
        self.len = segment.len
        self.len_dir = segment.len_dir
        self.ds = ds
        self.is_headwater = segment.is_headwater
        self.up = up
        # if self.up is not None:
        #     self.up = [x for x in up if x != 0]
        self.slope = segment.slope
        self.sinuosity = segment.sinuosity
        self.stream_drop = segment.stream_drop
        self.uparea = segment.uparea
        self.coords = segment.coords
        self.crs = segment.crs
        self.segment = segment

    def convert_coords_to_wgs84(self):
        """Converts gdf coordinates to WGS84."""
        coords = self.coords
        source_crs = "EPSG:32618"
        target_crs = "EPSG:4326"  # WGS84
        gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=[Point(coord) for coord in coords], crs=source_crs)  # type: ignore
        gdf = gdf.to_crs(target_crs)  # type: ignore
        self.coords = [(point.x, point.y) for point in gdf.geometry]

    def calculate_sinuosity(self, curve_length: float):
        """Calculates the sinuosity of the edge.

        Parameters
        ----------
        curve_length : float
            _description_

        Returns
        -------
        _type_
            _description_
        """
        euclidean_distance = Point(self.coords[0]).distance(Point(self.coords[-1]))
        self.len_dir = euclidean_distance
        # distances.append(euclidean_distance)
        return curve_length / euclidean_distance if euclidean_distance != 0 else 1

    def calculate_drainage_area(self, idx: int, segment_das: dict[str, float]) -> None:
        """Calculates the drainage area of the edge.

        Parameters
        ----------
        idx : int
            the specific index of the edge
        segment_das : dict[str, float]
            A dictionary containing the drainage areas of each segment
        """
        if idx == -1:
            self.uparea = self.segment.uparea
        else:
            if not self.segment.up:
                prev_up_area = 0
            else:
                try:
                    prev_up_area = sum(segment_das[seg] for seg in self.segment.up)
                except KeyError:
                    prev_up_area = 0
                    log.info("Missing upstream branch. Treating as head node")

            ratio = (self.len * (idx + 1)) / self.segment.transformed_line.length
            area_difference = self.segment.uparea - prev_up_area
            self.uparea = prev_up_area + (area_difference * ratio)


class Segment:
    """
    A class to represent a river segment with geographical and hydrological attributes.

    Parameters
    ----------
    row : dict
        A dictionary containing the segment's data. Expected keys are:
        'COMID' (int): Segment ID.
        'order_' (int): Stream order.
        'lengthkm' (float): Length of the segment in kilometers.
        'lengthdir' (float): Direct length of the segment in kilometers.
        'NextDownID' (int): ID of the downstream segment.
        'maxup' (int): Maximum number of upstream segments.
        'up1', 'up2', 'up3', 'up4' (int): IDs of upstream segments.
        'slope' (float): Slope of the segment.
        'sinuosity' (float): Sinuosity of the segment.
        'strmDrop_t' (float): Total stream drop.
        'uparea' (float): Upstream catchment area.
    segment_coords : list of tuples
        List of coordinate tuples (e.g., [(x1, y1), (x2, y2), ...]) representing the segment's geometry.
    crs : str or CRS object
        Coordinate reference system of the segment.

    Attributes
    ----------
    id : int
        Segment ID.
    order : int
        Stream order.
    len : float
        Length of the segment in meters.
    len_dir : float
        Direct length of the segment in meters.
    ds : int
        ID of the downstream segment.
    is_headwater : bool
        True if the segment is a headwater segment, otherwise False.
    up : list of int
        IDs of the upstream segments.
    slope : float
        Slope of the segment.
    sinuosity : float
        Sinuosity of the segment.
    stream_drop : float
        Total stream drop.
    uparea : float
        Upstream catchment area.
    coords : list of tuples
        List of coordinate tuples representing the segment's geometry.
    crs : str or CRS object
        Coordinate reference system of the segment.
    transformed_line : type (optional)
        Transformed line geometry, initialized as None.
    edge_len : type (optional)
        Edge length, initialized as None.
    """

    def __init__(self, row, segment_coords, crs):
        self.id = row["COMID"]
        self.order = row["order"]
        self.len = row["lengthkm"] * 1000  # to meters
        self.len_dir = row["lengthdir"] * 1000  # to meters
        self.ds = row["NextDownID"]
        self.is_headwater = False
        if row["maxup"] > 0:
            up = [row["up1"], row["up2"], row["up3"], row["up4"]]
            self.up = [x for x in up if x != 0]
        else:
            if row["order"] == 1:
                self.is_headwater = True
            self.up = []
        self.slope = row["slope"]
        self.sinuosity = row["sinuosity"]
        self.stream_drop = row["strmDrop_t"]
        self.uparea = row["uparea"]
        self.coords = segment_coords
        self.crs = crs
        self.transformed_line = None
        self.edge_len = None


def get_edge_counts(segments: list[Segment], dx: float, buffer: float) -> dict[str, int]:
    """A function to calculate the number of edges for each segment.

    Parameters
    ----------
    segments : list[Segment]
        A list of merit segments.
    dx : float
        the length of each edge
    buffer : float
        a buffer of which to allow the edge length to deviate from dx

    Returns
    -------
    dict[str, int]
        A dictionary containing the number of edges for each segment.
    """
    edge_counts = {}

    for segment in tqdm(
        segments,
        desc="Creating edges",
        ncols=140,
        ascii=True,
    ):
        try:
            line = LineString(segment.coords)
        except TypeError:
            log.info(f"TypeError for segment {segment.id}. Fusing MultiLineString")
            if segment.coords.geom_type == "MultiLineString":
                multiline = MultiLineString(segment.coords)
                # Create a list of points from all LineStrings in the MultiLineString
                points_list = [point for linestring in multiline.geoms for point in linestring.coords]
                # Convert points to a single line
                line = LineString(points_list)
        source_crs = segment.crs
        target_crs = "EPSG:32618"

        gdf = gpd.GeoDataFrame(geometry=[line], crs=source_crs)  # type: ignore
        gdf = gdf.to_crs(target_crs)  # Transform to the target CRS
        transformed_line = gdf.geometry.iloc[0]  # type: ignore # Get the transformed LineString

        length = transformed_line.length
        num_edges = length // dx

        if num_edges == 0:
            num_edges = 1
            if dx - length < buffer:
                edge_len = length
            else:
                edge_len = dx
            edge_counts[segment.id] = num_edges
        else:
            # Calculate actual segment length
            edge_len = length / num_edges

            # Calculate buffer deviation
            buf_dev = edge_len - dx
            buf_dev_abs = abs(buf_dev)

            # Adjust the number of segments until they are within the buffer
            while buf_dev_abs > buffer:
                if buf_dev > dx:
                    num_edges -= 1
                else:
                    num_edges += 1
                edge_len = length / num_edges
                buf_dev = edge_len - dx
                buf_dev_abs = abs(edge_len - dx)
            edge_counts[segment.id] = int(num_edges)

        segment.transformed_line = transformed_line
        segment.edge_len = edge_len  # type: ignore

    return edge_counts


def get_upstream_ids(segment: Segment, edge_counts: dict[str, int]) -> list[str]:
    """_summary_

    Parameters
    ----------
    segment : Segment
        A merit river segment
    edge_counts : _type_
        The number of edges to break the segment into

    Returns
    -------
    list[str]
        The upstream IDs of the segment.
    """
    if segment.up is None:
        return []
    try:
        up_ids = [f"{up}_{edge_counts[up] - 1}" for up in segment.up]
    except KeyError:
        log.error(f"KeyError with segment {segment.id}")
        return []  # temp fix that will kill this river network and make this edge a stream with order 1
    return up_ids


def segments_to_edges(segment: Segment, edge_counts: dict[str, int], segment_das) -> list[Edge]:
    """
    Converts segments to edges

    Parameters
    ----------
    segment : Segment
        A merit river segment
    edge_counts : dict[str, int]
        The number of edges to break the segment into
    segment_das : dict[str, float]
        A dictionary containing the drainage areas of each segment

    Returns
    -------
    list[Edge]
        A list of edges
    """
    edges = []
    num_edges = edge_counts[segment.id]
    up_ids = get_upstream_ids(segment, edge_counts)

    """Iterating through dx edges"""
    if num_edges == 1:
        """Setting the small reaches to dx"""
        edge = Edge(
            segment,
            up=up_ids,
            ds=f"{segment.ds}_0",
            edge_id=f"{segment.id}_{0}",
        )
        edge.coords = list(segment.transformed_line.interpolate(segment.edge_len * num_edges).coords) + [  # type: ignore
            segment.transformed_line.coords[-1]  # type: ignore
        ]
        edge.len = segment.edge_len
        edge.calculate_sinuosity(edge.len)  # type: ignore
        edge.len_dir = segment.edge_len / edge.sinuosity  # This is FAKE as we're setting the len manually
        edge.calculate_drainage_area(-1, segment_das)
        edges.append(edge)
    else:
        for i in range(num_edges):
            if i == 0:
                # Create the first edge
                edge = Edge(
                    segment,
                    up=up_ids,
                    ds=f"{segment.id}_{i + 1}",
                    edge_id=f"{segment.id}_{i}",
                )
            else:
                # Create subsequent edges
                edge = Edge(
                    segment,
                    up=[f"{segment.id}_{i - 1}"],
                    ds=f"{segment.id}_{i + 1}" if i < num_edges - 1 else f"{segment.ds}_0",
                    edge_id=f"{segment.id}_{i}",
                )
            edge.coords = list(segment.transformed_line.interpolate(segment.edge_len * i).coords) + list(  # type: ignore
                segment.transformed_line.interpolate(segment.edge_len * (i + 1)).coords  # type: ignore
            )
            edge.len = segment.edge_len
            edge.sinuosity = edge.calculate_sinuosity(segment.edge_len)  # type: ignore
            edge.calculate_drainage_area(i, segment_das)
            edges.append(edge)
    [edge.convert_coords_to_wgs84() for edge in edges]  # Convert back to WGS84
    return edges


def data_to_csv(data_list: list[Any]) -> pd.DataFrame:
    """Writing the edges list to a csv

    Parameters
    ----------
    data_list : list
        List of Edge objects

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing the edge data
    """
    data_dicts = []
    for data in data_list:
        edge_dict = {
            "id": data.id,
            "merit_basin": data.merit_basin,
            "order": data.order,
            "len": data.len,
            "len_dir": data.len_dir,
            "ds": data.ds,
            "is_headwater": data.is_headwater,
            "up": data.up,
            "slope": data.slope,
            "sinuosity": data.sinuosity,
            "stream_drop": data.stream_drop,
            "uparea": data.uparea,
            "coords": data.coords,
            "crs": data.crs,
        }
        data_dicts.append(edge_dict)
    df = pd.DataFrame(data_dicts)
    return df


def _find_flowlines(cfg: DictConfig) -> Path:
    """
    A function to find the correct flowline of all MERIT basins using glob

    Parameters
    ----------
    cfg : DictConfig
        The cfg object

    Returns
    -------
    Path
        The file that we're going to create flowline connectivity for

    Raises
    ------
    IndexError
        Raised if no flowlines are found with your MERIT region code
    """
    flowline_path = Path(cfg.save_paths.flow_lines)
    region_id = f"_{cfg.zone}_"
    matching_file = flowline_path.glob(f"*{region_id}*.shp")
    try:
        found_file = list(matching_file)[0]
        return found_file
    except IndexError as e:
        raise IndexError(f"No flowlines found using: *{region_id}*.shp") from e
