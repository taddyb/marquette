import logging

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from tqdm import tqdm

log = logging.getLogger(__name__)


class HydroFabricEdge:
    def __init__(self, reach, crs, from_node=None, to_node=None, id=None):
        self.id = id
        self.elevation = reach.elev_mean
        self.from_node = from_node
        self.to_node = to_node
        self.order = reach.StreamOrde
        self.len = reach.LENGTHKM
        self.area_sqkm = reach.AreaSqKM
        self.drainage_area = reach.TotDASqKM
        self.wbareatype = reach.wbareatype
        self.roughness = reach.roughness
        self.crs = crs
        self.geometry = reach.geometry

    def convert_coords_to_wgs84(self):
        coords = self.coords
        source_crs = "EPSG:32618"
        target_crs = "EPSG:4326"  # WGS84
        gdf = gpd.GeoDataFrame(
            geometry=[Point(coord) for coord in coords], crs=source_crs
        )
        gdf = gdf.to_crs(target_crs)
        self.coords = [(point.x, point.y) for point in gdf.geometry]

    def calculate_sinuosity(self, curve_length):
        euclidean_distance = Point(self.coords[0]).distance(Point(self.coords[-1]))
        self.len_dir = euclidean_distance
        # distances.append(euclidean_distance)
        return curve_length / euclidean_distance if euclidean_distance != 0 else 1

    def calculate_drainage_area(self, idx, segment_das):
        if idx == -1:
            self.uparea = self.segment.uparea
        else:
            if not self.segment.up:
                prev_up_area = 0
            else:
                prev_up_area = sum(segment_das[seg] for seg in self.segment.up)

            ratio = (self.len * (idx + 1)) / self.segment.transformed_line.length
            area_difference = self.segment.uparea - prev_up_area
            self.uparea = prev_up_area + (area_difference * ratio)


class HydroFabricSegment:
    def __init__(self, row, segment_coords, crs):
        self.id = row["COMID"]
        self.order = row["StreamOrde"]
        self.len = row["LENGTHKM"] * 1000  # to meters
        self.uparea = row["TotDASqKM"]
        self.slope = row["slope"]
        self.roughness = row["roughness"]
        self.coords = segment_coords
        self.crs = crs
        self.transformed_line = None
        self.edge_len = None


def get_edge_counts(segments, dx, buffer):
    edge_counts = {}

    for segment in tqdm(segments):
        try:
            line = LineString(segment.coords)
        except TypeError:
            import traceback

            error_message = traceback.format_exc()
            log.info(f"TypeError for segment {segment.id}. Fusing MultiLineString")
            # log.info(f"Details: {error_message}")
            if segment.coords.geom_type == "MultiLineString":
                multiline = MultiLineString(segment.coords)
                # Create a list of points from all LineStrings in the MultiLineString
                points_list = [
                    point
                    for linestring in multiline.geoms
                    for point in linestring.coords
                ]
                # Convert points to a single line
                line = LineString(points_list)
        source_crs = segment.crs
        target_crs = "EPSG:32618"

        gdf = gpd.GeoDataFrame(geometry=[line], crs=source_crs)
        gdf = gdf.to_crs(target_crs)  # Transform to the target CRS
        transformed_line = gdf.geometry.iloc[0]  # Get the transformed LineString

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
        segment.edge_len = edge_len

    return edge_counts


def get_upstream_ids(segment, edge_counts):
    if segment.up is None:
        return []
    try:
        up_ids = [f"{up}_{edge_counts[up] - 1}" for up in segment.up]
    except KeyError:
        log.error(f"KeyError with segment {segment.id}")
    return up_ids


def segments_to_edges(segment, edge_counts, segment_das):
    edges = []
    num_edges = edge_counts[segment.id]
    # up_ids = get_upstream_ids(segment, edge_counts)

    """Iterating through dx edges"""
    if num_edges == 1:
        """Setting the small reaches to dx"""
        edge = HydroFabricEdge(
            segment,
            ds=f"{segment.ds}_0",
            edge_id=f"{segment.id}_{0}",
        )
        edge.coords = list(
            segment.transformed_line.interpolate(segment.edge_len * num_edges).coords
        ) + [segment.transformed_line.coords[-1]]
        edge.len = segment.edge_len
        edge.calculate_sinuosity(edge.len)
        edge.len_dir = (
            segment.edge_len / edge.sinuosity
        )  # This is FAKE as we're setting the len manually
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
                    ds=f"{segment.id}_{i + 1}"
                    if i < num_edges - 1
                    else f"{segment.ds}_0",
                    edge_id=f"{segment.id}_{i}",
                )
            edge.coords = list(
                segment.transformed_line.interpolate(segment.edge_len * i).coords
            ) + list(
                segment.transformed_line.interpolate(segment.edge_len * (i + 1)).coords
            )
            edge.len = segment.edge_len
            edge.sinuosity = edge.calculate_sinuosity(segment.edge_len)
            edge.calculate_drainage_area(i, segment_das)
            edges.append(edge)
    [edge.convert_coords_to_wgs84() for edge in edges]  # Convert back to WGS84
    return edges


def data_to_csv(data_list):
    """
    writing the edges list to disk
    :param edges:
    :return:
    """
    data_dicts = []
    for data in data_list:
        edge_dict = {
            "id": data.id,
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