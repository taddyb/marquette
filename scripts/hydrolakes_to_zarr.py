import argparse
from pathlib import Path

import geopandas as gpd
import numcodecs
import zarr
from shapely.wkt import dumps
from tqdm import tqdm


def split_hydrolakes(input_path: Path, output_path: Path) -> None:
    
    print(f"Reading gdf file: {input_path}")
    gdf = gpd.read_file(filename=input_path)
    
    print("Writing geometries")
    geometries = gdf["geometry"].apply(lambda geom: dumps(geom)).values

    # Create a Zarr store
    root: zarr.Group = zarr.open_group(output_path, mode="a")

    # Create datasets for each column
    for column in tqdm(
        gdf.columns,
        desc="writing gdf to zarr",
        ncols=140,
        ascii=True,
    ):
        if column == "geometry":
            root.array(column, data=geometries, dtype=object, object_codec=numcodecs.VLenUTF8())
            root.attrs["crs"] = gdf.crs.to_string()
        else:
            data = gdf[column].values
            if data.dtype == 'O':
                # Saving as an object
                root.array(column, data=geometries, dtype=object, object_codec=numcodecs.VLenUTF8())
            else:
                root.array(column, data=data)

    print(f"Processing complete! Zarr store saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a shapefile to a Zarr store")
    parser.add_argument("input_shp", type=Path, help="Path to the input shapefile")
    parser.add_argument(
        "output_zarr", type=Path, help="Path to save the output Zarr store"
    )

    args = parser.parse_args()

    input_path = Path(args.input_shp)
    output_path = Path(args.output_zarr)
    if not input_path.exists():
        raise FileNotFoundError(f"Input shapefile not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output Zarr store already exists: {output_path}")

    split_hydrolakes(input_path, output_path)
