# Creating a MERIT river graph from provided data

!!! info 

    This page documentst the `create_edges()` function

## Discretization of MERIT flowlines

Given the provided MERIT flowline data, you will notice that there is no standard size/length of these flowlines. Some of the catchments are large (50+ km reaches) others are much smaller (0.5 km reaches).

To get around this, we break up the MERIT river connectivity into smaller pieces based 2 km segments with a 666 m buffer. 

This data processing is done with dask/zarr for optimal speed and chunked data storage (fast read/writes).

## What is being saved to zarr stores

The following data is being saved:

| Attribute               | Description                                    | Data Type    | Example   |
| :---------------------: | ---------------------------------------------- |------------- | --------- |
| `id`                    | The edge id                                    | `str`        | 7402309_1 |
| `merit_basin`           | The edge's merit basin                         | `int`        | 7402309   |
| `segment_sorting_index` | An ordering index for the edge (Drainage Area) | `int`        | 2         |
| `order`                 | Edge stream order                              | `int`        | 1         |
| `len`                   | Edge stream length                             | `float`      | 2000.0    |
| `len_dir`               | Edge stream Euclidean Length                   | `float`      | 2000.0    |
| `ds`                    | Downstream connection                          | `str`        | 7402309_2 |
| `up`                    | Upstream connection                            | `list[str]`  | 7402308_2 |
| `up_merit`              | Upstream merit basin                           | `list[int]`  | 7402308   |
| `slope`                 | Edge stream slope                              | `float`      | 0.002     |
| `sinuosity`             | Edge stream sinuosity                          | `float`      | i         |
| `stream_drop`           | Difference in elevation over the edge length   | `float`      | i         |
| `uparea`                | Upstream drainage area                         | `float`      | 283.29    |
| `coords`                | Edge map coordinates                           | `str`        | i         |
| `crs`                   | Edge map projection                            | `CRS object` | i         |
