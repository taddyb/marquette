# import icechunk
import xarray as xr
import zarr
import s3fs

def main():
    # storage_config = icechunk.StorageConfig.s3_from_env(
    #     bucket="mhpi-spatial",
    #     prefix="marquette/merit/quickstart",
    #     region="us-east-2",
    #     endpoint_url=None
    # )
    # store = icechunk.IcechunkStore.create(storage_config)
    ds = xr.open_zarr("/projects/mhpi/data/MERIT/streamflow/zarr/merit_conus_v6.18_snow/74")

    ds1 = ds.isel(time=slice(None, 18))  # part 1
    ds1.to_zarr('s3://mhpi-spatial/marquette/merit/test1/', mode='w')
    # storage_config = icechunk.StorageConfig.s3_from_env(
    #     bucket="mhpi-spatial",
    #     prefix="marquette/merit/quickstart",
    #     region="us-east-2",
    #     endpoint_url=None,
    # )
    # store = icechunk.IcechunkStore.create(storage_config)
    # group = zarr.group(store)
    # array = group.create("my_array", shape=10, dtype=int)
    # array[:] = 1
    # store.commit("first commit")
    
if __name__ == "__main__":
    main()
