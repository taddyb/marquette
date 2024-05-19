# Quick Start:

## Prerequisites

The first step to using Marquette is creating a Python environment with all of the required dependencies.

!!! warning "uv support"

    UV support is still experimental

=== "Anaconda"
    ``` bash
    conda create -n marquette python=3.11
    conda activate marquette
    pip install -r requirements.txt
    ```
=== "uv"
    ``` bash
    uv venv
    uv pip sync requirements.txt
    ```

## Data

The whole purpose of Marquette is to format your hydrofabric data into a format that is usable for $\delta$MC river routing. To get the full formatted data, you will need:

### Hydrofabric Data

Data required for MERIT BASINS is available [here](https://www.reachhydro.org/home/records/google-drive-linux-instructions)

### Observational Data

!!! failure "Observational Data Availability"

    Observational point data will be made available upon release of the latest $\delta$MC code

### Streamflow Data

!!! failure "Streamflow Data Availability"

    Streamflow data will become available upon request

### Attribute Data

!!! failure "Attribute Data Availability"

    Streamflow data will become available upon request


## Running the module

To run the module, prepare a configuration file then run:

``` bash
python -m marquette
```

To run over multiple MERIT zones, run using hydra multirun

``` bash
python -m marquette --multirun zone=71,72,73
```



