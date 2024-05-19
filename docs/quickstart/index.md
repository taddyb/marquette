# Quick Start:

## Prerequisites

The first step to using Marquette is creating a Python environment with all of the required dependencies.

!!! warning "uv support"

    UV support is still experimental

=== "Anaconda"
    ```sh
    conda create -n marquette python=3.11
    conda activate marquette
    pip install -r requirements.txt
    ```
=== "uv"
    ```sh
    uv venv
    uv pip sync requirements.txt
    ```

## Data

### Hydrofabric:

Data required for MERIT BASINS is available [here](https://www.reachhydro.org/home/records/google-drive-linux-instructions)

!!! failure "Observational Data Availability"

    Observational point data will be made available upon release of the latest $\delta$MC code

!!! failure "Streamflow Data Availability"

    Streamflow data will become available upon request

## Running the module

To run the module, prepare a configuration file then run:

```sh
python -m marquette
```

To run over multiple MERIT zones, run using hydra multirun

```sh
python -m marquette --multirun zone=71,72,73
```



