# Clustered FL Experiments

This repository is organized as a small Python package around clustered federated learning experiments on `FashionMNIST`.

## Layout

- `fl_cluster_bench/`: active source code
- `fl_cluster_bench/methods/`: experiment method runners
- `legacy/`: historical standalone scripts kept for reference
- `data/`: dataset cache
- `run_experiment.py`: thin root-level launcher

## Run

```powershell
python run_experiment.py --methods fedavg,loss_dpmm_cluster
```

Or:

```powershell
python -m fl_cluster_bench.cli --methods fedavg,loss_dpmm_cluster
```

## Notes

- `legacy/` scripts are not the primary entrypoints anymore.
- `methods/fesem.py` and `methods/cfl.py` are placeholders.
