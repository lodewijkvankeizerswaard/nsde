# Source Distribution Estimation using Normalizing Flows
This repository contains the source code used to produce the results of Part 1 of the master thesis: ``Population-Level Density Estimation Using Normalizing Flows and Hyper Posteriors".

## Installation
To install the environment run:
```
conda env create -f environment.yml
```
To use the wandb logger, also install wandb:
```
pip install wandb==0.13.5
```

## Usage
To run the experiments, use the `main.py` script. For example, to train a normalizing flow on the 2D simulators using affine autoregressive normalizing flows, run:
```
python main.py --problem 2d --loss-function marginal --model naive --marginal affine-autoregressive --device cuda --problem2d-marginal mixture --marginal-layers 12
```

To run multiple experiments, use the `batch_run.py` script. For example, to train a normalizing flow on the 2D simulators using affine autoregressive normalizing flows, run:
```
python batch_run.py --run_file runfiles/example_runfile.md
```


## Files
The following files are included in this repository:
- `batch_run.py`: A script to submit multiple experiment runs to a cluster.
- `main.py`: The main file to run the experiments.
- `training.py`: The training procedures and loss functions.
- `marginals.py`: The model definitions.
- `problems.py`: The simulator definitions.
- `roc_auc.py`: The ROC AUC metric.
- `utils.py`: Utility functions.
- `extra_dists.py`: Extra distributions for various purposes.
- `get_plots.py`: A script to generate the plots from the paper (you need to train your own models).
- `plot_marginals.ipynb`: A notebook to plot the marginals of the simulators.
- `environment.yml`: The conda environment file.


