# H-CAP: context-aware activity prediction using heterogeneous graphs

## Prerequisite
- Install Python 3.10
- Install requirements using pip: `pip install -r requirements.txt`

## How to run
To start code, run: `hyper_parameters_search.py`.
According to the methodology described in the paper, the code:
- reads the IGs file (`.g` in `dataset` folder);
- discovers, for each event, its set of contextual events;
- encodes the prefix-IGs and the contextual events;
- runs the hyper-parameter search of the GNN using Optuna.
