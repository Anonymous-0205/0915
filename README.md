# STAGE
## Introduction
This is the code repository for the double-blind review. Structure-adaptive graph-encoded policy gradient (STAGE) is a multi-agent reinforcement learning based algorithm tailored for MuRES problem in uncertain topological environments. The STAGE algorithm consists of two main modules: (1) Bi-scale GAT encoder, which fuses a $k$-hop local graph attention network (GAT) with a distance-augmented long-range GAT. The augmented graph is updated whenever newly observed edges shorten shortest-path distances, enabling the encoder to capture both neighborhood and long-range structural changes. (2) Entropy-regularized counterfactual policy gradient, in which a structure-aware centralized critic learns both team return and graph-information signals, and decentralized actors are trained via counterfactual marginalization with entropy regularization.
## Dependencies
- Python 3.9
- Pandas
- NumPy
- Matplotlib
- PyTorch
- tqdm
- torch_geometric
- collections

## Description
- `STAGE`  
  The example of training our STAGE algorithm.

- `search_utils.py`  
  Implements the training pipeline for multi-agent reinforcement learning (MARL) algorithms in a specified environment and multi-robot system (MRS) configuration.

- `classic_env.py`  
  The environment of running the MuRES problem under uncertainty.
