# LS-GFN
Official Code for Local Search GFlowNets (ICLR 2024 Spotlight)


### Environment Setup
To install dependecies, please run the command `pip install -r requirement.txt`.
Note that python version should be < 3.8 for running RNA-Binding tasks. You should install `pyg` with the following command
```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

### Code references
Our implementation is based on "Towards Understanding and Improving GFlowNet Training" (https://github.com/maxwshen/gflownet). 

### Our contribution (in terms of codes)

We extend codebase with RNA-binding tasks designed by FLEXS (https://github.com/samsinai/FLEXS) 

We implement detailed balance (DB), sub-trajectory balance (SubTB), and our method LS-GFN on top of DB, SubTB, TB, MaxEnt and GTB. 

We also implement various state-of-the-art baselines including RL approaches (A2C-Entropy, SQL, PPO) and recent MCMC approaches (MARS)

### Large files

You can download additional large files by following link: https://drive.google.com/drive/folders/1JobUWGowoiQxGWVz3pipdfcjhtdY4CVq?usp=sharing

These files should be placed in `datasets`


### Main Experiments
You can run the following command to validate the effectiveness of LS-GFN on various biochemical tasks.
As a default setting, we choose TB as a training objective and apply deterministic filtering to determine whether to accept or reject refined samples.

```
# LS-GFN
python main.py --setting qm9str --model gfn --ls --seed 0

# GFN (TB)
python main.py --setting qm9str --model gfn --seed 0
```

### Other Baselines
Beyond GFN baselines, we also implement reward-maximization methods as baselines. Baselines can be executed by setting `--model` option.
- Available Options: `mars, a2c, sql, ppo`
```
# MARS
python main.py --setting tfbind8 --model mars --seed 0

# A2C
python main.py --setting tfbind8 --model a2c --seed 0

# Soft Q-Learning
python main.py --setting tfbind8 --model sql --seed 0

# PPO
python main.py --setting tfbind8 --model ppo --seed 0
```

### Additional Experiments
You can change various biochemical tasks to evaluate the performance of LS-GFN by setting `--setting` option.
- Available Options: `qm9str, sehstr, tfbind8, rna1, rna2, rna3`
```
python main.py --setting <setting> --model gfn --ls --seed 0
```

You can change GFlowNet training objectives to evaluate the performance of Logit-GFN by setting `--loss_type` option.
- Available Options: `TB (Default), MaxEnt, DB, SubTB, GTB`
```
python main.py --setting qm9str --model gfn --ls --loss_type <loss_type> --seed 0
```

You can change the filtering strategies during local search by setting `--filtering` option.
- Available Options: `deterministic (Default), stochastic`
```
python main.py --setting qm9str --model gfn --ls --filtering <filtering_strategies> --seed 0
```

You can adjust the number of iterations per batch ($I$) by setting `num_iterations` option (Default: 7).
```
python main.py --setting qm9str --model gfn --ls --num_iterations <I> --seed 0
```

You can adjust the number of backtracking and reconstruction steps ($K$) by setting `num_back_forth_steps` option. (Default: $K=\lfloor(L+1)/2\rfloor$)
```
python main.py --setting qm9str --model gfn --ls --num_back_forth_steps <K> --seed 0
```
