# Generalized Bayesian Likelihood-Free Inference Using Scoring Rules Estimators 

For the non SG-MCMC based experiments, please see [here](https://github.com/LoryPack/GenBayes_LikelihoodFree_ScoringRules). 

Code for the paper: __Generalized Bayesian Likelihood-Free Inference Using Scoring Rules Estimators__,
which can be found [here](https://arxiv.org/abs/2104.03889).

# Installation instructions
Note: Only tested for python 3.8.13

1. `pip install -r requirements.txt`
2. `pip install -e .`

# Notes
## `examples`
Please see examples folder for basic inference setups

# Code References
The files `src/sampler/gradient_estimation.py`, `src/sampler/gradient_estimation.py` and `src/sampler/gradient_estimation.py` are modified from the original repo located at:
https://github.com/jeremiecoullon/SGMCMCJax.
Original file copyright (c) 2022 Jeremie Coullon, licensed under the Apache 2.0 License.

The files `src/mamba/gridsearch.py`, `src/mamba/kdf.py`, `src/mamba/mambda.py`,`src/mamba/sampler.py` and `src/mamba/util.py` are modified from the original repo located at:
https://github.com/jeremiecoullon/SGMCMC_bandit_tuning.
Original file copyright (c) 2021 Jeremie Coullon.