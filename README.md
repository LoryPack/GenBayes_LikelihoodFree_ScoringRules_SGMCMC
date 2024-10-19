# Generalized Bayesian Likelihood-Free Inference Using Scoring Rules Estimators 

For the non SG-MCMC based experiments, please see [here](https://github.com/LoryPack/GenBayes_LikelihoodFree_ScoringRules). 

Code for the paper: __Generalized Bayesian Likelihood-Free Inference Using Scoring Rules Estimators__,
which can be found [here](https://doi.org/10.1214/24-EJS2283).

# Installation instructions
Note: Only tested for python 3.8.13

1. `pip install -r requirements.txt`
2. `pip install -e .`

# Experiments
1. Exp-1 : Univariate g-and-k model, Figures 1 and 2
2. Exp-2 : Univariate g-and-k model (Concentration), Figure 3
3. Exp-3 : Multivariate g-and-k model (Well-specified, misspecified concentrations), Figures 4 and 5
4. Exp-4 : Linear Lorenz-96 model, Figure 6
5. Exp-5 : Neural Lorenz-96 model, Figure 7

# Code References
The files `src/sampler/gradient_estimation.py`, `src/sampler/gradient_estimation.py` and `src/sampler/gradient_estimation.py` are modified from the original repo located at:
https://github.com/jeremiecoullon/SGMCMCJax.
Original file copyright (c) 2022 Jeremie Coullon, licensed under the Apache 2.0 License.

The files `src/mamba/gridsearch.py`, `src/mamba/kdf.py`, `src/mamba/mambda.py`,`src/mamba/sampler.py` and `src/mamba/util.py` are modified from the original repo located at:
https://github.com/jeremiecoullon/SGMCMC_bandit_tuning.
Original file copyright (c) 2021 Jeremie Coullon.

# Citation
Please use the following `.bib` entry:
```bibtex
@article{pacchiardi2024generalized,
        author = {Lorenzo Pacchiardi and Sherman Khoo and Ritabrata Dutta},
        title = {{Generalized Bayesian likelihood-free inference}},
        volume = {18},
        journal = {Electronic Journal of Statistics},
        number = {2},
        publisher = {Institute of Mathematical Statistics and Bernoulli Society},
        pages = {3628 -- 3686},
        year = {2024},
        doi = {10.1214/24-EJS2283},
        URL = {https://doi.org/10.1214/24-EJS2283}
}
```
