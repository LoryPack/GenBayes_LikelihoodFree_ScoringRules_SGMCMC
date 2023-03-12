from collections import namedtuple
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import jit, lax, random

from sgmcmcjax.types import PRNGKey, PyTree, SamplerState, SVRGState
import torch
import numpy as np
from ..utils import transform_neural_lorenz_parameter, GradientNANException

def build_neural_l96_gradient_estimation_fn(
    joint_log_prob, data
) -> Tuple[Callable, Callable]:
    """Build a custom gradient estimator for l96 model

    Args:
        model (torch.nn.module): LFI model 
        data (Tuple): tuple of data. It should either have a single array (for unsupervised problems) or have two arrays (for supervised problems)

    Returns:
        Tuple[Callable, Callable]: gradient estimation function and gradient initialisation function
    """
    def init_gradient(key: PRNGKey, param: PyTree) -> Tuple[PyTree, SVRGState]:
        param_torch_unconstrained = torch.from_numpy(np.asarray(param, dtype=np.float64))
        param_torch_unconstrained = transform_neural_lorenz_parameter(param_torch_unconstrained)

        op = joint_log_prob(param_torch_unconstrained=param_torch_unconstrained, obs=data)
        op.backward()

        param_grad = torch.cat([param_torch_unconstrained[0].grad.reshape(-1)] + [p.grad.flatten() for p in param_torch_unconstrained[1]]).numpy()
        if np.isnan(param_grad).any():
            raise GradientNANException()
        param_grad = jnp.array(param_grad)

        return param_grad, SVRGState()
    
    def estimate_gradient(
        i: int, key: PRNGKey, param: PyTree, svrg_state: SVRGState = SVRGState()
    ) -> Tuple[PyTree, SVRGState]:
        return init_gradient(key, param)

    return estimate_gradient, init_gradient

def build_gradient_estimation_fn(
    joint_log_prob, data
) -> Tuple[Callable, Callable]:
    """Build a standard gradient estimator

    Args:
        model (torch.nn.module): LFI model 
        data (Tuple): tuple of data. It should either have a single array (for unsupervised problems) or have two arrays (for supervised problems)

    Returns:
        Tuple[Callable, Callable]: gradient estimation function and gradient initialisation function
    """
    def init_gradient(key: PRNGKey, param: PyTree) -> Tuple[PyTree, SVRGState]:
        param_torch_unconstrained = torch.from_numpy(np.asarray(param, dtype=np.float64))
        param_torch_unconstrained.requires_grad = True

        op = joint_log_prob(param_torch_unconstrained, data)
        op.backward()

        param_grad = param_torch_unconstrained.grad.numpy()
        if np.isnan(param_grad).any():
            raise GradientNANException()
        param_grad = jnp.array(param_grad)

        return param_grad, SVRGState()
    
    def estimate_gradient(
        i: int, key: PRNGKey, param: PyTree, svrg_state: SVRGState = SVRGState()
    ) -> Tuple[PyTree, SVRGState]:
        return init_gradient(key, param)

    return estimate_gradient, init_gradient