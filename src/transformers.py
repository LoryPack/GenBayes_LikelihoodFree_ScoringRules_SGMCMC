import numpy as np
import torch

# The transformers are used in the MCMC inference scheme, in order to run MCMC of an unbounded transformed
# space in case the original space is bounded. It therefore also implements the jacobian terms which appear in
# the acceptance rate.


class BoundedVarTransformer:
    """
    See ref: https://mc-stan.org/docs/2_27/reference-manual/logit-transform-jacobian-section.html

    This scaler implements both lower bounded and two sided bounded transformations according to the provided bounds.
    It works on 1d vectors. You need to specify separately the lower and upper bounds in two arrays with the same length
    of the objects on which the transformations will be applied (likely the parameters on which MCMC is conducted for
    this function).

    Note for Sherman: this works on numpy arrays; either we need to convert torch -> numpy and back, or we re-write
    this such that it works on torch directly. Up to you.

    If the bounds for a given variable are both None, it is assumed to be unbounded; if instead the
    lower bound is given and the upper bound is None, it is assumed to be lower bounded. Finally, if both bounds are
    given, it is assumed to be bounded on both sides.
    """

    def __init__(self, lower_bound, upper_bound):
        """
        Parameters
        ----------
        lower_bound : np.ndarray
            Array of the same length of the variable to which the transformation will be applied, containing lower
            bounds of the variable. Each entry of the array can be either None or a number (see above).
        upper_bound
            Array of the same length of the variable to which the transformation will be applied, containing upper
            bounds of the variable. Each entry of the array can be either None or a number (see above).
        """
        # upper and lower bounds need to be numpy arrays with size the size of the variable
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if not hasattr(lower_bound, "shape") or not hasattr(upper_bound, "shape"):
            raise RuntimeError("Provided lower and upper bounds need to be arrays.")
        elif hasattr(lower_bound, "shape") and hasattr(upper_bound, "shape") and lower_bound.shape != upper_bound.shape:
            raise RuntimeError("Provided lower and upper bounds need to have same shape.")

        # note that == None checks if the array is None element wise.
        self.unbounded_vars = np.logical_and(np.equal(lower_bound, None), np.equal(upper_bound, None))
        self.lower_bounded_vars = np.logical_and(np.not_equal(lower_bound, None), np.equal(upper_bound, None))
        self.upper_bounded_vars = np.logical_and(np.equal(lower_bound, None), np.not_equal(upper_bound, None))
        self.two_sided_bounded_vars = np.logical_and(np.not_equal(lower_bound, None), np.not_equal(upper_bound, None))
        if self.upper_bounded_vars.any():
            raise NotImplementedError("We do not yet implement the transformation for upper bounded random variables")

        self.lower_bound_lower_bounded = self.lower_bound[self.lower_bounded_vars].astype("float32")
        self.lower_bound_two_sided = self.lower_bound[self.two_sided_bounded_vars].astype("float32")
        self.upper_bound_two_sided = self.upper_bound[self.two_sided_bounded_vars].astype("float32")

    @staticmethod
    def logit(x):
        return np.log(x) - np.log(1 - x)

    def _check_data_in_bounds(self, X):
        # Takes as input 1d or 2d arrays
        X = np.atleast_2d(X)  # convert to 2d if needed
        if np.any(X[:, self.lower_bounded_vars] <= self.lower_bound_lower_bounded):
            raise RuntimeError("The provided data are out of the bounds.")
        if (X[:, self.two_sided_bounded_vars] <= self.lower_bound[self.two_sided_bounded_vars]).any() or (
                X[:, self.two_sided_bounded_vars] >= self.upper_bound_two_sided).any():
            raise RuntimeError("The provided data is out of the bounds.")

    def _apply_nonlinear_transf(self, X):
        # apply the different transformations to the different kind of variables. Takes as input 1d or 2d arrays
        squeeze = len(X.shape) == 1
        X = np.atleast_2d(X)
        X_transf = X.copy()
        X_transf[:, self.lower_bounded_vars] = np.log(X[:, self.lower_bounded_vars] - self.lower_bound_lower_bounded)
        X_transf[:, self.two_sided_bounded_vars] = self.logit(
            (X[:, self.two_sided_bounded_vars] - self.lower_bound_two_sided) / (
                    self.upper_bound_two_sided - self.lower_bound_two_sided))
        return X_transf.squeeze() if squeeze else X_transf

    def _apply_inverse_nonlinear_transf(self, X):
        # inverse transformation. Different trasformations applied to different kind of variables.
        # Takes as input 1d or 2d arrays
        squeeze = len(X.shape) == 1
        X = np.atleast_2d(X)
        inv_X = X.copy()
        inv_X[:, self.two_sided_bounded_vars] = (self.upper_bound_two_sided - self.lower_bound_two_sided) * np.exp(
            X[:, self.two_sided_bounded_vars]) / (1 + np.exp(
            X[:, self.two_sided_bounded_vars])) + self.lower_bound_two_sided
        inv_X[:, self.lower_bounded_vars] = np.exp(X[:, self.lower_bounded_vars]) + self.lower_bound_lower_bounded
        return inv_X.squeeze() if squeeze else inv_X

    def _jac_log_det(self, x):
        # computes the jacobian log determinant. Takes as input arrays.
        results = np.zeros_like(x)
        results[self.two_sided_bounded_vars] = np.log(
            (self.upper_bound_two_sided - self.lower_bound_two_sided).astype("float64") / (
                    (x[self.two_sided_bounded_vars] - self.lower_bound_two_sided) * (
                    self.upper_bound_two_sided - x[self.two_sided_bounded_vars])))
        results[self.lower_bounded_vars] = - np.log(x[self.lower_bounded_vars] - self.lower_bound_lower_bounded)
        return np.sum(results)

    def _jac_log_det_inverse_transform(self, x):
        # computes the log determinant of jacobian evaluated in the inverse transformation. Takes as input arrays.
        results = np.zeros_like(x)
        results[self.lower_bounded_vars] = - x[self.lower_bounded_vars]
        # two sided: need some tricks to avoid numerical issues:
        results[self.two_sided_bounded_vars] = - np.log(
            self.upper_bound_two_sided - self.lower_bound_two_sided)

        indices = x[self.two_sided_bounded_vars] < 100  # for avoiding numerical overflow
        res_b = np.copy(x)[self.two_sided_bounded_vars]
        res_b[indices] = np.log(1 + np.exp(x[self.two_sided_bounded_vars][indices]))
        results[self.two_sided_bounded_vars] += res_b

        indices = x[self.two_sided_bounded_vars] > - 100  # for avoiding numerical overflow
        res_c = np.copy(- x)[self.two_sided_bounded_vars]
        res_c[indices] = np.log(1 + np.exp(- x[self.two_sided_bounded_vars][indices]))
        results[self.two_sided_bounded_vars] += res_c

        # res = res_b + res_c - res_a

        return np.sum(results)

    @staticmethod
    def _array_from_list(x):
        return np.array(x).reshape(-1)

    @staticmethod
    def _list_from_array(x_arr, x):
        # transforms the array x to the list structure that contains x
        x_new = [None] * len(x)
        for i in range(len(x)):
            if isinstance(x[i], np.ndarray):
                x_new[i] = np.array(x_arr[i].reshape(x[i].shape))
            else:
                x_new[i] = x_arr[i]
        return x_new

    def transform(self, x, use_torch=False):
        """Scale features of x according to feature_range.

        Parameters
        ----------
        x : list of length n_parameters
            Input data that will be transformed.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        if use_torch:
            bounds = torch.tensor(np.stack([self.lower_bound, self.upper_bound],axis=1))
            diff = bounds[:,1] - bounds[:,0]
            return torch.special.logit((x-bounds[:,0])/diff)
        else:
            # convert data to array:
            x_arr = self._array_from_list(x)

            # need to check if we can apply the log first:
            self._check_data_in_bounds(x_arr)

            # we transform the data with the log transformation:
            x_arr = self._apply_nonlinear_transf(x_arr)

            # convert back to the list structure:
            x = self._list_from_array(x_arr, x)

            return x

    def inverse_transform(self, x, use_torch=False):
        """Undo the scaling of x according to feature_range.

        Parameters
        ----------
        x : list of len n_parameters
            Input data that will be transformed. It cannot be sparse.
        OR if use_torch=True
            Torch.tensor


        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        # now apply the inverse transform

        if use_torch:
            if np.isnan(self.lower_bound.astype(float)).any() or np.isnan(self.upper_bound.astype(float)).any():
                raise TypeError("Not implemented for unbounded!")

            bounds = torch.tensor(np.stack([self.lower_bound, self.upper_bound],axis=1))
            diff = bounds[:,1] - bounds[:,0]
            return bounds[:,0] + diff * torch.special.expit(x)

        else:
            x_arr = self._array_from_list(x)
            inv_x = self._apply_inverse_nonlinear_transf(x_arr)

            # convert back to the list structure:
            inv_x = self._list_from_array(inv_x, x)

            return inv_x

    def jac_log_det(self, x):
        """Returns the log determinant of the Jacobian: :math:`\log |J_t(x)|`.

        Parameters
        ----------
        x : list of len n_parameters
            Input data, living in the original space (with optional bounds).
        Returns
        -------
        res : float
            log determinant of the jacobian
        """
        x = self._array_from_list(x)
        self._check_data_in_bounds(x)

        return self._jac_log_det(x)

    def jac_log_det_inverse_transform(self, x, use_torch=False):
        """Returns the log determinant of the Jacobian evaluated in the inverse transform:
        :math:`\log |J_t(t^{-1}(x))| = - \log |J_{t^{-1}}(x)|`

        Parameters
        ----------
        x : list of len n_parameters
            Input data, living in the transformed space (spanning the whole :math:`R^d`).

        OR if use_torch=True
            Torch.tensor

        Returns
        -------
        res : float
            log determinant of the jacobian evaluated in :math:`t^{-1}(x)`
        """

        if use_torch:
            return torch.log(
                    torch.abs(
                        torch.det(
                            torch.autograd.functional.jacobian(lambda x: self.inverse_transform(x, use_torch=True),
                                                                x,
                                                                create_graph=True)
                                )
                            )
                        )
        else:
            x = self._array_from_list(x)
            return self._jac_log_det_inverse_transform(x)

class DummyTransformer:
    """Dummy transformer which does nothing, and for which the jacobian is 1"""

    def __init__(self):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x, use_torch=False):
        return x

    def jac_log_det(self, x):
        return 0

    def jac_log_det_inverse_transform(self, x):
        return 0

    def log_det_gradient(self, x):
        """Needed for running SGMCMC on transformed spaces"""
        return 0