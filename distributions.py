import theano.tensor as TT
import numpy as np

TINY = 1e-8

class Distribution(object):

    @property
    def dim(self):
        raise NotImplementedError

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two distributions
        """
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        raise NotImplementedError

    def entropy(self, dist_info):
        raise NotImplementedError

    def log_likelihood_sym(self, x_var, dist_info_vars):
        raise NotImplementedError

    def log_likelihood(self, xs, dist_info):
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        raise NotImplementedError


def from_onehot_sym(x_var):
    ret = TT.zeros((x_var.shape[0],), x_var.dtype)
    nonzero_n, nonzero_a = TT.nonzero(x_var)[:2]
    ret = TT.set_subtensor(ret[nonzero_n], nonzero_a.astype('uint8'))
    return ret


def from_onehot(x_var):
    ret = np.zeros((len(x_var),), 'int32')
    nonzero_n, nonzero_a = np.nonzero(x_var)
    ret[nonzero_n] = nonzero_a
    return ret


class Categorical(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two categorical distributions
        """
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * A
        return TT.sum(
            old_prob_var * (TT.log(old_prob_var + TINY) - TT.log(new_prob_var + TINY)),
            axis=-1
        )

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two categorical distributions
        """
        old_prob = old_dist_info["prob"]
        new_prob = new_dist_info["prob"]
        return np.sum(
            old_prob * (np.log(old_prob + TINY) - np.log(new_prob + TINY)),
            axis=-1
        )

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * A
        N = old_prob_var.shape[0]
        x_inds = from_onehot_sym(x_var)
        return (new_prob_var[TT.arange(N), x_inds] + TINY) / (old_prob_var[TT.arange(N), x_inds] + TINY)

    def entropy(self, info):
        probs = info["prob"]
        return -np.sum(probs * np.log(probs + TINY), axis=1)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        probs = dist_info_vars["prob"]
        # Assume layout is N * A
        x_var = from_onehot_sym(x_var)
        N = probs.shape[0]
        return TT.log(probs[TT.arange(N), x_var] + TINY)

    def log_likelihood(self, xs, dist_info):
        probs = dist_info["prob"]
        # Assume layout is N * A
        N = probs.shape[0]
        return np.log(probs[np.arange(N), from_onehot(np.asarray(xs))] + TINY)

    @property
    def dist_info_keys(self):
        return ["prob"]
        

class DiagonalGaussian(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_means = old_dist_info_vars["mean"]
        old_log_stds = old_dist_info_vars["log_std"]
        new_means = new_dist_info_vars["mean"]
        new_log_stds = new_dist_info_vars["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = TT.exp(old_log_stds)
        new_std = TT.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = TT.square(old_means - new_means) + \
                    TT.square(old_std) - TT.square(new_std)
        denominator = 2 * TT.square(new_std) + 1e-8
        return TT.sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return TT.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        means = dist_info_vars["mean"]
        log_stds = dist_info_vars["log_std"]
        zs = (x_var - means) / TT.exp(log_stds)
        return - TT.sum(log_stds, axis=-1) - \
               0.5 * TT.sum(TT.square(zs), axis=-1) - \
               0.5 * means.shape[-1] * np.log(2 * np.pi)

    def sample(self, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * means.shape[-1] * np.log(2 * np.pi)

    def entropy(self, dist_info):
        log_stds = dist_info["log_std"]
        return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    @property
    def dist_info_keys(self):
        return ["mean", "log_std"]