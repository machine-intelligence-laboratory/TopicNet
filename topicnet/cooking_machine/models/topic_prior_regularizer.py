import numpy as np
import warnings
from .base_regularizer import BaseRegularizer


class TopicPriorRegularizer(BaseRegularizer):
    """
    TopicPriorRegularizer adds prior beta_t to every column
    in Phi matrix of ARTM model. Thus every phi_wt has
    preassigned prior probability of being attached to topic t.

    If beta is balanced with respect to apriori collection balance,
    topics become better and save n_t balance.

    """  # noqa: W291
    def __init__(self, name, tau, num_topics=None, beta=1):
        """

        Parameters
        ----------
        name : str
            Regularizer name
        tau : float
            Regularizer influence degree
        num_topics : int
            Number of topics for uniform sampling
        beta : float or list or np.array
            Prior for columns of Phi matrix (Default value = 1)

        """
        super().__init__(name, tau)

        beta_is_n_dim = isinstance(beta, (list, np.ndarray))
        if beta_is_n_dim and (num_topics is not None) and len(beta) != num_topics:
            raise ValueError('Beta dimension doesn\'t equal num_topics.')
        if num_topics is None and not beta_is_n_dim:
            warnings.warn('Num topics set to 1.')
            num_topics = 1

        if beta_is_n_dim:
            if np.sum(np.array(beta)) == 0:
                raise ValueError('Incorrect input beta: at least one value must be greater zero.')
            if np.min(np.array(beta)) < 0:
                raise ValueError('Incorrect input beta: all values must be greater or equal zero.')

            self.beta = np.array(beta)
            self.beta = self.beta / np.sum(self.beta)
        else:
            self.beta = np.ones(num_topics)

    def grad(self, pwt, nwt):
        grad_array = np.repeat([self.beta * self.tau], pwt.shape[0], axis=0)

        return grad_array


class TopicPriorSampledRegularizer(BaseRegularizer):
    """
    TopicPriorSampleRegularizer adds prior beta_t to every column
    in Phi matrix of ARTM model. Thus every phi_wt has
    preassigned prior probability of being attached to topic t.

    Beta vector is sampled from
    Dirichlet distribution with parameter beta_prior.
    By varying beta_prior one can apply different degrees of balance to model.
    Beta_prior influence:
        1 - fully random balance
        << 1 - uniform distribution of topics size
        >> 1 - highly unbalanced distribution of topics size

    If beta is balanced with respect to apriori collection balance,
    topics become better and save n_t balance.

    """  # noqa: W291
    def __init__(self, name, tau, num_topics=None, beta_prior=(), random_seed=42):
        """

        Parameters
        ----------
        name : str
            Regularizer name
        tau : float
            Regularizer influence degree
        num_topics : int
            Number of topics for uniform sampling
        beta_prior : list or np.array
            Prior for Dirichlet distribution to sample beta parameter
        random_seed : int
            Random seed for Dirichlet distribution (Default value = 42)

        """
        super().__init__(name, tau)

        if num_topics is None and len(beta_prior) == 0:
            warnings.warn('Num topics set to 1.')
            num_topics = 1

        beta_prior_is_n_dim = isinstance(beta_prior, (list, np.ndarray))
        if len(beta_prior) != 0 and beta_prior_is_n_dim:
            if np.sum(np.array(beta_prior)) == 0:
                raise ValueError(
                    'Incorrect input beta_prior: at least one value must be greater zero.'
                )
            if np.min(np.array(beta_prior)) < 0:
                raise ValueError(
                    'Incorrect input beta_prior: all values must be greater or equal zero.'
                )

            self.beta = np.random.RandomState(random_seed).dirichlet(beta_prior)
        else:
            self.beta = np.random.RandomState(random_seed).dirichlet([1 for _ in range(num_topics)])

    def grad(self, pwt, nwt):
        grad_array = np.repeat([self.beta * self.tau], pwt.shape[0], axis=0)

        return grad_array
