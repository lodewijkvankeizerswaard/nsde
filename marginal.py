"""
This file contains the marginal estimators. 
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from problems import MarginalProblem
from extra_dists import rejection_sampling_uniform

# MODEL_MAP can be found at the bottom of this file

# A map to get the normalizing flow from command line arguments
FLOW_MAP = {
    'planar': T.planar,
    'spline': T.spline,
    'spline-coupling': T.spline_coupling,
    'polynomial': T.polynomial,
    'sylvester': T.sylvester,
    'affine-coupling': T.affine_coupling,
    'affine-autoregressive': T.affine_autoregressive,
    'spline-autoregressive': T.spline_autoregressive,
}

class MarginalEstimator(nn.Module):
    """ Base class for marginal estimators. """
    def __init__(self):
        super().__init__()

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, *args) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def add_arguments():
        arguments = {
            "--marginal": {
                "type": str,
                "default": "spline",
                "help": "Type of flow to use",
            },
            "--marginal-layers": {
                "type": int,
                "default": 5,
                "help": "Number of flows to use",
            },
        }
        return arguments

    def base_dist(self) -> dist.Distribution:
        base_dist_shape = tuple(self.param_shape)
        return dist.MultivariateNormal(torch.zeros(base_dist_shape, device=self.device()), 
                                torch.eye(base_dist_shape[0], device=self.device()))

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

class Naive(MarginalEstimator):
    """ Normalizing flow model used to directly model data or parameters.
        Only has a marginal property, which can either model the observed data,
        or the parameters, by sampling from the marginal and using those for the
        the parameters of the problem likelihood function. """
    def __init__(self, args, problem: MarginalProblem):
        super().__init__()
        if args.loss_function == "marginal":
            # We are learning the data directly
            self.flow_shape = problem.event_shape
        elif args.loss_function == "likelihood" or args.loss_function == "default":
            # We are learning the parameters directly
            self.flow_shape = problem.param_shape

        flow_type = FLOW_MAP[args.marginal]

        self.flow_list = [flow_type(input_dim=self.flow_shape[0]) for _ in range(args.marginal_layers)]
        self.marginal_module = T.ComposeTransformModule(self.flow_list)  # Register parameters
        self.marginal = dist.TransformedDistribution(self.base_dist(), self.flow_list)

    def to(self, device: torch.device):
        for flow in self.flow_list:
            flow.to(device)
        super().to(device)
        self.marginal.base_dist = self.base_dist()
        return self

    def device(self):
        return next(self.flow_list[0].parameters()).device
    
    def base_dist(self) -> dist.Distribution:
        base_dist_shape = tuple(self.flow_shape)
        return dist.MultivariateNormal(torch.zeros(base_dist_shape, device=self.device()), 
                                torch.eye(base_dist_shape[0], device=self.device()))


    def clear_cache(self):
        for flow in self.flow_list:
            flow.clear_cache()

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return self.marginal.log_prob(d)
    
    @staticmethod
    def add_arguments():
        # Unbound use of super: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        arguments = super(Naive, Naive).add_arguments() 
        return arguments

class VAEFlow(MarginalEstimator):
    """ Normalizing flow model based on a VAE used to model the posterior distribution over theta. """
    has_rs_theta = True
    def __init__(self, args, problem: MarginalProblem):
        super().__init__()
        self.param_shape = problem.param_shape
        self.data_shape = problem.event_shape
        x_dim = np.prod(*problem.event_shape)
        p_dim = np.prod(*problem.param_shape)

        self._build_marginal(args, p_dim)
        self._build_posterior(args, x_dim, p_dim)

    def _build_marginal(self, args, p_dim):
        
        if args.marginal in FLOW_MAP:
            marginal_type = FLOW_MAP[args.marginal]
            self.marginal_list = [marginal_type(p_dim) for _ in range(args.marginal_layers)]
            self.marginal_module = T.ComposeTransformModule(self.marginal_list)  # Register parameters
            self.marginal = dist.TransformedDistribution(self.base_dist(), self.marginal_list)

        else:
            raise NotImplementedError("Marginal type not implemented: {}".format(args.marginal))

    def _build_posterior(self, args, x_dim, p_dim):
        if args.posterior == "dense":
            hidden_layers = [32 for _ in range(args.posterior_layers)]
            self.posterior_module = pyro.nn.DenseNN(x_dim, hidden_layers, param_dims=[p_dim, p_dim])
            self.posterior_type = "dense"
        
        elif args.posterior in FLOW_MAP:
            assert x_dim == p_dim, "Posterior flow must have same input and output dimension"
            posterior_type = FLOW_MAP[args.posterior]
            self.post_list = [posterior_type(p_dim) for _ in range(args.posterior_layers)]
            self.posterior_module = T.ComposeTransformModule(self.post_list)
            self.posterior = dist.TransformedDistribution(self.base_dist(), self.post_list)
            self.posterior_type = "flow"

        else:
            raise NotImplementedError("Posterior type not implemented: {}".format(args.posterior))

    def clear_cache(self):
        if hasattr(self, "marginal_list"):
            for flow in self.marginal_list:
                flow.clear_cache()
        if hasattr(self, "post_list"):
            for flow in self.post_list:
                flow.clear_cache()

    def to(self, device):
        super().to(device)
        if hasattr(self, "marginal"):
            base_dist = self.base_dist()
            self.marginal.base_dist = base_dist
        super().to(device)
    
    @staticmethod
    def add_arguments():
        # Unbound use of super: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        arguments = super(VAEFlow, VAEFlow).add_arguments() 
        arguments = arguments | {
            "--posterior": {
                "type": str,
                "default": "dense",
                "choices": ["dense"] + list(FLOW_MAP.keys())
            },
            "--posterior-layers": {
                "type": int,
                "default": 2
            }
        }
        return arguments

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Compute the posterior conditioned on x
        if self.posterior_type == "dense":
            theta_mu, theta_sigma = self.posterior_module(data)
            theta_sigma = torch.nn.functional.softplus(theta_sigma)

            nan_mask = ~torch.any(theta_mu.isnan(),dim=1)
            if not torch.all(nan_mask):
                print("[Warning] NaNs in VAE forward (theta_mu): {} of {}".format(torch.sum(~nan_mask), theta_mu.shape[0]))
                theta_mu = theta_mu[nan_mask]
                theta_sigma = theta_sigma[nan_mask]

            q_theta_x = dist.Normal(theta_mu, theta_sigma)

            theta_x = q_theta_x.rsample()

        elif self.posterior_type == "flow":
            theta_x = self.posterior_module(data)

        # Sample from the posterior
        return theta_x

    def rs_theta(self, data, N) -> torch.Tensor:
        """
        Computes N samples from the posterior distribution over theta using rejection sampling.

        Args:
            data (torch.Tensor): The observed data.
            N (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor of shape (N, param_shape[0]) containing the generated samples.
        """
        if self.posterior_type == "dense":
            # Get a random subset of the data
            perm = torch.randperm(data.shape[0])
            idx = perm[:N]

            # Compute the posterior conditioned on x
            theta_mu, theta_sigma = self.posterior(data[idx])
            theta_sigma = torch.nn.functional.softplus(theta_sigma)

            # Create callable for the log probability of the posterior
            q_theta_x = dist.Normal(theta_mu, theta_sigma)
            fn_theta = lambda theta: q_theta_x.log_prob(theta)

        elif self.posterior_type == "flow":
            # Since the posterior is a flow, we can use the log probability of the flow
            # We do not need to precompute the posterior based on a number of samples
            post = self.posterior.copy()
            fn_theta = lambda theta: post.log_prob(theta)

        rs_theta = rejection_sampling_uniform(fn_theta, (-5, 5), N, self.param_shape[0], log_prob=True)
        return rs_theta


    def plot_posterior_vectorfield(self, grid):
        """
        Plots the posterior vector field for the current model and the given grid.

        Args:
            grid (torch.Tensor): A 2D tensor of shape (n_points, 2) representing the grid of points
                where the posterior will be evaluated.

        Returns:
            numpy.ndarray: A 4D numpy array of shape (1, height, width, 3) representing the plotted image.
        """
        if self.posterior_type == "dense":
            theta_mu, theta_sigma = self.posterior_module(grid)
            theta_sigma = torch.nn.functional.softplus(theta_sigma)

            theta_mu = theta_mu.detach().cpu().numpy()
            theta_sigma = theta_sigma.detach().cpu().numpy()

            vector_value = theta_mu
            vector_color = theta_sigma.sum(axis=1)

        elif self.posterior_type == "flow":
            transformed_dist = self.posterior_module(grid)
            vector_value = transformed_dist.detach().cpu().numpy()
            vector_color = np.ones(grid.shape[0])

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].set_title("Absolute posterior vectors")
        ax[1].set_title("Relative posterior vectors")

        ax[0].quiver(grid[:, 0], 
                grid[:, 1], 
                vector_value[:, 0], 
                vector_value[:, 1], 
                vector_color,
                scale=1, 
                scale_units='xy', 
                angles='xy', 
                color='r', 
                alpha=0.5)
        vector_value_rel = vector_value - grid.numpy()
        ax[1].quiver(grid[:, 0],
                    grid[:, 1],
                    vector_value_rel[:, 0],
                    vector_value_rel[:, 1],
                    vector_color,
                    scale=1,
                    scale_units='xy',
                    angles='xy',
                    color='r',
                    alpha=0.5)
        

        # plt.colorbar(sm)
        fig.suptitle(f"Posterior vector field {self.posterior_type}")

        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        return image


MODEL_MAP = {
    "naive": Naive,
    "vae-flow": VAEFlow,
}
    