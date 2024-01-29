"""
This file contains the simulators used in the project. The problems are used to
generate data, and to compare the true marginal with the approximate marginal.
"""


from __future__ import annotations

import numpy as np
import torch
import torch.distributions as dist

import seaborn as sns
import matplotlib.pyplot as plt

from extra_dists import MixtureMarginal, ClippedUniform, TorusMarginal, MixtureMarginal2D, GaussianNoiseLikelihood, GaussianAbsoluteLikelihood

# PROBLEM_MAP can be found at the bottom of this file

color_map = {
    'true_marginal': 'black',
    'approximate_marginal': 'tab:blue',
    'data': 'blue',
    'cmap': 'Blues'
}

class MarginalProblem(object):
    """ A base class for the marginal learning problems used in this project.
        These problems include a way to generate data, and to compare the
        approximate marginal with the true marginal. """
    _event_shape = torch.Size([1])
    _param_shape = torch.Size([1])

    def __init__(self):
        pass

    @property
    def event_shape(self) -> torch.Size:
        return self._event_shape

    @property
    def param_shape(self) -> torch.Size:
        return self._param_shape

    @staticmethod
    def construct_problem(args) -> MarginalProblem:
        raise NotImplementedError

    def likelihood(self, theta: torch.Tensor) -> dist.Distribution:
        """ The likelihood of the data given the parameters. """
        raise NotImplementedError

    def log_prob(self, d: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """ The conditional log probability of the data given the parameters. """
        raise NotImplementedError

    def generate_data(self, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        """ Generate data from the problem. """
        N = torch.Size([N]) if isinstance(N, int) else N
        theta = self.p_theta.sample(N)
        d = self.likelihood(theta).sample()
        d = d.view(N + self.event_shape)
        theta = theta.view(N + self.param_shape)
        return theta, d

    @staticmethod
    def add_arguments():
        """ Add problem specific arguments to the parser. The default is to add
            no arguments. """
        return {}

    @staticmethod
    def fig_to_array(fig) -> np.ndarray:
        """Convert a Matplotlib figure to a 4D numpy array with NHWC channels"""
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        return image
    
class Problem1D(MarginalProblem):
    """The goal of this problem is to recover a mariginal distribution
       over the parameter. """
    name = "Problem1D"
    _event_shape = torch.Size([1])
    _param_shape = torch.Size([1])
    
    def __init__(self, p_theta: dist.Distribution):
        # Settings for this problem
        super().__init__()
        self.p_theta = p_theta

    @staticmethod
    def construct_problem(args) -> MarginalProblem:
        if args.problem1d_marginal == "mixture" or args.problem1d_marginal == "default":
            pi = torch.tensor([0.5, 0.5])
            mu = torch.tensor([-1.0, 1.0])
            sigma = torch.tensor([0.5, 0.5])
            p_theta = MixtureMarginal(pi, mu, sigma)
        elif args.problem1d_marginal == "uniform":
            p_theta = ClippedUniform(-5, 5)
        else:
            raise ValueError(f"Unknown problem1d-marginal {args.problem1d_marginal}.")
        
        p = Problem1D(p_theta)
        if args.problem1d_likelihood == "gaussian-absolute":
            likelihood = GaussianAbsoluteLikelihood(p.event_shape, sigma=args.problem1d_likelihood_std)
        elif args.problem1d_likelihood == "gaussian-noise" or args.problem1d_likelihood == "default":
            likelihood = GaussianNoiseLikelihood(p.event_shape, sigma=args.problem1d_likelihood_std)
        else:
            raise ValueError(f"Unknown problem1d-likelihood {args.problem1d_likelihood}.")
        
        p._likelihood = likelihood
        return p

    def likelihood(self, theta: torch.Tensor) -> dist.Distribution:
        p_d_theta = self._likelihood.likelihood(theta)
        return p_d_theta

    def log_prob(self, d: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        log_p_d_theta = self.likelihood(theta).log_prob(d)
        return log_p_d_theta

    def log_prob_true(self, theta: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        log_p_d_theta = self.likelihood(theta).log_prob(d)
        log_p_theta = self.p_theta.log_prob(theta)
        return log_p_d_theta + log_p_theta

    @staticmethod
    def add_arguments():
        # Unbound use of super: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        arguments = super(Problem1D, Problem1D).add_arguments()
        arguments = arguments | {
            "--problem1d-marginal": {
                "type": str,
                "default": "default",
                "help": f"The marginal of the problem.",
                "choices": ["default", "mixture", "uniform"],
            },
            "--problem1d-likelihood": {
                "type": str,
                "default": "default",
                "help": f"The likelihood of the problem.",
                "choices": ["default", "gaussian-absolute", "gaussian-noise"],
            },
            "--problem1d-likelihood-std": {
                "type": float,
                "default": 0.1,
                "help": f"The standard deviation of the likelihood.",
            },
        }
        return arguments

    ### Plotting functions
    def plot_problem(self, nr_samples: int = 1000):
        global color_map

        theta = self.p_theta.sample(torch.Size([nr_samples]))
        data = self.likelihood(theta).sample().cpu().detach().numpy()
        theta = theta.cpu().detach().numpy()

        sns.kdeplot(theta, label=r"$p(\theta)$", color=color_map['true_marginal'])
        sns.kdeplot(data, label=r"$p(d|\theta)$", color=color_map['data'])
        plt.title("Problem1D")
        plt.legend()
        plt.show()

    def marginal_comparison(self,
                            marginal: dist.Distribution,
                            draft: bool = False,
                            plot: bool = False,
                            return_fig: bool = False
                            ) -> np.ndarray:
        """Plot the true marginal and the approximate marginal"""
        global color_map
        # First check if we can sample from the marginal
        try:
            approx_samples = marginal.sample(torch.Size([10000])).cpu().detach().numpy()
        except KeyError:
            # print("This distribution does not enable sampling.")
            return None
        
        
        # Plot the true marginal
        fig = plt.figure()
        true_samples = self.p_theta.sample(torch.Size([10000])).cpu().detach().numpy()
        sns.kdeplot(true_samples, 
                     label="True marginal",
                     color=color_map["true_marginal"],)

        # Plot the approximate marginal
        sns.kdeplot(approx_samples,
                     label="Approximate marginal",
                     color=color_map["approximate_marginal"],)

        plt.legend()

        if plot:
            plt.show()

        if return_fig:
            return MarginalProblem.fig_to_array(fig), fig
        
        return MarginalProblem.fig_to_array(fig)

    def log_prob_comparison(self,
                            marginal: dist.Distribution,
                            draft: bool = False,
                            plot: bool = False
                            ) -> np.ndarray:
        global color_map

        x = torch.linspace(-10, 10, 1000)
        # First check if we can obtain the log prob
        try:
            y_approx = marginal.log_prob(x.unsqueeze(-1))
        except KeyError:
            # print("This distribution does not enable scoring.")
            return None
        
        # Plot the true log prob
        fig = plt.figure()
        y_true = self.p_theta.log_prob(x)
        plt.plot(x, y_true, label="True log prob", color=color_map["true_marginal"])

        # Plot the approximate marginal
        y_approx = y_approx.cpu().detach().numpy()
        plt.plot(x, y_approx, label="Approximate log prob", color=color_map["approximate_marginal"])
        plt.legend()

        arr = MarginalProblem.fig_to_array(fig)

        if plot:
            plt.show()
        
        return arr
    
    def log_prob_comparison_2(self,
                              marginal: dist.Distribution,
                                draft: bool = False,
                                plot: bool = False,
                                return_fig: bool = False
                                ) -> np.ndarray:
        global color_map

        x = torch.linspace(-10, 10, 1000)
        # First check if we can obtain the log prob
        try:
            y_approx = marginal.log_prob(x.unsqueeze(-1))
        except KeyError:
            # print("This distribution does not enable scoring.")
            return None
        
        # Plot the true log prob
        fig = plt.figure(figsize=(3, 3))
        y_true = self.p_theta.log_prob(x).exp()
        plt.plot(x, y_true, color=color_map["true_marginal"])

        # Plot the approximate marginal
        y_approx = np.exp(y_approx.cpu().detach().numpy())
        plt.plot(x, y_approx, color=color_map["approximate_marginal"])

        arr = MarginalProblem.fig_to_array(fig)

        if plot:
            plt.show()

        if return_fig:
            return arr, fig
        
        return arr


class Problem2D(MarginalProblem):
    """ A marginal distribution on the 2D torus or a 2D mixture of Gaussians."""
    name = "Problem2D"
    _event_shape = torch.Size([2])
    _param_shape = torch.Size([2])

    def __init__(self, p_theta: dist.Distribution):
        super().__init__()
        self.p_theta = p_theta

    @staticmethod
    def construct_problem(args=None) -> MarginalProblem:
        if args == None or args.problem2d_marginal == "default" or args.problem2d_marginal == "torus":
            center = torch.tensor([0.0, 0.0])
            radius = 2.0
            std = 0.1

            p_theta = TorusMarginal(center, radius, std)

        elif args.problem2d_marginal == "mixture":
            mus = torch.Tensor([[-2.0, 0.0], [2.0, 0.0]])
            sigma = torch.Tensor([0.1, 0.1])

            p_theta = MixtureMarginal2D(mus, sigma)

        else:
            raise ValueError(f"Unknown problem2d-marginal: {args.problem2d_marginal}")
        
        p = Problem2D(p_theta)

        if args.problem2d_likelihood == "gaussian-absolute":
            sigma = args.problem2d_likelihood_std
            likelihood = GaussianAbsoluteLikelihood(p.event_shape, sigma)

        elif args.problem2d_likelihood == "gaussian-noise" or args.problem2d_likelihood == "default":
            sigma = args.problem2d_likelihood_std
            likelihood = GaussianNoiseLikelihood(p.event_shape, sigma)

        else:
            raise ValueError(f"Unknown problem2d-likelihood {args.problem2d_likelihood}.")

        p._likelihood = likelihood
        return p
    
    def likelihood(self, theta: torch.Tensor) -> dist.Distribution:
        p_d_theta = self._likelihood.likelihood(theta)
        return p_d_theta

    @staticmethod
    def add_arguments():
        arguments = {
            "--problem2d-marginal": {
                "type": str,
                "default": "default",
                "help": "Variation of the problem to use.",
                "choices": ["default", "torus", "mixture"]
            },
            "--problem2d-likelihood": {
                "type": str,
                "default": "default",
                "help": "The likelihood function for the plane2d problem. ",
                "choices": ["default", "gaussian-absolute", "gaussian-noise"],
            },
            "--problem2d-likelihood-std": {
                "type": float,
                "default": 0.1,
                "help": "The standard deviation of the likelihood.",
            },
        }
        return arguments

    def marginal_comparison(self,
                            marginal: dist.Distribution,
                            draft: bool = False,
                            plot: bool = False,
                            return_fig: bool = False
                            ) -> np.ndarray | None:
        global color_map
        # Fist check if this marignal supports sampling
        try:
            approx_samples = marginal.sample(torch.Size([10000]))
        except KeyError:
            # print("The approximate marginal does not support sampling.")
            return None
        
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title("True marginal")
        ax[1].set_title("Approximate marginal")

        # Plot the true marginal
        true_samples = self.p_theta.sample(torch.Size([10000])).cpu().detach().numpy()
        sns.kdeplot(x=true_samples[:, 0],
                    y=true_samples[:, 1],
                    label="True marginal",
                    color=color_map["true_marginal"],
                    fill=True,
                    ax=ax[0])
        # plt.legend()
        

        # Plot the approximate marginal
        approx_samples = approx_samples.cpu().detach().numpy()
        try: 
            sns.kdeplot(x=approx_samples[:, 0],
                        y=approx_samples[:, 1],
                        label="Approximate marginal",
                        color=color_map["approximate_marginal"],
                        fill=True,
                        ax=ax[1])
        except ValueError:
            print("Plotting the approximate marginal failed.")

        if plot:
            plt.show()

        if return_fig:
            return MarginalProblem.fig_to_array(fig), fig
        
        return MarginalProblem.fig_to_array(fig)

    def log_prob_comparison(self,
                            marginal: dist.Distribution,
                            draft: bool = False,
                            plot: bool = False
                            ) -> np.ndarray | None:
        """ Compare the true probability with the approximate marginal
            in the relevant region. THIS IS NOT LOGPROB."""

        x = torch.linspace(-5, 5, 100)
        y = torch.linspace(-5, 5, 100)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([X, Y], dim=-1).view(-1, 2)

        # First check if we can obtain the log prob
        try:
            log_prob_approx = marginal.log_prob(grid)
        except KeyError:
            # print("This distribution does not enable scoring.")
            return None

        fig, axes = plt.subplots(1, 2)

        # Plot the true log prob
        log_prob_true = self.p_theta.log_prob(grid)
        log_prob_true = log_prob_true.view(100, 100)
        axes[0].contourf(X, Y, log_prob_true.exp().detach().numpy(), cmap=color_map['cmap'])
        axes[0].set_title("True log prob")

        # Plot the approximate marginal
        log_prob_approx = log_prob_approx.view(100, 100)
        axes[1].contourf(X, Y, log_prob_approx.exp().detach().numpy(), cmap=color_map['cmap'])
        axes[1].set_title("Approximate log prob")

        arr = MarginalProblem.fig_to_array(fig)

        if plot:
            plt.show()
        
        return arr
    
    def log_prob_comparison_2(self,
                            marginal: dist.Distribution,
                            draft: bool = False,
                            plot: bool = False,
                            return_fig: bool = False
                            ) -> np.ndarray | None:
        """ Compare the true probability with the approximate marginal
            in the relevant region. THIS IS NOT LOGPROB."""
        size = 1000
        x = torch.linspace(-5, 5, size)
        y = torch.linspace(-5, 5, size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([X, Y], dim=-1).view(-1, 2)

        # First check if we can obtain the log prob
        try:
            log_prob_approx = marginal.log_prob(grid)
        except KeyError:
            # print("This distribution does not enable scoring.")
            return None

        fig, axes = plt.subplots(1, 1, figsize=(3, 3))

        # Plot the true log prob
        log_prob_true = self.p_theta.log_prob(grid)
        log_prob_true = log_prob_true.view(size, size)
        log_prob_approx = log_prob_approx.view(size, size)
        axes.contour(X, Y, log_prob_true.exp().detach().numpy(), 1, colors=color_map['true_marginal'], alpha=0.7)
        # Remove the outer most contours
        axes.collections[0].remove()

        axes.contourf(X, Y, log_prob_approx.exp().detach().numpy(), cmap=color_map['cmap'])

        arr = MarginalProblem.fig_to_array(fig)

        if plot:
            plt.show()

        if return_fig:
            return arr, fig
        
        return arr
    

PROBLEM_MAP = {
    "1D".lower(): Problem1D,
    "2D".lower(): Problem2D,
}