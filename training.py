"""
This file contains the training and testing functions for the marginal estimation problem.
The `train` function is used to train a model on a given problem. The different loss functions
are defined in `LOSS_FUNCTIONS`. 
"""


import torch
import torch.nn as nn
from copy import deepcopy
import pyro.distributions.transforms as T
from tqdm import tqdm
import argparse

from utils import logger
from problems import MarginalProblem
from marginal import MarginalEstimator
from roc_auc import compute_roc_auc

def marginal_logprob_loss(data: torch.Tensor,
                          model: MarginalEstimator,
                          problem: MarginalProblem) -> torch.Tensor:
    """Loss function that directly models the data with the marginal property of the model. 
        Args:
            data(torch.Tensor): Data to be modelled
            model(MarginalEstimator): Model that has a marginal property
            problem(MarginalProblem): Problem to be solved
            
        Returns:
            torch.Tensor: Loss value
            dict: Dictionary with additional information
    """

    # Check if model has a marginal property
    assert hasattr(model, "marginal"), "Model must have a marginal attribute"

    # Get the log probability of the data under the marginal
    logprob = model.marginal.log_prob(data).mean()

    return -logprob, {}

def likelihood_logprob_loss(data: torch.Tensor,
                            model: MarginalEstimator,
                            problem: MarginalProblem) -> torch.Tensor:
    """Loss function that models the data by the problem likelihood, parameterized by the model (marginal). 
        Args:
            data(torch.Tensor): Data to be modelled
            model(MarginalEstimator): Model that has a marginal property
            problem(MarginalProblem): Problem to be solved, must have a likelihood attribute
            
        Returns:
            torch.Tensor: Loss value
            dict: Dictionary with additional information
    """
    # Check if model has a marginal property, and if problem has a likelihood
    assert hasattr(problem, "likelihood"), "Problem must have a likelihood attribute"
    assert hasattr(model, "marginal"), "Model must have a marginal attribute"

    # Get the likelihood parameters from the models marginal
    theta = model.marginal.rsample((data.shape[0], ))

    # Get the log probability of the data under the likelihood
    logprob = problem.likelihood(theta).log_prob(data)
    logprob = torch.logsumexp(logprob, dim=0) - torch.log(torch.tensor(data.shape[0], dtype=torch.float32))
    return -logprob, {}

def vae_loss(data: torch.Tensor,
             model: MarginalEstimator,
             problem: MarginalProblem) -> torch.Tensor:
    """Loss function that models the data by the problem likelihood, parameterized by the model (marginal). 
        This loss function also uses the posterior of the model (VAE).
        Args:
            data(torch.Tensor): Data to be modelled
            model(MarginalEstimator): Model that has a marginal property
            problem(MarginalProblem): Problem to be solved, must have a likelihood attribute
            
        Returns:
            torch.Tensor: Loss value
            dict: Dictionary with additional information
    """

    # Check if model has a marginal property and if problem has a likelihood
    assert hasattr(problem, "likelihood"), "Problem must have a likelihood attribute"
    assert hasattr(model, "marginal"), "Model must have a marginal attribute"

    # Get the likelihood parameters from the models posterior
    theta = model.forward(data)

    # Get the log probability of the data under the likelihood (reconstruction loss)
    logprob = -problem.likelihood(theta).log_prob(data).mean()

    # Get the log probability of the parameters under the marginal
    reg = -model.marginal.log_prob(theta).mean()

    loss = logprob + reg
    return loss, {"rec": logprob.item(), "reg": reg.item()}


LOSS_FUNCTIONS = {
    "default": marginal_logprob_loss,
    "marginal": marginal_logprob_loss,
    "likelihood": likelihood_logprob_loss,
    "vae": vae_loss,
}

def train(args: argparse.Namespace,
          problem: MarginalProblem,
          model: nn.Module,
          writer: logger,
          roc_model: MarginalEstimator | None = None
          ) -> tuple[torch.nn.Module, list[float]]:
    """
    Trains a model using the given arguments, problem, and data.

    Args:
        args (argparse.Namespace): The command-line arguments.
        problem (MarginalProblem): The marginal problem.
        model (nn.Module): The model to train.
        writer (logger): The logger for writing logs and saving models.
        roc_model (MarginalEstimator | None, optional): The ROC model for intermediate evaluation. Defaults to None.

    Returns:
        tuple[torch.nn.Module, list[float]]: The trained model and the list of validation losses.
    """
    train_theta, train_data = problem.generate_data(args.problem_size)
    validation_theta, validation_data = problem.generate_data(int(args.problem_size / 10))

    if args.loss_function == 'marginal':
        train_loader = torch.utils.data.DataLoader(train_theta, batch_size=args.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_theta, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    
    if args.save_data:
        writer.save_data("train_data", train_data)

    # Setup the training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.early_stopping, verbose=True, 
                                                           min_lr=args.lr/1000, eps=1e-08)
    loss_function = LOSS_FUNCTIONS[args.loss_function]
    model.to(args.device)
    validation_data = validation_data.to(args.device)
    rs_theta = torch.tensor([])

    # Train the model flow
    best_loss = 0
    for step in (pbar := tqdm(range(args.epochs), desc="Epochs", position=0, disable=not args.verbose)):
        train_loss, train_info = train_step(train_loader, model, problem, loss_function, optimizer, args, step, writer)
        val_loss, val_info = eval_step(validation_loader, model, problem, loss_function, args)

        # Resample the parameters for the model (not used in our experiments)
        if args.rs_theta and model.has_rs_theta and not args.resample:
            rs_theta = torch.cat((rs_theta, model.rs_theta(data, 10)))

        # Resample the data (not used in our experiments)
        if args.resample:
            data = problem.generate_data(args.problem_size)

        pbar.set_description(f"Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f}")
        writer.log_scalar("train_loss", train_loss, step)
        writer.log_scalar("val_loss", val_loss, step)

        # Record best model
        if step == 0 or val_loss < best_loss:
            best_loss = val_loss.item()
            best_model = deepcopy(model.state_dict())
            writer.save_model(model, f"best_model.pt")
            prev_worse = False

        if args.intermediate_evaluation != 0 and \
           ((step - 1) % args.intermediate_evaluation == 0 or step == args.epochs - 1) and \
           roc_model is not None:
            roc_model.load_state_dict(best_model)
            roc_auc_val = compute_roc_auc(problem, roc_model, args.test_samples)
            writer.log_scalar("roc_auc_val", roc_auc_val, step)

        # Log the scalars
        for name, value in train_info.items():
            writer.log_scalar(name, value, step)
        for name, value in val_info.items():
            writer.log_scalar(name, value, step)
        
        # Early stopping
        if args.early_stopping > 0:
            scheduler.step(val_loss)
            if scheduler._last_lr[0] <= args.lr/1000:
                break

    model.load_state_dict(best_model)
    return {"Loss": val_loss} | val_info


def train_step(train_loader: torch.utils.data.DataLoader,
               model: MarginalEstimator,
               problem: MarginalProblem,
               loss_function: callable,
               optimizer: torch.optim.Optimizer,
               args: argparse.Namespace,
               epoch: int = 0,
               log: logger | None = None) -> tuple[torch.Tensor, dict[str, any]]:
    # Set all gradients to zero
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(train_loader, leave=False, desc="Training", position=1, disable=not args.verbose)):
        # Move the batch to the device
        batch = batch.to(args.device)

        # Compute the loss
        loss, info = loss_function(batch, model, problem)

        # Backpropagate the loss
        loss.backward()
    
        # Clip the gradients
        if args.clip_grad > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        # Clear the model cache
        model.clear_cache()

        # Update the total loss
        total_loss += loss.item()

        # Log the scalars
        if log != None:
            step = epoch * len(train_loader) + i
            for name, value in info.items():
                log.log_scalar("train_" + name + "_it", value, step)

    total_loss /= len(train_loader)

    return total_loss, {}

def eval_step(dataloader: torch.Tensor,
              model: MarginalEstimator,
              problem: MarginalProblem,
              loss_function: callable,
              args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, any]]:
    # Set model to evaluation mode
    model.eval()
    model.clear_cache()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader, leave=False, desc="Validation", position=1, disable=not args.verbose)):
        # Move the batch to the device
        batch = batch.to(args.device)

        # Compute the loss
        loss, info = loss_function(batch, model, problem)
        total_loss += loss.item()

    total_loss /= len(dataloader)

    return loss, {}

def test(args: argparse.Namespace,
         problem: MarginalProblem,
         model: nn.Module,
         writer) -> dict[str, any]:

    test_theta, test_data = problem.generate_data(args.test_samples)
    if args.save_data:
        writer.save_data("test_data", test_data)

    loss_function = LOSS_FUNCTIONS[args.loss_function]
    test_loss, info = loss_function(test_data, model, problem)

    summary = {"test_loss": test_loss}
    for name, value in info.items():
        summary[name] = value
    return summary
