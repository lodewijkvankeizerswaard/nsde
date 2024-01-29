"""
This file contains some utility functions for the project. The functions are
used in multiple places, and are therefore collected here. The functions are
used for logging, setting the seed, and parsing command line arguments.
"""

import argparse
import datetime
import torch
import numpy as np
import pandas as pd
import os
import json
from typing import Union, Any

try:
    import wandb
    wandb_message = ""
except ImportError:
    wandb_message = "Please make sure you have installed `wandb`. \
                    This is not included in the environment files."
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser(model_problem_arguments: dict = {}):
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str.lower, default="1d")
    parser.add_argument("--loss-function", type=str.lower, default="default", choices=["default", "marginal", "likelihood", "vae"])
    parser.add_argument("--resample", action='store_true')
    parser.add_argument("--rs-theta", action='store_true')
    parser.add_argument("--problem-size", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)

    parser.add_argument("--model", type=str.lower, default="naive")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping", type=int, default=0)
    parser.add_argument("--intermediate-evaluation", type=int, default=0, help="Evaluate every n epochs")

    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--logger-type", type=str, default="tensorboard", choices=["tensorboard", "wandb"])

    parser.add_argument("--plot-posterior", action="store_true", default=False)
    parser.add_argument("--plot-marginal", action="store_true", default=False)
    parser.add_argument("--plot-rs-theta", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--runid', type=str, default=None, help="Used to identify a runbatch in the database")

    for argument, options in model_problem_arguments.items():
        parser.add_argument(argument, **options)

    return parser


def initialize_model_name_from_args(args: argparse.Namespace):
    """A function for initializing the name of the model from
       the cmd-line arguments."""
    if not args.debug:
        name = "runs/"
    else:
        name = "runs_debug/"

    if args.name != None and args.name != "":
        name += f"{args.problem}/{args.name}/"
    else:
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name += f"{args.problem}/{time}/"
    return name

def dump_config(config: Union[dict[str, Any], argparse.Namespace], path: str):
    # Convert args to dict if necessary
    if not isinstance(config, dict):
        config = vars(config)

    # Make sure the directory exists
    dir = os.path.split(path)[0]
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Dump the config
    json_string = json.dumps(config, indent=4, default=lambda o: '<not serializable>')
    with open(path, 'w') as f:
        f.write(json_string)

class logger(object):
    """A general object to handle all logging logic, and the saving of models."""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log_type = args.logger_type

        if self.log_type == "tensorboard":
            self.writer = SummaryWriter(args.name)

        elif self.log_type == "wandb":
            if wandb_message:
                raise ImportError(wandb_message)
            os.environ["WANDB_SILENT"] = "true"

            # TODO make this modular
            wandb.init(project="thesis", entity="lodewijk", name=args.name, config=args)

        else:
            raise ValueError(f"Unknown logger type: {self.log_type}")
        
        # Save model directory
        self.model_directory = os.path.join(args.name, "models")
        self.data_directory = os.path.join(args.name, "data")

        # Make sure the directories exist
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        # Save the config
        config_file_name = os.path.join(args.name, "config.json")
        dump_config(args, config_file_name)

        # Create a logging file location
        self.log_file = os.path.join(self.model_directory, "log.txt")

        self.log(f"Logging to: {self.model_directory}")

        self.summary_called = False

    def log(self, print_string: str, verbose: bool = True):
        """A function for logging to the console and to a file."""
        if verbose:
            print(print_string)
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                print_string = print_string if print_string[-1] == "\n" else print_string + "\n"
                f.write(print_string)

    def log_hparams(self, hparams: argparse.Namespace):
        if self.log_type == "tensorboard":
            self.tb_hparams = vars(hparams)
        elif self.log_type == "wandb":
            # Already done
            pass

    def log_scalar(self, name: str, value: Any, timestep: int):
        if self.log_type == "tensorboard":
            self.writer.add_scalar(name, value, timestep)
        elif self.log_type == "wandb":
            wandb.log({name: value})

    def summary(self, layout: dict):
        if self.log_type == "tensorboard":
            self.writer.add_hparams(self.tb_hparams, layout)
        elif self.log_type == "wandb":
            for name, value in layout.items():
                wandb.run.summary[name] = value

        self.summary_called = True

    def log_image(self, name: str, image: Any, timestep: int = 0):
        if self.log_type == "tensorboard":
            self.writer.add_image(name, image, dataformats="NHWC", global_step=timestep)
        elif self.log_type == "wandb":
            img = wandb.Image(image, caption=name)
            wandb.log({name: img})

    def log_images(self, image_dict: dict[str, Any], timestep: Union[int, list[int], None] = None):
        if isinstance(timestep, list):
            assert len(timestep) == len(image_dict), "Timestep and image_dict must have the same size"
        else:
            timestep = [timestep] * len(image_dict)

        if self.log_type == "tensorboard":
            for (name, image), t in zip(image_dict.items(), timestep):
                self.writer.add_image(name, image, dataformats="NHWC", global_step=t)
        elif self.log_type == "wandb":
            for (name, image), t in zip(image_dict.items(), timestep):
                img = wandb.Image(image, caption=name)
                wandb.log({name: img})

    def save_data(self, name: str, data: torch.Tensor):
        # print(f"Saving {name} to {self.data_directory}")
        data_file_name = os.path.join(self.data_directory, f"{name}.pt")
        if os.path.exists(data_file_name):
            print(f"WARNING: {data_file_name} already exists, overwriting...")
        torch.save(data, data_file_name)

    def save_dataframe(self, name: str, df: pd.DataFrame):
        # print(f"Saving {name} to {self.data_directory}")
        data_file_name = os.path.join(self.data_directory, f"{name}.csv")
        if os.path.exists(data_file_name):
            print(f"WARNING: {data_file_name} already exists, overwriting...")
        df.to_csv(data_file_name)

    def save_model(self, model: torch.nn.Module, name: str = "model.pt"):
        # print(f"Saving model to {self.model_directory}")
        model_file_name = os.path.join(self.model_directory, name)
        torch.save(model.state_dict(), model_file_name)

    def __del__(self):
        if not self.summary_called:
            self.summary({})
        if self.log_type == "tensorboard":
            self.writer.close()

    @staticmethod
    def load_model(args: argparse.Namespace) -> torch.nn.Module:
        model_file_name = os.path.join(args.name, "models", "model.pt")
        model = torch.load(model_file_name)
        return model