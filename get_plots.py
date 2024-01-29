"""
This file generates the plots for the report. It loads the models from the
runs defined below and then saves the plots to the img directory. It also
saves the roc_auc_score to the same directory.
"""

import os
import torch
import json
import argparse
import matplotlib.image
import matplotlib.pyplot as plt

from utils import set_seed
from problems import PROBLEM_MAP
from marginal import MODEL_MAP

import warnings
warnings.filterwarnings('ignore')


DIR_PREFIX = ""

RUNS = [
    # Insert run directories, names, and scores here in the format
    # ('run/directory/', "Name of the run", score) 
]

def save_img(array, path):
    matplotlib.image.imsave(path, array)

if __name__ == "__main__":
    # Loop over the runs defined above
    for run in RUNS:
        run, name, roc_auc_score = run
        run = os.path.join(DIR_PREFIX, run)
        print(name)

        # Load the cmd line args
        config_path = os.path.join(run, 'config.json')
        with open(config_path) as f:
            args = json.load(f)

        args = argparse.Namespace(**args)
        set_seed(args.seed)

        # Construct problem 
        problem = PROBLEM_MAP[args.problem].construct_problem(args)

        # Get emtpy model from args
        model = MODEL_MAP[args.model](args, problem)
        model_dir = [run]

        #Load the model
        for i, mdir in enumerate(model_dir):
            mdir = os.path.join(mdir, 'models')

            # Load the model
            model_file = os.path.join(mdir, 'model.pt')
            model_state_dict = torch.load(model_file)
            model.load_state_dict(model_state_dict)

            # Move the model to the CPU and log the final marginal and probability
            model.to("cpu")

            # Create the img dir
            img_dir = os.path.join(f"{DIR_PREFIX}img", name.replace(' ', '-'))
            os.makedirs(img_dir, exist_ok=True)

            _, logprob2 = problem.log_prob_comparison_2(model.marginal, return_fig=True)
            if logprob2 is not None:
                plt.tight_layout()
                logprob2.savefig(os.path.join(img_dir, f'pdf_{i}.png'))
               
            # Save the score
            with open(os.path.join(img_dir, f'roc_{i}.txt'), 'w') as f:
                f.write(f"{roc_auc_score}")

        print('#'*30)





    