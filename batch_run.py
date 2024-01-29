""" 
This script is used to generate jobfiles and submit jobs to a cluster using SLURM.
An example run_file is provided in the `runfiles` directory. The run_file contains 
a list of commands to be executed. Each command is a string that can be executed
from the command line. The script will generate a jobfile for each command and submit
it to the cluster. The jobfiles will be stored in the `jobfiles` directory. The output
of each job will be stored in the `jobfiles_out` directory. The jobfiles and output
files will be named using the current timestamp and a runid. The runid is a 4-digit
number that is used to distinguish between different runs of the same command.
"""

import os
import datetime

GENERAL_ARGS="--logger-type wandb --epochs 300 --lr 0.0001 --problem-size 10000 --batch-size 1024 --device cpu --runid 22"
SEEDS=[353165, 426035, 599936, 984300, 102110, 587783, 708221, 954608, 701673, 168733]

def get_jobfile(cmd: str, jobname: str = None, outfile: str = None) -> str:
    """
    Generate a jobfile script for submitting a job to a cluster using SLURM.

    Args:
        cmd (str): The command to be executed in the job.
        jobname (str, optional): The name of the job. If not provided, the current timestamp will be used.
        outfile (str, optional): The name of the output file. If not provided, it will be derived from the jobname.

    Returns:
        str: The generated jobfile script as a string.
    """
    if jobname is None:
        jobname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if outfile is None:
        outfile = jobname + ".out"
    cwd = os.getcwd()

    jobfile = f"""#!/bin/bash

    ## Resource Request
    #SBATCH --job-name={jobname}
    #SBATCH --output=jobfiles_out/{outfile}.out
    #SBATCH --time=0-01:00
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=20G
    #SBATCH --gpus=0

    ## Job Steps
    cd {cwd}
    source activate sde
    srun {cmd}
    """

    return jobfile.replace("    ", "")

def get_command_list(filename: str) -> list[str]:
    """
    Reads a file and returns a list of non-empty lines that do not start with '#'.

    Args:
        filename (str): The path of the file to read.

    Returns:
        list[str]: A list of non-empty lines from the file.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != ""]
    lines = [line for line in lines if not line.startswith("#")]
    return lines

def commands_to_seeds(commands: list[str], seeds: list[int]) -> list[str]:
    """
    Generates a list of command strings with seeds.

    Args:
        commands (list[str]): List of command strings.
        seeds (list[int]): List of seed values.

    Returns:
        list[str]: List of command strings with seeds.
    """
    command_seed_list = [cmd + f" --seed {seed}" for cmd in commands for seed in seeds]
    return command_seed_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_file", type=str, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    # Get commands from the run_file
    commands = get_command_list(args.run_file)
    commands = commands_to_seeds(commands, SEEDS)
    commands = [cmd + " " + GENERAL_ARGS for cmd in commands]

    # Create jobfiles and submit jobs
    for i, cmd in enumerate(commands):
        rid = str(i).zfill(4)
        dt = datetime.datetime.now()
        jobname = dt.strftime('%H%M') + '-' + rid
        outfile = dt.strftime("%Y%m%d-%H%M%S") + '-' + rid + ".out"
        jobfile = dt.strftime("%Y%m%d-%H%M%S") + '-' + rid + ".job"

        jobcontent = get_jobfile(cmd, jobname, outfile)
        if args.dry:
            print(cmd)
            continue

        with open(f"jobfiles/{jobfile}", "w") as f:
            f.write(jobcontent)

        os.system(f"sbatch jobfiles/{jobfile}")
