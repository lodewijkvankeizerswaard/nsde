import torch
from roc_auc import compute_roc_auc
from utils import get_parser, initialize_model_name_from_args, logger, set_seed
from problems import PROBLEM_MAP
from marginal import MODEL_MAP
from training import train, test


def get_model_problem_arguments() -> dict:
    """
    Returns a dictionary containing all the arguments required by the models and problems
    defined in the MODEL_MAP and PROBLEM_MAP dictionaries respectively.
    """
    arguments = {}
    for Model in MODEL_MAP.values():
        arguments = arguments | Model.add_arguments()

    for Problem in PROBLEM_MAP.values():
        arguments = arguments | Problem.add_arguments()
    
    return arguments


def main():
    # Load the problem and model arguments for the parser
    arguments = get_model_problem_arguments()
    parser = get_parser(arguments)

    # Parse the arguments and initialize the model name
    args = parser.parse_args()
    args.name = initialize_model_name_from_args(args)

    if args.seed != 0:
        set_seed(args.seed)

    # Initialize the logger and log the hyperparameters
    writer = logger(args)
    writer.log_hparams(args)

    # Initialize the problem and model
    problem = PROBLEM_MAP[args.problem].construct_problem(args)
    model = MODEL_MAP[args.model](args, problem)
    roc_model = MODEL_MAP[args.model](args, problem)

    # Log the model
    writer.log(str(model))

    # Train the model and save it
    train_summary = train(args, problem, model, writer, roc_model=roc_model)
    writer.save_model(model)

    # Move the model to the CPU and log the final marginal and log probability
    model.to("cpu")
    marginal_plot = problem.marginal_comparison(model.marginal)
    if marginal_plot is not None:
        writer.log_image("marginal_final", marginal_plot, 0)
    logprob = problem.log_prob_comparison(model.marginal)
    if logprob is not None:
        writer.log_image("logprob_final", logprob, 0)

    # Test the model and log the results 
    test_summary = test(args, problem, model, writer)

    # Log the ROC AUC score
    roc_auc = compute_roc_auc(problem, model, num_samples=args.test_samples)
    writer.log_scalar('roc_auc', roc_auc, 0)
    test_summary["roc_auc"] = roc_auc

    writer.summary(test_summary)

if __name__ == "__main__":
    main()