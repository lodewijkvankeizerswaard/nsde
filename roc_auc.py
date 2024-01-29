""""
This file contains the code for computing the ROC AUC score for a given problem and model.
More information on the ROC AUC score can be found below. The `__main__` function contains
an example of how to use this function, and how to visualize the results.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from problems import MarginalProblem
from marginal import MarginalEstimator



def compute_roc_auc(problem: MarginalProblem,
                    model: MarginalEstimator,
                    num_samples: int = 1000,
                    return_debug: bool = False,
                    seed: int = 42):
    """
    Computes the ROC AUC score for a given problem and model, based on the three classifiers
    defined in `classifiers`. For further reference on the roc_auc score, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    For further reference on the used classifiers, see:
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    Args:
        problem (MarginalProblem): The problem to compute the ROC AUC score for.
        model (MarginalEstimator): The model to compute the ROC AUC score for.
        num_samples (int): The number of samples to use for computing the ROC AUC score.
        return_debug (bool): Whether to return debug information along with the ROC AUC score.
        seed (int): The seed to use for reproducibility.

    Returns:
        float: The ROC AUC score for the given problem and model.
        dict(optional): A dictionary containing debug information (if `return_debug` is True).
    """
    classifiers = {
        'gaussian-process': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=seed),
        "neural-network": MLPClassifier(alpha=1, max_iter=1000, random_state=seed),
        "rbf-svm": SVC(random_state=seed, probability=True),
    }

    true_samples = problem.p_theta.sample((num_samples, ))
    model_samples = model.marginal.sample((num_samples, )).squeeze().cpu()


    true_labels = torch.ones(num_samples)
    model_labels = torch.zeros(num_samples)

    X = torch.cat([true_samples, model_samples], dim=0).view(num_samples * 2, *problem.param_shape)
    y = torch.cat([true_labels, model_labels], dim=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    # Train a classifier on the samples
    scores = []
    classifier_accuracies = {}
    for name, cls in classifiers.items():
        # Fit the classifier
        cls.fit(X_train, y_train)

        # Compute the probabilities of each class: true(1) and model(0)
        y_train_probs = cls.predict_proba(X_train)
        y_test_probs = cls.predict_proba(X_test)
        accuracy = cls.score(X_test, y_test)
        classifier_accuracies[name + "_acc"] = accuracy

        train_score = roc_auc_score(y_train, y_train_probs[:, 1])
        test_score = roc_auc_score(y_test, y_test_probs[:, 1])

        # If the model is better than random on the training set, but worse than random on the test set,
        # set the test score to 0.5. We do not want overfitting to drag the score down.
        if test_score < 0.5:
            test_score = 0.5

        scores.append(test_score)

    if return_debug:
        debug = {
            'X': X,
            'y': y,
            'scores': scores,
        } | classifier_accuracies
        return np.mean(scores), classifiers, debug

    return np.mean(scores)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay

    class DummyModel():
        def __init__(self, mean: torch.Tensor, std: torch.Tensor):
            self.marginal = torch.distributions.Normal(mean, std)

    class DummyProblem():
        def __init__(self, mean: torch.Tensor, std: torch.Tensor):
            self.p_theta = torch.distributions.Normal(mean, std)

    def show_roc_auc(N, model, problem):
        _, classifiers, debug = compute_roc_auc(problem, model, num_samples=N, return_debug=True)

        X = debug['X']
        y = debug['y']
        scores = debug['scores']

        models = [c for c in classifiers.values()]
    
        print("The roc auc of the two distributions is:")
        for name, score in zip(classifiers.keys(), scores):
            print(f"{name}: {score}")

        cm = plt.cm.RdBu
        _, ax = plt.subplots(1, len(models) , figsize=(6 * len(models), 6), squeeze=False)
        for i, (name, model) in enumerate(zip(classifiers.keys(), models)):
            ax[0, i].set_title(name)
            ax[0, i].scatter(X[:, 0], X[:, 1], c=y, cmap=cm)
            # Decision boundary
            DecisionBoundaryDisplay.from_estimator(
                model, X, cmap=cm, alpha=0.8, ax=ax[0, i], eps=0.5
            )
        plt.show()


    N = 1000
    # Distinguishable distributions which should have a high ROC AUC (1.0)
    mean_model = torch.tensor([0.0, 0.0])
    std_model = torch.tensor([1.0, 1.0])

    mean_problem = torch.tensor([10.0, 10.0])
    std_problem = torch.tensor([1.0, 1.0])

    model = DummyModel(mean_model, std_model)
    problem = DummyProblem(mean_problem, std_problem)

    show_roc_auc(N, model, problem)


    # Indistinguishable distributions which should have a low ROC AUC (0.5)
    mean_model = torch.tensor([0.0, 0.0])
    std_model = torch.tensor([1.0, 1.0])

    mean_problem = torch.tensor([0.0, 0.0])
    std_problem = torch.tensor([1.0, 1.0])

    model = DummyModel(mean_model, std_model)
    problem = DummyProblem(mean_problem, std_problem)

    show_roc_auc(N, model, problem)
    
