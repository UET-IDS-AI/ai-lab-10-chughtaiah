"""
AI_stats_lab.py

Instructor Solution
Lab: Bias-Variance Tradeoff
"""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# ============================================================
# Question 1: Model Complexity and Generalization
# ============================================================

def generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42):
    """
    Generate nonlinear data:
        y = sin(2*pi*x) + Gaussian noise
    """
    rng = np.random.RandomState(random_state)

    X = rng.rand(n_samples, 1)

    y_clean = np.sin(2 * np.pi * X)

    noise_values = rng.normal(
        loc=0.0,
        scale=noise,
        size=(n_samples, 1)
    )

    y = y_clean + noise_values

    return X, y.ravel()


def create_polynomial_model(degree):
    """
    Create polynomial regression pipeline.
    """
    model = Pipeline([
        ("polynomial_features", PolynomialFeatures(
            degree=degree,
            include_bias=False
        )),
        ("linear_regression", LinearRegression())
    ])

    return model


def evaluate_polynomial_degrees(X, y, degrees, test_size=0.3, random_state=0):
    """
    Evaluate different polynomial degrees using train/dev MSE.
    """
    X_train, X_dev, y_train, y_dev = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    train_errors = []
    dev_errors = []

    for degree in degrees:
        model = create_polynomial_model(degree)

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        dev_predictions = model.predict(X_dev)

        train_mse = mean_squared_error(y_train, train_predictions)
        dev_mse = mean_squared_error(y_dev, dev_predictions)

        train_errors.append(train_mse)
        dev_errors.append(dev_mse)

    best_index = int(np.argmin(dev_errors))
    best_degree = degrees[best_index]

    return {
        "degrees": degrees,
        "train_errors": train_errors,
        "dev_errors": dev_errors,
        "best_degree": best_degree
    }


def diagnose_from_errors(
    train_error,
    dev_error,
    high_error_threshold=0.15,
    gap_threshold=0.05
):
    """
    Diagnose high bias, high variance, both, or good fit.
    """
    generalization_gap = dev_error - train_error

    if train_error > high_error_threshold and generalization_gap <= gap_threshold:
        diagnosis = "high_bias"

    elif train_error <= high_error_threshold and generalization_gap > gap_threshold:
        diagnosis = "high_variance"

    elif train_error > high_error_threshold and generalization_gap > gap_threshold:
        diagnosis = "high_bias_and_high_variance"

    else:
        diagnosis = "good_fit"

    return {
        "train_error": train_error,
        "dev_error": dev_error,
        "generalization_gap": generalization_gap,
        "diagnosis": diagnosis
    }


# ============================================================
# Question 2: Regularization and Model Improvement
# ============================================================

def regularization_comparison(X_train, y_train, X_dev, y_dev, alphas):
    """
    Compare Ridge models using different alpha values.
    """
    train_errors = []
    dev_errors = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        dev_predictions = model.predict(X_dev)

        train_mse = mean_squared_error(y_train, train_predictions)
        dev_mse = mean_squared_error(y_dev, dev_predictions)

        train_errors.append(train_mse)
        dev_errors.append(dev_mse)

    best_index = int(np.argmin(dev_errors))
    best_alpha = alphas[best_index]

    return {
        "alphas": alphas,
        "train_errors": train_errors,
        "dev_errors": dev_errors,
        "best_alpha": best_alpha
    }


def recommend_action(diagnosis):
    """
    Recommend action based on bias/variance diagnosis.
    """
    if diagnosis == "high_bias":
        return "increase_model_complexity"

    if diagnosis == "high_variance":
        return "add_regularization_or_more_data"

    if diagnosis == "high_bias_and_high_variance":
        return "increase_complexity_then_regularize"

    if diagnosis == "good_fit":
        return "keep_model_or_minor_tuning"

    return "unknown_diagnosis"


if __name__ == "__main__":
    X, y = generate_nonlinear_data()
    print("Instructor solution loaded.")
