import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from logistic_regression import LogisticRegression

def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X: DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X: DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    train = X.sample(frac=train_proportion)
    test = X.loc[X.index.difference(train.index)]
    return train, y.loc[train.index], test, y.loc[test.index]


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting ROC curve of logistic regression over SA heart disease data
    X_train_np = np.hstack([np.ones((X_train.shape[0], 1)), X_train.to_numpy()])
    X_test_np = np.hstack([np.ones((X_test.shape[0], 1)), X_test.to_numpy()])
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    model = LogisticRegression()
    model.fit(X_train_np, y_train_np)
    y_prob = model.predict_proba(X_test_np)

    fpr, tpr, thresholds = roc_curve(y_test_np, y_prob)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines'))
    fig1.show()

    best_idx = np.argmax(tpr - fpr)
    alpha_star = thresholds[best_idx]

    y_pred = (y_prob >= alpha_star).astype(int)
    test_error = np.mean(y_pred != y_test_np)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    for penalty in ["l1", "l2"]:
        train_errors = []
        test_errors = []

        for lam in lambdas:
            reg_model = LogisticRegression(penalty=penalty, lam=lam)
            reg_model.fit(X_train_np, y_train_np)

            p_train = reg_model.predict_proba(X_train_np)
            p_test = reg_model.predict_proba(X_test_np)

            train_errors.append(np.mean((p_train >= 0.5).astype(int) != y_train_np))
            test_errors.append(np.mean((p_test >= 0.5).astype(int) != y_test_np))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=lambdas, y=train_errors, mode='lines+markers'))
        fig2.add_trace(go.Scatter(x=lambdas, y=test_errors, mode='lines+markers'))
        fig2.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_logistic_regression()