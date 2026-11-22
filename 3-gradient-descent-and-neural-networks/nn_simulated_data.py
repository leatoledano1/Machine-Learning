import numpy as np
import pandas as pd
from typing import Tuple
from nn_presentation_utils import custom, plot_decision_boundary, animate_decision_boundary
from nn_loss_functions import accuracy
from nn_modules import FullyConnectedLayer, ReLU, CrossEntropyLoss
from neural_network import NeuralNetwork
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from nn_utils import split_train_test

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "simple_white"


def generate_nonlinear_data(
        samples_per_class: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        train_proportion: float = 0.8) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a multiclass non linearly-separable dataset. Adopted from Stanford CS231 course code.

    Parameters:
    -----------
    samples_per_class: int, default = 100
        Number of samples per class

    n_features: int, default = 2
        Data dimensionality

    n_classes: int, default = 2
        Number of classes to generate

    train_proportion: float, default=0.8
        Proportion of samples to be used for train set

    Returns:
    --------
    train_X : ndarray of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : ndarray of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : ndarray of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : ndarray of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    X, y = np.zeros((samples_per_class*n_classes, n_features)), np.zeros(samples_per_class*n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * 0.2  # theta
        X[ix], y[ix] = np.c_[r * np.sin(t), r * np.cos(t)], j

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=train_proportion)
    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()


if __name__ == '__main__':
    np.random.seed(0)

    # Generate and visualize dataset
    n_features, n_classes = 2, 3
    train_X, train_y, test_X, test_y = generate_nonlinear_data(
        samples_per_class=500, n_features=n_features, n_classes=n_classes, train_proportion=0.8)

    go.Figure(data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                               marker=dict(color=train_y, colorscale=custom, line=dict(color="black", width=1)))],
              layout=go.Layout(title=r"$\text{Train Data}$", xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400))\
        .write_image(f"figures/nonlinear_data.png")


    #LIMITS
    lims = [
        (float(train_X[:, 0].min()) - 0.1, float(train_X[:, 0].max()) + 0.1),
        (float(train_X[:, 1].min()) - 0.1, float(train_X[:, 1].max()) + 0.1)
    ]

    # ---------------------------------------------------------------------------------------------#
    # Question 1: Fitting simple network with two hidden layers                                    #
    # ---------------------------------------------------------------------------------------------#
    mlp_q1 = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=16, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=16, output_dim=16, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=16, output_dim=n_classes, activation=None, include_intercept=True)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, tol=1e-5)
    )
    mlp_q1.fit(train_X, train_y)

    preds_q1 = mlp_q1.predict(test_X)
    acc_q1 = accuracy(test_y, preds_q1)
    print(f"Q1 Test Accuracy: {acc_q1}")

    fig_q1 = plot_decision_boundary(mlp_q1, lims, train_X, train_y, title="(2 Hidden Layers, Width 16)")
    fig_q1.write_image("figures/q1_decision_boundary.png")
    fig_q1.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 2: Fitting a network with no hidden layers                                          #
    # ---------------------------------------------------------------------------------------------#
    mlp_q2 = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=n_classes, activation=None, include_intercept=True)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, tol=1e-5)
    )
    mlp_q2.fit(train_X, train_y)

    preds_q2 = mlp_q2.predict(test_X)
    acc_q2 = accuracy(test_y, preds_q2)
    print(f"Q2 Test Accuracy: {acc_q2}")

    fig_q2 = plot_decision_boundary(mlp_q2, lims, train_X, train_y, title="(No Hidden Layers)")
    fig_q2.write_image("figures/q2_decision_boundary.png")
    fig_q2.show()


    # ---------------------------------------------------------------------------------------------#
    # Question 3+4: Plotting network convergence process                                           #
    # ---------------------------------------------------------------------------------------------#
    recorded_losses = []
    recorded_grad_norms = []
    recorded_weights = []


    def tracking_callback(solver, weights, val, grad, t, eta, delta, **kwargs):
        recorded_losses.append(float(val[0]))
        recorded_grad_norms.append(float(np.linalg.norm(grad)))
        if t % 100 == 0 or t == 1:
            recorded_weights.append(np.copy(weights))


    mlp_q3 = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=16, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=16, output_dim=16, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=16, output_dim=n_classes, activation=None, include_intercept=True)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, tol=1e-5, callback=tracking_callback)
    )
    mlp_q3.fit(train_X, train_y)

    fig_convergence = make_subplots(specs=[[{"secondary_y": True}]])
    iterations = list(range(1, len(recorded_losses) + 1))

    fig_convergence.add_trace(go.Scatter(x=iterations, y=recorded_losses, name="Loss Value", mode="lines"),
                              secondary_y=False)
    fig_convergence.add_trace(go.Scatter(x=iterations, y=recorded_grad_norms, name="Gradient Norm", mode="lines"),
                              secondary_y=True)

    fig_convergence.update_layout(title="Network Convergence Profile (Width 16)", xaxis_title="Iteration")
    fig_convergence.update_yaxes(title_text="Loss", secondary_y=False)
    fig_convergence.update_yaxes(title_text="Gradient Norm", secondary_y=True)
    fig_convergence.write_image("figures/q3_convergence_profile.png")

    animate_decision_boundary(mlp_q3, recorded_weights, lims, train_X, train_y,
                              title="Width 16 Model", save_name="figures/q3_decision_animation.gif")

    recorded_losses_q4 = []
    recorded_grad_norms_q4 = []
    recorded_weights_q4 = []
    fig_convergence.show()


    def tracking_callback_q4(solver, weights, val, grad, t, eta, delta, **kwargs):
        recorded_losses_q4.append(float(val[0]))
        recorded_grad_norms_q4.append(float(np.linalg.norm(grad)))
        if t % 100 == 0 or t == 1:
            recorded_weights_q4.append(np.copy(weights))


    mlp_q4 = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=6, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=6, output_dim=6, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=6, output_dim=n_classes, activation=None, include_intercept=True)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=5000, tol=1e-5, callback=tracking_callback_q4)
    )
    mlp_q4.fit(train_X, train_y)

    acc_q4 = accuracy(test_y, mlp_q4.predict(test_X))
    print(f"Q4 Test Accuracy: {acc_q4}")

    fig_q4 = plot_decision_boundary(mlp_q4, lims, train_X, train_y, title="(2 Hidden Layers, Width 6)")
    fig_q4.write_image("figures/q4_decision_boundary.png")

    animate_decision_boundary(mlp_q4, recorded_weights_q4, lims, train_X, train_y,
                              title="Width 6 Model", save_name="figures/q4_decision_animation.gif")

    fig_q4.show()