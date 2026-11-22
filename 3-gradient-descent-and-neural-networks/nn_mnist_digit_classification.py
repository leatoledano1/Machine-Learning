import time
import numpy as np
import gzip
from typing import Tuple

import os


from nn_loss_functions import accuracy, softmax
from nn_modules import FullyConnectedLayer, ReLU, CrossEntropyLoss
from neural_network import NeuralNetwork
from gradient_descent import GradientDescent  # use implementation from previous exercise
from learning_rate import FixedLR  # use implementation from previous exercise
from stochastic_gradient_descent import StochasticGradientDescent
from nn_utils import confusion_matrix


import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('mnist-train-images.gz'),
            load_labels('mnist-train-labels.gz'),
            load_images('mnist-test-images.gz'),
            load_labels('mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray", )\
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False)\
        .update_xaxes(showticklabels=False)\
        .update_yaxes(showticklabels=False)


if __name__ == '__main__':
    os.makedirs("figures", exist_ok=True)
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    losses_q5 = []
    grad_norms_q5 = []

    def callback_q5(solver, weights, val, grad, t, eta, delta, **kwargs):
        losses_q5.append(float(val[0]))
        grad_norms_q5.append(float(np.linalg.norm(grad)))


    mlp_mnist = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=64, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=64, output_dim=n_classes, activation=None, include_intercept=True)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256,
                                         callback=callback_q5)
    )

    print("Training Network for Q5...")
    mlp_mnist.fit(train_X, train_y)

    preds_test = mlp_mnist.predict(test_X)
    acc_test = accuracy(test_y, preds_test)
    print(f"Question 5 - MNIST Test Accuracy: {acc_test:.4f}")

    fig_conv = make_subplots(specs=[[{"secondary_y": True}]])
    iters = list(range(1, len(losses_q5) + 1))
    fig_conv.add_trace(go.Scatter(x=iters, y=losses_q5, name="Loss", mode="lines"), secondary_y=False)
    fig_conv.add_trace(go.Scatter(x=iters, y=grad_norms_q5, name="Gradient Norm", mode="lines"), secondary_y=True)
    fig_conv.update_layout(title="MNIST SGD Convergence Process", xaxis_title="Iteration")
    fig_conv.update_yaxes(title_text="Loss", secondary_y=False)
    fig_conv.update_yaxes(title_text="Gradient Norm", secondary_y=True)
    fig_conv.write_image("figures/mnist_convergence.png")

    conf_mat = confusion_matrix(test_y, preds_test)
    fig_conf = ff.create_annotated_heatmap(z=conf_mat, x=list(range(10)), y=list(range(10)), colorscale='Blues')
    fig_conf.update_layout(title="Confusion Matrix on MNIST Test Set", xaxis_title="Predicted Label",
                           yaxis_title="True Label")
    fig_conf.write_image("figures/mnist_confusion_matrix.png")

    fig_conv.show()
    fig_conf.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    mlp_linear = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=n_classes, activation=None, include_intercept=True)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256)
    )

    print("Training Network for Q8...")
    mlp_linear.fit(train_X, train_y)
    preds_linear = mlp_linear.predict(test_X)
    acc_linear = accuracy(test_y, preds_linear)
    print(f"Question 8 - Linear Model Test Accuracy: {acc_linear:.4f}")

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    indices_of_7 = np.where(test_y == 7)[0]
    X_7 = test_X[indices_of_7]

    logits_7 = mlp_mnist.compute_prediction(X_7)
    probs_7 = softmax(logits_7)

    confidences = np.max(probs_7, axis=-1)

    sorted_indices = np.argsort(confidences)
    least_confident_idx = sorted_indices[:64]
    most_confident_idx = sorted_indices[-64:]

    fig_most = plot_images_grid(X_7[most_confident_idx], title="64 Most Confident '7' Predictions")
    fig_most.write_image("figures/most_confident_7.png")

    fig_least = plot_images_grid(X_7[least_confident_idx], title="64 Least Confident '7' Predictions")
    fig_least.write_image("figures/least_confident_7.png")
    fig_least.show()
    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    sub_X, sub_y = train_X[:2500], train_y[:2500]

    gd_time_vs_loss = []
    sgd_time_vs_loss = []

    start_time_gd = time.time()


    def callback_gd(solver, weights, val, grad, t, eta, delta, **kwargs):
        elapsed = time.time() - start_time_gd
        gd_time_vs_loss.append([elapsed, float(val[0])])


    mlp_gd_bench = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=64, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=64, output_dim=n_classes, activation=None, include_intercept=True)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(learning_rate=FixedLR(0.1), max_iter=1000, tol=1e-10, callback=callback_gd)
    )
    print("Benchmarking GD...")
    mlp_gd_bench.fit(sub_X, sub_y)

    start_time_sgd = time.time()


    def callback_sgd(solver, weights, val, grad, t, eta, delta, **kwargs):
        elapsed = time.time() - start_time_sgd
        sgd_time_vs_loss.append([elapsed, float(val[0])])


    mlp_sgd_bench = NeuralNetwork(
        modules=[
            FullyConnectedLayer(input_dim=n_features, output_dim=64, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU(), include_intercept=True),
            FullyConnectedLayer(input_dim=64, output_dim=n_classes, activation=None, include_intercept=True)
        ],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=1000, batch_size=64, tol=1e-10,
                                         callback=callback_sgd)
    )
    print("Benchmarking SGD...")
    mlp_sgd_bench.fit(sub_X, sub_y)

    gd_data = np.array(gd_time_vs_loss)
    sgd_data = np.array(sgd_time_vs_loss)

    fig_bench_gd = go.Figure(go.Scatter(x=gd_data[:, 0], y=gd_data[:, 1], mode='lines', name='GD'))
    fig_bench_gd.update_layout(title="GD: Running Time vs Loss", xaxis_title="Time (seconds)", yaxis_title="Loss")
    fig_bench_gd.write_image("figures/bench_gd_time_loss.png")

    fig_bench_sgd = go.Figure(
        go.Scatter(x=sgd_data[:, 0], y=sgd_data[:, 1], mode='lines', name='SGD', line=dict(color='orange')))
    fig_bench_sgd.update_layout(title="SGD: Running Time vs Loss", xaxis_title="Time (seconds)", yaxis_title="Loss")
    fig_bench_sgd.write_image("figures/bench_sgd_time_loss.png")

    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=gd_data[:, 0], y=gd_data[:, 1], mode='lines', name='Full GD'))
    fig_combined.add_trace(
        go.Scatter(x=sgd_data[:, 0], y=sgd_data[:, 1], mode='lines', name='Stochastic GD (Batch=64)'))
    fig_combined.update_layout(title="Comparison: Running Time vs Loss (GD vs SGD)", xaxis_title="Time (seconds)",
                               yaxis_title="Loss")
    fig_combined.write_image("figures/bench_combined_time_loss.png")

    fig_bench_gd.show()
    fig_bench_sgd.show()
    fig_combined.show()