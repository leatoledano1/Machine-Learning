import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule

from gradient_descent import GradientDescent
from learning_rate import FixedLR, ExponentialLR

import plotly.graph_objects as go

from modules import L2, L1


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from nn_presentation_utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_list = []

    def callback(solver=None, weights=None, val=None, grad=None, t=None, eta=None, delta=None, **kwargs):
        # שליפת הערכים בצורה בטוחה בין אם הם הגיעו כארגומנטים רגילים או בתוך kwargs
        actual_val = val if val is not None else kwargs.get("val")
        actual_weights = weights if weights is not None else kwargs.get("weights")

        values.append(float(actual_val[0]))
        weights_list.append(np.copy(actual_weights))

    return callback, values, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # Optimize L1 and L2 modules using Gradient Descent for each eta
    for module_name, module_class in [("L1", L1), ("L2", L2)]:
        fig_convergence = go.Figure()

        for eta in etas:
            module = module_class(weights=np.copy(init))
            callback, values, weights_list = get_gd_state_recorder_callback()

            values.insert(0, float(module.compute_output()[0]))
            weights_list.insert(0, np.copy(init))

            gd = GradientDescent(
                learning_rate=FixedLR(eta),
                max_iter=1000,
                tol=1e-5,
                out_type="last",
                callback=callback
            )
            gd.fit(module, X=np.array([]), y=np.array([]))

            weights_arr = np.array(weights_list)

            # Plot descent path for each setting and observe L1 vs L2 differences
            fig_path = plot_descent_path(
                module=module_class,
                descent_path=weights_arr,
                title=f"for {module_name} with eta={eta}"
            )
            fig_path.show()

            fig_convergence.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="lines+markers",
                name=f"eta={eta}"
            ))

            print(f"[{module_name} | eta={eta}] Lowest loss: {min(values)}")

        # Plot convergence rate (norm vs iteration) for all learning rates
        fig_convergence.update_layout(
            title=f"GD Convergence Rate for {module_name}",
            xaxis_title="Iteration",
            yaxis_title="Loss Value"
        )
        fig_convergence.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig_convergence = go.Figure()

    for gamma in gammas:
        module = L1(weights=np.copy(init))
        callback, values, weights_list = get_gd_state_recorder_callback()

        values.insert(0, float(module.compute_output()[0]))
        weights_list.insert(0, np.copy(init))

        gd = GradientDescent(
            learning_rate=ExponentialLR(base_lr=eta, decay_rate=gamma),
            max_iter=1000,
            tol=1e-5,
            out_type="last",
            callback=callback
        )
        gd.fit(module, X=np.array([]), y=np.array([]))

        weights_arr = np.array(weights_list)

        # Plot algorithm's convergence for the different values of gamma
        fig_convergence.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode="lines+markers",
            name=f"gamma={gamma}"
        ))

        # Plot descent path for gamma=0.95
        if gamma == 0.95:
            fig_path = plot_descent_path(
                module=L1,
                descent_path=weights_arr,
                title=f"for L1 with Exponential LR (eta={eta}, gamma=0.95)"
            )
            fig_path.show()

    fig_convergence.update_layout(
        title="L1 Convergence with Exponential Decay Learning Rates",
        xaxis_title="Iteration",
        yaxis_title="Loss Value"
    )
    fig_convergence.show()


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()