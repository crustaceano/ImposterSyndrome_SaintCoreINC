import platform
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from src.oracles import ExponentialLossL2Oracle, LogCoshL2Oracle


def make_regression_data(m=2000, n=30, noise=0.1, random_state=42):
    rng = np.random.default_rng(random_state)
    A = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    b = A @ x_true + noise * rng.normal(size=m)
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)
    return A, b


def make_classification_data(m=2000, n=30, random_state=43):
    rng = np.random.default_rng(random_state)
    A = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    logits = A @ x_true + 0.5 * rng.normal(size=m)
    b = np.where(logits >= 0.0, 1.0, -1.0)
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)
    return A, b


def make_team_oracles(m=2000, n=30, regcoef=1e-2, random_state=42):
    A_reg, b_reg = make_regression_data(m=m, n=n, random_state=random_state)
    A_cls, b_cls = make_classification_data(m=m, n=n, random_state=random_state + 1)
    return {
        "Log-Cosh": LogCoshL2Oracle(A_reg, b_reg, regcoef),
        "Exponential": ExponentialLossL2Oracle(A_cls, b_cls, regcoef),
    }


def make_scaled_oracle(oracle):
    scales = np.ones(oracle.n)
    mid = oracle.n // 2
    scales[:mid] = 1000.0
    scales[mid:] = 1.0 / 1000.0
    A_scaled = oracle.A * scales
    return oracle.__class__(A_scaled, oracle.b.copy(), oracle.regcoef)


def make_anomaly_oracle(oracle, fraction=0.01, multiplier=100.0, random_state=44):
    rng = np.random.default_rng(random_state)
    A = np.asarray(oracle.A, dtype=float).copy()
    b = oracle.b.copy()
    count = max(1, int(fraction * oracle.m))
    idx = rng.choice(oracle.m, size=count, replace=False)
    A[idx] *= multiplier
    b[idx] *= -1.0
    return oracle.__class__(A, b, oracle.regcoef), idx


def find_reference_minimum(oracle, x0=None, maxiter=1000):
    if x0 is None:
        x0 = np.zeros(oracle.n)

    result = minimize(
        oracle.func,
        x0,
        jac=oracle.grad,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10},
    )
    return result.x, float(result.fun), result


def add_residual(history, f_star, eps=1e-16):
    history = dict(history)
    history["residual"] = np.maximum(np.asarray(history["func"]) - f_star, eps)
    history["log_residual"] = np.log(history["residual"])
    return history


def plot_histories(histories, y_key="func", title=None, logy=False, xlabel="Эффективные эпохи"):
    plt.figure(figsize=(9, 5))
    for label, hist in histories.items():
        y = hist[y_key]
        if logy:
            plt.semilogy(hist["epoch"], y, linewidth=2.4, label=label)
        else:
            plt.plot(hist["epoch"], y, linewidth=2.4, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(y_key)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_time_histories(histories, y_key="func", title=None, logy=False):
    plot_histories(histories, y_key=y_key, title=title, logy=logy, xlabel="Время, секунды")
    ax = plt.gca()
    for line, hist in zip(ax.lines, histories.values()):
        line.set_xdata(hist["time"])


def hardware_description():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

