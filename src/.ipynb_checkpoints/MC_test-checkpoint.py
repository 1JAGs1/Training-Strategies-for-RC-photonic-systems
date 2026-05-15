import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from pathlib import Path

def linear_memory_curve(R, u, max_delay=50, washout=100, split=0.8, ridge_alpha=1e-6):

    R = np.asarray(R, dtype=float)
    u = np.asarray(u, dtype=float).reshape(-1)

    assert R.shape[0] == len(u),

    delays = []
    capacities = []
    nrmse_list = []

    # start late enough that all tested delays are valid
    start = washout + max_delay
    t_idx = np.arange(start, len(u))

    for d in range(1, max_delay + 1):
        X = R[t_idx, :]          # current reservoir states
        y = u[t_idx - d]         # delayed target input

        n_samples = len(y)
        n_train = int(split * n_samples)

        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Centre using train stats only
        X_mean = X_train.mean(axis=0, keepdims=True)
        y_mean = y_train.mean()

        X_train_c = X_train - X_mean
        X_test_c  = X_test  - X_mean
        y_train_c = y_train - y_mean
        y_test_c  = y_test  - y_mean

        model = Ridge(alpha=ridge_alpha, fit_intercept=False)
        model.fit(X_train_c, y_train_c)
        y_pred_c = model.predict(X_test_c)

        mse = np.mean((y_test_c - y_pred_c)**2)
        var = np.var(y_test_c)

        if var <= 0:
            C_d = 0.0
            nrmse = np.nan
        else:
            C_d = max(0.0, 1.0 - mse / var)   # held-out memory score
            nrmse = np.sqrt(mse / var)

        delays.append(d)
        capacities.append(C_d)
        nrmse_list.append(nrmse)

    delays = np.array(delays)
    capacities = np.array(capacities)
    nrmse_list = np.array(nrmse_list)

    MC_est = capacities.sum()
    return delays, capacities, nrmse_list, MC_est


# def plot_memory_curve(delays, capacities, MC_est):
#     plt.figure(figsize=(8, 4))
#     plt.plot(delays, capacities, marker='o', linewidth=1)
#     plt.xlabel("Delay d (input cycles)")
#     plt.ylabel("Memory score C_d")
#     plt.title(f"Linear memory curve   MC_est = {MC_est:.2f}")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


def plot_memory_curve(
    delays,
    capacities,
    MC_est,
    save_path=None,
    show=True
):
  

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)

    ax.plot(
        delays,
        capacities,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        markeredgewidth=0.8
    )

    ax.set_xlabel("Delay, $d$ (input cycles)", fontsize=11)
    ax.set_ylabel("Memory score, $C_d$", fontsize=11)

    # Report-style annotation instead of large title
    ax.text(
        0.97, 0.93,
        rf"$MC_{{est}} = {MC_est:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.7",
            linewidth=0.8
        )
    )

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(axis="both", which="minor", length=3)

    # Clean report-style borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax