"""Synthetic-data generation and experiment plots."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from src.lasso import ista, fista, objective


def make_sparse_regression(
    n: int = 250,
    p: int = 500,
    n_nonzero: int = 15,
    noise_std: float = 0.1,
    seed: int = 7,
    correlation: float = 0.0,
):
    """Generate a sparse linear model with optional shared-factor correlation.

    Each feature is standardized to approximately zero mean and unit variance.
    For correlation in [0, 1), features combine independent noise with a
    shared latent factor. This is a simple controlled design, not a full
    correlation-matrix benchmark.
    """
    if not 0 <= correlation < 1:
        raise ValueError("correlation must be in [0, 1)")
    rng = np.random.default_rng(seed)
    independent = rng.normal(size=(n, p))
    common = rng.normal(size=(n, 1))
    X = np.sqrt(1.0 - correlation) * independent + np.sqrt(correlation) * common
    X -= X.mean(axis=0, keepdims=True)
    X /= np.maximum(X.std(axis=0, keepdims=True), 1e-12)

    beta_true = np.zeros(p)
    indices = rng.choice(p, size=n_nonzero, replace=False)
    beta_true[indices] = rng.choice([-1.0, 1.0], size=n_nonzero) * rng.uniform(0.8, 2.0, size=n_nonzero)
    y = X @ beta_true + rng.normal(scale=noise_std, size=n)
    y -= y.mean()
    return X, y, beta_true


def run_comparison(output_dir="results", seed=7, lam=0.1, correlation=0.0):
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    X, y, beta_true = make_sparse_regression(seed=seed, correlation=correlation)
    common = dict(lam=lam, max_iter=20_000, tol=1e-10)
    result_ista = ista(X, y, **common)
    result_fista = fista(X, y, **common)

    # A practical reference objective: the lower final value from these runs.
    # This is not a certified optimum.
    reference = min(result_ista.objective_history[-1], result_fista.objective_history[-1])
    gap_i = np.maximum(result_ista.objective_history - reference, np.finfo(float).tiny)
    gap_f = np.maximum(result_fista.objective_history - reference, np.finfo(float).tiny)

    plt.figure()
    plt.semilogy(np.arange(len(gap_i)), gap_i, label="ISTA")
    plt.semilogy(np.arange(len(gap_f)), gap_f, label="FISTA")
    plt.xlabel("Iteration")
    plt.ylabel(r"$F(\beta_k)-F_{\mathrm{ref}}$")
    plt.title("ISTA vs FISTA: objective gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "objective_gap.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(np.arange(len(result_ista.objective_history)), result_ista.objective_history, label="ISTA")
    plt.plot(np.arange(len(result_fista.objective_history)), result_fista.objective_history, label="FISTA")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("ISTA vs FISTA: objective values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "objective_values.png", dpi=180)
    plt.close()

    rows = [
        ("ISTA", result_ista.iterations, result_ista.objective_history[-1],
         np.count_nonzero(np.abs(result_ista.beta) > 1e-8), result_ista.converged),
        ("FISTA", result_fista.iterations, result_fista.objective_history[-1],
         np.count_nonzero(np.abs(result_fista.beta) > 1e-8), result_fista.converged),
    ]
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "iterations", "final_objective", "nonzero_coefficients", "converged"])
        writer.writerows(rows)

    print(f"n={X.shape[0]}, p={X.shape[1]}, lambda={lam}, correlation={correlation}")
    print(f"L={result_ista.lipschitz:.8g}")
    print("algorithm  iterations  final objective  nonzero coefficients  converged")
    for name, iters, val, nonzero, converged in rows:
        print(f"{name:9s}  {iters:10d}  {val:15.8g}  {nonzero:20d}  {converged}")
    print(f"Figures and summary saved to: {output_dir.resolve()}")
    return result_ista, result_fista


if __name__ == "__main__":
    run_comparison()
