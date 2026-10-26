"""From-scratch ISTA and FISTA for the Lasso objective."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np
Array = np.ndarray


def objective(X: Array, y: Array, beta: Array, lam: float) -> float:
    """F(beta) = ||X beta - y||^2 / (2 n) + lam * ||beta||_1."""
    n = X.shape[0]
    residual = X @ beta - y
    return float((residual @ residual) / (2.0 * n) + lam * np.linalg.norm(beta, 1))


def smooth_loss(X: Array, y: Array, beta: Array) -> float:
    """Quadratic loss f(beta) = ||X beta - y||^2 / (2 n)."""
    n = X.shape[0]
    residual = X @ beta - y
    return float((residual @ residual) / (2.0 * n))


def gradient(X: Array, y: Array, beta: Array) -> Array:
    """Gradient of the smooth quadratic loss."""
    n = X.shape[0]
    return (X.T @ (X @ beta - y)) / n


def lipschitz_constant(X: Array) -> float:
    """L = ||X||_2^2 / n, the Lipschitz constant of the gradient."""
    n = X.shape[0]
    spectral_norm = np.linalg.norm(X, ord=2)
    L = float(spectral_norm * spectral_norm / n)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("The design matrix must have a positive finite spectral norm.")
    return L


def soft_thresholding(z: Array, threshold: float) -> Array:
    """Coordinate-wise prox for threshold * ||.||_1."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


@dataclass
class SolverResult:
    beta: Array
    objective_history: Array
    iterations: int
    converged: bool
    lipschitz: float


def ista(
    X: Array,
    y: Array,
    lam: float,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    beta0: Optional[Array] = None,
    callback: Optional[Callable[[int, Array, float], None]] = None,
) -> SolverResult:
    """Run ISTA with step size 1/L.

    Stops when the relative change in objective is <= tol. The history includes
    the initial objective at iteration 0.
    """
    _validate_inputs(X, y, lam, max_iter, tol)
    p = X.shape[1]
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float, copy=True)
    if beta.shape != (p,):
        raise ValueError(f"beta0 must have shape ({p},)")
    L = lipschitz_constant(X)
    history = [objective(X, y, beta, lam)]
    converged = False

    for k in range(1, max_iter + 1):
        previous = beta.copy()
        beta = soft_thresholding(previous - gradient(X, y, previous) / L, lam / L)
        value = objective(X, y, beta, lam)
        history.append(value)
        if callback is not None:
            callback(k, beta, value)
        scale = max(1.0, abs(history[-2]))
        if abs(history[-2] - value) / scale <= tol:
            converged = True
            break

    return SolverResult(beta, np.asarray(history), len(history) - 1, converged, L)


def fista(
    X: Array,
    y: Array,
    lam: float,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    beta0: Optional[Array] = None,
    callback: Optional[Callable[[int, Array, float], None]] = None,
) -> SolverResult:
    """Run standard Beck-Teboulle FISTA with step size 1/L.

    Standard FISTA is not monotone: its objective may increase between
    consecutive iterations. Stopping uses the relative objective change.
    """
    _validate_inputs(X, y, lam, max_iter, tol)
    p = X.shape[1]
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float, copy=True)
    if beta.shape != (p,):
        raise ValueError(f"beta0 must have shape ({p},)")
    z = beta.copy()
    t = 1.0
    L = lipschitz_constant(X)
    history = [objective(X, y, beta, lam)]
    converged = False

    for k in range(1, max_iter + 1):
        previous_beta = beta.copy()
        beta = soft_thresholding(z - gradient(X, y, z) / L, lam / L)
        value = objective(X, y, beta, lam)
        history.append(value)
        if callback is not None:
            callback(k, beta, value)

        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = beta + ((t - 1.0) / t_next) * (beta - previous_beta)
        t = t_next

        scale = max(1.0, abs(history[-2]))
        if abs(history[-2] - value) / scale <= tol:
            converged = True
            break

    return SolverResult(beta, np.asarray(history), len(history) - 1, converged, L)


def _validate_inputs(X: Array, y: Array, lam: float, max_iter: int, tol: float) -> None:
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array with one value per row of X")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have at least one row and one column")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("lam must be finite and non-negative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol < 0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and non-negative")
