import unittest
import numpy as np

from src.lasso import (
    objective, gradient, soft_thresholding, ista, fista, lipschitz_constant
)


class TestLasso(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.normal(size=(30, 12))
        self.beta = rng.normal(size=12)
        self.y = self.X @ self.beta + 0.05 * rng.normal(size=30)

    def test_soft_thresholding(self):
        z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        got = soft_thresholding(z, 1.0)
        expected = np.array([-2.0, 0.0, 0.0, 0.0, 2.0])
        np.testing.assert_allclose(got, expected)

    def test_gradient_finite_difference(self):
        eps = 1e-6
        analytic = gradient(self.X, self.y, self.beta)
        numerical = np.zeros_like(self.beta)
        for j in range(len(self.beta)):
            step = np.zeros_like(self.beta)
            step[j] = eps
            fp = np.sum((self.X @ (self.beta + step) - self.y) ** 2) / (2 * len(self.y))
            fm = np.sum((self.X @ (self.beta - step) - self.y) ** 2) / (2 * len(self.y))
            numerical[j] = (fp - fm) / (2 * eps)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)

    def test_ista_fista_reach_similar_objective(self):
        lam = 0.2
        ri = ista(self.X, self.y, lam, max_iter=20000, tol=1e-11)
        rf = fista(self.X, self.y, lam, max_iter=20000, tol=1e-11)
        self.assertLessEqual(abs(ri.objective_history[-1] - rf.objective_history[-1]), 1e-5)

    def test_lipschitz_positive(self):
        self.assertGreater(lipschitz_constant(self.X), 0.0)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            soft_thresholding(np.array([1.0]), -1.0)


if __name__ == "__main__":
    unittest.main()
