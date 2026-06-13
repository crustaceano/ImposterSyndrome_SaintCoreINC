import numpy as np
import scipy.sparse as sp


_LOG2 = np.log(2.0)
_EXP_CLIP = 709.0


def _as_1d(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _slice_rows(A, idx):
    if idx is None:
        return A
    return A[idx]


def matmat_ATsA(A, s):
    s = np.asarray(s, dtype=float).reshape(-1)
    if sp.issparse(A):
        return A.T @ sp.diags(s) @ A
    return A.T @ (s[:, None] * A)


class BaseStochasticOracle:
    """Базовый интерфейс для конечных сумм с L2-регуляризацией."""

    def __init__(self, A, b, regcoef=0.0):
        self.A = A.tocsr() if sp.issparse(A) else np.asarray(A, dtype=float)
        self.b = _as_1d(b)
        self.regcoef = float(regcoef)
        self._m, self._n = self.A.shape
        if self.b.size != self._m:
            raise ValueError("Размер b должен совпадать с числом строк A.")

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    def loss_values(self, x, batch_idx=None):
        raise NotImplementedError

    def loss_grad(self, x, batch_idx=None):
        """Градиент среднего loss по батчу без L2-регуляризации."""
        raise NotImplementedError

    def individual_loss_grad(self, x, idx):
        """Матрица градиентов отдельных loss: shape = (len(idx), n)."""
        raise NotImplementedError

    def func(self, x, batch_idx=None):
        x = _as_1d(x)
        return float(np.mean(self.loss_values(x, batch_idx)) + 0.5 * self.regcoef * x.dot(x))

    def grad(self, x, batch_idx=None):
        x = _as_1d(x)
        return self.loss_grad(x, batch_idx) + self.regcoef * x

    def hess(self, x):
        raise NotImplementedError

    def hess_vec(self, x, v):
        return self.hess(x).dot(v)


class LogCoshL2Oracle(BaseStochasticOracle):
    """F(x) = mean(log(cosh(Ax - b))) + regcoef / 2 * ||x||^2."""

    def loss_values(self, x, batch_idx=None):
        A = _slice_rows(self.A, batch_idx)
        b = self.b if batch_idx is None else self.b[batch_idx]
        r = A.dot(x) - b
        return np.logaddexp(r, -r) - _LOG2

    def loss_grad(self, x, batch_idx=None):
        A = _slice_rows(self.A, batch_idx)
        b = self.b if batch_idx is None else self.b[batch_idx]
        r = A.dot(x) - b
        t = np.tanh(r)
        return np.asarray(A.T.dot(t)).reshape(-1) / t.size

    def individual_loss_grad(self, x, idx):
        idx = np.asarray(idx, dtype=int)
        A = _slice_rows(self.A, idx)
        r = A.dot(x) - self.b[idx]
        coef = np.tanh(r)
        if sp.issparse(A):
            return A.multiply(coef[:, None]).toarray()
        return coef[:, None] * A

    def hess(self, x):
        r = self.A.dot(x) - self.b
        s = (1.0 - np.tanh(r) ** 2) / self.m
        return matmat_ATsA(self.A, s) + self.regcoef * np.eye(self.n)


class ExponentialLossL2Oracle(BaseStochasticOracle):
    """F(x) = mean(exp(-b_i <a_i, x>)) + regcoef / 2 * ||x||^2, b_i in {-1, 1}."""

    def loss_values(self, x, batch_idx=None):
        A = _slice_rows(self.A, batch_idx)
        b = self.b if batch_idx is None else self.b[batch_idx]
        margins = b * A.dot(x)
        return np.exp(np.clip(-margins, -_EXP_CLIP, _EXP_CLIP))

    def loss_grad(self, x, batch_idx=None):
        A = _slice_rows(self.A, batch_idx)
        b = self.b if batch_idx is None else self.b[batch_idx]
        margins = b * A.dot(x)
        weights = -b * np.exp(np.clip(-margins, -_EXP_CLIP, _EXP_CLIP))
        return np.asarray(A.T.dot(weights)).reshape(-1) / weights.size

    def individual_loss_grad(self, x, idx):
        idx = np.asarray(idx, dtype=int)
        A = _slice_rows(self.A, idx)
        b = self.b[idx]
        margins = b * A.dot(x)
        coef = -b * np.exp(np.clip(-margins, -_EXP_CLIP, _EXP_CLIP))
        if sp.issparse(A):
            return A.multiply(coef[:, None]).toarray()
        return coef[:, None] * A

    def hess(self, x):
        margins = self.b * self.A.dot(x)
        s = np.exp(np.clip(-margins, -_EXP_CLIP, _EXP_CLIP)) / self.m
        return matmat_ATsA(self.A, s) + self.regcoef * np.eye(self.n)


def make_oracle(kind, A, b, regcoef):
    if kind == "log_cosh":
        return LogCoshL2Oracle(A, b, regcoef)
    if kind == "exponential":
        return ExponentialLossL2Oracle(A, b, regcoef)
    raise ValueError("kind должен быть 'log_cosh' или 'exponential'.")

