from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np
from qtensor.tensor.tensor import Tensor

try:
    from scipy.sparse.linalg import ArpackNoConvergence
    from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
    from scipy.sparse.linalg import eigs as scipy_eigs
    from scipy.sparse.linalg import expm_multiply as scipy_expm_multiply
    from scipy.sparse.linalg import eigsh as scipy_eigsh
except Exception:  # pragma: no cover - optional dependency
    ArpackNoConvergence = None
    ScipyLinearOperator = None
    scipy_eigs = None
    scipy_expm_multiply = None
    scipy_eigsh = None


class AbstractLinearOperator(ABC):
    """
    Abstract tensor-native linear operator.

    This class is designed for matrix-free algorithms such as Lanczos or
    Arnoldi, where the operator is never formed as a dense matrix. Instead,
    subclasses implement `apply(x)` which maps an input Tensor to an output
    Tensor with compatible index structure.

    Intended use cases include:
        - one-site effective Hamiltonians in DMRG
        - two-site effective Hamiltonians in DMRG
        - three-site effective Hamiltonians in future extensions
        - other abstract matrix-free tensor-network operators
    """

    def __init__(
        self,
        *,
        domain_inds: tuple[Any, ...],
        codomain_inds: tuple[Any, ...] | None = None,
        dtype: Any | None = None,
        is_hermitian: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        self._domain_inds = tuple(domain_inds)
        self._codomain_inds = tuple(domain_inds if codomain_inds is None else codomain_inds)
        self._dtype = dtype
        self._is_hermitian = is_hermitian
        self.metadata = {} if metadata is None else dict(metadata)

    @property
    def domain_inds(self) -> tuple[Any, ...]:
        """Index structure expected for input tensors."""
        return self._domain_inds

    @property
    def codomain_inds(self) -> tuple[Any, ...]:
        """Index structure returned by the operator action."""
        return self._codomain_inds

    @property
    def dtype(self) -> Any | None:
        """Numerical dtype associated with the operator."""
        return self._dtype

    @property
    def is_hermitian(self) -> bool:
        """Whether the operator is intended to be Hermitian."""
        return self._is_hermitian

    def check_input(self, x: Tensor) -> None:
        """
        Validate that `x` is compatible with the operator domain.
        """
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected Tensor input, got {type(x)!r}")
        if x.inds != self.domain_inds:
            raise ValueError(
                "Input tensor indices do not match operator domain.\n"
                f"Expected: {self.domain_inds}\n"
                f"Got:      {x.inds}"
            )

    def check_output(self, y: Tensor) -> None:
        """
        Validate that the operator output matches the declared codomain.
        """
        if not isinstance(y, Tensor):
            raise TypeError(f"Operator output must be a Tensor, got {type(y)!r}")
        if y.inds != self.codomain_inds:
            raise ValueError(
                "Output tensor indices do not match operator codomain.\n"
                f"Expected: {self.codomain_inds}\n"
                f"Got:      {y.inds}"
            )

    @abstractmethod
    def apply(self, x: Tensor) -> Tensor:
        """
        Apply the linear operator to `x`.

        Subclasses should implement the actual matrix-free action here.
        """
        raise NotImplementedError

    def apply_data(self, x_data):
        """
        Apply the operator directly to raw array data with the operator's
        domain shape. Subclasses may override this to avoid Tensor wrapping in
        Krylov hot loops.
        """
        x = Tensor(x_data, self.domain_inds)
        return self.apply(x).data

    def __call__(self, x: Tensor) -> Tensor:
        y = self.apply(x)
        self.check_output(y)
        return y

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.__call__(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"domain_inds={self.domain_inds}, "
            f"codomain_inds={self.codomain_inds}, "
            f"dtype={self.dtype}, "
            f"is_hermitian={self.is_hermitian})"
        )


class CallableLinearOperator(AbstractLinearOperator):
    """
    Minimal concrete wrapper around a callable.

    This is useful for quickly exposing a matrix-free action as an
    `AbstractLinearOperator` before defining a more structured subclass.
    """

    def __init__(
        self,
        fn,
        *,
        domain_inds: tuple[Any, ...],
        codomain_inds: tuple[Any, ...] | None = None,
        dtype: Any | None = None,
        is_hermitian: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(
            domain_inds=domain_inds,
            codomain_inds=codomain_inds,
            dtype=dtype,
            is_hermitian=is_hermitian,
            metadata=metadata,
        )
        self.fn = fn

    def apply(self, x: Tensor) -> Tensor:
        self.check_input(x)
        return self.fn(x)

    def apply_data(self, x_data):
        y = self.fn(Tensor(x_data, self.domain_inds))
        self.check_output(y)
        return y.data


class EffectiveOneSiteHamiltonian(AbstractLinearOperator):
    """
    Matrix-free effective one-site Hamiltonian.

    This operator represents the standard DMRG effective Hamiltonian acting on
    a single-site tensor `A`. It is defined by a left environment, one local
    MPO tensor, and a right environment, and never forms a dense matrix.

    The operator acts on a Tensor with indices
        (left_link, site, right_link)
    and returns a Tensor with the same index structure.
    """

    def __init__(
        self,
        *,
        left_env: Tensor,
        W: Tensor,
        right_env: Tensor,
        domain_inds: tuple[Any, ...],
        codomain_inds: tuple[Any, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        dtype = jnp.result_type(left_env.data, W.data, right_env.data)
        super().__init__(
            domain_inds=domain_inds,
            codomain_inds=codomain_inds,
            dtype=dtype,
            is_hermitian=True,
            metadata=metadata,
        )
        self.left_env = left_env
        self.W = W
        self.right_env = right_env

    def apply(self, x: Tensor) -> Tensor:
        """
        Apply the effective one-site Hamiltonian to a one-site tensor `x`.

        The input tensor is expected to carry the same index structure as
        `domain_inds`, typically
            (left_link, site, right_link).

        Internally, `x` is first aligned with the ket physical leg used by the
        effective-Hamiltonian network. After contraction, the resulting tensor
        naturally carries the bra-link legs of the environments and the MPO
        `site_in` leg. These are then renamed back to `codomain_inds` so the
        operator acts within a single tensor space.
        """
        self.check_input(x)

        y_data = jnp.einsum(
            "apb,pijq,xqy,bjy->aix",
            self.left_env.data,
            self.W.data,
            self.right_env.data,
            x.data,
            optimize="auto",
        )
        return Tensor(y_data, self.codomain_inds)

    def apply_data(self, x_data):
        return jnp.einsum(
            "apb,pijq,xqy,bjy->aix",
            self.left_env.data,
            self.W.data,
            self.right_env.data,
            x_data,
            optimize="auto",
        )

    def __repr__(self) -> str:
        return (
            f"EffectiveOneSiteHamiltonian("
            f"domain_inds={self.domain_inds}, "
            f"codomain_inds={self.codomain_inds}, "
            f"dtype={self.dtype})"
        )

class EffectiveTwoSiteHamiltonian(AbstractLinearOperator):
    """
    Matrix-free effective two-site Hamiltonian.

    This operator represents the standard DMRG effective Hamiltonian acting on
    a two-site tensor `theta`. It is defined by a left environment, two local
    MPO tensors, and a right environment, and never forms a dense matrix.

    Expected network structure:
                      |     |
        left_env -- W1 -- W2 -- right_env
                      |     |
                    theta  theta

    The operator acts on a Tensor with indices
        (left_link, site_1, site_2, right_link)
    and returns a Tensor with the same index structure.
    """

    def __init__(
        self,
        *,
        left_env: Tensor,
        W1: Tensor,
        W2: Tensor,
        right_env: Tensor,
        domain_inds: tuple[Any, ...],
        codomain_inds: tuple[Any, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        dtype = jnp.result_type(
            left_env.data,
            W1.data,
            W2.data,
            right_env.data,
        )
        super().__init__(
            domain_inds=domain_inds,
            codomain_inds=codomain_inds,
            dtype=dtype,
            is_hermitian=True,
            metadata=metadata,
        )
        self.left_env = left_env
        self.W1 = W1
        self.W2 = W2
        self.right_env = right_env

    def apply(self, x: Tensor) -> Tensor:
        """
        Apply the effective two-site Hamiltonian to a two-site tensor `x`.

        The input tensor is expected to carry the same index structure as
        `domain_inds`, typically
            (left_link, site_1, site_2, right_link).

        Internally, `x` is first aligned with the ket legs used by the
        effective-Hamiltonian network:
            - local site legs are matched to the MPO `site_out` legs
            - the left/right virtual legs already match the ket-link legs of
              the environments by construction

        After contraction, the resulting tensor naturally carries the bra-link
        legs of the environments and the MPO `site_in` legs. These are then
        renamed back to `codomain_inds` so the operator acts within a single
        tensor space.
        """
        self.check_input(x)

        y_data = jnp.einsum(
            "apb,pijq,qklr,xry,bjly->aikx",
            self.left_env.data,
            self.W1.data,
            self.W2.data,
            self.right_env.data,
            x.data,
            optimize="auto",
        )
        return Tensor(y_data, self.codomain_inds)

    def apply_data(self, x_data):
        return jnp.einsum(
            "apb,pijq,qklr,xry,bjly->aikx",
            self.left_env.data,
            self.W1.data,
            self.W2.data,
            self.right_env.data,
            x_data,
            optimize="auto",
        )

    def __repr__(self) -> str:
        return (
            f"EffectiveTwoSiteHamiltonian("
            f"domain_inds={self.domain_inds}, "
            f"codomain_inds={self.codomain_inds}, "
            f"dtype={self.dtype})"
        )


class EffectiveZeroSiteHamiltonian(AbstractLinearOperator):
    """
    Matrix-free effective zero-site Hamiltonian acting on a bond-center matrix.

    The operator is defined by the left and right environments on the same bond.
    If the bond-center matrix carries data `C[b, y]`, the action is

        (K C)[a, x] = sum_{p, b, y} L[a, p, b] R[x, p, y] C[b, y]

    where `L` and `R` are the left and right environments.
    """

    def __init__(
        self,
        *,
        left_env: Tensor,
        right_env: Tensor,
        domain_inds: tuple[Any, ...],
        codomain_inds: tuple[Any, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        dtype = jnp.result_type(left_env.data, right_env.data)
        super().__init__(
            domain_inds=domain_inds,
            codomain_inds=codomain_inds,
            dtype=dtype,
            is_hermitian=True,
            metadata=metadata,
        )
        self.left_env = left_env
        self.right_env = right_env

    def apply(self, x: Tensor) -> Tensor:
        self.check_input(x)
        y_data = jnp.einsum(
            "apb,xpy,by->ax",
            self.left_env.data,
            self.right_env.data,
            x.data,
            optimize="auto",
        )
        return Tensor(y_data, self.codomain_inds)

    def apply_data(self, x_data):
        return jnp.einsum(
            "apb,xpy,by->ax",
            self.left_env.data,
            self.right_env.data,
            x_data,
            optimize="auto",
        )

    def __repr__(self) -> str:
        return (
            f"EffectiveZeroSiteHamiltonian("
            f"domain_inds={self.domain_inds}, "
            f"codomain_inds={self.codomain_inds}, "
            f"dtype={self.dtype})"
        )


# === Tensor-native Krylov solvers ===

def _array_inner(x_data, y_data):
    return jnp.vdot(x_data.reshape(-1), y_data.reshape(-1))


def _array_norm(x_data):
    val = _array_inner(x_data, x_data)
    return jnp.sqrt(jnp.real(val))


def _flatten_array(x_data):
    return x_data.reshape(-1)


def _krylov_expm_action_matrix(hessenberg, rhs, scale=1):
    evals, evecs = jnp.linalg.eigh(hessenberg)
    coeffs = jnp.conjugate(evecs).T @ rhs
    return evecs @ (jnp.exp(scale * evals) * coeffs)


def _lanczos_factorization(
    A: AbstractLinearOperator,
    x0: Tensor,
    maxiter: int,
    tol: float,
):
    A.check_input(x0)
    x0_data = x0.data
    beta0 = _array_norm(x0_data)
    if beta0 == 0:
        raise ValueError("Initial tensor x0 must have nonzero norm")

    q_data = x0_data / beta0
    q_prev_flat = None
    beta_prev = jnp.array(0.0, dtype=jnp.result_type(beta0))

    basis_flat: list[Any] = []
    alphas = []
    betas = []

    for k in range(maxiter):
        q_flat = _flatten_array(q_data)
        basis_flat.append(q_flat)
        z_data = A.apply_data(q_data)
        z_flat = _flatten_array(z_data)

        alpha = jnp.vdot(q_flat, z_flat)
        alphas.append(alpha)

        z_flat = z_flat - alpha * q_flat
        if q_prev_flat is not None:
            z_flat = z_flat - beta_prev * q_prev_flat

        for qi_flat in basis_flat:
            hij = jnp.vdot(qi_flat, z_flat)
            z_flat = z_flat - hij * qi_flat
        z_data = z_flat.reshape(x0_data.shape)

        beta = _array_norm(z_data)
        if k < maxiter - 1:
            betas.append(beta)

        if beta <= tol:
            break

        q_prev_flat = q_flat
        q_data = z_data / beta
        beta_prev = beta

    m = len(alphas)
    T = jnp.zeros((m, m), dtype=jnp.result_type(*alphas, *betas if betas else [0.0]))
    for i in range(m):
        T = T.at[i, i].set(alphas[i])
        if i < m - 1:
            T = T.at[i, i + 1].set(betas[i])
            T = T.at[i + 1, i].set(jnp.conj(betas[i]))

    return {
        "basis_vectors": tuple(basis_flat),
        "alphas": tuple(alphas),
        "betas": tuple(betas),
        "tridiagonal": T,
        "iterations": m,
        "initial_norm": beta0,
        "shape": x0_data.shape,
    }


def krylov_expm_multiply(
    A: AbstractLinearOperator,
    x: Tensor,
    dt: complex,
    maxiter: int = 10,
    tol: float = 1e-12,
    normalize: bool = False,
):
    """
    Approximate exp(dt A) x using a Lanczos projection built from
    `lanczos_lowest_eigenpair`.
    """
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if not A.is_hermitian:
        raise ValueError("krylov_expm_multiply requires a Hermitian operator")

    _, _, info = lanczos_lowest_eigenpair(
        A,
        x,
        maxiter=maxiter,
        tol=tol,
        max_restarts=0,
    )
    T = info["tridiagonal"]
    beta = info["initial_norm"]
    basis_flat = info["basis_vectors"]
    x_shape = info["shape"]

    scale = jnp.asarray(dt, dtype=jnp.result_type(x.data, dt))
    e1 = jnp.zeros((T.shape[0],), dtype=jnp.result_type(T, scale))
    e1 = e1.at[0].set(beta)
    coeffs = _krylov_expm_action_matrix(T, e1, scale=scale)

    y_data = jnp.zeros_like(x.data, dtype=jnp.result_type(x.data, coeffs))
    for c, q_flat in zip(coeffs, basis_flat):
        y_data = y_data + c * q_flat.reshape(x_shape)

    if normalize:
        nrm = jnp.linalg.norm(y_data.reshape(-1))
        if nrm > 0:
            y_data = y_data / nrm

    return Tensor(y_data, x.inds)


def scipy_expm_multiply_operator(
    A: AbstractLinearOperator,
    x: Tensor,
    dt: complex,
    normalize: bool = False,
    traceA=None,
):
    """
    Apply the matrix exponential of an abstract operator using
    scipy.sparse.linalg.expm_multiply.
    """
    if scipy_expm_multiply is None or ScipyLinearOperator is None:
        raise RuntimeError(
            "scipy.sparse.linalg.expm_multiply is not available"
        )

    A.check_input(x)
    shape = x.shape
    dim = x.size
    scale = dt
    dtype = np.result_type(
        np.asarray(x.data),
        np.asarray(scale),
        np.complex128,
    )

    def apply_scaled(v, scale_factor):
        v_arr = jnp.asarray(v).reshape(shape)
        y_arr = A.apply_data(v_arr).reshape(-1)
        y_arr = scale_factor * y_arr
        return np.asarray(y_arr, dtype=dtype)

    def matvec(v):
        return apply_scaled(v, scale)

    def rmatvec(v):
        return apply_scaled(v, np.conjugate(scale))

    def matmat(X):
        cols = [matvec(col) for col in np.asarray(X).T]
        return np.column_stack(cols)

    def rmatmat(X):
        cols = [rmatvec(col) for col in np.asarray(X).T]
        return np.column_stack(cols)

    linop = ScipyLinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=dtype,
    )
    x_vec = np.asarray(x.data.reshape(-1), dtype=dtype)
    y_vec = scipy_expm_multiply(linop, x_vec, traceA=traceA)
    y_data = jnp.asarray(y_vec).reshape(shape)

    if normalize:
        nrm = jnp.linalg.norm(y_data.reshape(-1))
        if nrm > 0:
            y_data = y_data / nrm

    return Tensor(y_data, x.inds)


def expm_multiply_operator(
    A: AbstractLinearOperator,
    x: Tensor,
    dt: complex,
    method: str = "krylov",
    normalize: bool = False,
    maxiter: int = 20,
    tol: float = 1e-12,
    traceA=None,
):
    """
    Apply the local TDVP propagator using either a Krylov method or SciPy's
    expm_multiply backend.
    """
    if method == "krylov":
        return krylov_expm_multiply(
            A,
            x,
            dt=dt,
            maxiter=maxiter,
            tol=tol,
            normalize=normalize,
        )
    if method == "scipy":
        return scipy_expm_multiply_operator(
            A,
            x,
            dt=dt,
            normalize=normalize,
            traceA=traceA,
        )
    raise ValueError("method must be 'krylov' or 'scipy'")


def scipy_lowest_eigsh(
    A: AbstractLinearOperator,
    x0: Tensor,
    tol: float = 1e-10,
    maxiter: int | None = None,
):
    """
    Compute the lowest eigenpair of a Hermitian abstract operator using
    scipy.sparse.linalg.eigsh.
    """
    if scipy_eigsh is None or ScipyLinearOperator is None:
        raise RuntimeError("scipy.sparse.linalg.eigsh is not available")
    if not A.is_hermitian:
        raise ValueError("scipy_lowest_eigsh requires a Hermitian operator")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if maxiter is not None and maxiter <= 0:
        raise ValueError("maxiter must be positive or None")

    A.check_input(x0)
    shape = x0.shape
    dim = x0.size
    dtype = np.result_type(np.asarray(x0.data), np.complex128)

    def matvec(v):
        v_arr = jnp.asarray(v).reshape(shape)
        y_arr = A.apply_data(v_arr).reshape(-1)
        return np.asarray(y_arr, dtype=dtype)

    def rmatvec(v):
        v_arr = jnp.asarray(v).reshape(shape)
        y_arr = A.apply_data(v_arr).reshape(-1)
        return np.asarray(y_arr, dtype=dtype)

    linop = ScipyLinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=dtype,
    )
    v0 = np.asarray(x0.data.reshape(-1), dtype=dtype)
    ncv = min(dim, max(20, 2 * 1 + 1))
    try:
        evals, evecs = scipy_eigsh(
            linop,
            k=1,
            which="SA",
            v0=v0,
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
        )
    except ArpackNoConvergence as exc:
        if getattr(exc, "eigenvalues", None) is not None and len(exc.eigenvalues) > 0:
            evals = exc.eigenvalues
            evecs = exc.eigenvectors
        else:
            retry_maxiter = max(100, 4 * (maxiter if maxiter is not None else 20))
            evals, evecs = scipy_eigsh(
                linop,
                k=1,
                which="SA",
                v0=v0,
                tol=tol,
                maxiter=retry_maxiter,
                ncv=ncv,
            )
    eigval = jnp.asarray(evals[0])
    eigvec_data = jnp.asarray(evecs[:, 0]).reshape(shape)
    nrm = _array_norm(eigvec_data)
    if nrm > 0:
        eigvec_data = eigvec_data / nrm
    eigvec = Tensor(eigvec_data, x0.inds)
    info = {
        "solver": "scipy_eigsh",
        "iterations": maxiter,
        "shape": shape,
    }
    return eigval, eigvec, info


def scipy_lowest_eigs(
    A: AbstractLinearOperator,
    x0: Tensor,
    tol: float = 1e-10,
    maxiter: int | None = None,
):
    """
    Compute the eigenpair with smallest real part using
    scipy.sparse.linalg.eigs.
    """
    if scipy_eigs is None or ScipyLinearOperator is None:
        raise RuntimeError("scipy.sparse.linalg.eigs is not available")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if maxiter is not None and maxiter <= 0:
        raise ValueError("maxiter must be positive or None")

    A.check_input(x0)
    shape = x0.shape
    dim = x0.size
    dtype = np.result_type(np.asarray(x0.data), np.complex128)

    def matvec(v):
        v_arr = jnp.asarray(v).reshape(shape)
        y_arr = A.apply_data(v_arr).reshape(-1)
        return np.asarray(y_arr, dtype=dtype)

    linop = ScipyLinearOperator(
        shape=(dim, dim),
        matvec=matvec,
        dtype=dtype,
    )
    v0 = np.asarray(x0.data.reshape(-1), dtype=dtype)
    ncv = min(dim, max(20, 2 * 1 + 1))
    try:
        evals, evecs = scipy_eigs(
            linop,
            k=1,
            which="SR",
            v0=v0,
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
        )
    except ArpackNoConvergence as exc:
        if getattr(exc, "eigenvalues", None) is not None and len(exc.eigenvalues) > 0:
            evals = exc.eigenvalues
            evecs = exc.eigenvectors
        else:
            retry_maxiter = max(100, 4 * (maxiter if maxiter is not None else 20))
            evals, evecs = scipy_eigs(
                linop,
                k=1,
                which="SR",
                v0=v0,
                tol=tol,
                maxiter=retry_maxiter,
                ncv=ncv,
            )
    eigval = jnp.asarray(evals[0])
    eigvec_data = jnp.asarray(evecs[:, 0]).reshape(shape)
    nrm = _array_norm(eigvec_data)
    if nrm > 0:
        eigvec_data = eigvec_data / nrm
    eigvec = Tensor(eigvec_data, x0.inds)
    info = {
        "solver": "scipy_eigs",
        "iterations": maxiter,
        "shape": shape,
    }
    return eigval, eigvec, info


def lanczos_lowest_eigenpair(
    A: AbstractLinearOperator,
    x0: Tensor,
    maxiter: int = 20,
    tol: float = 1e-10,
    max_restarts: int = 0,
):
    """
    Approximate the lowest eigenpair of a Hermitian abstract linear operator
    using a basic Lanczos iteration.

    Parameters
    ----------
    A : AbstractLinearOperator
        Hermitian matrix-free operator.
    x0 : Tensor
        Initial tensor in the operator domain.
    maxiter : int
        Maximum Lanczos iterations.
    tol : float
        Early-stop tolerance based on the subdiagonal coefficient.

    Returns
    -------
    eigval : scalar
        Approximate lowest Ritz eigenvalue.
    eigvec : Tensor
        Approximate lowest Ritz eigenvector with the same index structure as x0.
    info : dict
        Lightweight diagnostics including the tridiagonal matrix coefficients.
    """
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if not A.is_hermitian:
        raise ValueError("Lanczos requires a Hermitian operator")

    A.check_input(x0)
    x0_inds = x0.inds

    def _run_once(x0_local: Tensor):
        info = _lanczos_factorization(A, x0_local, maxiter=maxiter, tol=tol)
        T = info["tridiagonal"]
        evals, evecs = jnp.linalg.eigh(T)
        idx = jnp.argmin(jnp.real(evals))
        eigval = evals[idx]
        coeffs = evecs[:, idx]

        data = jnp.zeros_like(x0_local.data)
        for c, qv_flat in zip(coeffs, info["basis_vectors"]):
            data = data + c * qv_flat.reshape(x0_local.data.shape)
        nrm = _array_norm(data)
        if nrm > 0:
            data = data / nrm

        eigvec = Tensor(data, x0_inds)

        return eigval, eigvec, info

    eigval, eigvec, info = _run_once(x0)
    for _ in range(max_restarts):
        eigval, eigvec, info = _run_once(eigvec)
    return eigval, eigvec, info


def arnoldi_eigenpair(
    A: AbstractLinearOperator,
    x0: Tensor,
    maxiter: int = 20,
    tol: float = 1e-10,
    which: str = "LM",
    max_restarts: int = 0,
):
    """
    Basic Arnoldi iteration for a general (not necessarily Hermitian)
    abstract linear operator.

    Parameters
    ----------
    A : AbstractLinearOperator
        Matrix-free operator.
    x0 : Tensor
        Initial tensor in the operator domain.
    maxiter : int
        Maximum Arnoldi iterations.
    tol : float
        Early-stop tolerance based on the next Krylov vector norm.
    which : str
        Ritz value selection rule. Supported values:
            - "LM": largest magnitude
            - "SM": smallest magnitude
            - "LR": largest real part
            - "SR": smallest real part

    Returns
    -------
    eigval : scalar
        Selected Ritz eigenvalue.
    eigvec : Tensor
        Corresponding Ritz vector lifted back to the tensor space.
    info : dict
        Lightweight diagnostics including the Hessenberg matrix.
    """
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if which not in ("LM", "SM", "LR", "SR"):
        raise ValueError("which must be one of 'LM', 'SM', 'LR', 'SR'")

    A.check_input(x0)
    x0_inds = x0.inds

    def _run_once(x0_local: Tensor):
        x0_data = x0_local.data
        beta = _array_norm(x0_data)
        if beta == 0:
            raise ValueError("Initial tensor x0 must have nonzero norm")

        q0_data = x0_data / beta
        basis_data: list[Any] = [q0_data]
        H = jnp.zeros((maxiter + 1, maxiter), dtype=A.dtype if A.dtype is not None else q0_data.dtype)

        m = 0
        for j in range(maxiter):
            v_data = A.apply_data(basis_data[j])

            # classical GS (first pass)
            for i in range(j + 1):
                hij = _array_inner(basis_data[i], v_data)
                H = H.at[i, j].set(hij)
                v_data = v_data - hij * basis_data[i]

            # full reorthogonalization (second pass)
            for i in range(j + 1):
                hij = _array_inner(basis_data[i], v_data)
                H = H.at[i, j].add(hij)
                v_data = v_data - hij * basis_data[i]

            h_next = _array_norm(v_data)
            H = H.at[j + 1, j].set(h_next)
            m = j + 1

            if h_next <= tol or j == maxiter - 1:
                break

            basis_data.append(v_data / h_next)

        Hm = H[:m, :m]
        evals, evecs = jnp.linalg.eig(Hm)

        if which == "LM":
            idx = jnp.argmax(jnp.abs(evals))
        elif which == "SM":
            idx = jnp.argmin(jnp.abs(evals))
        elif which == "LR":
            idx = jnp.argmax(jnp.real(evals))
        else:
            idx = jnp.argmin(jnp.real(evals))

        eigval = evals[idx]
        coeffs = evecs[:, idx]

        data = jnp.zeros_like(basis_data[0])
        for c, qv_data in zip(coeffs, basis_data[:m]):
            data = data + c * qv_data

        nrm = _array_norm(data)
        if nrm > 0:
            data = data / nrm

        eigvec = Tensor(data, x0_inds)

        info = {
            "iterations": m,
            "hessenberg": H[: m + 1, :m],
        }
        return eigval, eigvec, info

    eigval, eigvec, info = _run_once(x0)
    for _ in range(max_restarts):
        eigval, eigvec, info = _run_once(eigvec)
    return eigval, eigvec, info
