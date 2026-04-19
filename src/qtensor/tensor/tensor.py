from __future__ import annotations
import jax.numpy as jnp
from typing import Tuple
from qtensor.tensor.index import Index


class Tensor:
    """
    Basic tensor object used in tensor networks.

    A Tensor consists of:
        - data : a JAX array
        - inds : a tuple of Index objects describing each axis
    """

    def __init__(self, data: jnp.ndarray, inds: Tuple[Index, ...]):

        # rank check
        if data.ndim != len(inds):
            raise ValueError(
                f"Tensor rank mismatch: data.ndim={data.ndim}, len(inds)={len(inds)}"
            )

        # dimension check
        for axis, ind in enumerate(inds):
            if data.shape[axis] != ind.dim:
                raise ValueError(
                    f"Dimension mismatch at axis {axis}: "
                    f"data.shape={data.shape[axis]} vs index.dim={ind.dim}"
                )

        self.data = data
        self.inds = inds

    # -------------------------------------------------
    # basic properties
    # -------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype


    # -------------------------------------------------
    # index utilities
    # -------------------------------------------------

    def ind_pos(self, ind: Index) -> int:
        """Return axis position of an Index."""
        try:
            return self.inds.index(ind)
        except ValueError:
            raise ValueError("Index not found in tensor")

    def has_ind(self, ind: Index) -> bool:
        """Check if tensor contains an Index."""
        return ind in self.inds

    # -------------------------------------------------
    # basic tensor operations
    # -------------------------------------------------
    
    def get_ind(self, name: str) -> Index:
        """Return the Index with the given name."""
        for ind in self.inds:
            if ind.name == name:
                return ind
        raise ValueError(f"Index with name '{name}' not found")

    def replace_ind(self, old: Index, new: Index) -> Tensor:
        """
        Replace an Index with another Index (dimensions must match).
        Returns a new Tensor.
        """
        if old not in self.inds:
            raise ValueError("Old index not found in tensor")

        axis = self.ind_pos(old)

        if new.dim != old.dim:
            raise ValueError(
                f"Cannot replace index with different dimension: {old.dim} != {new.dim}"
            )

        new_inds = list(self.inds)
        new_inds[axis] = new

        return Tensor(self.data, tuple(new_inds))


    def rename_inds(self, mapping: dict[Index, Index]) -> "Tensor":
        """
        Replace multiple indices according to a mapping.
        Dimensions of replaced indices must match.
        Returns a new Tensor.
        """
        new_inds = []
        for ind in self.inds:
            if ind in mapping:
                new_ind = mapping[ind]
                if new_ind.dim != ind.dim:
                    raise ValueError(
                        f"Cannot replace index with different dimension: {ind.dim} != {new_ind.dim}"
                    )
                new_inds.append(new_ind)
            else:
                new_inds.append(ind)
        
        if len(set(new_inds)) != len(new_inds):
            raise ValueError("Renaming indices produced duplicate indices")
        
        return Tensor(self.data, tuple(new_inds))
    
    def prime_inds(self, inds: Tuple[Index, ...] | None = None, n: int = 1) -> "Tensor":
        """
        Prime selected indices. If inds is None, prime all indices.
        Returns a new Tensor.
        """
        if inds is None:
            inds = self.inds
        target = set(inds)
        mapping = {ind: ind.prime(n) for ind in self.inds if ind in target}
        return self.rename_inds(mapping)

    def unprime_inds(self, inds: Tuple[Index, ...] | None = None, n: int = 1) -> "Tensor":
        """
        Unprime selected indices. If inds is None, unprime all indices.
        Returns a new Tensor.
        """
        if inds is None:
            inds = self.inds
        target = set(inds)
        mapping = {ind: ind.unprime(n) for ind in self.inds if ind in target}
        return self.rename_inds(mapping)

    # -------------------------------------------------
    # basic tensor operations
    # -------------------------------------------------

    def _check_same_structure(self, other: "Tensor"):
        if not isinstance(other, Tensor):
            raise TypeError("Operand must be a Tensor")
        if self.inds != other.inds:
            raise ValueError("Tensor indices must match for arithmetic operations")

    def __add__(self, other: "Tensor") -> "Tensor":
        """Elementwise tensor addition (indices must match)."""
        self._check_same_structure(other)
        return Tensor(self.data + other.data, self.inds)

    def __sub__(self, other: "Tensor") -> "Tensor":
        """Elementwise tensor subtraction (indices must match)."""
        self._check_same_structure(other)
        return Tensor(self.data - other.data, self.inds)

    def __neg__(self) -> "Tensor":
        """Unary negation."""
        return Tensor(-self.data, self.inds)

    def __mul__(self, other) -> "Tensor":
        """Scalar multiplication: Tensor * scalar."""
        if isinstance(other, Tensor):
            raise TypeError("Tensor * Tensor is not defined. Use contract().")
        return Tensor(self.data * other, self.inds)
    
    def __truediv__(self, other) -> "Tensor":
        """Scalar division: Tensor / scalar."""
        if other == 0:
            raise ZeroDivisionError("division by zero")
        if isinstance(other, Tensor):
            raise TypeError("Tensor / Tensor is not defined. Use contract().")
        return Tensor(self.data / other, self.inds)

    def __matmul__(self, other):
        """
        Lazy tensor-network contraction.

        `A @ B @ C` builds a contraction chain and materializes it only when the
        resulting tensor data or indices are accessed. Materialization uses the
        general `contract(*tensors)` routine, so shared indices are detected
        automatically and the contraction path is optimized globally across the
        whole chain.
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        return TensorContraction((self, other))

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        return TensorContraction((other, self))

    def __rmul__(self, other) -> "Tensor":
        """Scalar multiplication: scalar * Tensor."""
        return Tensor(other * self.data, self.inds)

    def copy(self) -> "Tensor":
        """
        Return a shallow copy of the Tensor with copied array data.
        
        Note:
        For JAX arrays this should be understood as a logical copy of the
        Tensor container, not a strict guarantee of independent physical
        storage.
        """
        return Tensor(jnp.array(self.data, copy=True), self.inds)

    def astype(self, dtype) -> "Tensor":
        """Return a new Tensor with array cast to the given dtype."""
        return Tensor(self.data.astype(dtype), self.inds)

    def conj(self) -> "Tensor":
        """Return the complex conjugate of the Tensor."""
        return Tensor(jnp.conjugate(self.data), self.inds)

    def norm(self):
        """Return the Frobenius norm of the tensor."""
        return jnp.linalg.norm(self.data)

    def max(self):
        """Return the maximum element value of the tensor."""
        return jnp.max(self.data)
    
    def min(self):
        """Return the minimum element value of the tensor."""
        return jnp.min(self.data)

    def permute(self, new_inds: Tuple[Index, ...]) -> Tensor:
        """
        Permute tensor axes according to a new index ordering.
        Returns a new Tensor. 
        """

        if set(new_inds) != set(self.inds):
            raise ValueError("new_inds must contain the same indices")

        perm = tuple(self.ind_pos(ind) for ind in new_inds)
        new_data = jnp.transpose(self.data, perm)

        return Tensor(new_data, new_inds)

    def to_array(self, *inds: Index) -> jnp.ndarray:
        """
        Return the raw array with axes ordered according to `inds`.

        Examples
        --------
        `A.to_array(i, j, k)` returns `A.data` permuted so that its axes are in
        the order `(i, j, k)`.
        """
        if len(inds) == 1 and isinstance(inds[0], tuple):
            inds = inds[0]
        if len(inds) == 0:
            return self.data
        return self.permute(tuple(inds)).data
    
    # -------------------------------------------------
    # printing
    # -------------------------------------------------

    def __repr__(self):
        ind_str = ", ".join([repr(i) for i in self.inds])
        return f"Tensor(shape={self.shape}, inds=[{ind_str}])"


class TensorContraction(Tensor):
    """
    Lazy contraction chain produced by `Tensor.__matmul__`.

    The chain keeps the original tensor list and calls `contract(*tensors)` only
    when the result is needed. This preserves global contraction-path
    optimization for expressions like `A @ B @ C @ D`.
    """

    def __init__(
        self,
        tensors: Tuple[Tensor, ...],
        optimize: str | bool = "auto",
    ):
        flat_tensors = []
        for tensor in tensors:
            if isinstance(tensor, TensorContraction):
                flat_tensors.extend(tensor._tensors)
            elif isinstance(tensor, Tensor):
                flat_tensors.append(tensor)
            else:
                raise TypeError("TensorContraction operands must be Tensor objects")

        if len(flat_tensors) < 2:
            raise ValueError("TensorContraction requires at least two tensors")

        self._tensors = tuple(flat_tensors)
        self._optimize = optimize
        self._materialized: Tensor | None = None

    def materialize(self) -> Tensor:
        if self._materialized is None:
            from qtensor.tensor.contract import contract

            self._materialized = contract(*self._tensors, optimize=self._optimize)
        return self._materialized

    def compute(self) -> Tensor:
        return self.materialize()

    def eval(self) -> Tensor:
        return self.materialize()
    
    @property
    def data(self):
        return self.materialize().data

    @property
    def inds(self):
        return self.materialize().inds

    @property
    def shape(self):
        return self.materialize().shape

    @property
    def ndim(self):
        return self.materialize().ndim

    @property
    def size(self):
        return self.materialize().size

    @property
    def dtype(self):
        return self.materialize().dtype

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        return TensorContraction(self._tensors + (other,), optimize=self._optimize)

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        return TensorContraction((other,) + self._tensors, optimize=self._optimize)

    def __repr__(self):
        if self._materialized is None:
            return (
                f"TensorContraction(ntensors={len(self._tensors)}, "
                f"optimize={self._optimize!r})"
            )
        return repr(self._materialized)
