from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class SiteType:
    """Base class for local physical Hilbert spaces."""

    name: str
    basis_labels: Tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.basis_labels)

    def states(self) -> tuple[str, ...]:
        """Return the canonical basis labels for this site type."""
        return self.basis_labels

    def ops(self) -> tuple[str, ...]:
        """Return the available local operator names."""
        return tuple(self._ops().keys())

    def state(self, label: str):
        """Return the local basis vector corresponding to `label`."""
        aliases = self._state_aliases()
        if label not in aliases:
            raise ValueError(
                f"Unknown state label '{label}' for {self.name}. "
                f"Allowed labels: {sorted(aliases.keys())}"
            )

        idx = aliases[label]
        vec = jnp.zeros((self.dim,))
        return vec.at[idx].set(1.0)

    def eye(self):
        """Return the identity operator on the local Hilbert space."""
        return self.op("I")

    def projector(self, label: str):
        """Return the rank-1 projector onto the named local state."""
        vec = self.state(label)
        return jnp.outer(vec, jnp.conjugate(vec))

    def op(self, name: str, *args):
        """Return a local operator. Some operators may accept parameters."""
        ops = self._ops()

        if name not in ops:
            raise ValueError(
                f"Unknown operator '{name}' for {self.name}. "
                f"Allowed operators: {sorted(ops.keys())}"
            )

        op_obj = ops[name]

        if callable(op_obj):
            return op_obj(*args)

        if args:
            raise ValueError(f"Operator '{name}' for {self.name} does not take parameters")

        return op_obj

    def _state_aliases(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.basis_labels)}

    def _ops(self) -> dict[str, jnp.ndarray]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class QubitSite(SiteType):
    """Qubit local Hilbert space with common states and single-qubit gates."""

    def __init__(self):
        object.__setattr__(self, "name", "Qubit")
        object.__setattr__(self, "basis_labels", ("0", "1"))

    def _state_vectors(self):
        s2 = jnp.sqrt(2.0)
        return {
            "0": jnp.array([1.0, 0.0]),
            "1": jnp.array([0.0, 1.0]),
            "up": jnp.array([1.0, 0.0]),
            "dn": jnp.array([0.0, 1.0]),
            "+z": jnp.array([1.0, 0.0]),
            "-z": jnp.array([0.0, 1.0]),
            "X+": jnp.array([1.0, 1.0]) / s2,
            "X-": jnp.array([1.0, -1.0]) / s2,
            "+x": jnp.array([1.0, 1.0]) / s2,
            "-x": jnp.array([1.0, -1.0]) / s2,
            "Y+": jnp.array([1.0, 1.0j]) / s2,
            "Y-": jnp.array([1.0, -1.0j]) / s2,
            "+y": jnp.array([1.0, 1.0j]) / s2,
            "-y": jnp.array([1.0, -1.0j]) / s2,
        }

    def state(self, label: str):
        """Return computational, X, and Y eigenstates."""
        basis = self._state_vectors()

        if label not in basis:
            raise ValueError(
                f"Unknown state label '{label}' for {self.name}. "
                f"Allowed labels: {sorted(basis.keys())}"
            )

        return basis[label]

    def _state_aliases(self) -> dict[str, int]:
        return {
            "0": 0,
            "1": 1,
            "up": 0,
            "dn": 1,
            "+z": 0,
            "-z": 1,
        }

    def _ops(self):
        I = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        X = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        Y = jnp.array([[0.0, -1.0j], [1.0j, 0.0]])
        Z = jnp.array([[1.0, 0.0], [0.0, -1.0]])

        P0 = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        P1 = jnp.array([[0.0, 0.0], [0.0, 1.0]])

        def rx(theta):
            return jnp.cos(theta / 2) * I - 1j * jnp.sin(theta / 2) * X

        def ry(theta):
            return jnp.cos(theta / 2) * I - 1j * jnp.sin(theta / 2) * Y

        def rz(theta):
            return jnp.cos(theta / 2) * I - 1j * jnp.sin(theta / 2) * Z

        def phase(phi):
            return jnp.array([
                [1.0, 0.0],
                [0.0, jnp.exp(1j * phi)],
            ])

        def rot(axis: str, theta):
            axis_map = {"x": X, "y": Y, "z": Z, "X": X, "Y": Y, "Z": Z}

            if axis not in axis_map:
                raise ValueError("axis must be one of 'x','y','z'")

            sigma = axis_map[axis]
            return jnp.cos(theta / 2) * I - 1j * jnp.sin(theta / 2) * sigma

        return {
            "Id": I,
            "I": I,

            "X": X,
            "Y": Y,
            "Z": Z,

            "P0": P0,
            "P1": P1,

            "Sx": 0.5 * X,
            "Sy": 0.5 * Y,
            "Sz": 0.5 * Z,

            "Sp": jnp.array([[0.0, 1.0], [0.0, 0.0]]),
            "Sm": jnp.array([[0.0, 0.0], [1.0, 0.0]]),

            "Rx": rx,
            "Ry": ry,
            "Rz": rz,

            "Phase": phase,
            "Rot": rot,
        }
