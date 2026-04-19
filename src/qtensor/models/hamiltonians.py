from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from qtensor.models.sites import SiteType
from qtensor.mps.mpo import MPO
from qtensor.tensor.index import Index
from qtensor.tensor.tensor import Tensor



def ising_mpo(
    site_types: Sequence[SiteType],
    J: float = 1.0,
    h: float = 0.0,
    g: float = 0.0,
    link_prefix: str = "link",
    site_prefix: str = "site",
) -> MPO:
    """
    Construct the open-boundary transverse-field Ising MPO

        H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i - g * sum_i Z_i

    with MPO bond dimension 3.

    Parameters
    ----------
    site_types : Sequence[SiteType]
        Local site types. Each site must provide operators "I", "X", and "Z".
    J : float
        Nearest-neighbor ZZ coupling.
    h : float
        Transverse-field strength multiplying X.
    g : float
        Longitudinal-field strength multiplying Z.
    """
    if len(site_types) == 0:
        raise ValueError("ising_mpo requires at least one site")

    for n, stype in enumerate(site_types):
        for op_name in ("I", "X", "Z"):
            try:
                op = stype.op(op_name)
            except Exception as exc:
                raise ValueError(
                    f"Site type at site {n} does not provide required operator {op_name!r}"
                ) from exc
            if op.shape != (stype.dim, stype.dim):
                raise ValueError(
                    f"Operator {op_name!r} at site {n} has shape {op.shape}, "
                    f"expected ({stype.dim}, {stype.dim})"
                )

    nsites = len(site_types)

    if nsites == 1:
        stype = site_types[0]
        d = stype.dim
        left_link = Index(1, f"{link_prefix}-0", tags=("link", "0"))
        right_link = Index(1, f"{link_prefix}-1", tags=("link", "1"))
        site_in = Index(d, f"{site_prefix}_in-0", tags=("site_in", "0", stype.name))
        site_out = Index(d, f"{site_prefix}_out-0", tags=("site_out", "0", stype.name))

        X = stype.op("X")
        Z = stype.op("Z")
        data = (-h * X - g * Z).reshape(1, d, d, 1)
        return MPO((Tensor(data, (left_link, site_in, site_out, right_link)),), tuple(site_types), gauge_center=0)

    links = []
    for i in range(nsites + 1):
        dim = 1 if (i == 0 or i == nsites) else 3
        links.append(Index(dim, f"{link_prefix}-{i}", tags=("link", f"{i}")))

    tensors = []
    for n, stype in enumerate(site_types):
        d = stype.dim
        site_in = Index(d, f"{site_prefix}_in-{n}", tags=("site_in", f"{n}", stype.name))
        site_out = Index(d, f"{site_prefix}_out-{n}", tags=("site_out", f"{n}", stype.name))

        I = stype.op("I")
        X = stype.op("X")
        Z = stype.op("Z")
        onsite = -h * X - g * Z
        dtype = jnp.result_type(I, X, Z, J, h, g)

        if n == 0:
            # Left boundary row vector: [ onsite, -J Z, I ]
            data = jnp.zeros((1, d, d, 3), dtype=dtype)
            data = data.at[0, :, :, 0].set(onsite)
            data = data.at[0, :, :, 1].set(-J * Z)
            data = data.at[0, :, :, 2].set(I)
        elif n == nsites - 1:
            # Right boundary column vector: [ I, Z, onsite ]^T
            data = jnp.zeros((3, d, d, 1), dtype=dtype)
            data = data.at[0, :, :, 0].set(I)
            data = data.at[1, :, :, 0].set(Z)
            data = data.at[2, :, :, 0].set(onsite)
        else:
            # Bulk operator-valued matrix:
            # [[ I,  0,      0     ],
            #  [ Z,  0,      0     ],
            #  [ onsite, -J Z, I   ]]
            data = jnp.zeros((3, d, d, 3), dtype=dtype)
            data = data.at[0, :, :, 0].set(I)
            data = data.at[1, :, :, 0].set(Z)
            data = data.at[2, :, :, 0].set(onsite)
            data = data.at[2, :, :, 1].set(-J * Z)
            data = data.at[2, :, :, 2].set(I)

        tensors.append(Tensor(data, (links[n], site_in, site_out, links[n + 1])))

    return MPO(tuple(tensors), tuple(site_types), gauge_center=None)


def pxp_mpo(
    site_types: Sequence[SiteType],
    Omega: float = 1.0,
    link_prefix: str = "link",
    site_prefix: str = "site",
) -> MPO:
    """
    Construct the open-boundary PXP MPO

        H = Omega * (X_0 P_1 + sum_{i=1}^{N-2} P_{i-1} X_i P_{i+1} + P_{N-2} X_{N-1})

    where P = |0><0| is the projector onto the local ``"0"`` state.

    For a single site, this reduces to

        H = Omega * X_0

    and the MPO bond dimension is 4 for N >= 3.
    """
    if len(site_types) == 0:
        raise ValueError("pxp_mpo requires at least one site")

    for n, stype in enumerate(site_types):
        for op_name in ("I", "X", "P0"):
            try:
                op = stype.op(op_name)
            except Exception as exc:
                raise ValueError(
                    f"Site type at site {n} does not provide required operator {op_name!r}"
                ) from exc
            if op.shape != (stype.dim, stype.dim):
                raise ValueError(
                    f"Operator {op_name!r} at site {n} has shape {op.shape}, "
                    f"expected ({stype.dim}, {stype.dim})"
                )

    nsites = len(site_types)

    if nsites == 1:
        stype = site_types[0]
        d = stype.dim
        left_link = Index(1, f"{link_prefix}-0", tags=("link", "0"))
        right_link = Index(1, f"{link_prefix}-1", tags=("link", "1"))
        site_in = Index(d, f"{site_prefix}_in-0", tags=("site_in", "0", stype.name))
        site_out = Index(d, f"{site_prefix}_out-0", tags=("site_out", "0", stype.name))
        X = stype.op("X")
        data = (Omega * X).reshape(1, d, d, 1)
        return MPO(
            (Tensor(data, (left_link, site_in, site_out, right_link)),),
            tuple(site_types),
            gauge_center=0,
        )

    links = []
    for i in range(nsites + 1):
        dim = 1 if (i == 0 or i == nsites) else 4
        links.append(Index(dim, f"{link_prefix}-{i}", tags=("link", f"{i}")))

    tensors = []
    for n, stype in enumerate(site_types):
        d = stype.dim
        site_in = Index(d, f"{site_prefix}_in-{n}", tags=("site_in", f"{n}", stype.name))
        site_out = Index(d, f"{site_prefix}_out-{n}", tags=("site_out", f"{n}", stype.name))

        I = stype.op("I")
        X = Omega * stype.op("X")
        P = stype.op("P0")
        dtype = jnp.result_type(I, X, P, Omega)

        if n == 0:
            # Left boundary row: [done, need_P, need_X_then_P, idle]
            # Enables the left-edge term X_0 P_1 and the interior starts P_i X_{i+1} P_{i+2}.
            data = jnp.zeros((1, d, d, 4), dtype=dtype)
            data = data.at[0, :, :, 1].set(X)
            data = data.at[0, :, :, 2].set(P)
            data = data.at[0, :, :, 3].set(I)
        elif n == nsites - 1:
            # Right boundary column:
            #   state 0 -> I          (propagate completed terms)
            #   state 1 -> P          (finish ... X P)
            #   state 2 -> X          (finish right-edge term ... P X)
            data = jnp.zeros((4, d, d, 1), dtype=dtype)
            data = data.at[0, :, :, 0].set(I)
            data = data.at[1, :, :, 0].set(P)
            data = data.at[2, :, :, 0].set(X)
        else:
            # Bulk automaton states:
            #   0: completed term / propagate identity
            #   1: waiting for right projector P after placing X
            #   2: waiting for X then P after placing left projector P
            #   3: idle / can start a new term
            data = jnp.zeros((4, d, d, 4), dtype=dtype)
            data = data.at[0, :, :, 0].set(I)
            data = data.at[1, :, :, 0].set(P)
            data = data.at[2, :, :, 1].set(X)
            data = data.at[3, :, :, 2].set(P)
            data = data.at[3, :, :, 3].set(I)

        tensors.append(Tensor(data, (links[n], site_in, site_out, links[n + 1])))

    return MPO(tuple(tensors), tuple(site_types), gauge_center=None)


def cluster_ising_mpo(
    site_types: Sequence[SiteType],
    hx: float = 1.0,
    hz: float = 1.0, 
    Jzz: float = 1.0, 
    Jzxz: float = 1.0, 
    link_prefix: str = "link",
    site_prefix: str = "site",
) -> MPO:
    """
    Construct the open-boundary Cluster Ising MPO

        H = sum_{i=0}^{N-1} (- hx X_i - hz Z_i) - hzz sum_{i=0}^{N-2} Z_i Z_{i+1} 
            -Jzxz sum_{i=1}^{N-2} Z_{i-1} X_i Z_{i+1}
            
    and the MPO bond dimension is 4 for N >= 3.
    """
    if len(site_types) == 0:
        raise ValueError("cluster_ising_mpo requires at least one site")

    for n, stype in enumerate(site_types):
        for op_name in ("I", "X", "Z"):
            try:
                op = stype.op(op_name)
            except Exception as exc:
                raise ValueError(
                    f"Site type at site {n} does not provide required operator {op_name!r}"
                ) from exc
            if op.shape != (stype.dim, stype.dim):
                raise ValueError(
                    f"Operator {op_name!r} at site {n} has shape {op.shape}, "
                    f"expected ({stype.dim}, {stype.dim})"
                )

    nsites = len(site_types)

    if nsites == 1:
        stype = site_types[0]
        d = stype.dim
        left_link = Index(1, f"{link_prefix}-0", tags=("link", "0"))
        right_link = Index(1, f"{link_prefix}-1", tags=("link", "1"))
        site_in = Index(d, f"{site_prefix}_in-0", tags=("site_in", "0", stype.name))
        site_out = Index(d, f"{site_prefix}_out-0", tags=("site_out", "0", stype.name))
        onsite = -hx * stype.op("X") - hz * stype.op("Z")
        data = onsite.reshape(1, d, d, 1)
        return MPO(
            (Tensor(data, (left_link, site_in, site_out, right_link)),),
            tuple(site_types),
            gauge_center=0,
        )

    links = []
    for i in range(nsites + 1):
        dim = 1 if (i == 0 or i == nsites) else 4
        links.append(Index(dim, f"{link_prefix}-{i}", tags=("link", f"{i}")))

    tensors = []
    for n, stype in enumerate(site_types):
        d = stype.dim
        site_in = Index(d, f"{site_prefix}_in-{n}", tags=("site_in", f"{n}", stype.name))
        site_out = Index(d, f"{site_prefix}_out-{n}", tags=("site_out", f"{n}", stype.name))

        I = stype.op("I")
        X = stype.op("X")
        Z = stype.op("Z")
        onsite = -hx * X - hz * Z
        dtype = jnp.result_type(I, X, Z, hx, hz, Jzz, Jzxz)

        if n == 0:
            # Left boundary row with state order:
            #   0: done / completed term
            #   1: waiting for final Z after placing X in a ZXZ term
            #   2: after placing the leading Z of either ZZ or ZXZ
            #   3: idle / can start a new term later
            data = jnp.zeros((1, d, d, 4), dtype=dtype)
            data = data.at[0, :, :, 0].set(onsite)
            data = data.at[0, :, :, 2].set(Z)
            data = data.at[0, :, :, 3].set(I)
        elif n == nsites - 1:
            data = jnp.zeros((4, d, d, 1), dtype=dtype)
            data = data.at[0, :, :, 0].set(I)
            data = data.at[1, :, :, 0].set(Z)
            data = data.at[2, :, :, 0].set(-Jzz * Z)
            data = data.at[3, :, :, 0].set(onsite)
        else:
            data = jnp.zeros((4, d, d, 4), dtype=dtype)
            data = data.at[0, :, :, 0].set(I)
            data = data.at[1, :, :, 0].set(Z)
            data = data.at[2, :, :, 0].set(-Jzz * Z)
            data = data.at[2, :, :, 1].set(-Jzxz * X)
            data = data.at[3, :, :, 0].set(onsite)
            data = data.at[3, :, :, 2].set(Z)
            data = data.at[3, :, :, 3].set(I)

        tensors.append(Tensor(data, (links[n], site_in, site_out, links[n + 1])))

    return MPO(tuple(tensors), tuple(site_types), gauge_center=None)

