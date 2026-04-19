from __future__ import annotations

from math import prod
from typing import Iterable

import jax.numpy as jnp

from qtensor.tensor.index import Index
from qtensor.tensor.tensor import Tensor


# -----------------------------
# internal helpers
# -----------------------------



def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    return tuple(x)



def _infer_right_inds(T: Tensor, left_inds: tuple[Index, ...]) -> tuple[Index, ...]:
    left_set = set(left_inds)
    return tuple(ind for ind in T.inds if ind not in left_set)



def _check_partition(T: Tensor, left_inds: tuple[Index, ...], right_inds: tuple[Index, ...]):
    all_inds = left_inds + right_inds
    if set(all_inds) != set(T.inds) or len(all_inds) != len(T.inds):
        raise ValueError("left_inds and right_inds must form a partition of T.inds")



def _matrixize(T: Tensor, left_inds: tuple[Index, ...], right_inds: tuple[Index, ...]):
    _check_partition(T, left_inds, right_inds)

    perm_inds = left_inds + right_inds
    T_perm = T.permute(perm_inds)

    dl = prod(ind.dim for ind in left_inds) if left_inds else 1
    dr = prod(ind.dim for ind in right_inds) if right_inds else 1

    mat = T_perm.data.reshape(dl, dr)
    return T_perm, mat


def _matrix_to_tensor(mat, inds: tuple[Index, ...]):
    shape = tuple(ind.dim for ind in inds)
    return Tensor(mat.reshape(shape), inds)


def _make_bond_pair(dim: int, base_name: str):
    bond = Index(dim=dim, name=base_name, tags=("bond",))
    bond_p = Index(dim=dim, name=base_name + "_p", tags=("bond",))
    return bond, bond_p



# -----------------------------
# SVD
# -----------------------------


def tensor_svd(
    T: Tensor,
    left_inds: Iterable[Index],
    right_inds: Iterable[Index] | None = None,
    bond_name: str = "svd_bond",
    full_matrices: bool = False,
):
    left_inds = _as_tuple(left_inds)

    if right_inds is None:
        right_inds = _infer_right_inds(T, left_inds)
    else:
        right_inds = _as_tuple(right_inds)

    _, mat = _matrixize(T, left_inds, right_inds)

    U, S, Vh = jnp.linalg.svd(mat, full_matrices=full_matrices)

    rank = S.shape[0]
    bond, bond_p = _make_bond_pair(rank, bond_name)

    U_tensor = Tensor(
        U.reshape(*(i.dim for i in left_inds), rank),
        left_inds + (bond,),
    )

    S_tensor = Tensor(
        jnp.diag(S),
        (bond, bond_p),
    )

    Vh_tensor = Tensor(
        Vh.reshape(rank, *(i.dim for i in right_inds)),
        (bond_p,) + right_inds,
    )

    return U_tensor, S_tensor, Vh_tensor



def truncated_svd(
    T: Tensor,
    left_inds: Iterable[Index],
    right_inds: Iterable[Index] | None = None,
    bond_name: str = "svd_bond",
    cutoff: float | None = None,
    max_bond: int | None = None,
    full_matrices: bool = False,
):
    left_inds = _as_tuple(left_inds)

    if right_inds is None:
        right_inds = _infer_right_inds(T, left_inds)
    else:
        right_inds = _as_tuple(right_inds)

    _, mat = _matrixize(T, left_inds, right_inds)
    U, svals, Vh = jnp.linalg.svd(mat, full_matrices=full_matrices)

    keep = svals.shape[0]

    if cutoff is not None:
        keep = min(keep, int(jnp.sum(svals > cutoff)))

    if max_bond is not None:
        keep = min(keep, max_bond)

    keep = max(1, keep)

    new_bond, new_bond_p = _make_bond_pair(keep, bond_name)
    kept_svals = svals[:keep]

    U_trunc = Tensor(
        U[:, :keep].reshape(*(i.dim for i in left_inds), keep),
        left_inds + (new_bond,),
    )

    S_trunc = Tensor(
        jnp.diag(kept_svals),
        (new_bond, new_bond_p),
    )

    Vh_trunc = Tensor(
        Vh[:keep, :].reshape(keep, *(i.dim for i in right_inds)),
        (new_bond_p,) + right_inds,
    )

    info = {
        "old_bond_dim": int(svals.shape[0]),
        "new_bond_dim": int(keep),
        "discarded_weight": jnp.sum(svals[keep:] ** 2) if keep < svals.shape[0] else jnp.array(0.0, dtype=svals.dtype),
        "singular_values": kept_svals,
    }

    return U_trunc, S_trunc, Vh_trunc, info


# -----------------------------
# QR / RQ
# -----------------------------


def tensor_qr(
    T: Tensor,
    left_inds: Iterable[Index],
    right_inds: Iterable[Index] | None = None,
    bond_name: str = "qr_bond",
    mode: str = "reduced",
):
    left_inds = _as_tuple(left_inds)

    if right_inds is None:
        right_inds = _infer_right_inds(T, left_inds)
    else:
        right_inds = _as_tuple(right_inds)

    _, mat = _matrixize(T, left_inds, right_inds)

    Q, R = jnp.linalg.qr(mat, mode=mode)

    rank = Q.shape[1]
    bond = Index(dim=rank, name=bond_name, tags=("bond",))

    Q_tensor = Tensor(
        Q.reshape(*(i.dim for i in left_inds), rank),
        left_inds + (bond,),
    )

    R_tensor = Tensor(
        R.reshape(rank, *(i.dim for i in right_inds)),
        (bond,) + right_inds,
    )

    return Q_tensor, R_tensor



def _matrix_rq(mat):
    q1, r1 = jnp.linalg.qr(jnp.flipud(mat).T, mode="reduced")
    r = jnp.flipud(r1.T)
    r = jnp.fliplr(r)
    q = jnp.flipud(q1.T)
    return r, q



def tensor_rq(
    T: Tensor,
    left_inds: Iterable[Index],
    right_inds: Iterable[Index] | None = None,
    bond_name: str = "rq_bond",
):
    left_inds = _as_tuple(left_inds)

    if right_inds is None:
        right_inds = _infer_right_inds(T, left_inds)
    else:
        right_inds = _as_tuple(right_inds)

    _, mat = _matrixize(T, left_inds, right_inds)

    R, Q = _matrix_rq(mat)

    rank = Q.shape[0]
    bond = Index(dim=rank, name=bond_name, tags=("bond",))

    R_tensor = Tensor(
        R.reshape(*(i.dim for i in left_inds), rank),
        left_inds + (bond,),
    )

    Q_tensor = Tensor(
        Q.reshape(rank, *(i.dim for i in right_inds)),
        (bond,) + right_inds,
    )

    return R_tensor, Q_tensor


# -----------------------------
# eigen decompositions
# -----------------------------


def tensor_eig(
    T: Tensor,
    left_inds: Iterable[Index],
    right_inds: Iterable[Index] | None = None,
    bond_name: str = "eig_bond",
):
    left_inds = _as_tuple(left_inds)

    if right_inds is None:
        right_inds = _infer_right_inds(T, left_inds)
    else:
        right_inds = _as_tuple(right_inds)

    _, mat = _matrixize(T, left_inds, right_inds)

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("tensor_eig requires square matrix")

    evals, evecs = jnp.linalg.eig(mat)

    rank = evals.shape[0]
    bond, _ = _make_bond_pair(rank, bond_name)

    evecs_tensor = Tensor(
        evecs.reshape(*(i.dim for i in left_inds), rank),
        left_inds + (bond,),
    )

    return evals, evecs_tensor



def tensor_eigh(
    T: Tensor,
    left_inds: Iterable[Index],
    right_inds: Iterable[Index] | None = None,
    bond_name: str = "eigh_bond",
):
    left_inds = _as_tuple(left_inds)

    if right_inds is None:
        right_inds = _infer_right_inds(T, left_inds)
    else:
        right_inds = _as_tuple(right_inds)

    _, mat = _matrixize(T, left_inds, right_inds)

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("tensor_eigh requires square matrix")

    evals, evecs = jnp.linalg.eigh(mat)

    rank = evals.shape[0]
    bond, _ = _make_bond_pair(rank, bond_name)

    evecs_tensor = Tensor(
        evecs.reshape(*(i.dim for i in left_inds), rank),
        left_inds + (bond,),
    )

    return evals, evecs_tensor



def truncated_eig(
    T: Tensor,
    left_inds: Iterable[Index],
    right_inds: Iterable[Index] | None = None,
    bond_name: str = "eig_bond",
    cutoff: float | None = None,
    max_bond: int | None = None,
):
    evals, evecs = tensor_eig(
        T,
        left_inds=left_inds,
        right_inds=right_inds,
        bond_name=bond_name,
    )

    order = jnp.argsort(jnp.abs(evals))[::-1]
    evals = evals[order]
    evecs_data = evecs.data[..., order]

    keep = evals.shape[0]

    if cutoff is not None:
        keep = min(keep, int(jnp.sum(jnp.abs(evals) > cutoff)))

    if max_bond is not None:
        keep = min(keep, max_bond)

    keep = max(1, keep)

    new_bond, _ = _make_bond_pair(keep, bond_name)

    evals_trunc = evals[:keep]
    evecs_trunc = Tensor(
        evecs_data[..., :keep],
        evecs.inds[:-1] + (new_bond,),
    )

    info = {
        "old_bond_dim": int(evals.shape[0]),
        "new_bond_dim": int(keep),
    }

    return evals_trunc, evecs_trunc, info



def truncated_eigh(
    T: Tensor,
    left_inds: Iterable[Index],
    right_inds: Iterable[Index] | None = None,
    bond_name: str = "eigh_bond",
    cutoff: float | None = None,
    max_bond: int | None = None,
):
    evals, evecs = tensor_eigh(
        T,
        left_inds=left_inds,
        right_inds=right_inds,
        bond_name=bond_name,
    )

    order = jnp.argsort(evals)[::-1]
    evals = evals[order]
    evecs_data = evecs.data[..., order]

    keep = evals.shape[0]

    if cutoff is not None:
        keep = min(keep, int(jnp.sum(evals > cutoff)))

    if max_bond is not None:
        keep = min(keep, max_bond)

    keep = max(1, keep)

    new_bond, _ = _make_bond_pair(keep, bond_name)

    evals_trunc = evals[:keep]
    evecs_trunc = Tensor(
        evecs_data[..., :keep],
        evecs.inds[:-1] + (new_bond,),
    )

    info = {
        "old_bond_dim": int(evals.shape[0]),
        "new_bond_dim": int(keep),
        "discarded_weight": jnp.sum(evals[keep:]) if keep < evals.shape[0] else jnp.array(0.0, dtype=evals.dtype),
    }

    return evals_trunc, evecs_trunc, info
