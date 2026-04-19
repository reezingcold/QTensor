from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from qtensor.models.sites import SiteType
from qtensor.tensor.contract import contract
from qtensor.tensor.index import Index
from qtensor.tensor.linalg import tensor_qr, truncated_svd
from qtensor.tensor.tensor import Tensor


def _site_indices(site_types: Sequence[SiteType], link_dims: Sequence[int], link_prefix: str, site_prefix: str):
    links = [
        Index(int(dim), f"{link_prefix}-{i}", tags=("link", f"{i}"))
        for i, dim in enumerate(link_dims)
    ]
    site_in_inds = []
    site_out_inds = []
    for n, stype in enumerate(site_types):
        d = stype.dim
        site_in_inds.append(Index(d, f"{site_prefix}_in-{n}", tags=("site_in", f"{n}", stype.name)))
        site_out_inds.append(Index(d, f"{site_prefix}_out-{n}", tags=("site_out", f"{n}", stype.name)))
    return links, tuple(site_in_inds), tuple(site_out_inds)


def _mpo_from_data(
    datas: Sequence[jnp.ndarray],
    site_types: Sequence[SiteType],
    link_prefix: str = "link",
    site_prefix: str = "site",
) -> "MPO":
    if len(datas) != len(site_types):
        raise ValueError("datas and site_types must have the same length")
    link_dims = [datas[0].shape[0]]
    link_dims.extend(data.shape[3] for data in datas)
    links, site_in_inds, site_out_inds = _site_indices(site_types, link_dims, link_prefix, site_prefix)
    tensors = []
    for n, data in enumerate(datas):
        tensors.append(Tensor(data, (links[n], site_in_inds[n], site_out_inds[n], links[n + 1])))
    return MPO(tuple(tensors), tuple(site_types), gauge_center=0)


def _mirror_mpo_data(W: "MPO") -> list[jnp.ndarray]:
    datas = []
    for T in reversed(W.tensors):
        data = jnp.transpose(T.data, (3, 1, 2, 0))
        if data.shape[0] > 1:
            data = data[::-1, :, :, :]
        if data.shape[3] > 1:
            data = data[:, :, :, ::-1]
        datas.append(data)
    return datas


def _block_left_qr_site(data: jnp.ndarray, is_first: bool):
    left_dim, din, dout, right_dim = data.shape
    if right_dim == 1:
        return data, jnp.ones((1, 1), dtype=data.dtype)

    row_slice = slice(None) if is_first else slice(0, left_dim - 1)
    V0 = data[row_slice, :, :, 0:1]
    Vm = data[row_slice, :, :, 1 : right_dim - 1]
    rows = V0.shape[0] * din * dout
    scale = jnp.sqrt(float(din))

    q0 = V0
    q0_metric = (V0 / scale).reshape(rows, 1)
    vm_mat = Vm.reshape(rows, max(0, right_dim - 2)) / scale
    proj = jnp.conjugate(q0_metric).T @ vm_mat
    residual = vm_mat - q0_metric @ proj

    if residual.shape[1] == 0:
        rank_mid = 0
        Qmid_mat = jnp.zeros((rows, 0), dtype=data.dtype)
        Rmid = jnp.zeros((0, 0), dtype=data.dtype)
    else:
        U, svals, Vh = jnp.linalg.svd(residual, full_matrices=False)
        rank_mid = int(jnp.sum(svals > 1e-12))
        Qmid_mat = U[:, :rank_mid]
        Rmid = jnp.diag(svals[:rank_mid]) @ Vh[:rank_mid, :]

    qdata = jnp.zeros((left_dim, din, dout, rank_mid + 2), dtype=data.dtype)
    qdata = qdata.at[row_slice, :, :, 0:1].set(q0)
    if rank_mid > 0:
        qmid = (scale * Qmid_mat).reshape(V0.shape[0], din, dout, rank_mid)
        qdata = qdata.at[row_slice, :, :, 1 : rank_mid + 1].set(qmid)
    qdata = qdata.at[:, :, :, rank_mid + 1].set(data[:, :, :, right_dim - 1])
    if not is_first and left_dim > 1:
        qdata = qdata.at[left_dim - 1, :, :, : rank_mid + 1].set(0)

    R = jnp.zeros((rank_mid + 2, right_dim), dtype=data.dtype)
    R = R.at[0, 0].set(1)
    if proj.shape[1] > 0:
        R = R.at[0, 1 : right_dim - 1].set(proj[0])
    if rank_mid > 0:
        R = R.at[1 : rank_mid + 1, 1 : right_dim - 1].set(Rmid)
    R = R.at[rank_mid + 1, right_dim - 1].set(1)
    return qdata, R


def _apply_left_matrix(data: jnp.ndarray, mat: jnp.ndarray) -> jnp.ndarray:
    return jnp.tensordot(mat, data, axes=((1,), (0,)))


def _apply_right_matrix(data: jnp.ndarray, mat: jnp.ndarray) -> jnp.ndarray:
    return jnp.tensordot(data, mat, axes=((3,), (0,)))


def _compress_mpo_mps_style(
    W: "MPO",
    cutoff: float = 0.0,
    max_bond: int | None = None,
) -> "MPO":
    if len(W) == 1:
        return W.copy()

    out = W.copy()
    for n in range(len(out) - 1):
        out = left_canonicalize_site(out, n)

    tensors = list(out.tensors)
    for n in range(len(out) - 2, -1, -1):
        theta = contract(tensors[n], tensors[n + 1])
        U, S, Vh, _ = truncated_svd(
            theta,
            left_inds=(theta.inds[0], theta.inds[1], theta.inds[2]),
            bond_name=f"mpo-compress-{n}",
            cutoff=cutoff,
            max_bond=max_bond,
        )
        new_link = Index(
            U.inds[3].dim,
            out.right_link_ind(n).name,
            tags=("link", f"{n + 1}"),
            prime_level=out.right_link_ind(n).prime_level,
        )
        tensors[n] = U.replace_ind(U.inds[3], new_link)
        tensors[n + 1] = contract(S, Vh).replace_ind(S.inds[0], new_link)

    return MPO(tuple(tensors), W.site_types, gauge_center=0)


def add_regular_mpo(A: "MPO", B: "MPO") -> "MPO":
    if len(A) != len(B):
        raise ValueError("MPO lengths must match for addition")
    if len(A) == 1:
        return A + B

    datas = []
    for n, (TA, TB) in enumerate(zip(A.tensors, B.tensors)):
        da = TA.data
        db = TB.data
        la, di, do, ra = da.shape
        lb, _, _, rb = db.shape
        left_dim = la + lb - 1 if n > 0 else 1
        right_dim = ra + rb - 1 if n < len(A) - 1 else 1
        data = jnp.zeros((left_dim, di, do, right_dim), dtype=jnp.result_type(da, db))

        def map_left_a(idx):
            if n == 0:
                return 0
            if idx == 0:
                return 0
            if idx == la - 1:
                return left_dim - 1
            return idx

        def map_left_b(idx):
            if n == 0:
                return 0
            if idx == 0:
                return 0
            if idx == lb - 1:
                return left_dim - 1
            return (la - 1) + idx

        def map_right_a(idx):
            if n == len(A) - 1:
                return 0
            if idx == 0:
                return 0
            if idx == ra - 1:
                return right_dim - 1
            return idx

        def map_right_b(idx):
            if n == len(A) - 1:
                return 0
            if idx == 0:
                return 0
            if idx == rb - 1:
                return right_dim - 1
            return (ra - 1) + idx

        for i in range(la):
            for j in range(ra):
                data = data.at[map_left_a(i), :, :, map_right_a(j)].add(da[i, :, :, j])
        for i in range(lb):
            for j in range(rb):
                li = map_left_b(i)
                rj = map_right_b(j)
                shared_start = li == 0 and rj == 0
                shared_final = li == left_dim - 1 and rj == right_dim - 1
                shared_d = li == 0 and rj == right_dim - 1
                if shared_d:
                    data = data.at[li, :, :, rj].add(db[i, :, :, j])
                elif shared_start or shared_final:
                    if not bool(jnp.allclose(data[li, :, :, rj], 0)):
                        continue
                    data = data.at[li, :, :, rj].set(db[i, :, :, j])
                else:
                    data = data.at[li, :, :, rj].add(db[i, :, :, j])
        datas.append(data)
    return _mpo_from_data(datas, A.site_types, link_prefix="link_add", site_prefix="site")


def _left_canonicalize_regular(W: "MPO") -> "MPO":
    datas = [jnp.array(T.data, copy=True) for T in W.tensors]
    carry = jnp.ones((1, 1), dtype=datas[0].dtype)
    out = []
    for n, data in enumerate(datas[:-1]):
        work = _apply_left_matrix(data, carry)
        qdata, carry = _block_left_qr_site(work, is_first=(n == 0))
        out.append(qdata)
    out.append(_apply_left_matrix(datas[-1], carry))
    return _mpo_from_data(out, W.site_types, link_prefix="link_lcan", site_prefix="site")


def _right_canonicalize_regular(W: "MPO") -> "MPO":
    mirrored = _mpo_from_data(_mirror_mpo_data(W), tuple(reversed(W.site_types)), link_prefix="link_m", site_prefix="site_m")
    mirrored = _left_canonicalize_regular(mirrored)
    return _mpo_from_data(_mirror_mpo_data(mirrored), W.site_types, link_prefix="link_rcan", site_prefix="site")


def compress_mpo_optimal(
    W: "MPO",
    cutoff: float = 0.0,
    max_bond: int | None = None,
) -> "MPO":
    if len(W) <= 1:
        return W.copy()

    WR = _right_canonicalize_regular(W)
    out_datas = []
    carry = jnp.ones((1, 1), dtype=WR[0].dtype)

    for n in range(len(WR) - 1):
        work = _apply_left_matrix(WR[n].data, carry)
        qdata, R = _block_left_qr_site(work, is_first=(n == 0))

        middle = R[1:-1, 1:-1]
        t_row = R[0, 1:-1]
        if middle.size == 0:
            keep = 0
            U = jnp.zeros((0, 0), dtype=R.dtype)
            svals = jnp.zeros((0,), dtype=R.dtype)
            Vh = jnp.zeros((0, 0), dtype=R.dtype)
        else:
            U, svals, Vh = jnp.linalg.svd(middle, full_matrices=False)
            keep = svals.shape[0]
            if cutoff is not None:
                keep = min(keep, int(jnp.sum(svals > cutoff)))
            if max_bond is not None:
                keep = min(keep, max(0, max_bond - 2))
            keep = max(0, keep)
            U = U[:, :keep]
            svals = svals[:keep]
            Vh = Vh[:keep, :]

        new_right_dim = keep + 2
        Ublock = jnp.zeros((qdata.shape[3], new_right_dim), dtype=qdata.dtype)
        Ublock = Ublock.at[0, 0].set(1)
        Ublock = Ublock.at[-1, -1].set(1)
        if keep > 0:
            Ublock = Ublock.at[1:-1, 1:-1].set(U)
        out_datas.append(_apply_right_matrix(qdata, Ublock))

        next_carry = jnp.zeros((new_right_dim, R.shape[1]), dtype=R.dtype)
        next_carry = next_carry.at[0, 0].set(1)
        next_carry = next_carry.at[0, 1:-1].set(t_row)
        next_carry = next_carry.at[-1, -1].set(1)
        if keep > 0:
            next_carry = next_carry.at[1:-1, 1:-1].set(jnp.diag(svals) @ Vh)
        carry = next_carry

    out_datas.append(_apply_left_matrix(WR[-1].data, carry))
    return _mpo_from_data(out_datas, W.site_types, link_prefix="link_cmp", site_prefix="site")


class MPO:
    """
    Open-boundary matrix product operator.

    Leg convention for each site tensor is fixed as:
        (left_link, site_in, site_out, right_link)

    For an MPO with N sites, we use N+1 bond positions:
        link_0, link_1, ..., link_N

    Site n therefore carries indices:
        (link_n, site_in_n, site_out_n, link_{n+1})

    The boundary links link_0 and link_N must have dimension 1.
    """

    def __init__(
        self,
        tensors: Sequence[Tensor],
        site_types: Sequence[SiteType],
        gauge_center: int | None = None,
    ):
        if len(tensors) == 0:
            raise ValueError("MPO must contain at least one tensor")

        if len(tensors) != len(site_types):
            raise ValueError("tensors and site_types must have the same length")

        if gauge_center is not None and not (0 <= gauge_center < len(tensors)):
            raise ValueError("gauge_center must be None or a valid site index")

        self.tensors = tuple(tensors)
        self.site_types = tuple(site_types)
        self.gauge_center = gauge_center
        self._check_mpo_structure()

    # -------------------------------------------------
    # basic protocol
    # -------------------------------------------------

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, site: int) -> Tensor:
        return self.tensors[site]

    def site_type(self, site: int) -> SiteType:
        return self.site_types[site]

    def copy(self) -> "MPO":
        return MPO(tuple(T.copy() for T in self.tensors), self.site_types, self.gauge_center)

    def prime(self, n: int = 1) -> "MPO":
        if n < 0:
            raise ValueError("n must be non-negative")
        return MPO(tuple(T.prime_inds(n) for T in self.tensors), self.site_types, self.gauge_center)

    def prime_link_inds(self, n: int = 1) -> "MPO":
        if n < 0:
            raise ValueError("n must be non-negative")

        new_tensors = []
        for T in self.tensors:
            new_inds = tuple(ind.prime(n) if "link" in ind.tags else ind for ind in T.inds)
            new_tensors.append(Tensor(T.data, new_inds))

        return MPO(tuple(new_tensors), self.site_types, self.gauge_center)

    def prime_site_in_inds(self, n: int = 1) -> "MPO":
        if n < 0:
            raise ValueError("n must be non-negative")

        new_tensors = []
        for T in self.tensors:
            new_inds = tuple(ind.prime(n) if "site_in" in ind.tags else ind for ind in T.inds)
            new_tensors.append(Tensor(T.data, new_inds))

        return MPO(tuple(new_tensors), self.site_types, self.gauge_center)

    def prime_site_out_inds(self, n: int = 1) -> "MPO":
        if n < 0:
            raise ValueError("n must be non-negative")

        new_tensors = []
        for T in self.tensors:
            new_inds = tuple(ind.prime(n) if "site_out" in ind.tags else ind for ind in T.inds)
            new_tensors.append(Tensor(T.data, new_inds))

        return MPO(tuple(new_tensors), self.site_types, self.gauge_center)

    def prime_site_inds(self, n: int = 1) -> "MPO":
        if n < 0:
            raise ValueError("n must be non-negative")

        new_tensors = []
        for T in self.tensors:
            new_inds = tuple(
                ind.prime(n) if ("site_in" in ind.tags or "site_out" in ind.tags) else ind
                for ind in T.inds
            )
            new_tensors.append(Tensor(T.data, new_inds))

        return MPO(tuple(new_tensors), self.site_types, self.gauge_center)

    def conj(self) -> "MPO":
        return MPO(tuple(T.conj() for T in self.tensors), self.site_types, self.gauge_center)

    def __repr__(self) -> str:
        return (
            f"MPO(nsites={len(self)}, "
            f"site_dims={self.site_dims()}, link_dims={self.link_dims()}, "
            f"gauge_center={self.gauge_center})"
        )

    def __mul__(self, scalar) -> "MPO":
        if not jnp.isscalar(scalar):
            return NotImplemented
        target = 0 if self.gauge_center is None else self.gauge_center
        new_tensors = list(self.tensors)
        new_tensors[target] = new_tensors[target] * scalar
        return MPO(tuple(new_tensors), self.site_types, self.gauge_center)

    def __rmul__(self, scalar) -> "MPO":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "MPO":
        if not jnp.isscalar(scalar):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("division by zero")
        return self * (1.0 / scalar)

    def __add__(self, other: "MPO") -> "MPO":
        if not isinstance(other, MPO):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("MPO lengths must match for addition")

        for n in range(len(self)):
            if self.site_dim(n) != other.site_dim(n):
                raise ValueError(
                    f"Site dimension mismatch at site {n}: "
                    f"{self.site_dim(n)} != {other.site_dim(n)}"
                )
            if self.site_type(n).name != other.site_type(n).name:
                raise ValueError(
                    f"Site type mismatch at site {n}: "
                    f"{self.site_type(n).name} != {other.site_type(n).name}"
                )

        if len(self) == 1:
            data = self[0].data + other[0].data
            return MPO((Tensor(data, self[0].inds),), self.site_types, gauge_center=0)

        new_tensors = []

        A0, B0 = self[0], other[0]
        data0 = jnp.concatenate([A0.data, B0.data], axis=3)
        right_link0 = Index(
            data0.shape[3],
            self.right_link_ind(0).name,
            self.right_link_ind(0).tags,
            self.right_link_ind(0).prime_level,
        )
        new_tensors.append(
            Tensor(
                data0,
                (self.left_link_ind(0), self.site_in_ind(0), self.site_out_ind(0), right_link0),
            )
        )

        for n in range(1, len(self) - 1):
            A = self[n]
            B = other[n]
            la, di, do, ra = A.shape
            lb, _, _, rb = B.shape
            dtype = jnp.result_type(A.data, B.data)

            data = jnp.zeros((la + lb, di, do, ra + rb), dtype=dtype)
            data = data.at[:la, :, :, :ra].set(A.data)
            data = data.at[la:, :, :, ra:].set(B.data)

            left_link = new_tensors[-1].inds[3]
            right_link = Index(
                ra + rb,
                self.right_link_ind(n).name,
                self.right_link_ind(n).tags,
                self.right_link_ind(n).prime_level,
            )
            new_tensors.append(
                Tensor(data, (left_link, self.site_in_ind(n), self.site_out_ind(n), right_link))
            )

        A_last, B_last = self[-1], other[-1]
        data_last = jnp.concatenate([A_last.data, B_last.data], axis=0)
        left_link_last = new_tensors[-1].inds[3]
        new_tensors.append(
            Tensor(
                data_last,
                (
                    left_link_last,
                    self.site_in_ind(len(self) - 1),
                    self.site_out_ind(len(self) - 1),
                    self.right_link_ind(len(self) - 1),
                ),
            )
        )
        return MPO(tuple(new_tensors), self.site_types, gauge_center=0)

    def __sub__(self, other: "MPO") -> "MPO":
        if not isinstance(other, MPO):
            return NotImplemented
        return self + ((-1.0) * other)

    # -------------------------------------------------
    # structural helpers
    # -------------------------------------------------

    def left_link_ind(self, site: int) -> Index:
        return self.tensors[site].inds[0]

    def site_in_ind(self, site: int) -> Index:
        return self.tensors[site].inds[1]

    def site_out_ind(self, site: int) -> Index:
        return self.tensors[site].inds[2]

    def right_link_ind(self, site: int) -> Index:
        return self.tensors[site].inds[3]

    def site_in_inds(self) -> tuple[Index, ...]:
        return tuple(self.site_in_ind(n) for n in range(len(self)))

    def site_out_inds(self) -> tuple[Index, ...]:
        return tuple(self.site_out_ind(n) for n in range(len(self)))

    def site_inds(self) -> tuple[tuple[Index, Index], ...]:
        return tuple((self.site_in_ind(n), self.site_out_ind(n)) for n in range(len(self)))

    def site_dim(self, site: int) -> int:
        # `_check_mpo_structure` enforces that input/output physical legs have
        # the same dimension and both match `site_types[site].dim`.
        return self.site_in_ind(site).dim

    def link_inds(self) -> tuple[Index, ...]:
        links = [self.left_link_ind(0)]
        links.extend(self.right_link_ind(n) for n in range(len(self)))
        return tuple(links)
    
    def link_ind(self, n: int) -> Index:
        return self.link_inds()[n]

    def link_dim(self, link: int) -> int:
        return self.link_inds()[link].dim

    def site_dims(self) -> tuple[int, ...]:
        return tuple(self.site_in_ind(n).dim for n in range(len(self)))

    def link_dims(self) -> tuple[int, ...]:
        return tuple(ind.dim for ind in self.link_inds())

    def _check_mpo_structure(self) -> None:
        for n, T in enumerate(self.tensors):
            if T.ndim != 4:
                raise ValueError(
                    f"Each MPO tensor must have rank 4, but site {n} has rank {T.ndim}"
                )

            ll, site_in, site_out, rl = T.inds

            if "link" not in ll.tags:
                raise ValueError(f"Site {n} left index must carry tag 'link'")
            if "site_in" not in site_in.tags:
                raise ValueError(f"Site {n} input physical index must carry tag 'site_in'")
            if "site_out" not in site_out.tags:
                raise ValueError(f"Site {n} output physical index must carry tag 'site_out'")
            if "link" not in rl.tags:
                raise ValueError(f"Site {n} right index must carry tag 'link'")

            if site_in.dim != self.site_types[n].dim:
                raise ValueError(
                    f"Input site dimension mismatch at site {n}: "
                    f"{site_in.dim} != {self.site_types[n].dim}"
                )
            if site_out.dim != self.site_types[n].dim:
                raise ValueError(
                    f"Output site dimension mismatch at site {n}: "
                    f"{site_out.dim} != {self.site_types[n].dim}"
                )

        for n in range(len(self.tensors) - 1):
            if self.right_link_ind(n) != self.left_link_ind(n + 1):
                raise ValueError(
                    f"Link mismatch between sites {n} and {n + 1}: "
                    f"{self.right_link_ind(n)} != {self.left_link_ind(n + 1)}"
                )

        if self.left_link_ind(0).dim != 1:
            raise ValueError("Left boundary link dimension must be 1")
        if self.right_link_ind(len(self) - 1).dim != 1:
            raise ValueError("Right boundary link dimension must be 1")

    # -------------------------------------------------
    # immutable updates
    # -------------------------------------------------

    def replace_site(self, site: int, tensor: Tensor) -> "MPO":
        new_tensors = list(self.tensors)
        new_tensors[site] = tensor
        return MPO(tuple(new_tensors), self.site_types, gauge_center=None)

    def truncate(
        self,
        cutoff: float = 0.0,
        max_bond: int | None = None,
        method: str = "regular",
    ) -> "MPO":
        return compress_mpo(self, cutoff=cutoff, max_bond=max_bond, method=method)

    # -------------------------------------------------
    # constructors
    # -------------------------------------------------

    @classmethod
    def identity(
        cls,
        site_types: Sequence[SiteType],
        link_prefix: str = "link",
        site_prefix: str = "site",
    ) -> "MPO":
        if len(site_types) == 0:
            raise ValueError("identity MPO requires at least one site")

        nsites = len(site_types)
        links = [Index(1, f"{link_prefix}-{i}", tags=("link", f"{i}")) for i in range(nsites + 1)]

        tensors = []
        for n, stype in enumerate(site_types):
            d = stype.dim
            site_in = Index(d, f"{site_prefix}_in-{n}", tags=("site_in", f"{n}", stype.name))
            site_out = Index(d, f"{site_prefix}_out-{n}", tags=("site_out", f"{n}", stype.name))

            data = jnp.eye(d).reshape(1, d, d, 1)
            tensors.append(Tensor(data, (links[n], site_in, site_out, links[n + 1])))

        # Bond-dimension-1 product operators are trivially compatible with
        # using site 0 as a convenient initial gauge-center marker.
        return cls(tuple(tensors), tuple(site_types), gauge_center=0)

    @classmethod
    def product_operator(
        cls,
        site_types: Sequence[SiteType],
        op_names: Sequence[str],
        link_prefix: str = "link",
        site_prefix: str = "site",
    ) -> "MPO":
        if len(site_types) != len(op_names):
            raise ValueError("site_types and op_names must have the same length")
        if len(site_types) == 0:
            raise ValueError("product_operator requires at least one site")

        nsites = len(site_types)
        links = [Index(1, f"{link_prefix}-{i}", tags=("link", f"{i}")) for i in range(nsites + 1)]

        tensors = []
        for n, (stype, op_name) in enumerate(zip(site_types, op_names)):
            d = stype.dim
            site_in = Index(d, f"{site_prefix}_in-{n}", tags=("site_in", f"{n}", stype.name))
            site_out = Index(d, f"{site_prefix}_out-{n}", tags=("site_out", f"{n}", stype.name))

            op = stype.op(op_name)
            data = op.reshape(1, d, d, 1)
            tensors.append(Tensor(data, (links[n], site_in, site_out, links[n + 1])))

        # Bond-dimension-1 product operators are trivially compatible with
        # using site 0 as a convenient initial gauge-center marker.
        return cls(tuple(tensors), tuple(site_types), gauge_center=0)

    # -------------------------------------------------
    # dense conversion
    # -------------------------------------------------

    def to_tensor(self) -> Tensor:
        """
        Contract the full OBC MPO into a single Tensor over all input and output
        physical indices.

        Returns a Tensor with indices
            self.site_in_inds() + self.site_out_inds()
        after removing the two boundary links of dimension 1.
        """
        out = self.tensors[0]
        for n in range(1, len(self.tensors)):
            out = contract(out, self.tensors[n])

        expected_inds = (
            self.left_link_ind(0),
            *self.site_in_inds(),
            *self.site_out_inds(),
            self.right_link_ind(len(self) - 1),
        )
        out = out.permute(expected_inds)

        if self.left_link_ind(0).dim != 1 or self.right_link_ind(len(self) - 1).dim != 1:
            raise ValueError(
                "Boundary link dimensions must be 1 to convert an OBC MPO to a full operator tensor"
            )

        squeezed = jnp.squeeze(out.data, axis=(0, out.data.ndim - 1))
        return Tensor(squeezed, self.site_in_inds() + self.site_out_inds())


def left_canonicalize_site(W: MPO, n: int) -> MPO:
    if not (0 <= n < len(W) - 1):
        raise ValueError("n must satisfy 0 <= n < len(W) - 1")

    tensors = [T.copy() for T in W.tensors]
    A = tensors[n]
    Q, R = tensor_qr(A, left_inds=(A.inds[0], A.inds[1], A.inds[2]))
    qr_link = Q.inds[3]
    new_link = Index(
        qr_link.dim,
        W.right_link_ind(n).name,
        tags=("link", f"{n + 1}"),
        prime_level=W.right_link_ind(n).prime_level,
    )
    Q = Q.replace_ind(qr_link, new_link)
    tensors[n] = Q

    A_next = tensors[n + 1]
    new_next = contract(R, A_next)
    new_next = new_next.replace_ind(R.inds[0], new_link)
    tensors[n + 1] = new_next
    return MPO(tuple(tensors), W.site_types, gauge_center=n + 1)


def compress_mpo(
    W: MPO,
    cutoff: float = 0.0,
    max_bond: int | None = None,
    method: str = "auto",
) -> MPO:
    """
    Compress an MPO.

    `method="regular"` uses the finite-MPO canonical/truncation sweep for
    regular-form MPOs with distinguished start/final channels.
    `method="mps"` falls back to generic MPO-as-doubled-MPS compression.
    `method="auto"` currently chooses the regular sweep.
    """
    if method == "auto":
        method = "regular"
    if method == "regular":
        return compress_mpo_optimal(W, cutoff=cutoff, max_bond=max_bond)
    if method == "mps":
        return _compress_mpo_mps_style(W, cutoff=cutoff, max_bond=max_bond)
    raise ValueError("method must be one of 'auto', 'regular', or 'mps'")


def truncate_mpo(
    A: MPO,
    cutoff: float = 0.0,
    max_bond: int | None = None,
    method: str = "regular",
) -> MPO:
    if not isinstance(A, MPO):
        raise TypeError("truncate expects an MPO")
    return A.truncate(cutoff=cutoff, max_bond=max_bond, method=method)
