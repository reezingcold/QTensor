from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from qtensor.tensor.index import Index
from qtensor.tensor.tensor import Tensor
from qtensor.tensor.contract import contract
from qtensor.models.sites import SiteType
from qtensor.tensor.linalg import tensor_qr, tensor_rq, truncated_svd, tensor_svd



class MPS:
    """
    Open-boundary MPS.

    Leg convention for each site tensor is fixed as:
        (left_bond, physical, right_bond)

    For an MPS with N sites, we use N+1 bond positions:
        link_0, link_1, ..., link_N

    Site n therefore carries indices:
        (link_n, site_n, link_{n+1})

    The boundary bonds bond_0 and bond_N must have dimension 1.
         s0        s1        s2
         |         |         |
    b0 - A0 - b1 - A1 - b2 - A2 - b3 - ...
    """

    def __init__(
        self,
        tensors: Sequence[Tensor],
        site_types: Sequence[SiteType],
        gauge_center: int | None = None,
    ):
        if len(tensors) == 0:
            raise ValueError("MPS must contain at least one tensor")

        if len(tensors) != len(site_types):
            raise ValueError("tensors and site_types must have the same length")

        if gauge_center is not None and not (0 <= gauge_center < len(tensors)):
            raise ValueError("gauge_center must be None or a valid site index")

        self.tensors = tuple(tensors)
        self.site_types = tuple(site_types)
        self.gauge_center = gauge_center
        self._check_mps_structure()

    # -------------------------------------------------
    # basic protocol
    # -------------------------------------------------

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, site: int) -> Tensor:
        return self.tensors[site]

    def site_type(self, site: int) -> SiteType:
        return self.site_types[site]

    def copy(self) -> "MPS":
        return MPS(tuple(T.copy() for T in self.tensors), self.site_types, self.gauge_center)

    def prime(self, n: int = 1) -> "MPS":
        if n < 0:
            raise ValueError("n must be non-negative")
        return MPS(tuple(T.prime_inds(n) for T in self.tensors), self.site_types, self.gauge_center)

    def prime_link_inds(self, n: int = 1) -> "MPS":
        if n < 0:
            raise ValueError("n must be non-negative")

        new_tensors = []
        for T in self.tensors:
            new_inds = tuple(ind.prime(n) if "link" in ind.tags else ind for ind in T.inds)
            new_tensors.append(Tensor(T.data, new_inds))

        return MPS(tuple(new_tensors), self.site_types, self.gauge_center)

    def prime_site_inds(self, n: int = 1) -> "MPS":
        if n < 0:
            raise ValueError("n must be non-negative")

        new_tensors = []
        for T in self.tensors:
            new_inds = tuple(ind.prime(n) if "site" in ind.tags else ind for ind in T.inds)
            new_tensors.append(Tensor(T.data, new_inds))

        return MPS(tuple(new_tensors), self.site_types, self.gauge_center)

    def conj(self) -> "MPS":
        return MPS(tuple(T.conj() for T in self.tensors), self.site_types, self.gauge_center)

    def __repr__(self) -> str:
        return (
            f"MPS(nsites={len(self)}, "
            f"phys_dims={self.phys_dims()}, bond_dims={self.bond_dims()}, "
            f"gauge_center={self.gauge_center})"
        )

    # -------------------------------------------------
    # structural helpers
    # -------------------------------------------------

    def left_link_ind(self, site: int) -> Index:
        return self.tensors[site].inds[0]

    def phys_ind(self, site: int) -> Index:
        return self.tensors[site].inds[1]

    def right_link_ind(self, site: int) -> Index:
        return self.tensors[site].inds[2]

    def site_inds(self) -> tuple[Index, ...]:
        return tuple(self.phys_ind(n) for n in range(len(self)))
    
    def site_ind(self, n) -> Index:
        return self.phys_ind(n)

    def site_dim(self, site: int) -> int:
        return self.phys_ind(site).dim

    def link_inds(self) -> tuple[Index, ...]:
        links = [self.left_link_ind(0)]
        links.extend(self.right_link_ind(n) for n in range(len(self)))
        return tuple(links)
    
    def link_ind(self, n: int) -> Index:
        return self.link_inds()[n]

    def link_dim(self, link: int) -> int:
        return self.link_inds()[link].dim

    def phys_dims(self) -> tuple[int, ...]:
        return tuple(self.phys_ind(n).dim for n in range(len(self)))

    def bond_dims(self) -> tuple[int, ...]:
        return tuple(ind.dim for ind in self.link_inds())
    
    def link_dims(self) -> tuple[int, ...]:
        return tuple(ind.dim for ind in self.link_inds())

    def maxlinkdim(self) -> int:
        """Return the maximum bond dimension of the MPS."""
        return max(self.link_dims())

    def bond_entropy(self, bond: int | None = None, base: float = jnp.e) -> float:
        """
        Return the von Neumann entanglement entropy across a bond.

        Parameters
        ----------
        bond : int | None
            Bond position in the bipartition convention
                [0, ..., bond-1] | [bond, ..., N-1]
            so valid values satisfy 1 <= bond <= len(self) - 1.
            If None, use the half-chain cut `len(self) // 2`.
        base : float
            Logarithm base used in the entropy. The default is the natural base.

        Notes
        -----
        The entropy is computed from the Schmidt values across the requested
        cut. To do this robustly, the MPS is first moved into mixed-canonical
        form with orthogonality center at site `bond - 1`, then a two-site
        tensor crossing the cut is split with an SVD. If the cut is exactly at
        the orthogonality center boundary, the singular values are the Schmidt
        values.
        """
        if len(self) < 2:
            return 0.0

        if bond is None:
            bond = len(self) // 2

        if not (1 <= bond <= len(self) - 1):
            raise ValueError("bond must satisfy 1 <= bond <= len(self) - 1")

        if base <= 0 or base == 1:
            raise ValueError("base must be positive and not equal to 1")

        psi = move_center(self, bond - 1)
        A = psi[bond - 1]
        left_dim, phys_dim, right_dim = A.shape

        mat = A.data.reshape(left_dim * phys_dim, right_dim)
        s = jnp.linalg.svdvals(mat)
        p = jnp.abs(s**2)
        eps = jnp.finfo(p.dtype).eps
        p = p[p > eps]
        
        logp = jnp.log(p) / jnp.log(jnp.asarray(base, dtype=p.dtype))
        entropy = -jnp.sum(p * logp)
        return entropy

    def half_chain_entropy(self, base: float = jnp.e) -> float:
        """
        Return the entanglement entropy across the half-chain cut.

        For an N-site chain this uses the bipartition
            [0, ..., N//2 - 1] | [N//2, ..., N-1].
        """
        if len(self) < 2:
            return 0.0
        return self.bond_entropy(bond=len(self) // 2, base=base)

    def local_expect(self, op_name: str, *op_args, normalize: bool = True):
        """
        Return the expectation value of a one-site observable on every site.

        Parameters
        ----------
        op_name : str
            Local operator name understood by each site's `SiteType.op(...)`,
            for example `"Z"`.
        *op_args
            Optional operator arguments passed through to `SiteType.op`.
        normalize : bool
            If True, divide by `<psi|psi>` so the result is robust even when
            the MPS is not exactly normalized.

        Returns
        -------
        values : jax.Array
            Array of shape `(len(self),)` containing
            `[<O_0>, <O_1>, ..., <O_{N-1}>]`.
        """
        psi = move_center(self, 0)
        vals = []
        out_dtype = None

        for site in range(len(psi)):
            A = psi[site].data
            op = psi.site_type(site).op(op_name, *op_args)
            if out_dtype is None:
                out_dtype = jnp.result_type(op)
            val = jnp.einsum(
                "asb,st,atb->",
                jnp.conjugate(A),
                op,
                A,
                optimize="auto",
            )
            if normalize:
                norm = jnp.einsum(
                    "asb,asb->",
                    jnp.conjugate(A),
                    A,
                    optimize="auto",
                )
                if jnp.abs(norm) > 0:
                    val = val / norm
            
            vals.append(val)

            if site < len(psi) - 1:
                psi = left_canonicalize_site(psi, site)

        return jnp.asarray(vals, dtype=out_dtype)

    def _check_mps_structure(self) -> None:
        for n, T in enumerate(self.tensors):
            if T.ndim != 3:
                raise ValueError(
                    f"Each MPS tensor must have rank 3, but site {n} has rank {T.ndim}"
                )

            lb, phys, rb = T.inds

            if "link" not in lb.tags:
                raise ValueError(f"Site {n} left index must carry tag 'link'")
            if "site" not in phys.tags:
                raise ValueError(f"Site {n} physical index must carry tag 'site'")
            if "link" not in rb.tags:
                raise ValueError(f"Site {n} right index must carry tag 'link'")
            if phys.dim != self.site_types[n].dim:
                raise ValueError(
                    f"Physical dimension mismatch at site {n}: "
                    f"{phys.dim} != {self.site_types[n].dim}"
                )

        for n in range(len(self.tensors) - 1):
            if self.right_link_ind(n) != self.left_link_ind(n + 1):
                raise ValueError(
                    f"Bond mismatch between sites {n} and {n + 1}: "
                    f"{self.right_link_ind(n)} != {self.left_link_ind(n + 1)}"
                )

        if self.left_link_ind(0).dim != 1:
            raise ValueError("Left boundary bond dimension must be 1")
        if self.right_link_ind(len(self) - 1).dim != 1:
            raise ValueError("Right boundary bond dimension must be 1")

    # -------------------------------------------------
    # immutable updates
    # -------------------------------------------------

    def replace_site(self, site: int, tensor: Tensor) -> "MPS":
        new_tensors = list(self.tensors)
        new_tensors[site] = tensor
        return MPS(tuple(new_tensors), self.site_types, gauge_center=None)
    
    def to_tensor(self) -> Tensor:
        """
        Contract the full OBC MPS into a single Tensor over the physical site indices.

        For an MPS with site tensors A[n](link_n, site_n, link_{n+1}), this
        contracts all internal link indices and removes the two boundary links,
        which must both have dimension 1.
        """
        out = self.tensors[0]
        for n in range(1, len(self.tensors)):
            out = contract(out, self.tensors[n])

        expected_inds = (
            self.left_link_ind(0),
            *self.site_inds(),
            self.right_link_ind(len(self) - 1),
        )
        out = out.permute(expected_inds)

        if self.left_link_ind(0).dim != 1 or self.right_link_ind(len(self) - 1).dim != 1:
            raise ValueError(
                "Boundary link dimensions must be 1 to convert an OBC MPS to a full state tensor"
            )

        squeezed = jnp.squeeze(out.data, axis=(0, out.data.ndim - 1))
        return Tensor(squeezed, self.site_inds())
    
    # -------------------------------------------------
    # algebra
    # -------------------------------------------------
    def __mul__(self, scalar) -> "MPS":
        if not jnp.isscalar(scalar):
            return NotImplemented

        target = 0 if self.gauge_center is None else self.gauge_center
        new_tensors = list(self.tensors)
        new_tensors[target] = new_tensors[target] * scalar
        return MPS(tuple(new_tensors), self.site_types, self.gauge_center)
    
    def __rmul__(self, scalar) -> "MPS":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "MPS":
        if not jnp.isscalar(scalar):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("division by zero")
        return self * (1.0 / scalar)

    def __add__(self, other: "MPS") -> "MPS":
        if not isinstance(other, MPS):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("MPS lengths must match for addition")

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
            return MPS((Tensor(data, self[0].inds),), self.site_types, gauge_center=None)

        new_tensors = []

        # first site: concatenate along the right virtual leg
        A0, B0 = self[0], other[0]
        data0 = jnp.concatenate([A0.data, B0.data], axis=2)
        right_link0 = Index(
            data0.shape[2],
            self.right_link_ind(0).name,
            self.right_link_ind(0).tags,
            self.right_link_ind(0).prime_level,
        )
        new_tensors.append(
            Tensor(data0, (self.left_link_ind(0), self.phys_ind(0), right_link0))
        )

        # middle sites: block diagonal in the virtual bonds
        for n in range(1, len(self) - 1):
            A = self[n]
            B = other[n]
            la, d, ra = A.shape
            lb, _, rb = B.shape
            dtype = jnp.result_type(A.data, B.data)

            data = jnp.zeros((la + lb, d, ra + rb), dtype=dtype)
            data = data.at[:la, :, :ra].set(A.data)
            data = data.at[la:, :, ra:].set(B.data)

            left_link = new_tensors[-1].inds[2]
            right_link = Index(
                ra + rb,
                self.right_link_ind(n).name,
                self.right_link_ind(n).tags,
                self.right_link_ind(n).prime_level,
            )
            new_tensors.append(Tensor(data, (left_link, self.phys_ind(n), right_link)))

        # last site: concatenate along the left virtual leg
        A_last, B_last = self[-1], other[-1]
        data_last = jnp.concatenate([A_last.data, B_last.data], axis=0)
        left_link_last = new_tensors[-1].inds[2]
        new_tensors.append(
            Tensor(
                data_last,
                (
                    left_link_last,
                    self.phys_ind(len(self) - 1),
                    self.right_link_ind(len(self) - 1),
                ),
            )
        )

        return MPS(tuple(new_tensors), self.site_types, gauge_center=None)

    def __sub__(self, other: "MPS") -> "MPS":
        if not isinstance(other, MPS):
            return NotImplemented
        return self + ((-1.0) * other)
    
    # -------------------------------------------------
    # constructors
    # -------------------------------------------------

    @classmethod
    def product_state(
        cls,
        site_types: Sequence[SiteType],
        states: Sequence[str],
        link_prefix: str = "link",
        site_prefix: str = "site",
    ) -> "MPS":
        """
        Construct a product state MPS from site types and local state labels.

        Parameters
        ----------
        site_types : Sequence[SiteType]
            Local Hilbert space types (e.g. QubitSite, SpinHalfSite, ...).
        states : Sequence[str]
            Local state labels compatible with each site type.
        """

        if len(site_types) != len(states):
            raise ValueError("site_types and states must have the same length")

        nsites = len(site_types)
        if nsites == 0:
            raise ValueError("Product state must contain at least one site")

        # create link indices (N+1 of them)
        links = [
            Index(1, f"{link_prefix}-{i}", tags=("link", f"{i}"))
            for i in range(nsites + 1)
        ]

        tensors = []

        for n, (stype, state_label) in enumerate(zip(site_types, states)):
            # local state vector from SiteType
            vec = stype.state(state_label)
            d = stype.dim

            # (site_role, site_position, site_type)
            site_ind = Index(d, f"{site_prefix}-{n}", tags=("site", f"{n}", stype.name))

            data = jnp.zeros((1, d, 1), dtype=jnp.result_type(vec))
            data = data.at[0, :, 0].set(vec)

            left_link = links[n]
            right_link = links[n + 1]

            tensors.append(Tensor(data, (left_link, site_ind, right_link)))

        return cls(tuple(tensors), tuple(site_types), gauge_center=0)





# -------------------------------------------------
# MPS functions
# -------------------------------------------------

def mps_inner(phi: MPS, psi: MPS):
    """
    Compute the overlap <phi|psi>. 

    The two MPS do not need to share the same site Index objects. It is enough
    that they have the same number of sites and compatible local site spaces.
    Internally, the bra network is built by conjugating `phi`, priming only its
    link indices, and then aligning each bra site index with the corresponding
    ket site index of `psi`.
    """
    if len(phi) != len(psi):
        raise ValueError("MPS lengths must match for inner product")

    for n in range(len(phi)):
        if phi.site_dim(n) != psi.site_dim(n):
            raise ValueError(
                f"Site dimension mismatch at site {n}: "
                f"{phi.site_dim(n)} != {psi.site_dim(n)}"
            )
        if phi.site_type(n).name != psi.site_type(n).name:
            raise ValueError(
                f"Site type mismatch at site {n}: "
                f"{phi.site_type(n).name} != {psi.site_type(n).name}"
            )

    bra = phi.conj().prime_link_inds()

    aligned_bra_tensors = []
    for n in range(len(phi)):
        bra_tensor = bra[n].replace_ind(bra.phys_ind(n), psi.phys_ind(n))
        aligned_bra_tensors.append(bra_tensor)

    env = contract(aligned_bra_tensors[0], psi[0])

    for n in range(1, len(psi)):
        env = contract(env, aligned_bra_tensors[n])
        env = contract(env, psi[n])

    return env.data.squeeze()


def random_mps(
    site_types: Sequence[SiteType],
    maxlinkdim: int,
    dtype: str | object = jnp.float64,
    seed: int = 0,
    link_prefix: str = "link",
    site_prefix: str = "site",
    normalize: bool = True,
    gauge_center: int = 0,
) -> MPS:
    """
    Construct a random open-boundary MPS with bond dimensions capped by
    `maxlinkdim`.

    Parameters
    ----------
    site_types : Sequence[SiteType]
        Local Hilbert-space types for each site.
    maxlinkdim : int
        Maximum allowed internal bond dimension.
    dtype : str or dtype-like
        Either the strings "float" / "complex", or an explicit real/complex
        dtype such as `jnp.float32`, `jnp.float64`, `jnp.complex64`, or
        `jnp.complex128`.
    seed : int
        PRNG seed used to generate the random tensor entries.
    normalize : bool
        If True, normalize the full MPS so that <psi|psi> = 1.
    gauge_center : int
        Desired orthogonality center of the returned random MPS.
    """
    if len(site_types) == 0:
        raise ValueError("random_mps requires at least one site")
    if maxlinkdim <= 0:
        raise ValueError("maxlinkdim must be positive")
    # dtype validation and resolution
    if isinstance(dtype, str):
        if dtype == "float":
            resolved_dtype = jnp.dtype(jnp.float64)
        elif dtype == "complex":
            resolved_dtype = jnp.dtype(jnp.complex128)
        else:
            raise ValueError(
                "dtype string must be either 'float' or 'complex'"
            )
    else:
        try:
            resolved_dtype = jnp.dtype(dtype)
        except Exception as exc:
            raise ValueError(
                "dtype must be either 'float', 'complex', or a valid dtype object"
            ) from exc

        if jnp.dtype(resolved_dtype).kind not in ("f", "c"):
            raise ValueError(
                "dtype object must be a floating or complex floating type"
            )
    if not (0 <= gauge_center < len(site_types)):
        raise ValueError("gauge_center must satisfy 0 <= gauge_center < len(site_types)")

    nsites = len(site_types)
    phys_dims = [stype.dim for stype in site_types]

    # Canonical upper bound on each bond dimension:
    #   chi_n <= min(prod_{i < n} d_i, prod_{i >= n} d_i, maxlinkdim)
    left_dims = [1]
    for d in phys_dims:
        left_dims.append(left_dims[-1] * d)

    right_dims = [1] * (nsites + 1)
    for n in range(nsites - 1, -1, -1):
        right_dims[n] = right_dims[n + 1] * phys_dims[n]

    bond_dims = [1]
    for n in range(1, nsites):
        bond_dims.append(min(maxlinkdim, left_dims[n], right_dims[n]))
    bond_dims.append(1)

    links = [
        Index(bond_dims[i], f"{link_prefix}-{i}", tags=("link", f"{i}"))
        for i in range(nsites + 1)
    ]

    key = jax.random.PRNGKey(seed)
    tensors = []

    for n, stype in enumerate(site_types):
        d = stype.dim
        left_dim = bond_dims[n]
        right_dim = bond_dims[n + 1]

        site_ind = Index(d, f"{site_prefix}-{n}", tags=("site", f"{n}", stype.name))

        key, subkey = jax.random.split(key)
        shape = (left_dim, d, right_dim)

        if jnp.dtype(resolved_dtype).kind == "f":
            data = jax.random.normal(subkey, shape, dtype=resolved_dtype)
        else:
            real_dtype = jnp.float32 if jnp.dtype(resolved_dtype) == jnp.dtype(jnp.complex64) else jnp.float64
            key, subkey_im = jax.random.split(key)
            real = jax.random.normal(subkey, shape, dtype=real_dtype)
            imag = jax.random.normal(subkey_im, shape, dtype=real_dtype)
            data = (real + 1j * imag).astype(resolved_dtype)

        # Normalize each local tensor to avoid pathological norms.
        norm = jnp.linalg.norm(data)
        if norm > 0:
            data = data / norm

        tensors.append(Tensor(data, (links[n], site_ind, links[n + 1])))

    psi = MPS(tuple(tensors), tuple(site_types), gauge_center=None)

    # Put the random state into mixed-canonical form with the requested center.
    psi = move_center(psi, gauge_center)

    if normalize:
        norm = jnp.linalg.norm(psi.tensors[gauge_center].data)
        if norm > 0:
            scaled_tensors = list(psi.tensors)
            scaled_tensors[gauge_center] = scaled_tensors[gauge_center] * (1.0 / norm)
            psi = MPS(tuple(scaled_tensors), tuple(site_types), gauge_center=gauge_center)

    return psi


def left_canonicalize_site(psi: MPS, n: int) -> MPS:
    """
    Perform one left-canonicalization step at site n using a QR decomposition.

    This factors the site tensor

        A[n](link_n, site_n, link_{n+1}) = Q[n] R[n]

    with left legs `(link_n, site_n)`. The tensor `Q[n]` replaces site `n`,
    and `R[n]` is absorbed into site `n+1`.

    Important:
        `tensor_qr` introduces a fresh intermediate QR bond. For an MPS we want
        to preserve the original chain link index at position `n+1`, so the
        fresh QR bond is replaced by the original `link_{n+1}` on both updated
        tensors.

    The returned MPS has `gauge_center = n + 1`.
    """
    if not (0 <= n < len(psi) - 1):
        raise ValueError("n must satisfy 0 <= n < len(psi) - 1")

    tensors = [T.copy() for T in psi.tensors]

    A = tensors[n]
    left_link = A.inds[0]
    site_ind = A.inds[1]
    old_right_link = A.inds[2]

    Q, R = tensor_qr(A, left_inds=(left_link, site_ind))

    # Keep the original chain link at position n+1.
    qr_link = Q.inds[2]
    Q = Q.replace_ind(qr_link, old_right_link)
    tensors[n] = Q

    A_next = tensors[n + 1]
    new_next = contract(R, A_next)
    new_next = new_next.replace_ind(qr_link, old_right_link)
    tensors[n + 1] = new_next

    return MPS(tuple(tensors), psi.site_types, gauge_center=n + 1)



def left_canonicalize(psi: MPS, stop: int | None = None) -> MPS:
    """
    Bring an open-boundary MPS into left-canonical form up to a target site.

    If `psi.gauge_center` is known, the sweep starts from that center,
    assuming sites strictly to the left are already left-canonical.
    The returned MPS has `gauge_center = stop`.
    """
    if len(psi) == 1:
        return MPS(tuple(T.copy() for T in psi.tensors), psi.site_types, gauge_center=0)

    out = psi.copy()

    if stop is None:
        stop = len(out) - 1

    if not (0 <= stop <= len(out) - 1):
        raise ValueError("stop must satisfy 0 <= stop <= len(psi) - 1")

    for n in range(stop):
        out = left_canonicalize_site(out, n)

    return MPS(tuple(T.copy() for T in out.tensors), out.site_types, gauge_center=stop)


def right_canonicalize_site(psi: MPS, n: int) -> MPS:
    """
    Perform one right-canonicalization step at site n using an RQ decomposition.

    This factors the site tensor

        A[n](link_n, site_n, link_{n+1}) = R[n] Q[n]

    with right legs `(site_n, link_{n+1})`. The tensor `Q[n]` replaces site `n`,
    and `R[n]` is absorbed into site `n-1`.

    Important:
        `tensor_rq` introduces a fresh intermediate RQ bond. For an MPS we want
        to preserve the original chain link index at position `n`, so the fresh
        RQ bond is replaced by the original `link_n` on both updated tensors.

    The returned MPS has `gauge_center = n - 1`.
    """
    if not (0 < n <= len(psi) - 1):
        raise ValueError("n must satisfy 0 < n <= len(psi) - 1")

    tensors = [T.copy() for T in psi.tensors]

    A = tensors[n]
    old_left_link = A.inds[0]
    site_ind = A.inds[1]
    right_link = A.inds[2]

    R, Q = tensor_rq(A, left_inds=(old_left_link,))

    # Keep the original chain link at position n.
    rq_link = Q.inds[0]
    Q = Q.replace_ind(rq_link, old_left_link)
    tensors[n] = Q

    A_prev = tensors[n - 1]
    new_prev = contract(A_prev, R)
    new_prev = new_prev.replace_ind(rq_link, old_left_link)
    tensors[n - 1] = new_prev

    return MPS(tuple(tensors), psi.site_types, gauge_center=n - 1)


def right_canonicalize(psi: MPS, stop: int | None = None) -> MPS:
    """
    Bring an open-boundary MPS into right-canonical form up to a target site.

    If `psi.gauge_center` is known, the sweep starts from that center,
    assuming sites strictly to the right are already right-canonical.
    The returned MPS has `gauge_center = stop`.
    """
    if len(psi) == 1:
        return MPS(tuple(T.copy() for T in psi.tensors), psi.site_types, gauge_center=0)

    out = psi.copy()

    if stop is None:
        stop = 0

    if not (0 <= stop <= len(out) - 1):
        raise ValueError("stop must satisfy 0 <= stop <= len(psi) - 1")

    for n in range(len(out) - 1, stop, -1):
        out = right_canonicalize_site(out, n)

    return MPS(tuple(T.copy() for T in out.tensors), out.site_types, gauge_center=stop)


def move_center(psi: MPS, center: int) -> MPS:
    """
    Move the orthogonality center of an MPS to the target site.

    If the current `psi.gauge_center` is known, only the necessary partial sweep
    is performed. Otherwise the function falls back to a full sweep from the
    relevant boundary.
    """
    if not (0 <= center < len(psi)):
        raise ValueError("center must satisfy 0 <= center < len(psi)")

    if psi.gauge_center == center:
        return psi.copy()

    # Build the requested mixed gauge explicitly from the two boundaries.
    # This avoids trusting possibly stale gauge-center metadata.
    out = left_canonicalize(psi, stop=center)
    out = right_canonicalize(out, stop=center)
    return out


def check_gauge_center(psi: MPS, center: int, tol: float = 1e-10) -> bool:
    """
    Check whether an MPS is in mixed canonical form with orthogonality center
    at `center`.

    Concretely:
      - every site strictly left of `center` must be left-canonical,
      - every site strictly right of `center` must be right-canonical.

    The center site itself is not constrained.
    """
    if not (0 <= center < len(psi)):
        raise ValueError("center must satisfy 0 <= center < len(psi)")
    if tol < 0:
        raise ValueError("tol must be non-negative")

    def _left_isometry_error(A) -> float:
        mat = A.data.reshape(A.shape[0] * A.shape[1], A.shape[2])
        gram = jnp.conjugate(mat).T @ mat
        return float(jnp.linalg.norm(gram - jnp.eye(mat.shape[1], dtype=gram.dtype)))

    def _right_isometry_error(A) -> float:
        mat = A.data.reshape(A.shape[0], A.shape[1] * A.shape[2])
        gram = mat @ jnp.conjugate(mat).T
        return float(jnp.linalg.norm(gram - jnp.eye(mat.shape[0], dtype=gram.dtype)))

    for n in range(center):
        if _left_isometry_error(psi[n]) > tol:
            return False

    for n in range(center + 1, len(psi)):
        if _right_isometry_error(psi[n]) > tol:
            return False

    return True


def truncate_mps(
    psi: MPS,
    maxdim: int | None = None,
    cutoff: float = 0.0,
    center: int | None = None,
    normalize: bool = True,
) -> MPS:
    """
    Truncate the bond dimensions of an MPS without changing its current
    orthogonality center.

    Assumptions
    -----------
    The input MPS is assumed to already be in mixed-canonical form with respect
    to its gauge center. If `psi.gauge_center` is unknown, the caller may pass
    `center` explicitly and that site will be treated as the orthogonality center.

    Strategy
    --------
    - On the left of the gauge center, sweep from left to right. At each bond
      `(n, n+1)`, merge the two neighboring sites, perform a truncated SVD with
      left legs `(link_n, site_n)`, and absorb the diagonal singular-value
      tensor `S` into the right site.
    - On the right of the gauge center, sweep from right to left. At each bond
      `(n-1, n)`, merge the two neighboring sites, perform a truncated SVD with
      left legs `(link_{n-1}, site_{n-1})`, and absorb `S` into the left site.

    This preserves the gauge center position while reducing intermediate bond
    dimensions according to `maxdim` and `cutoff`.

    normalize : bool
        If True, normalize the MPS after truncation by absorbing the norm
        into the center tensor.
    """
    if maxdim is not None and maxdim <= 0:
        raise ValueError("maxdim must be positive or None")
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    if len(psi) == 1:
        return psi.copy()

    if center is None:
        center = psi.gauge_center

    if center is None:
        raise ValueError("truncate_mps requires either psi.gauge_center or an explicit center")

    if not (0 <= center < len(psi)):
        raise ValueError("center must satisfy 0 <= center < len(psi)")

    tensors = [T.copy() for T in psi.tensors]

    # -----------------------------------------------------------------
    # Left of the center: left-to-right sweep, absorb S into the right site.
    # -----------------------------------------------------------------
    for n in range(center):
        theta = contract(tensors[n], tensors[n + 1])

        left_link = tensors[n].inds[0]
        left_site = tensors[n].inds[1]
        old_mid_link = tensors[n].inds[2]
        right_site = tensors[n + 1].inds[1]
        right_link = tensors[n + 1].inds[2]

        U, S, Vh, info = truncated_svd(
            theta,
            left_inds=(left_link, left_site),
            max_bond=maxdim,
            cutoff=cutoff,
            bond_name=f"trunc_left_{n}",
        )

        new_mid_link = Index(
            dim=U.inds[2].dim,
            name=old_mid_link.name,
            tags=old_mid_link.tags,
            prime_level=old_mid_link.prime_level,
        )

        left_tensor = U.replace_ind(U.inds[2], new_mid_link)

        right_tensor = contract(S, Vh)
        right_tensor = right_tensor.replace_ind(right_tensor.inds[0], new_mid_link)

        expected_right_inds = (new_mid_link, right_site, right_link)
        right_tensor = right_tensor.permute(expected_right_inds)

        tensors[n] = left_tensor
        tensors[n + 1] = right_tensor

    # -----------------------------------------------------------------
    # Right of the center: right-to-left sweep, absorb S into the left site.
    # -----------------------------------------------------------------
    for n in range(len(psi) - 1, center, -1):
        theta = contract(tensors[n - 1], tensors[n])

        left_link = tensors[n - 1].inds[0]
        left_site = tensors[n - 1].inds[1]
        old_mid_link = tensors[n - 1].inds[2]
        right_site = tensors[n].inds[1]
        right_link = tensors[n].inds[2]

        U, S, Vh, info = truncated_svd(
            theta,
            left_inds=(left_link, left_site),
            max_bond=maxdim,
            cutoff=cutoff,
            bond_name=f"trunc_right_{n-1}",
        )

        new_mid_link = Index(
            dim=U.inds[2].dim,
            name=old_mid_link.name,
            tags=old_mid_link.tags,
            prime_level=old_mid_link.prime_level,
        )

        left_tensor = contract(U, S)
        left_tensor = left_tensor.replace_ind(left_tensor.inds[2], new_mid_link)
        left_tensor = left_tensor.permute((left_link, left_site, new_mid_link))

        right_tensor = Vh.replace_ind(Vh.inds[0], new_mid_link)
        expected_right_inds = (new_mid_link, right_site, right_link)
        right_tensor = right_tensor.permute(expected_right_inds)

        tensors[n - 1] = left_tensor
        tensors[n] = right_tensor

    out = MPS(tuple(tensors), psi.site_types, gauge_center=center)

    if normalize:
        # Since the MPS is in mixed-canonical form with known orthogonality
        # center, the full state norm is exactly the Frobenius norm of the
        # center tensor.
        norm = jnp.linalg.norm(out.tensors[center].data)
        if norm > 0:
            scaled = list(out.tensors)
            scaled[center] = scaled[center] * (1.0 / norm)
            out = MPS(tuple(scaled), psi.site_types, gauge_center=center)

    return out
