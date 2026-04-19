from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp

from qtensor.models.sites import SiteType
from qtensor.mps.mpo import MPO
from qtensor.tensor.index import Index
from qtensor.tensor.tensor import Tensor


Token = tuple[str, ...]
DONE_SIG = ("__done__",)


@dataclass(frozen=True)
class OpTerm:
    coeff: complex
    factors: tuple[tuple[str, int], ...]


class OpSum:
    """
    Lightweight AutoMPO-style operator sum builder.

    Example
    -------
    ```python
    os = OpSum()
    os += 0.3, "X", 0, "X", 2
    os += 1.0, "Z", 1
    H = os.to_mpo(sites)
    ```
    """

    def __init__(self):
        self.terms: list[OpTerm] = []

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms)

    def add_term(self, coeff, *opsites) -> "OpSum":
        if len(opsites) == 0 or len(opsites) % 2 != 0:
            raise ValueError("A term must be given as coeff, op1, site1, op2, site2, ...")

        factors = []
        for i in range(0, len(opsites), 2):
            op_name = opsites[i]
            site = opsites[i + 1]
            if not isinstance(op_name, str):
                raise TypeError("Operator names must be strings")
            if not isinstance(site, int):
                raise TypeError("Site indices must be integers")
            factors.append((op_name, site))

        self.terms.append(OpTerm(coeff=coeff, factors=tuple(factors)))
        return self

    def __iadd__(self, other):
        if not isinstance(other, tuple) or len(other) < 3:
            raise TypeError("Use `opsum += coeff, \"Op\", site, ...`")
        coeff = other[0]
        return self.add_term(coeff, *other[1:])

    def to_dense(self, site_types: Sequence[SiteType]):
        if len(site_types) == 0:
            raise ValueError("site_types must contain at least one site")

        nsites = len(site_types)
        local_ids = [stype.eye() for stype in site_types]
        dtype = jnp.result_type(
            *(op for op in local_ids),
            *(term.coeff for term in self.terms) if self.terms else (1.0,),
        )
        dim = 1
        for stype in site_types:
            dim *= stype.dim

        H = jnp.zeros((dim, dim), dtype=dtype)
        for term in self.terms:
            local_ops = [jnp.asarray(op, dtype=dtype) for op in local_ids]
            for op_name, site in term.factors:
                if not (0 <= site < nsites):
                    raise ValueError(f"Site index {site} out of range for {nsites} sites")
                op = jnp.asarray(site_types[site].op(op_name), dtype=dtype)
                local_ops[site] = local_ops[site] @ op

            dense_term = local_ops[0]
            for op in local_ops[1:]:
                dense_term = jnp.kron(dense_term, op)
            H = H + term.coeff * dense_term
        return H

    def to_mpo(
        self,
        site_types: Sequence[SiteType],
        cutoff: float = 0.0,
        max_bond: int | None = None,
        compress: bool = False,
        link_prefix: str = "link",
        site_prefix: str = "site",
    ) -> MPO:
        return to_mpo(
            self,
            site_types,
            cutoff=cutoff,
            max_bond=max_bond,
            compress=compress,
            link_prefix=link_prefix,
            site_prefix=site_prefix,
        )


def _normalize_term(term: OpTerm, site_types: Sequence[SiteType]) -> tuple[int, Token, tuple[Token, ...]] | None:
    by_site: dict[int, list[str]] = {}
    for op_name, site in term.factors:
        if not (0 <= site < len(site_types)):
            raise ValueError(f"Site index {site} out of range for {len(site_types)} sites")
        by_site.setdefault(site, []).append(op_name)

    if not by_site:
        return None

    support = sorted(by_site)
    start = support[0]
    stop = support[-1]
    tokens: list[Token] = []
    for site in range(start, stop + 1):
        tokens.append(tuple(by_site.get(site, ())))

    first = tokens[0]
    rest = tuple(tokens[1:])
    return start, first, rest


def _token_op(stype: SiteType, token: Token, coeff=1.0):
    op = jnp.asarray(stype.eye())
    for name in token:
        op = op @ jnp.asarray(stype.op(name))
    return coeff * op


def _build_fsm_terms(
    opsum: OpSum,
    site_types: Sequence[SiteType],
) -> tuple[dict[int, list[tuple[tuple[Token, ...], complex]]], dict[int, set[tuple[Token, ...]]]]:
    starts: dict[int, list[tuple[tuple[Token, ...], complex]]] = {}
    residuals_by_bond: dict[int, set[tuple[Token, ...]]] = {}

    for term in opsum:
        normalized = _normalize_term(term, site_types)
        if normalized is None:
            continue
        start, first, rest = normalized
        full = (first,) + rest
        starts.setdefault(start, []).append((full, term.coeff))
        for offset in range(1, len(full)):
            bond = start + offset
            residuals_by_bond.setdefault(bond, set()).add(full[offset:])

    return starts, residuals_by_bond


def to_mpo(
    opsum: OpSum,
    site_types: Sequence[SiteType],
    cutoff: float = 0.0,
    max_bond: int | None = None,
    compress: bool = False,
    link_prefix: str = "link",
    site_prefix: str = "site",
) -> MPO:
    """
    Convert an `OpSum` into an MPO using a finite-state-machine construction.

    The MPO is built directly from symbolic local-term continuations rather
    than by summing term MPOs one by one.
    """
    if not isinstance(opsum, OpSum):
        raise TypeError("opsum must be an OpSum")
    if len(site_types) == 0:
        raise ValueError("site_types must contain at least one site")
    if len(opsum) == 0:
        zero = MPO.identity(site_types, link_prefix=link_prefix, site_prefix=site_prefix)
        return 0.0 * zero

    nsites = len(site_types)
    starts, residuals_by_bond = _build_fsm_terms(opsum, site_types)

    sig_to_local_by_bond: dict[int, dict[tuple, int]] = {}
    residual_to_sig_by_bond: dict[int, dict[tuple[Token, ...], tuple]] = {}
    sig_order_by_bond: dict[int, list[tuple]] = {}

    for bond in range(nsites - 1, -1, -1):
        residuals = sorted(residuals_by_bond.get(bond, set()), key=lambda seq: (len(seq), seq))
        sig_to_local: dict[tuple, int] = {}
        residual_to_sig: dict[tuple[Token, ...], tuple] = {}
        sig_order: list[tuple] = []
        for residual in residuals:
            next_sig = DONE_SIG if len(residual) == 1 else residual_to_sig_by_bond[bond + 1][residual[1:]]
            sig = (residual[0], next_sig)
            residual_to_sig[residual] = sig
            if sig not in sig_to_local:
                sig_to_local[sig] = len(sig_order)
                sig_order.append(sig)
        sig_to_local_by_bond[bond] = sig_to_local
        residual_to_sig_by_bond[bond] = residual_to_sig
        sig_order_by_bond[bond] = sig_order

    link_dims = [1]
    for bond in range(1, nsites):
        link_dims.append(2 + len(sig_order_by_bond.get(bond, [])))
    link_dims.append(1)
    links = [Index(dim, f"{link_prefix}-{i}", tags=("link", f"{i}")) for i, dim in enumerate(link_dims)]

    tensors = []
    for n, stype in enumerate(site_types):
        d = stype.dim
        dtype = jnp.result_type(stype.eye(), *(term.coeff for term in opsum))
        ldim = link_dims[n]
        rdim = link_dims[n + 1]
        data = jnp.zeros((ldim, d, d, rdim), dtype=dtype)

        if n == 0:
            idle_right = 1 if rdim > 1 else None
            if idle_right is not None:
                data = data.at[0, :, :, idle_right].set(stype.eye())
            for seq, coeff in starts.get(n, []):
                op = _token_op(stype, seq[0], coeff=coeff)
                if len(seq) == 1:
                    if n == nsites - 1:
                        data = data.at[0, :, :, 0].add(op)
                    else:
                        data = data.at[0, :, :, 0].add(op)
                else:
                    sig = residual_to_sig_by_bond[n + 1][seq[1:]]
                    dst = 2 + sig_to_local_by_bond[n + 1][sig]
                    data = data.at[0, :, :, dst].add(op)
        elif n == nsites - 1:
            data = data.at[0, :, :, 0].set(stype.eye())
            for sig_idx, sig in enumerate(sig_order_by_bond.get(n, [])):
                token, next_sig = sig
                if next_sig != DONE_SIG:
                    raise ValueError("Invalid finite-state machine: unfinished residual at final site")
                data = data.at[2 + sig_idx, :, :, 0].add(_token_op(stype, token))
            for seq, coeff in starts.get(n, []):
                if len(seq) != 1:
                    raise ValueError("Invalid term extends beyond the chain")
                data = data.at[1, :, :, 0].add(_token_op(stype, seq[0], coeff=coeff))
        else:
            data = data.at[0, :, :, 0].set(stype.eye())
            data = data.at[1, :, :, 1].set(stype.eye())
            for sig_idx, sig in enumerate(sig_order_by_bond.get(n, [])):
                token, next_sig = sig
                dst = 0 if next_sig == DONE_SIG else 2 + sig_to_local_by_bond[n + 1][next_sig]
                data = data.at[2 + sig_idx, :, :, dst].add(_token_op(stype, token))
            for seq, coeff in starts.get(n, []):
                op = _token_op(stype, seq[0], coeff=coeff)
                if len(seq) == 1:
                    data = data.at[1, :, :, 0].add(op)
                else:
                    sig = residual_to_sig_by_bond[n + 1][seq[1:]]
                    dst = 2 + sig_to_local_by_bond[n + 1][sig]
                    data = data.at[1, :, :, dst].add(op)

        site_in = Index(d, f"{site_prefix}_in-{n}", tags=("site_in", f"{n}", stype.name))
        site_out = Index(d, f"{site_prefix}_out-{n}", tags=("site_out", f"{n}", stype.name))
        tensors.append(Tensor(data, (links[n], site_in, site_out, links[n + 1])))

    out = MPO(tuple(tensors), tuple(site_types), gauge_center=0)
    if compress:
        out = out.truncate(cutoff=cutoff, max_bond=max_bond, method="regular")
    return out
