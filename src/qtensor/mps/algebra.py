from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from qtensor.models.sites import SiteType
from qtensor.tensor.contract import contract
from qtensor.tensor.index import Index
from qtensor.tensor.tensor import Tensor
from qtensor.mps.mps import MPS
from qtensor.mps.mpo import MPO


########################################################################################
# MPS expectations, ...
########################################################################################

def inner(*args):
    """
    Unified inner-product / matrix-element interface.

    Supported call signatures
    -------------------------
    inner(phi, psi)
        Returns <phi|psi> for two MPS objects.

    inner(phi, W, psi)
        Returns <phi|W|psi> for MPS `phi`, MPO `W`, and MPS `psi`.

    inner(phi, W1, W2, psi)
        Returns <phi|W1 W2|psi> for MPS `phi`, MPOs `W1`, `W2`, and MPS `psi`.
    """
    if len(args) == 2:
        phi, psi = args
        if not isinstance(phi, MPS) or not isinstance(psi, MPS):
            raise TypeError("inner(phi, psi) requires two MPS objects")

        from mps.mps import mps_inner
        return mps_inner(phi, psi)

    if len(args) == 3:
        phi, W, psi = args
        if not isinstance(phi, MPS) or not isinstance(W, MPO) or not isinstance(psi, MPS):
            raise TypeError("inner(phi, W, psi) requires (MPS, MPO, MPS)")

        if len(W) != len(phi) or len(W) != len(psi):
            raise ValueError("MPO and MPS lengths must match for matrix element")

        for n in range(len(psi)):
            if phi.site_dim(n) != psi.site_dim(n):
                raise ValueError(
                    f"MPS site dimension mismatch at site {n}: "
                    f"{phi.site_dim(n)} != {psi.site_dim(n)}"
                )
            if W.site_dim(n) != psi.site_dim(n):
                raise ValueError(
                    f"Site dimension mismatch at site {n}: "
                    f"{W.site_dim(n)} != {psi.site_dim(n)}"
                )
            if phi.site_type(n).name != psi.site_type(n).name:
                raise ValueError(
                    f"MPS site type mismatch at site {n}: "
                    f"{phi.site_type(n).name} != {psi.site_type(n).name}"
                )
            if W.site_type(n).name != psi.site_type(n).name:
                raise ValueError(
                    f"Site type mismatch at site {n}: "
                    f"{W.site_type(n).name} != {psi.site_type(n).name}"
                )

        bra = phi.conj().prime_link_inds()

        aligned_bra = []
        aligned_ket = []
        for n in range(len(psi)):
            bra_n = bra[n].replace_ind(bra.site_inds()[n], W.site_in_ind(n))
            ket_n = psi[n].replace_ind(psi.site_inds()[n], W.site_out_ind(n))
            aligned_bra.append(bra_n)
            aligned_ket.append(ket_n)

        env = contract(aligned_bra[0], W[0], aligned_ket[0])

        for n in range(1, len(psi)):
            env = contract(env, aligned_bra[n], W[n], aligned_ket[n])

        return env.data.squeeze()

    if len(args) == 4:
        phi, W1, W2, psi = args
        if (
            not isinstance(phi, MPS)
            or not isinstance(W1, MPO)
            or not isinstance(W2, MPO)
            or not isinstance(psi, MPS)
        ):
            raise TypeError("inner(phi, W1, W2, psi) requires (MPS, MPO, MPO, MPS)")

        if len(W1) != len(phi) or len(W2) != len(phi) or len(phi) != len(psi):
            raise ValueError("MPO and MPS lengths must match for matrix element")

        for n in range(len(psi)):
            if phi.site_dim(n) != psi.site_dim(n):
                raise ValueError(
                    f"MPS site dimension mismatch at site {n}: "
                    f"{phi.site_dim(n)} != {psi.site_dim(n)}"
                )
            if W1.site_dim(n) != psi.site_dim(n):
                raise ValueError(
                    f"Site dimension mismatch for W1 at site {n}: "
                    f"{W1.site_dim(n)} != {psi.site_dim(n)}"
                )
            if W2.site_dim(n) != psi.site_dim(n):
                raise ValueError(
                    f"Site dimension mismatch for W2 at site {n}: "
                    f"{W2.site_dim(n)} != {psi.site_dim(n)}"
                )
            if phi.site_type(n).name != psi.site_type(n).name:
                raise ValueError(
                    f"MPS site type mismatch at site {n}: "
                    f"{phi.site_type(n).name} != {psi.site_type(n).name}"
                )
            if W1.site_type(n).name != psi.site_type(n).name:
                raise ValueError(
                    f"Site type mismatch for W1 at site {n}: "
                    f"{W1.site_type(n).name} != {psi.site_type(n).name}"
                )
            if W2.site_type(n).name != psi.site_type(n).name:
                raise ValueError(
                    f"Site type mismatch for W2 at site {n}: "
                    f"{W2.site_type(n).name} != {psi.site_type(n).name}"
                )

        bra = phi.conj().prime_link_inds()

        aligned_bra = []
        aligned_ket = []
        for n in range(len(psi)):
            bra_n = bra[n].replace_ind(bra.site_inds()[n], W1.site_in_ind(n))
            ket_n = psi[n].replace_ind(psi.site_inds()[n], W2.site_out_ind(n))
            aligned_bra.append(bra_n)
            aligned_ket.append(ket_n)

        W1_tensors = []
        W2_tensors = []

        # Keep the W1 MPO bond structure as-is, but rebuild W2 with a fresh set
        # of MPO bond indices so that W1 and W2 do not accidentally contract
        # with each other when they are in fact the same MPO object.
        w2_link_inds = []
        for n in range(len(psi) + 1):
            if n == 0 or n == len(psi):
                w2_link_inds.append(W2.left_link_ind(0) if n == 0 else W2.right_link_ind(len(psi) - 1))
            else:
                src = W2.right_link_ind(n - 1)
                w2_link_inds.append(
                    Index(
                        dim=src.dim,
                        name=f"__inner_w2_link_{n}",
                        tags=("inner", "w2", f"bond-{n}"),
                    )
                )

        for n in range(len(psi)):
            w12_mid = Index(
                dim=W1.site_dim(n),
                name=f"__inner_w12_mid_site_{n}",
                tags=("inner", "mid", f"site-{n}"),
            )

            W1_n = W1[n].replace_ind(W1.site_out_ind(n), w12_mid)

            W2_n = W2[n]
            W2_n = W2_n.replace_ind(W2.left_link_ind(n), w2_link_inds[n])
            W2_n = W2_n.replace_ind(W2.right_link_ind(n), w2_link_inds[n + 1])
            W2_n = W2_n.replace_ind(W2.site_in_ind(n), w12_mid)

            W1_tensors.append(W1_n)
            W2_tensors.append(W2_n)

        env = contract(aligned_bra[0], W1_tensors[0], W2_tensors[0], aligned_ket[0])

        for n in range(1, len(psi)):
            env = contract(env, aligned_bra[n], W1_tensors[n], W2_tensors[n], aligned_ket[n])

        return env.data.squeeze()

    raise TypeError(
        "inner expects one of (MPS, MPS), (MPS, MPO, MPS), or (MPS, MPO, MPO, MPS)"
    )


def expect(W: MPO, psi: MPS):
    """Return <psi|W|psi>."""
    return inner(psi, W, psi)


def variance(H: MPO, psi: MPS):
    """Return <psi|H^2|psi>-<psi|H|psi>^2."""
    return inner(psi, H, H, psi) - inner(psi, H, psi)**2


########################################################################################
# MPO-MPS contract, zip_up, density_matrix, fitting, src, ...
########################################################################################

def apply(A: MPO, psi: MPS, method="src"):
    return None