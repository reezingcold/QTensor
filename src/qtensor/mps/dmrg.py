from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import time
import jax
import jax.numpy as jnp


from qtensor.mps.algebra import expect
from qtensor.mps.mpo import MPO
from qtensor.mps.mps import MPS, move_center
from qtensor.tensor.contract import contract
from qtensor.tensor.index import Index
from qtensor.tensor.linalg import tensor_qr, tensor_rq, truncated_svd
from qtensor.tensor.sparse import (
    EffectiveOneSiteHamiltonian,
    EffectiveTwoSiteHamiltonian,
    arnoldi_eigenpair,
    lanczos_lowest_eigenpair,
    scipy_lowest_eigs,
    scipy_lowest_eigsh,
)
from qtensor.tensor.tensor import Tensor


# Helper functions for bra site tensors
def _prime_link_inds_of_tensor(tensor: Tensor) -> Tensor:
    """
    Return a copy of `tensor` whose link indices are primed by +1.

    The rule used here is tag-based: any index whose tags contain the string
    "link" is treated as a virtual bond index and gets its `prime_level`
    incremented by one. All other indices are left unchanged.
    """
    new_inds = []
    for ind in tensor.inds:
        if "link" in ind.tags:
            new_inds.append(
                Index(
                    dim=ind.dim,
                    name=ind.name,
                    tags=ind.tags,
                    prime_level=ind.prime_level + 1,
                )
            )
        else:
            new_inds.append(ind)
    return Tensor(tensor.data, tuple(new_inds))


def _make_bra_site_tensor(tensor: Tensor) -> Tensor:
    """
    Construct the bra version of a single MPS site tensor by conjugating it and
    priming only its link indices.
    """
    primed = _prime_link_inds_of_tensor(tensor)
    return Tensor(jnp.conjugate(primed.data), primed.inds)


def _make_left_env_site_tensors(psi_site: Tensor, mpo_site: Tensor, env: Tensor) -> tuple[Tensor, Tensor]:
    """
    Build the bra/ket tensors used in a left-environment update in one shot.

    The returned tensors are aligned so that:
        - their left link indices match the open bra/ket links of `env`
        - their physical legs match the MPO site-in / site-out legs
    """
    left_bra_ind = env.inds[0]
    mpo_in_ind = mpo_site.inds[1]
    left_ket_ind = env.inds[2]
    right_ind = psi_site.inds[2]

    bra_n = Tensor(
        jnp.conjugate(psi_site.data),
        inds=(left_bra_ind, mpo_in_ind, Index(
            dim=right_ind.dim,
            name=right_ind.name,
            tags=right_ind.tags,
            prime_level=right_ind.prime_level + 1,
        )),
    )
    ket_n = Tensor(
        psi_site.data,
        inds=(left_ket_ind, mpo_site.inds[2], right_ind),
    )
    return bra_n, ket_n



def _make_right_env_site_tensors(psi_site: Tensor, mpo_site: Tensor, env: Tensor) -> tuple[Tensor, Tensor]:
    """
    Build the bra/ket tensors used in a right-environment update in one shot.

    The returned tensors are aligned so that:
        - their right link indices match the open bra/ket links of `env`
        - their physical legs match the MPO site-in / site-out legs
    """
    left_ind = psi_site.inds[0]
    right_bra_ind = env.inds[0]
    mpo_in_ind = mpo_site.inds[1]
    right_ket_ind = env.inds[2]

    bra_n = Tensor(
        jnp.conjugate(psi_site.data),
        inds=(Index(
            dim=left_ind.dim,
            name=left_ind.name,
            tags=left_ind.tags,
            prime_level=left_ind.prime_level + 1,
        ), mpo_in_ind, right_bra_ind),
    )
    ket_n = Tensor(
        psi_site.data,
        inds=(left_ind, mpo_site.inds[2], right_ket_ind),
    )
    return bra_n, ket_n


def _contract_left_environment_update(
    env: Tensor,
    bra_n: Tensor,
    mpo_n: Tensor,
    ket_n: Tensor,
) -> Tensor:
    data = jnp.einsum(
        "apb,aix,pijq,bjy->xqy",
        env.data,
        bra_n.data,
        mpo_n.data,
        ket_n.data,
        optimize="auto",
    )
    return Tensor(data, (bra_n.inds[2], mpo_n.inds[3], ket_n.inds[2]))


def _contract_right_environment_update(
    bra_n: Tensor,
    mpo_n: Tensor,
    ket_n: Tensor,
    env: Tensor,
) -> Tensor:
    data = jnp.einsum(
        "xia,pijq,yjb,aqb->xpy",
        bra_n.data,
        mpo_n.data,
        ket_n.data,
        env.data,
        optimize="auto",
    )
    return Tensor(data, (bra_n.inds[0], mpo_n.inds[0], ket_n.inds[0]))


@dataclass
class SweepStats:
    sweep: int
    direction: str
    energy: float | complex | None = None
    max_trunc_error: float | None = None
    max_bond_dim: int | None = None
    nsteps: int = 0


@dataclass
class StepStats:
    sweep: int
    bond: int
    direction: str
    energy: float | complex | None = None
    trunc_error: float | None = None
    kept_bond_dim: int | None = None


@dataclass
class ProfileStats:
    times: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, elapsed: float, count: int = 1) -> None:
        self.times[key] = self.times.get(key, 0.0) + elapsed
        self.counts[key] = self.counts.get(key, 0) + count

    def merge(self, other: "ProfileStats") -> None:
        for key, value in other.times.items():
            self.times[key] = self.times.get(key, 0.0) + value
        for key, value in other.counts.items():
            self.counts[key] = self.counts.get(key, 0) + value

    def as_dict(self, total_runtime: float | None = None) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        keys = set(self.times) | set(self.counts)
        for key in sorted(keys):
            total = self.times.get(key, 0.0)
            count = self.counts.get(key, 0)
            entry: dict[str, float | int] = {
                "total": total,
                "count": count,
                "avg": (total / count) if count else 0.0,
            }
            if total_runtime is not None and total_runtime > 0:
                entry["fraction"] = total / total_runtime
            out[key] = entry
        return out


@dataclass
class TwoSiteDMRGEngine:
    H: MPO
    psi: MPS
    maxdim: int
    cutoff: float = 0.0
    eigensolver: str = "lanczos"
    ishermitian: bool = True
    outputlevel: int = 0
    krylov_maxiter: int = 30
    krylov_tol: float = 1e-10
    krylov_max_restarts: int = 2
    store_step_history: bool = True
    store_sweep_history: bool = True
    profile: bool = False
    energy_tol: float | None = None
    energy: float | complex | None = None
    variance: float | None = None
    sweep: int = 0
    center: int | None = None
    dtype: Any | None = None
    left_envs: list[Any] = field(default_factory=list)
    right_envs: list[Any] = field(default_factory=list)
    left_env_valid_upto: int = 0
    right_env_valid_from: int = 0
    sweep_history: list[SweepStats] = field(default_factory=list)
    step_history: list[StepStats] = field(default_factory=list)
    profile_stats: ProfileStats = field(default_factory=ProfileStats)
    run_time_total: float = 0.0

    def __post_init__(self) -> None:
        if len(self.H) != len(self.psi):
            raise ValueError("H and psi must have the same number of sites")
        if self.maxdim <= 0:
            raise ValueError("maxdim must be positive")
        if self.cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        if self.krylov_maxiter <= 0:
            raise ValueError("krylov_maxiter must be positive")
        if self.krylov_tol < 0:
            raise ValueError("krylov_tol must be non-negative")
        if self.krylov_max_restarts < 0:
            raise ValueError("krylov_max_restarts must be non-negative")
        if self.energy_tol is not None and self.energy_tol < 0:
            raise ValueError("energy_tol must be non-negative or None")
        if self.eigensolver not in ("dense", "lanczos", "arnoldi", "scipy", "scipy_eigsh"):
            raise ValueError(
                "eigensolver must be one of 'dense', 'lanczos', 'arnoldi', 'scipy', or 'scipy_eigsh'"
            )

        self.dtype = jnp.result_type(self.psi.tensors[0].data, self.H.tensors[0].data)
        self.psi = move_center(self.psi, 0)
        self.center = self.psi.gauge_center
        self.initialize_environments()
        self.energy = expect(self.H, self.psi)

    def initialize_environments(self):
        """
        Environment convention:
            - left_envs[i]  stores the contraction of sites [0, ..., i-1]
            - right_envs[i] stores the contraction of sites [i, ..., N-1]

        This method is intended for one-time initialization. Afterwards, the
        cached environments are updated incrementally during sweeps.
        """
        nsites = len(self.psi)

        self.left_envs = [None] * (nsites + 1)
        self.right_envs = [None] * (nsites + 1)

        left_env0 = Tensor(
            jnp.ones((1, 1, 1), dtype=self.dtype),
            inds=(
                Index(
                    dim=self.psi[0].inds[0].dim,
                    name=self.psi[0].inds[0].name,
                    tags=self.psi[0].inds[0].tags,
                    prime_level=self.psi[0].inds[0].prime_level + 1,
                ),
                self.H.left_link_ind(0),
                self.psi.left_link_ind(0),
            ),
        )
        self.left_envs[0] = left_env0
        self.left_env_valid_upto = 0

        right_boundary = Tensor(
            jnp.ones((1, 1, 1), dtype=self.dtype),
            inds=(
                Index(
                    dim=self.psi[nsites - 1].inds[2].dim,
                    name=self.psi[nsites - 1].inds[2].name,
                    tags=self.psi[nsites - 1].inds[2].tags,
                    prime_level=self.psi[nsites - 1].inds[2].prime_level + 1,
                ),
                self.H.right_link_ind(nsites - 1),
                self.psi.right_link_ind(nsites - 1),
            ),
        )
        self.right_envs[nsites] = right_boundary
        self.right_env_valid_from = nsites

        env = right_boundary
        for n in range(nsites - 1, -1, -1):
            bra_n, ket_n = _make_right_env_site_tensors(self.psi[n], self.H[n], env)
            env = _contract_right_environment_update(bra_n, self.H[n], ket_n, env)
            self.right_envs[n] = env
        self.right_env_valid_from = 0

    def _block_until_ready(self, value) -> None:
        try:
            value.block_until_ready()
        except AttributeError:
            jax.block_until_ready(value)

    def _record_profile(self, key: str, t0: float, value=None, count: int = 1) -> None:
        if not self.profile:
            return
        if value is not None:
            self._block_until_ready(value)
        self.profile_stats.add(key, time.perf_counter() - t0, count=count)

    def _instrument_operator(self, op, key: str):
        if not self.profile:
            return op

        original_apply_data = op.apply_data

        def profiled_apply_data(x_data, _orig=original_apply_data):
            t0 = time.perf_counter()
            y_data = _orig(x_data)
            self._record_profile(key, t0, y_data)
            return y_data

        op.apply_data = profiled_apply_data
        return op

    def profile_summary(self) -> dict[str, dict[str, float | int]]:
        return self.profile_stats.as_dict(total_runtime=self.run_time_total)

    def _print_profile_summary(self) -> None:
        summary = self.profile_summary()
        if not summary:
            return
        print("    profile:")
        for key, entry in summary.items():
            frac = entry.get("fraction", 0.0)
            print(
                f"      {key}: total={entry['total']:.6f}s, "
                f"count={entry['count']}, avg={entry['avg']:.6f}s, frac={frac:.3f}"
            )

    def update_left_environment(self, site: int):
        nsites = len(self.psi)
        if not (0 <= site < nsites):
            raise ValueError("site must satisfy 0 <= site < len(psi)")
        if site > self.left_env_valid_upto or self.left_envs[site] is None:
            raise ValueError(
                f"left_envs[{site}] is not available for incremental update"
            )

        env = self.left_envs[site]
        bra_n, ket_n = _make_left_env_site_tensors(self.psi[site], self.H[site], env)
        t0 = time.perf_counter()
        self.left_envs[site + 1] = _contract_left_environment_update(
            env, bra_n, self.H[site], ket_n
        )
        self._record_profile("left_env_update", t0, self.left_envs[site + 1].data)
        self.left_env_valid_upto = max(self.left_env_valid_upto, site + 1)

    def update_right_environment(self, site: int):
        nsites = len(self.psi)
        if not (0 <= site < nsites):
            raise ValueError("site must satisfy 0 <= site < len(psi)")
        if site + 1 < self.right_env_valid_from or self.right_envs[site + 1] is None:
            raise ValueError(
                f"right_envs[{site + 1}] is not available for incremental update"
            )

        env = self.right_envs[site + 1]
        bra_n, ket_n = _make_right_env_site_tensors(self.psi[site], self.H[site], env)
        t0 = time.perf_counter()
        self.right_envs[site] = _contract_right_environment_update(
            bra_n, self.H[site], ket_n, env
        )
        self._record_profile("right_env_update", t0, self.right_envs[site].data)
        self.right_env_valid_from = min(self.right_env_valid_from, site)

    def prepare_left_to_right_sweep(self) -> None:
        # After initialization or a completed right-to-left sweep, the
        # orthogonality center should already be at site 0.
        if self.psi.gauge_center != 0:
            raise ValueError("Left-to-right sweep expected gauge center at site 0")
        if self.left_env_valid_upto < 0 or self.left_envs[0] is None:
            raise ValueError("left_envs[0] is not initialized")
        if self.right_env_valid_from > len(self.psi) or self.right_envs[-1] is None:
            raise ValueError("right boundary environment is not initialized")

    def prepare_right_to_left_sweep(self) -> None:
        # After a completed left-to-right sweep, the orthogonality center should
        # already be at the last site.
        if self.psi.gauge_center != len(self.psi) - 1:
            raise ValueError(
                "Right-to-left sweep expected gauge center at the last site"
            )
        if self.right_env_valid_from > len(self.psi) or self.right_envs[-1] is None:
            raise ValueError("right boundary environment is not initialized")

    def sweep_left_to_right(self) -> SweepStats:
        stats = SweepStats(sweep=self.sweep, direction="left_to_right")
        max_trunc = 0.0
        max_bond = 0
        self.prepare_left_to_right_sweep()

        for bond in range(len(self.psi) - 1):
            step = self.step_two_site(bond, direction="left_to_right")
            if self.store_step_history:
                self.step_history.append(step)
            stats.nsteps += 1
            if step.trunc_error is not None:
                max_trunc = max(max_trunc, float(jnp.real(step.trunc_error)))
            if step.kept_bond_dim is not None:
                max_bond = max(max_bond, int(step.kept_bond_dim))

        stats.energy = self.energy
        stats.max_trunc_error = max_trunc
        stats.max_bond_dim = max_bond
        if self.store_sweep_history:
            self.sweep_history.append(stats)
        return stats

    def sweep_right_to_left(self) -> SweepStats:
        stats = SweepStats(sweep=self.sweep, direction="right_to_left")
        max_trunc = 0.0
        max_bond = 0
        self.prepare_right_to_left_sweep()

        for bond in range(len(self.psi) - 2, -1, -1):
            step = self.step_two_site(bond, direction="right_to_left")
            if self.store_step_history:
                self.step_history.append(step)
            stats.nsteps += 1
            if step.trunc_error is not None:
                max_trunc = max(max_trunc, float(jnp.real(step.trunc_error)))
            if step.kept_bond_dim is not None:
                max_bond = max(max_bond, int(step.kept_bond_dim))

        stats.energy = self.energy
        stats.max_trunc_error = max_trunc
        stats.max_bond_dim = max_bond
        if self.store_sweep_history:
            self.sweep_history.append(stats)
        return stats

    def run(self, nsweeps: int = 2) -> MPS:
        if nsweeps < 0:
            raise ValueError("nsweeps must be non-negative")

        prev_energy = self.energy

        for _ in range(nsweeps):
            t0 = time.perf_counter()

            stats_lr = self.sweep_left_to_right()
            stats_rl = self.sweep_right_to_left()

            # Force JAX sync for accurate timing
            try:
                self.energy.block_until_ready()
            except AttributeError:
                jax.block_until_ready(self.energy)

            t1 = time.perf_counter()
            self.sweep += 1
            self.run_time_total += t1 - t0

            dE_val = None
            if prev_energy is not None:
                dE = jnp.abs(self.energy - prev_energy)
                dE_val = float(jnp.asarray(dE))

            if self.outputlevel >= 1:
                engine_name = self.__class__.__name__
                msg = (
                    f"[{engine_name} Sweep {self.sweep}] "
                    f"E={self.energy}, "
                    f"dE={dE_val if dE_val is not None else 'N/A'}, "
                    f"maxlinkdim={self.psi.maxlinkdim()}, "
                    f"time={t1 - t0:.6f} s"
                )
                print(msg)

            if self.outputlevel >= 2:
                print(
                    f"    bond_dims={self.psi.bond_dims()}\n"
                    f"    L->R: max_bond_dim={stats_lr.max_bond_dim}, "
                    f"max_trunc_error={stats_lr.max_trunc_error}, nsteps={stats_lr.nsteps}\n"
                    f"    R->L: max_bond_dim={stats_rl.max_bond_dim}, "
                    f"max_trunc_error={stats_rl.max_trunc_error}, nsteps={stats_rl.nsteps}"
                )
            if self.outputlevel >= 3 and self.profile:
                self._print_profile_summary()

            if self.energy_tol is not None and dE_val is not None:
                if dE_val < self.energy_tol:
                    if self.outputlevel >= 1:
                        print(
                            f"[DMRG] Early stop at sweep {self.sweep}: "
                            f"|ΔE|={dE_val:.3e} < energy_tol={self.energy_tol:.3e}"
                        )
                    break

            prev_energy = self.energy

        return self.psi

    def build_effective_two_site_hamiltonian(
        self,
        bond: int,
        domain_inds: tuple[Index, ...],
    ):
        if not (0 <= bond < len(self.psi) - 1):
            raise ValueError("bond must satisfy 0 <= bond < len(psi) - 1")
        if bond > self.left_env_valid_upto or self.left_envs[bond] is None:
            raise ValueError(f"left_envs[{bond}] is not available")
        if bond + 2 < self.right_env_valid_from or self.right_envs[bond + 2] is None:
            raise ValueError(f"right_envs[{bond + 2}] is not available")
        op = EffectiveTwoSiteHamiltonian(
            left_env=self.left_envs[bond],
            W1=self.H[bond],
            W2=self.H[bond + 1],
            right_env=self.right_envs[bond + 2],
            domain_inds=domain_inds,
            codomain_inds=domain_inds,
            metadata={"bond": bond},
        )
        return self._instrument_operator(op, "heff_two_site_apply")

    def local_solver(self, bond: int, Heff, theta: Tensor, direction: str):
        if direction not in ("left_to_right", "right_to_left"):
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

        solver = self.eigensolver
        if solver == "scipy_eigsh":
            solver = "scipy"

        if solver == "dense":
            solver_key = "lanczos" if self.ishermitian else "arnoldi"
        elif solver == "lanczos":
            solver_key = "lanczos"
        elif solver == "arnoldi":
            solver_key = "arnoldi"
        elif solver == "scipy":
            solver_key = "scipy_eigsh" if self.ishermitian else "scipy_eigs"
        else:
            solver_key = solver
        t_solver = time.perf_counter()
        if solver == "dense":
            if self.ishermitian:
                energy, theta_opt, _ = lanczos_lowest_eigenpair(
                    Heff,
                    theta,
                    maxiter=self.krylov_maxiter,
                    tol=self.krylov_tol,
                    max_restarts=self.krylov_max_restarts,
                )
            else:
                energy, theta_opt, _ = arnoldi_eigenpair(
                    Heff,
                    theta,
                    maxiter=self.krylov_maxiter,
                    tol=self.krylov_tol,
                    which="SR",
                    max_restarts=self.krylov_max_restarts,
                )
        elif solver == "lanczos":
            energy, theta_opt, _ = lanczos_lowest_eigenpair(
                Heff,
                theta,
                maxiter=self.krylov_maxiter,
                tol=self.krylov_tol,
                max_restarts=self.krylov_max_restarts,
            )
        elif solver == "arnoldi":
            energy, theta_opt, _ = arnoldi_eigenpair(
                Heff,
                theta,
                maxiter=self.krylov_maxiter,
                tol=self.krylov_tol,
                which="SR",
                max_restarts=self.krylov_max_restarts,
            )
        elif solver == "scipy":
            if self.ishermitian:
                energy, theta_opt, _ = scipy_lowest_eigsh(
                    Heff,
                    theta,
                    tol=self.krylov_tol,
                    maxiter=self.krylov_maxiter,
                )
            else:
                energy, theta_opt, _ = scipy_lowest_eigs(
                    Heff,
                    theta,
                    tol=self.krylov_tol,
                    maxiter=self.krylov_maxiter,
                )
        else:
            raise ValueError(
                "eigensolver must be one of 'dense', 'lanczos', 'arnoldi', or 'scipy'"
            )
        self._record_profile(solver_key, t_solver, energy)

        old_mid_link = self.psi.right_link_ind(bond)
        left_inds = (theta_opt.inds[0], theta_opt.inds[1])
        t_svd = time.perf_counter()
        U, S, Vh, info = truncated_svd(
            theta_opt,
            left_inds=left_inds,
            bond_name=f"dmrg-bond-{bond}",
            cutoff=self.cutoff,
            max_bond=self.maxdim,
        )
        self._record_profile("truncated_svd", t_svd, U.data)

        s = info["singular_values"]
        s_norm = jnp.linalg.norm(s)
        if s_norm > 0:
            s = s / s_norm

        new_mid_link = Index(
            dim=info["new_bond_dim"],
            name=old_mid_link.name,
            tags=old_mid_link.tags,
            prime_level=old_mid_link.prime_level,
        )

        if direction == "left_to_right":
            A_left = U.replace_ind(U.inds[-1], new_mid_link)
            A_right_data = s[:, None, None] * Vh.data
            A_right = Tensor(A_right_data, (new_mid_link, Vh.inds[1], Vh.inds[2]))
            new_center = bond + 1
        else:
            A_left_data = U.data * s[None, None, :]
            A_left = Tensor(A_left_data, (U.inds[0], U.inds[1], new_mid_link))
            A_right = Vh.replace_ind(Vh.inds[0], new_mid_link)
            new_center = bond

        tensors = list(self.psi.tensors)
        tensors[bond] = A_left
        tensors[bond + 1] = A_right
        self.psi = MPS(tuple(tensors), self.psi.site_types, gauge_center=new_center)

        trunc_error = info["discarded_weight"]
        kept_bond_dim = info["new_bond_dim"]
        return theta_opt, energy, trunc_error, kept_bond_dim

    def step_two_site(self, bond: int, direction: str) -> StepStats:
        nsites = len(self.psi)
        if not (0 <= bond < nsites - 1):
            raise ValueError("bond must satisfy 0 <= bond < len(psi) - 1")
        if direction not in ("left_to_right", "right_to_left"):
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

        if bond > self.left_env_valid_upto or self.left_envs[bond] is None:
            raise ValueError(f"left_envs[{bond}] is not available")
        if bond + 2 < self.right_env_valid_from or self.right_envs[bond + 2] is None:
            raise ValueError(f"right_envs[{bond + 2}] is not available")

        t_step = time.perf_counter()
        theta = contract(self.psi[bond], self.psi[bond + 1])
        Heff = self.build_effective_two_site_hamiltonian(bond, theta.inds)
        _, energy, trunc_error, kept_bond_dim = self.local_solver(bond, Heff, theta, direction)
        self.energy = energy
        # Keep the engine center metadata explicitly synchronized.
        self.center = self.psi.gauge_center

        if direction == "left_to_right":
            self.left_env_valid_upto = min(self.left_env_valid_upto, bond)
            self.right_env_valid_from = max(self.right_env_valid_from, bond + 2)
            self.update_left_environment(bond)
        else:
            self.left_env_valid_upto = min(self.left_env_valid_upto, bond)
            self.right_env_valid_from = max(self.right_env_valid_from, bond + 2)
            self.update_right_environment(bond + 1)
        self._record_profile("two_site_step", t_step, self.energy)

        return StepStats(
            sweep=self.sweep,
            bond=bond,
            direction=direction,
            energy=energy,
            trunc_error=trunc_error,
            kept_bond_dim=kept_bond_dim,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "energy": self.energy,
            "variance": self.variance,
            "maxdim": self.maxdim,
            "cutoff": self.cutoff,
            "center": self.center,
            "dtype": str(self.dtype),
            "sweep": self.sweep,
            "n_left_envs": self.left_env_valid_upto + 1,
            "n_right_envs": len(self.right_envs) - self.right_env_valid_from,
            "n_step_records": len(self.step_history),
            "n_sweep_records": len(self.sweep_history),
            "gauge_center": self.psi.gauge_center,
            "profile": self.profile_summary(),
            "run_time_total": self.run_time_total,
        }

@dataclass
class OneSiteDMRGEngine(TwoSiteDMRGEngine):
    """
    One-site finite DMRG engine.

    This engine reuses the same environment convention and incremental
    environment-update logic as `TwoSiteDMRGEngine`, so it can continue from a
    converged two-site run without rebuilding the environment structure from
    scratch.
    """

    @classmethod
    def from_two_site_engine(cls, engine: TwoSiteDMRGEngine) -> "OneSiteDMRGEngine":
        out = cls(
            H=engine.H,
            psi=engine.psi.copy(),
            maxdim=engine.maxdim,
            cutoff=engine.cutoff,
            eigensolver=engine.eigensolver,
            ishermitian=engine.ishermitian,
            outputlevel=engine.outputlevel,
            krylov_maxiter=engine.krylov_maxiter,
            krylov_tol=engine.krylov_tol,
            krylov_max_restarts=engine.krylov_max_restarts,
            store_step_history=engine.store_step_history,
            store_sweep_history=engine.store_sweep_history,
            energy_tol=engine.energy_tol,
        )
        out.energy = engine.energy
        out.variance = engine.variance
        out.sweep = engine.sweep
        out.center = engine.center
        out.dtype = engine.dtype
        out.left_envs = list(engine.left_envs)
        out.right_envs = list(engine.right_envs)
        out.left_env_valid_upto = engine.left_env_valid_upto
        out.right_env_valid_from = engine.right_env_valid_from
        out.sweep_history = list(engine.sweep_history)
        out.step_history = list(engine.step_history)
        out.profile = engine.profile
        out.profile_stats = ProfileStats(
            times=dict(engine.profile_stats.times),
            counts=dict(engine.profile_stats.counts),
        )
        out.run_time_total = engine.run_time_total
        return out

    def build_effective_one_site_hamiltonian(self, site: int):
        if not (0 <= site < len(self.psi)):
            raise ValueError("site must satisfy 0 <= site < len(psi)")
        if site > self.left_env_valid_upto or self.left_envs[site] is None:
            raise ValueError(f"left_envs[{site}] is not available")
        if site + 1 < self.right_env_valid_from or self.right_envs[site + 1] is None:
            raise ValueError(f"right_envs[{site + 1}] is not available")

        A = self.psi[site]
        op = EffectiveOneSiteHamiltonian(
            left_env=self.left_envs[site],
            W=self.H[site],
            right_env=self.right_envs[site + 1],
            domain_inds=A.inds,
            codomain_inds=A.inds,
            metadata={"site": site},
        )
        return self._instrument_operator(op, "heff_one_site_apply")

    def local_solver(self, site: int, Heff, A: Tensor, direction: str):
        if direction not in ("left_to_right", "right_to_left"):
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

        solver = self.eigensolver
        if solver == "scipy_eigsh":
            solver = "scipy"

        if solver == "dense":
            solver_key = "lanczos" if self.ishermitian else "arnoldi"
        elif solver == "lanczos":
            solver_key = "lanczos"
        elif solver == "arnoldi":
            solver_key = "arnoldi"
        elif solver == "scipy":
            solver_key = "scipy_eigsh" if self.ishermitian else "scipy_eigs"
        else:
            solver_key = solver
        t_solver = time.perf_counter()
        if solver == "dense":
            if self.ishermitian:
                energy, A_opt, _ = lanczos_lowest_eigenpair(
                    Heff,
                    A,
                    maxiter=self.krylov_maxiter,
                    tol=self.krylov_tol,
                    max_restarts=self.krylov_max_restarts,
                )
            else:
                energy, A_opt, _ = arnoldi_eigenpair(
                    Heff,
                    A,
                    maxiter=self.krylov_maxiter,
                    tol=self.krylov_tol,
                    which="SR",
                    max_restarts=self.krylov_max_restarts,
                )
        elif solver == "lanczos":
            energy, A_opt, _ = lanczos_lowest_eigenpair(
                Heff,
                A,
                maxiter=self.krylov_maxiter,
                tol=self.krylov_tol,
                max_restarts=self.krylov_max_restarts,
            )
        elif solver == "arnoldi":
            energy, A_opt, _ = arnoldi_eigenpair(
                Heff,
                A,
                maxiter=self.krylov_maxiter,
                tol=self.krylov_tol,
                which="SR",
                max_restarts=self.krylov_max_restarts,
            )
        elif solver == "scipy":
            if self.ishermitian:
                energy, A_opt, _ = scipy_lowest_eigsh(
                    Heff,
                    A,
                    tol=self.krylov_tol,
                    maxiter=self.krylov_maxiter,
                )
            else:
                energy, A_opt, _ = scipy_lowest_eigs(
                    Heff,
                    A,
                    tol=self.krylov_tol,
                    maxiter=self.krylov_maxiter,
                )
        else:
            raise ValueError(
                "eigensolver must be one of 'dense', 'lanczos', 'arnoldi', or 'scipy'"
            )
        self._record_profile(solver_key, t_solver, energy)

        # Normalize the optimized one-site tensor before passing the gauge to a
        # neighboring site.
        # A_norm = jnp.linalg.norm(A_opt.data)
        # if A_norm > 0:
        #     A_opt = Tensor(A_opt.data / A_norm, A_opt.inds)

        # H_A = Heff(A_opt)
        # energy = jnp.vdot(A_opt.data.reshape(-1), H_A.data.reshape(-1))
        # energy = jnp.real(energy) if jnp.iscomplexobj(energy) else energy

        tensors = list(self.psi.tensors)

        if direction == "left_to_right":
            if site == len(self.psi) - 1:
                tensors[site] = A_opt
                new_center = site
                kept_bond_dim = int(self.psi.right_link_ind(site).dim)
            else:
                old_mid_link = self.psi.right_link_ind(site)
                Q, R = tensor_qr(
                    A_opt,
                    left_inds=(A_opt.inds[0], A_opt.inds[1]),
                    bond_name=f"dmrg1-bond-{site}",
                )
                new_mid_link = Index(
                    dim=Q.inds[-1].dim,
                    name=old_mid_link.name,
                    tags=old_mid_link.tags,
                    prime_level=old_mid_link.prime_level,
                )
                A_site = Q.replace_ind(Q.inds[-1], new_mid_link)
                R = R.replace_ind(R.inds[0], new_mid_link)
                A_next = contract(R, self.psi[site + 1]).replace_ind(R.inds[0], new_mid_link)
                tensors[site] = A_site
                tensors[site + 1] = A_next
                new_center = site + 1
                kept_bond_dim = int(new_mid_link.dim)
        else:
            if site == 0:
                tensors[site] = A_opt
                new_center = site
                kept_bond_dim = int(self.psi.left_link_ind(site).dim)
            else:
                old_left_link = self.psi.left_link_ind(site)
                R, Q = tensor_rq(
                    A_opt,
                    left_inds=(A_opt.inds[0],),
                    right_inds=(A_opt.inds[1], A_opt.inds[2]),
                    bond_name=f"dmrg1-bond-{site-1}",
                )
                new_mid_link = Index(
                    dim=Q.inds[0].dim,
                    name=old_left_link.name,
                    tags=old_left_link.tags,
                    prime_level=old_left_link.prime_level,
                )
                R = R.replace_ind(R.inds[-1], new_mid_link)
                A_site = Q.replace_ind(Q.inds[0], new_mid_link)
                A_prev = contract(self.psi[site - 1], R).replace_ind(R.inds[-1], new_mid_link)
                tensors[site - 1] = A_prev
                tensors[site] = A_site
                new_center = site - 1
                kept_bond_dim = int(new_mid_link.dim)

        self.psi = MPS(tuple(tensors), self.psi.site_types, gauge_center=new_center)
        trunc_error = 0.0
        return A_opt, energy, trunc_error, kept_bond_dim

    def step_one_site(self, site: int, direction: str) -> StepStats:
        if not (0 <= site < len(self.psi)):
            raise ValueError("site must satisfy 0 <= site < len(psi)")
        if direction not in ("left_to_right", "right_to_left"):
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")
        if site > self.left_env_valid_upto or self.left_envs[site] is None:
            raise ValueError(f"left_envs[{site}] is not available")
        if site + 1 < self.right_env_valid_from or self.right_envs[site + 1] is None:
            raise ValueError(f"right_envs[{site + 1}] is not available")

        t_step = time.perf_counter()
        Heff = self.build_effective_one_site_hamiltonian(site)
        A = self.psi[site]
        _, energy, trunc_error, kept_bond_dim = self.local_solver(
            site, Heff, A, direction
        )
        self.energy = energy
        self.center = self.psi.gauge_center

        if direction == "left_to_right":
            self.left_env_valid_upto = min(self.left_env_valid_upto, site)
            next_valid_right = site + 1 if site == len(self.psi) - 1 else site + 2
            self.right_env_valid_from = max(self.right_env_valid_from, next_valid_right)
            self.update_left_environment(site)
        else:
            left_upto = 0 if site == 0 else site - 1
            self.left_env_valid_upto = min(self.left_env_valid_upto, left_upto)
            self.right_env_valid_from = max(self.right_env_valid_from, site + 1)
            self.update_right_environment(site)
        self._record_profile("one_site_step", t_step, self.energy)

        return StepStats(
            sweep=self.sweep,
            bond=site,
            direction=direction,
            energy=energy,
            trunc_error=trunc_error,
            kept_bond_dim=kept_bond_dim,
        )

    def sweep_left_to_right(self) -> SweepStats:
        stats = SweepStats(sweep=self.sweep, direction="left_to_right")
        max_trunc = 0.0
        max_bond = 0
        self.prepare_left_to_right_sweep()

        for site in range(len(self.psi) - 1):
            step = self.step_one_site(site, direction="left_to_right")
            if self.store_step_history:
                self.step_history.append(step)
            stats.nsteps += 1
            if step.trunc_error is not None:
                max_trunc = max(max_trunc, float(jnp.real(step.trunc_error)))
            if step.kept_bond_dim is not None:
                max_bond = max(max_bond, int(step.kept_bond_dim))

        last = len(self.psi) - 1
        step = self.step_one_site(last, direction="left_to_right")
        if self.store_step_history:
            self.step_history.append(step)
        stats.nsteps += 1
        if step.trunc_error is not None:
            max_trunc = max(max_trunc, float(jnp.real(step.trunc_error)))
        if step.kept_bond_dim is not None:
            max_bond = max(max_bond, int(step.kept_bond_dim))

        stats.energy = self.energy
        stats.max_trunc_error = max_trunc
        stats.max_bond_dim = max_bond
        if self.store_sweep_history:
            self.sweep_history.append(stats)
        return stats

    def sweep_right_to_left(self) -> SweepStats:
        stats = SweepStats(sweep=self.sweep, direction="right_to_left")
        max_trunc = 0.0
        max_bond = 0
        self.prepare_right_to_left_sweep()

        for site in range(len(self.psi) - 1, 0, -1):
            step = self.step_one_site(site, direction="right_to_left")
            if self.store_step_history:
                self.step_history.append(step)
            stats.nsteps += 1
            if step.trunc_error is not None:
                max_trunc = max(max_trunc, float(jnp.real(step.trunc_error)))
            if step.kept_bond_dim is not None:
                max_bond = max(max_bond, int(step.kept_bond_dim))

        step = self.step_one_site(0, direction="right_to_left")
        if self.store_step_history:
            self.step_history.append(step)
        stats.nsteps += 1
        if step.trunc_error is not None:
            max_trunc = max(max_trunc, float(jnp.real(step.trunc_error)))
        if step.kept_bond_dim is not None:
            max_bond = max(max_bond, int(step.kept_bond_dim))

        stats.energy = self.energy
        stats.max_trunc_error = max_trunc
        stats.max_bond_dim = max_bond
        if self.store_sweep_history:
            self.sweep_history.append(stats)
        return stats





# high-level interface
def dmrg(
    H: MPO,
    psi: MPS,
    nsweeps_two_site: int = 2,
    nsweeps_one_site: int = 2,
    maxdim: int = 32,
    cutoff: float = 1e-10,
    eigensolver: str = "lanczos",
    ishermitian: bool = True,
    outputlevel: int = 0,
    krylov_maxiter: int = 10,
    krylov_tol: float = 1e-10,
    krylov_max_restarts: int = 2,
    krylov_maxiter_one_site: int = 5,
    krylov_tol_one_site: float = 1e-10,
    krylov_max_restarts_one_site: int = 2,
    store_step_history: bool = False,
    store_sweep_history: bool = False,
    profile: bool = False,
    energy_tol: float | None = None,
):
    """
    High-level DMRG driver that first runs two-site DMRG and then switches to
    one-site DMRG for final refinement.

    Parameters
    ----------
    H : MPO
        Hamiltonian
    psi : MPS
        Initial state (will be copied internally)
    nsweeps_two_site : int
        Number of sweeps for two-site DMRG
    nsweeps_one_site : int
        Number of sweeps for one-site DMRG
    maxdim : int
        Maximum bond dimension
    cutoff : float
        Truncation cutoff
    eigensolver : str
        'lanczos', 'arnoldi', 'dense', or 'scipy'
    ishermitian : bool
        If True, Hermitian solvers are used (`lanczos`/`eigsh`).
        If False, non-Hermitian solvers are used (`arnoldi`/`eigs`).
    outputlevel : int
        Verbosity level
    energy_tol : float | None
        If not None, stop early once the absolute change in energy between two
        consecutive full sweeps is smaller than this threshold.
    krylov_maxiter_one_site : int | None
        If provided, override the one-site DMRG Krylov maxiter setting.
    krylov_tol_one_site : float | None
        If provided, override the one-site DMRG Krylov tolerance.
    krylov_max_restarts_one_site : int | None
        If provided, override the one-site DMRG Krylov max restart count.

    Returns
    -------
    psi : MPS
        Optimized MPS
    energy : float
        Final ground state energy
    """

    # -------------------------------
    # Phase 1: Two-site DMRG
    # -------------------------------
    engine2 = TwoSiteDMRGEngine(
        H=H,
        psi=psi.copy(),
        maxdim=maxdim,
        cutoff=cutoff,
        eigensolver=eigensolver,
        ishermitian=ishermitian,
        outputlevel=outputlevel,
        krylov_maxiter=krylov_maxiter,
        krylov_tol=krylov_tol,
        krylov_max_restarts=krylov_max_restarts,
        store_step_history=store_step_history,
        store_sweep_history=store_sweep_history,
        profile=profile,
        energy_tol=energy_tol,
    )

    psi = engine2.run(nsweeps=nsweeps_two_site)

    if outputlevel >= 1:
        print("[DMRG] After two-site phase: E =", float(engine2.energy))

    if krylov_maxiter_one_site is None:
        krylov_maxiter_one_site = krylov_maxiter
    if krylov_tol_one_site is None:
        krylov_tol_one_site = krylov_tol
    if krylov_max_restarts_one_site is None:
        krylov_max_restarts_one_site = krylov_max_restarts

    # -------------------------------
    # Phase 2: One-site DMRG
    # -------------------------------
    engine1 = OneSiteDMRGEngine.from_two_site_engine(engine2)
    engine1.krylov_maxiter = krylov_maxiter_one_site
    engine1.krylov_tol = krylov_tol_one_site
    engine1.krylov_max_restarts = krylov_max_restarts_one_site
    engine1.profile = profile

    psi = engine1.run(nsweeps=nsweeps_one_site)

    if outputlevel >= 1:
        print("[DMRG] After one-site phase: E =", float(engine1.energy))

    return psi, engine1.energy
