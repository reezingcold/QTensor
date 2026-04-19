from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import time

import jax
import jax.numpy as jnp

from qtensor.mps.mpo import MPO
from qtensor.mps.mps import MPS, move_center
from qtensor.tensor.contract import contract
from qtensor.tensor.index import Index
from qtensor.tensor.linalg import tensor_qr, tensor_rq, truncated_svd
from qtensor.tensor.sparse import (
    EffectiveOneSiteHamiltonian,
    EffectiveTwoSiteHamiltonian,
    EffectiveZeroSiteHamiltonian,
    expm_multiply_operator,
)
from qtensor.tensor.tensor import Tensor


def _primed(ind: Index) -> Index:
    return Index(
        dim=ind.dim,
        name=ind.name,
        tags=ind.tags,
        prime_level=ind.prime_level + 1,
    )


def _copied_bond(ind: Index, suffix: str) -> Index:
    return Index(
        dim=ind.dim,
        name=f"{ind.name}_{suffix}",
        tags=ind.tags,
        prime_level=ind.prime_level,
    )


def _make_left_env_site_tensors(
    psi_site: Tensor, mpo_site: Tensor, env: Tensor
) -> tuple[Tensor, Tensor]:
    left_bra_ind = env.inds[0]
    mpo_in_ind = mpo_site.inds[1]
    left_ket_ind = env.inds[2]
    right_ind = psi_site.inds[2]

    bra_n = Tensor(
        jnp.conjugate(psi_site.data),
        inds=(left_bra_ind, mpo_in_ind, _primed(right_ind)),
    )
    ket_n = Tensor(
        psi_site.data,
        inds=(left_ket_ind, mpo_site.inds[2], right_ind),
    )
    return bra_n, ket_n


def _make_right_env_site_tensors(
    psi_site: Tensor, mpo_site: Tensor, env: Tensor
) -> tuple[Tensor, Tensor]:
    left_ind = psi_site.inds[0]
    right_bra_ind = env.inds[0]
    mpo_in_ind = mpo_site.inds[1]
    right_ket_ind = env.inds[2]

    bra_n = Tensor(
        jnp.conjugate(psi_site.data),
        inds=(_primed(left_ind), mpo_in_ind, right_bra_ind),
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
class TDVPStepStats:
    sweep: int
    site_or_bond: int
    direction: str
    time_step: complex
    norm: float | complex | None = None
    elapsed: float | None = None


@dataclass
class TDVPSweepStats:
    sweep: int
    direction: str
    time_step: complex
    nsteps: int = 0
    elapsed: float | None = None
    norm: float | complex | None = None


@dataclass
class TDVPEngine:
    H: MPO | Callable[[complex], MPO]
    psi: MPS
    dt: complex
    normalize: bool = False
    expm_backend: str = "krylov"
    expm_krylov_maxiter: int = 20
    expm_krylov_tol: float = 1e-12
    outputlevel: int = 0
    sweep: int = 0
    center: int | None = None
    evolved_time: complex = 0.0
    dtype: Any | None = None
    left_envs: list[Any] = field(default_factory=list)
    right_envs: list[Any] = field(default_factory=list)
    left_env_valid_upto: int = 0
    right_env_valid_from: int = 0
    sweep_history: list[TDVPSweepStats] = field(default_factory=list)
    step_history: list[TDVPStepStats] = field(default_factory=list)
    observer: Callable[..., Any] | None = None
    hamiltonian_provider: Callable[[complex], MPO] | None = field(init=False, default=None)
    time_dependent_hamiltonian: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.expm_backend not in ("scipy", "krylov"):
            raise ValueError("expm_backend must be 'scipy' or 'krylov'")
        if self.expm_krylov_maxiter <= 0:
            raise ValueError("expm_krylov_maxiter must be positive")
        if self.expm_krylov_tol < 0:
            raise ValueError("expm_krylov_tol must be non-negative")

        if callable(self.H):
            self.hamiltonian_provider = self.H
            self.time_dependent_hamiltonian = True
            self.H = self.hamiltonian_provider(self.evolved_time)
        else:
            self.hamiltonian_provider = None
            self.time_dependent_hamiltonian = False

        self._validate_hamiltonian(self.H)
        self.dtype = jnp.result_type(self.psi[0].data, self.H[0].data)
        self.psi = move_center(self.psi, 0)
        self.center = self.psi.gauge_center
        self.initialize_environments()

    def _validate_hamiltonian(self, H: MPO) -> None:
        if len(H) != len(self.psi):
            raise ValueError("H and psi must have the same number of sites")

    def observe(self, event: str, **payload) -> None:
        if self.observer is None:
            return
        self.observer(
            event=event,
            engine=self,
            psi=self.psi,
            H=self.H,
            time=self.evolved_time,
            sweep=self.sweep,
            **payload,
        )

    def refresh_hamiltonian(self, time_point: complex | None = None) -> None:
        if self.hamiltonian_provider is None:
            return
        if time_point is None:
            time_point = self.evolved_time
        H_new = self.hamiltonian_provider(time_point)
        self._validate_hamiltonian(H_new)
        self.H = H_new
        self.dtype = jnp.result_type(self.psi[0].data, self.H[0].data)
        self.initialize_environments()

    @property
    def half_dt(self):
        return self.dt / 2

    def _block_until_ready(self, value) -> None:
        try:
            value.block_until_ready()
        except AttributeError:
            jax.block_until_ready(value)

    def _set_site(self, site: int, tensor: Tensor, gauge_center: int) -> None:
        tensors = list(self.psi.tensors)
        tensors[site] = tensor
        self.psi = MPS(tuple(tensors), self.psi.site_types, gauge_center=gauge_center)
        self.center = gauge_center

    def _set_two_sites(
        self,
        site1: int,
        tensor1: Tensor,
        site2: int,
        tensor2: Tensor,
        gauge_center: int,
    ) -> None:
        tensors = list(self.psi.tensors)
        tensors[site1] = tensor1
        tensors[site2] = tensor2
        self.psi = MPS(tuple(tensors), self.psi.site_types, gauge_center=gauge_center)
        self.center = gauge_center

    def initialize_environments(self) -> None:
        nsites = len(self.psi)
        self.left_envs = [None] * (nsites + 1)
        self.right_envs = [None] * (nsites + 1)

        self.left_envs[0] = Tensor(
            jnp.ones((1, 1, 1), dtype=self.dtype),
            inds=(
                _primed(self.psi[0].inds[0]),
                self.H.left_link_ind(0),
                self.psi.left_link_ind(0),
            ),
        )
        self.left_env_valid_upto = 0

        self.right_envs[nsites] = Tensor(
            jnp.ones((1, 1, 1), dtype=self.dtype),
            inds=(
                _primed(self.psi[-1].inds[2]),
                self.H.right_link_ind(nsites - 1),
                self.psi.right_link_ind(nsites - 1),
            ),
        )
        self.right_env_valid_from = nsites

        env = self.right_envs[nsites]
        for site in range(nsites - 1, -1, -1):
            bra_n, ket_n = _make_right_env_site_tensors(self.psi[site], self.H[site], env)
            env = _contract_right_environment_update(bra_n, self.H[site], ket_n, env)
            self.right_envs[site] = env
        self.right_env_valid_from = 0

    def update_left_environment(self, site: int) -> None:
        if site > self.left_env_valid_upto or self.left_envs[site] is None:
            raise ValueError(f"left_envs[{site}] is not available")
        env = self.left_envs[site]
        bra_n, ket_n = _make_left_env_site_tensors(self.psi[site], self.H[site], env)
        self.left_envs[site + 1] = _contract_left_environment_update(
            env, bra_n, self.H[site], ket_n
        )
        self.left_env_valid_upto = max(self.left_env_valid_upto, site + 1)

    def update_right_environment(self, site: int) -> None:
        if site + 1 < self.right_env_valid_from or self.right_envs[site + 1] is None:
            raise ValueError(f"right_envs[{site + 1}] is not available")
        env = self.right_envs[site + 1]
        bra_n, ket_n = _make_right_env_site_tensors(self.psi[site], self.H[site], env)
        self.right_envs[site] = _contract_right_environment_update(
            bra_n, self.H[site], ket_n, env
        )
        self.right_env_valid_from = min(self.right_env_valid_from, site)

    def invalidate_from_site_change(self, site: int) -> None:
        self.left_env_valid_upto = min(self.left_env_valid_upto, site)
        self.right_env_valid_from = max(self.right_env_valid_from, site + 1)

    def invalidate_from_bond_change(self, bond: int) -> None:
        self.left_env_valid_upto = min(self.left_env_valid_upto, bond)
        self.right_env_valid_from = max(self.right_env_valid_from, bond + 2)

    def prepare_left_to_right_sweep(self) -> None:
        if self.psi.gauge_center != 0:
            raise ValueError("Left-to-right sweep expected gauge center at site 0")

    def prepare_right_to_left_sweep(self) -> None:
        if self.psi.gauge_center != len(self.psi) - 1:
            raise ValueError("Right-to-left sweep expected gauge center at the last site")

    def build_effective_one_site_hamiltonian(self, site: int) -> EffectiveOneSiteHamiltonian:
        if not (0 <= site < len(self.psi)):
            raise ValueError("site must satisfy 0 <= site < len(psi)")
        if site > self.left_env_valid_upto or self.left_envs[site] is None:
            raise ValueError(f"left_envs[{site}] is not available")
        if site + 1 < self.right_env_valid_from or self.right_envs[site + 1] is None:
            raise ValueError(f"right_envs[{site + 1}] is not available")

        A = self.psi[site]
        return EffectiveOneSiteHamiltonian(
            left_env=self.left_envs[site],
            W=self.H[site],
            right_env=self.right_envs[site + 1],
            domain_inds=A.inds,
            codomain_inds=A.inds,
            metadata={"site": site},
        )

    def build_effective_zero_site_hamiltonian(
        self,
        bond: int,
        domain_inds: tuple[Index, ...],
    ) -> EffectiveZeroSiteHamiltonian:
        if not (0 <= bond < len(self.psi) - 1):
            raise ValueError("bond must satisfy 0 <= bond < len(psi) - 1")
        if bond + 1 > self.left_env_valid_upto or self.left_envs[bond + 1] is None:
            raise ValueError(f"left_envs[{bond + 1}] is not available")
        if bond + 1 < self.right_env_valid_from or self.right_envs[bond + 1] is None:
            raise ValueError(f"right_envs[{bond + 1}] is not available")

        return EffectiveZeroSiteHamiltonian(
            left_env=self.left_envs[bond + 1],
            right_env=self.right_envs[bond + 1],
            domain_inds=domain_inds,
            codomain_inds=domain_inds,
            metadata={"bond": bond},
        )

    def evolve_one_site(self, site: int, dt: complex) -> None:
        Heff = self.build_effective_one_site_hamiltonian(site)
        A = expm_multiply_operator(
            Heff,
            self.psi[site],
            dt=dt,
            normalize=False,
            method=self.expm_backend,
            maxiter=self.expm_krylov_maxiter,
            tol=self.expm_krylov_tol,
        )
        self._set_site(site, A, gauge_center=site)
        self.invalidate_from_site_change(site)

    def evolve_zero_site(self, bond: int, C: Tensor, dt: complex) -> Tensor:
        K = self.build_effective_zero_site_hamiltonian(bond, C.inds)
        return expm_multiply_operator(
            K,
            C,
            dt=dt,
            normalize=False,
            method=self.expm_backend,
            maxiter=self.expm_krylov_maxiter,
            tol=self.expm_krylov_tol,
        )

    def maybe_normalize_left_boundary(self) -> None:
        if not self.normalize:
            return
        nrm = jnp.linalg.norm(self.psi[0].data)
        if nrm > 0:
            self._set_site(0, Tensor(self.psi[0].data / nrm, self.psi[0].inds), gauge_center=0)
            self.invalidate_from_site_change(0)

    def environment_cache(self) -> dict[str, Any]:
        return {
            "left_envs": list(self.left_envs),
            "right_envs": list(self.right_envs),
            "left_env_valid_upto": self.left_env_valid_upto,
            "right_env_valid_from": self.right_env_valid_from,
            "center": self.center,
            "gauge_center": self.psi.gauge_center,
            "evolved_time": self.evolved_time,
            "time_dependent_hamiltonian": self.time_dependent_hamiltonian,
        }

    def load_environment_cache(self, cache: dict[str, Any]) -> None:
        self.left_envs = list(cache["left_envs"])
        self.right_envs = list(cache["right_envs"])
        self.left_env_valid_upto = int(cache["left_env_valid_upto"])
        self.right_env_valid_from = int(cache["right_env_valid_from"])
        self.center = cache.get("center", self.center)
        self.evolved_time = cache.get("evolved_time", self.evolved_time)

    def summary(self) -> dict[str, Any]:
        return {
            "dt": self.dt,
            "normalize": self.normalize,
            "expm_backend": self.expm_backend,
            "expm_krylov_maxiter": self.expm_krylov_maxiter,
            "expm_krylov_tol": self.expm_krylov_tol,
            "time_dependent_hamiltonian": self.time_dependent_hamiltonian,
            "center": self.center,
            "gauge_center": self.psi.gauge_center,
            "evolved_time": self.evolved_time,
            "dtype": str(self.dtype),
            "sweep": self.sweep,
            "n_left_envs": self.left_env_valid_upto + 1,
            "n_right_envs": len(self.right_envs) - self.right_env_valid_from,
            "n_step_records": len(self.step_history),
            "n_sweep_records": len(self.sweep_history),
        }

    def run(self, nsweeps: int = 1) -> MPS:
        if nsweeps < 0:
            raise ValueError("nsweeps must be non-negative")

        for _ in range(nsweeps):
            self.refresh_hamiltonian(self.evolved_time)
            self.observe("time_step_start")
            t0 = time.perf_counter()
            self.sweep_left_to_right()
            self.sweep_right_to_left()
            self._block_until_ready(self.psi[self.psi.gauge_center].data)
            t1 = time.perf_counter()
            self.sweep += 1
            self.evolved_time += self.dt
            self.observe("time_step_end")

            if self.outputlevel >= 1:
                print(
                    f"[{self.__class__.__name__} sweep {self.sweep}] "
                    f"dt={self.dt}, evolved_time={self.evolved_time:.3f}, "
                    f"elapsed={t1 - t0:.6f} s"
                )

        return self.psi

    def sweep_left_to_right(self) -> TDVPSweepStats:
        raise NotImplementedError

    def sweep_right_to_left(self) -> TDVPSweepStats:
        raise NotImplementedError


@dataclass
class OneSiteTDVPEngine(TDVPEngine):
    @classmethod
    def from_two_site_engine(cls, engine: "TwoSiteTDVPEngine") -> "OneSiteTDVPEngine":
        out = cls(
            H=engine.H,
            psi=engine.psi.copy(),
            dt=engine.dt,
            normalize=engine.normalize,
            expm_backend=engine.expm_backend,
            expm_krylov_maxiter=engine.expm_krylov_maxiter,
            expm_krylov_tol=engine.expm_krylov_tol,
            outputlevel=engine.outputlevel,
            sweep=engine.sweep,
            center=engine.center,
            evolved_time=engine.evolved_time,
            observer=engine.observer,
        )
        out.psi = engine.psi.copy()
        out.center = engine.center
        out.hamiltonian_provider = engine.hamiltonian_provider
        out.time_dependent_hamiltonian = engine.time_dependent_hamiltonian
        out.load_environment_cache(engine.environment_cache())
        return out

    def _factor_left_center(self, site: int) -> Tensor:
        A = self.psi[site]
        old_right = A.inds[2]
        Q, R = tensor_qr(A, left_inds=(A.inds[0], A.inds[1]), bond_name=f"tdvp-qr-{site}")
        if Q.inds[2].dim != old_right.dim:
            raise ValueError("One-site TDVP QR changed the bond dimension unexpectedly")

        A_left = Q.replace_ind(Q.inds[2], old_right)
        self._set_site(site, A_left, gauge_center=site)
        self.invalidate_from_site_change(site)

        c_left = _copied_bond(old_right, f"cL{site}")
        return R.replace_ind(R.inds[0], c_left)

    def _absorb_center_right(self, site: int, C: Tensor) -> None:
        old_right = self.psi.right_link_ind(site)
        next_tensor = contract(C, self.psi[site + 1]).replace_ind(C.inds[0], old_right)
        self._set_site(site + 1, next_tensor, gauge_center=site + 1)
        self.invalidate_from_site_change(site + 1)

    def _factor_right_center(self, site: int) -> Tensor:
        A = self.psi[site]
        old_left = A.inds[0]
        R, Q = tensor_rq(A, left_inds=(old_left,), bond_name=f"tdvp-rq-{site}")
        if Q.inds[0].dim != old_left.dim:
            raise ValueError("One-site TDVP RQ changed the bond dimension unexpectedly")

        A_right = Q.replace_ind(Q.inds[0], old_left)
        self._set_site(site, A_right, gauge_center=site)
        self.invalidate_from_site_change(site)

        c_right = _copied_bond(old_left, f"cR{site}")
        return R.replace_ind(R.inds[1], c_right)

    def _absorb_center_left(self, site: int, C: Tensor) -> None:
        old_left = self.psi.left_link_ind(site + 1)
        prev_tensor = contract(self.psi[site], C).replace_ind(C.inds[1], old_left)
        self._set_site(site, prev_tensor, gauge_center=site)
        self.invalidate_from_site_change(site)

    def step_left_to_right(self, site: int) -> TDVPStepStats:
        t0 = time.perf_counter()
        self.evolve_one_site(site, self.half_dt)
        C = self._factor_left_center(site)
        self.update_left_environment(site)
        C = self.evolve_zero_site(site, C, -self.half_dt)
        self._absorb_center_right(site, C)
        self._block_until_ready(self.psi[self.psi.gauge_center].data)
        t1 = time.perf_counter()
        return TDVPStepStats(
            sweep=self.sweep,
            site_or_bond=site,
            direction="left_to_right",
            time_step=self.half_dt,
            norm=jnp.linalg.norm(self.psi[self.psi.gauge_center].data),
            elapsed=t1 - t0,
        )

    def step_right_to_left(self, site: int) -> TDVPStepStats:
        t0 = time.perf_counter()
        C = self._factor_right_center(site + 1)
        self.update_right_environment(site + 1)
        C = self.evolve_zero_site(site, C, -self.half_dt)
        self._absorb_center_left(site, C)
        self.evolve_one_site(site, self.half_dt)
        if site == 0:
            self.maybe_normalize_left_boundary()
        self._block_until_ready(self.psi[self.psi.gauge_center].data)
        t1 = time.perf_counter()
        return TDVPStepStats(
            sweep=self.sweep,
            site_or_bond=site,
            direction="right_to_left",
            time_step=self.half_dt,
            norm=jnp.linalg.norm(self.psi[self.psi.gauge_center].data),
            elapsed=t1 - t0,
        )

    def sweep_left_to_right(self) -> TDVPSweepStats:
        self.prepare_left_to_right_sweep()
        t0 = time.perf_counter()
        stats = TDVPSweepStats(
            sweep=self.sweep,
            direction="left_to_right",
            time_step=self.half_dt,
        )

        for site in range(len(self.psi) - 1):
            step = self.step_left_to_right(site)
            self.step_history.append(step)
            self.observe("step", stats=step)
            stats.nsteps += 1

        self.evolve_one_site(len(self.psi) - 1, self.dt)
        self._block_until_ready(self.psi[-1].data)
        t1 = time.perf_counter()
        stats.elapsed = t1 - t0
        stats.norm = jnp.linalg.norm(self.psi[self.psi.gauge_center].data)
        self.sweep_history.append(stats)
        self.observe("sweep_end", stats=stats)
        return stats

    def sweep_right_to_left(self) -> TDVPSweepStats:
        self.prepare_right_to_left_sweep()
        t0 = time.perf_counter()
        stats = TDVPSweepStats(
            sweep=self.sweep,
            direction="right_to_left",
            time_step=self.half_dt,
        )

        for site in range(len(self.psi) - 2, -1, -1):
            step = self.step_right_to_left(site)
            self.step_history.append(step)
            self.observe("step", stats=step)
            stats.nsteps += 1

        self._block_until_ready(self.psi[0].data)
        t1 = time.perf_counter()
        stats.elapsed = t1 - t0
        stats.norm = jnp.linalg.norm(self.psi[self.psi.gauge_center].data)
        self.sweep_history.append(stats)
        self.observe("sweep_end", stats=stats)
        return stats


@dataclass
class TwoSiteTDVPEngine(TDVPEngine):
    maxdim: int = 64
    cutoff: float = 0.0

    def __post_init__(self) -> None:
        if self.maxdim <= 0:
            raise ValueError("maxdim must be positive")
        if self.cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        super().__post_init__()

    def build_effective_two_site_hamiltonian(
        self,
        bond: int,
        domain_inds: tuple[Index, ...],
    ) -> EffectiveTwoSiteHamiltonian:
        if not (0 <= bond < len(self.psi) - 1):
            raise ValueError("bond must satisfy 0 <= bond < len(psi) - 1")
        if bond > self.left_env_valid_upto or self.left_envs[bond] is None:
            raise ValueError(f"left_envs[{bond}] is not available")
        if bond + 2 < self.right_env_valid_from or self.right_envs[bond + 2] is None:
            raise ValueError(f"right_envs[{bond + 2}] is not available")

        return EffectiveTwoSiteHamiltonian(
            left_env=self.left_envs[bond],
            W1=self.H[bond],
            W2=self.H[bond + 1],
            right_env=self.right_envs[bond + 2],
            domain_inds=domain_inds,
            codomain_inds=domain_inds,
            metadata={"bond": bond},
        )

    def evolve_two_site(self, bond: int, theta: Tensor, dt: complex) -> Tensor:
        Heff = self.build_effective_two_site_hamiltonian(bond, theta.inds)
        return expm_multiply_operator(
            Heff,
            theta,
            dt=dt,
            normalize=False,
            method=self.expm_backend,
            maxiter=self.expm_krylov_maxiter,
            tol=self.expm_krylov_tol,
        )

    def _split_theta_left(self, bond: int, theta: Tensor) -> None:
        old_mid_link = self.psi.right_link_ind(bond)
        U, _, Vh, info = truncated_svd(
            theta,
            left_inds=(theta.inds[0], theta.inds[1]),
            bond_name=f"tdvp-bond-{bond}",
            cutoff=self.cutoff,
            max_bond=self.maxdim,
        )
        s = info["singular_values"]
        new_mid_link = Index(
            dim=info["new_bond_dim"],
            name=old_mid_link.name,
            tags=old_mid_link.tags,
            prime_level=old_mid_link.prime_level,
        )
        A_left = U.replace_ind(U.inds[-1], new_mid_link)
        A_center = Tensor(
            s[:, None, None] * Vh.data,
            (new_mid_link, Vh.inds[1], Vh.inds[2]),
        )
        self._set_two_sites(bond, A_left, bond + 1, A_center, gauge_center=bond + 1)
        self.invalidate_from_bond_change(bond)

    def _split_theta_right(self, bond: int, theta: Tensor) -> None:
        old_mid_link = self.psi.right_link_ind(bond)
        U, _, Vh, info = truncated_svd(
            theta,
            left_inds=(theta.inds[0], theta.inds[1]),
            bond_name=f"tdvp-bond-{bond}",
            cutoff=self.cutoff,
            max_bond=self.maxdim,
        )
        s = info["singular_values"]
        new_mid_link = Index(
            dim=info["new_bond_dim"],
            name=old_mid_link.name,
            tags=old_mid_link.tags,
            prime_level=old_mid_link.prime_level,
        )
        A_center = Tensor(
            U.data * s[None, None, :],
            (U.inds[0], U.inds[1], new_mid_link),
        )
        A_right = Vh.replace_ind(Vh.inds[0], new_mid_link)
        self._set_two_sites(bond, A_center, bond + 1, A_right, gauge_center=bond)
        self.invalidate_from_bond_change(bond)

    def step_two_site(self, bond: int, direction: str) -> TDVPStepStats:
        if direction not in ("left_to_right", "right_to_left"):
            raise ValueError("direction must be 'left_to_right' or 'right_to_left'")

        t0 = time.perf_counter()
        theta = contract(self.psi[bond], self.psi[bond + 1])
        theta = self.evolve_two_site(bond, theta, self.half_dt)

        if direction == "left_to_right":
            self._split_theta_left(bond, theta)
            self.update_left_environment(bond)
            if bond + 1 < len(self.psi) - 1:
                self.evolve_one_site(bond + 1, -self.half_dt)
        else:
            self._split_theta_right(bond, theta)
            self.update_right_environment(bond + 1)
            if bond > 0:
                self.evolve_one_site(bond, -self.half_dt)
            else:
                self.maybe_normalize_left_boundary()

        self._block_until_ready(self.psi[self.psi.gauge_center].data)
        t1 = time.perf_counter()
        return TDVPStepStats(
            sweep=self.sweep,
            site_or_bond=bond,
            direction=direction,
            time_step=self.half_dt,
            norm=jnp.linalg.norm(self.psi[self.psi.gauge_center].data),
            elapsed=t1 - t0,
        )

    def sweep_left_to_right(self) -> TDVPSweepStats:
        self.prepare_left_to_right_sweep()
        t0 = time.perf_counter()
        stats = TDVPSweepStats(
            sweep=self.sweep,
            direction="left_to_right",
            time_step=self.half_dt,
        )

        for bond in range(len(self.psi) - 1):
            step = self.step_two_site(bond, direction="left_to_right")
            self.step_history.append(step)
            self.observe("step", stats=step)
            stats.nsteps += 1

        self._block_until_ready(self.psi[-1].data)
        t1 = time.perf_counter()
        stats.elapsed = t1 - t0
        stats.norm = jnp.linalg.norm(self.psi[self.psi.gauge_center].data)
        self.sweep_history.append(stats)
        self.observe("sweep_end", stats=stats)
        return stats

    def sweep_right_to_left(self) -> TDVPSweepStats:
        self.prepare_right_to_left_sweep()
        t0 = time.perf_counter()
        stats = TDVPSweepStats(
            sweep=self.sweep,
            direction="right_to_left",
            time_step=self.half_dt,
        )

        for bond in range(len(self.psi) - 2, -1, -1):
            step = self.step_two_site(bond, direction="right_to_left")
            self.step_history.append(step)
            self.observe("step", stats=step)
            stats.nsteps += 1

        self._block_until_ready(self.psi[0].data)
        t1 = time.perf_counter()
        stats.elapsed = t1 - t0
        stats.norm = jnp.linalg.norm(self.psi[self.psi.gauge_center].data)
        self.sweep_history.append(stats)
        self.observe("sweep_end", stats=stats)
        return stats


def tdvp(
    H: MPO | Callable[[complex], MPO],
    psi: MPS,
    dt: complex | float,
    nsweeps: int = 1,
    maxdim: int = 32,
    cutoff: float = 0.0,
    normalize: bool = False,
    expm_backend: str = "krylov",
    expm_krylov_maxiter: int = 20,
    expm_krylov_tol: float = 1e-12,
    outputlevel: int = 0,
    switch_to_one_site: bool = True,
    observer: Callable[..., Any] | None = None,
):
    """
    High-level TDVP driver.

    The evolution starts with the two-site TDVP engine. If `switch_to_one_site`
    is enabled, the driver automatically switches to one-site TDVP once the
    current MPS bond dimension reaches `maxdim`, so subsequent sweeps no longer
    attempt bond-dimension growth.

    Parameters
    ----------
    H : MPO or callable
        Hamiltonian MPO, or a callable `H(t)` returning the MPO to use at the
        beginning of each full TDVP time step.
    psi : MPS
        Initial state. The input is copied internally.
    dt : complex
        Time-step parameter for the local propagator `exp(dt * H_eff)`.
        Typical choices are `dt = -1j * delta_t` for real-time evolution and
        `dt = -tau` for imaginary-time evolution.
    nsweeps : int
        Number of full TDVP sweeps to perform.
    maxdim : int
        Maximum bond dimension used by the two-site phase.
    cutoff : float
        Singular-value cutoff used in the two-site split.
    normalize : bool
        If True, normalize on the left boundary at the end of each full sweep.
        This is typically useful for imaginary-time evolution.
    expm_backend : str
        Local exponential-action backend, either ``"krylov"`` or ``"scipy"``.
    expm_krylov_maxiter : int
        Maximum Krylov subspace size for the ``"krylov"`` backend.
    expm_krylov_tol : float
        Krylov stopping tolerance for the ``"krylov"`` backend.
    outputlevel : int
        Verbosity level.
    switch_to_one_site : bool
        If True, switch from two-site to one-site TDVP once `psi.maxlinkdim()`
        reaches `maxdim`.
    observer : callable | None
        Callback invoked during evolution. It receives keyword arguments
        including `event`, `engine`, `psi`, `H`, `time`, and optionally `stats`.

    Returns
    -------
    psi : MPS
        Evolved MPS after `nsweeps` full sweeps.
    engine : TDVPEngine
        Final active TDVP engine, useful for continuing evolution.
    """
    if nsweeps < 0:
        raise ValueError("nsweeps must be non-negative")

    engine: TDVPEngine = TwoSiteTDVPEngine(
        H=H,
        psi=psi.copy(),
        dt=dt,
        normalize=normalize,
        expm_backend=expm_backend,
        expm_krylov_maxiter=expm_krylov_maxiter,
        expm_krylov_tol=expm_krylov_tol,
        outputlevel=outputlevel,
        maxdim=maxdim,
        cutoff=cutoff,
        observer=observer,
    )

    for sweep in range(nsweeps):
        engine.run(nsweeps=1)

        if (
            switch_to_one_site
            and isinstance(engine, TwoSiteTDVPEngine)
            and engine.psi.maxlinkdim() >= maxdim
            and sweep + 1 < nsweeps
        ):
            if outputlevel >= 1:
                print(
                    "[TDVP] Switching to one-site TDVP after "
                    f"sweep {engine.sweep} at max bond dimension {engine.psi.maxlinkdim()}"
                )
            engine = OneSiteTDVPEngine.from_two_site_engine(engine)
            engine.outputlevel = outputlevel

    return engine.psi, engine
