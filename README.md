# QTensor

A tensor network package based on JAX. This package automatically supports automatic differentiation. 

## Installation

For users, please use this to install.
```
pip install git+https://github.com/reezingcold/QTensor.git
```

For developers, we recommand this. 
```
git clone https://github.com/reezingcold/QTensor.git
cd QTensor
pip install -e .
```

## Example

A simple example of DMRG in 1D Ising model. 

```python
import jax
jax.config.update("jax_enable_x64", True)
import qtensor as qt
import numpy as np
from scipy.sparse import linalg as splin
from functools import reduce

qeye = np.array([[1, 0], [0, 1]])
sigmax = np.array([[0, 1],[1, 0]])
sigmay = np.array([[0, -1j],[1j, 0]])
sigmaz = np.array([[1, 0],[0, -1]])

def tensor(lst): return reduce(np.kron, lst)

def ising_benchmark(hx, J=1, N=10):
    H = 0
    for i in range(N):
        cup = [qeye,] * N
        cup[i] = sigmax
        H += -hx * tensor(cup)
    for i in range(N-1):
        cup = [qeye,] * N
        cup[i], cup[i+1] = sigmaz, sigmaz
        H += -J * tensor(cup)
    Es_ed, psis_ed = splin.eigsh(H, k=2, which='SA')
    E0_ed, psi0_ed = Es_ed[0], psis_ed[:,0]
    Ox = 0
    for i in range(N):
        cup = [qeye,] * N
        cup[i] = sigmax
        Ox += tensor(cup)
    x_aver = (psi0_ed.T @ Ox @ psi0_ed)/N
    return E0_ed, x_aver



L = 10
sites = [qt.QubitSite(),] * L
h, g, J = 1, 0, 1
# ED result as benchmark
E0_ed, x_aver_ed = ising_benchmark(1, N=L)

# using OpSum() to construct H MPO automatically
opsum = qt.OpSum()
for i in range(L - 1):
    opsum += -J, "Z", i, "Z", i + 1

for i in range(L):
    if h != 0.0:
        opsum += -h, "X", i
    if g != 0.0:
        opsum += -g, "Z", i

H = opsum.to_mpo(sites)
print("H bond dimension:", H.link_dims())
psi0 = qt.random_mps(sites, 4)
print("psi0 bond dimension:", psi0.link_dims())
# doing dmrg
psi, energy = qt.dmrg(H, psi0, nsweeps_two_site=5, nsweeps_one_site=5, 
                      maxdim=32, cutoff=1e-10, outputlevel=1)

x_aver = np.mean(psi.local_expect("X"))

print(f"ground state energy ED  : {E0_ed}")
print(f"ground state energy DMRG: {energy}")

print(f"magnetization ED  : {x_aver_ed}")
print(f"magnetization DMRG: {x_aver}")
```
Result of the DMRG example code:
```
H bond dimension: (1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1)
psi0 bond dimension: (1, 2, 4, 4, 4, 4, 4, 4, 4, 2, 1)
[TwoSiteDMRGEngine Sweep 1] E=-12.381489999654626, dE=15.908732792844525, maxlinkdim=16, time=4.012653 s
[TwoSiteDMRGEngine Sweep 2] E=-12.381489999654761, dE=1.3500311979441904e-13, maxlinkdim=19, time=2.519036 s
[TwoSiteDMRGEngine Sweep 3] E=-12.381489999654763, dE=1.7763568394002505e-15, maxlinkdim=19, time=0.253595 s
[TwoSiteDMRGEngine Sweep 4] E=-12.381489999654772, dE=8.881784197001252e-15, maxlinkdim=19, time=0.185264 s
[TwoSiteDMRGEngine Sweep 5] E=-12.381489999654777, dE=5.329070518200751e-15, maxlinkdim=19, time=0.187930 s
[DMRG] After two-site phase: E = -12.381489999654777
[OneSiteDMRGEngine Sweep 6] E=-12.38148999965476, dE=1.7763568394002505e-14, maxlinkdim=19, time=2.267716 s
[OneSiteDMRGEngine Sweep 7] E=-12.381489999654773, dE=1.4210854715202004e-14, maxlinkdim=19, time=0.097398 s
[OneSiteDMRGEngine Sweep 8] E=-12.381489999654757, dE=1.5987211554602254e-14, maxlinkdim=19, time=0.093677 s
[OneSiteDMRGEngine Sweep 9] E=-12.38148999965476, dE=1.7763568394002505e-15, maxlinkdim=19, time=0.092206 s
[OneSiteDMRGEngine Sweep 10] E=-12.381489999654786, dE=2.6645352591003757e-14, maxlinkdim=19, time=0.093983 s
[DMRG] After one-site phase: E = -12.381489999654786
ground state energy ED  : -12.381489999654743
ground state energy DMRG: -12.381489999654786
magnetization ED  : 0.7322550547297129
magnetization DMRG: 0.7322550547297143
```
