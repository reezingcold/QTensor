from __future__ import annotations
from functools import lru_cache
import jax.numpy as jnp
from opt_einsum.parser import get_symbol
from typing import Dict, Tuple

from qtensor.tensor.index import Index
from qtensor.tensor.tensor import Tensor



@lru_cache(maxsize=4096)
def _build_einsum_eq_from_inds(
    tensor_inds: tuple[tuple[Index, ...], ...],
) -> tuple[str, tuple[Index, ...]]:
    """
    Build an einsum equation from tensor index tuples.

    Rule:
        - if an Index appears in two or more tensors, contract it
        - if an Index appears only once, keep it as an output index
    """
    if len(tensor_inds) == 0:
        raise ValueError("At least one tensor must be provided")

    # do not allow repeated indices inside a single tensor for now
    for inds in tensor_inds:
        if len(set(inds)) != len(inds):
            raise ValueError("Repeated indices inside one tensor are not supported yet")

    symbol_map: Dict[Index, str] = {}
    counts: Dict[Index, int] = {}

    # assign one einsum symbol to each unique Index
    for inds in tensor_inds:
        for ind in inds:
            if ind not in symbol_map:
                symbol_map[ind] = get_symbol(len(symbol_map))
                counts[ind] = 0
            counts[ind] += 1

    input_terms = []
    for inds in tensor_inds:
        term = "".join(symbol_map[ind] for ind in inds)
        input_terms.append(term)

    # free indices = those appearing exactly once
    # order convention:
    #   preserve the input tensor order, and within each tensor preserve
    #   the original index order of uncontracted indices.
    output_inds_list: list[Index] = []
    for inds in tensor_inds:
        for ind in inds:
            if counts[ind] == 1:
                output_inds_list.append(ind)

    output_inds = tuple(output_inds_list)
    output_term = "".join(symbol_map[ind] for ind in output_inds)

    equation = ",".join(input_terms) + "->" + output_term
    return equation, output_inds


def _build_einsum_eq(*tensors: "Tensor") -> tuple[str, tuple[Index, ...]]:
    return _build_einsum_eq_from_inds(tuple(tensor.inds for tensor in tensors))


def contract(*tensors: "Tensor", optimize: str | bool = "auto") -> Tensor:
    """
    Contract any number of tensors.

    Behavior:
        - if tensors share indices, those indices are contracted
        - if two tensors share no indices, they are combined by outer product
        - contraction path is optimized automatically via JAX einsum

    Parameters
    ----------
    *tensors
        Tensor objects to be contracted.
    optimize
        Passed to `jax.numpy.einsum`. Common choices are:
            - "auto"    : recommended default
            - "greedy"  : fast path search
            - "optimal" : more expensive path search
            - False      : no path optimization
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided")
    if len(tensors) == 1:
        return tensors[0]

    equation, output_inds = _build_einsum_eq(*tensors)
    arrays = [tensor.data for tensor in tensors]
    new_data = jnp.einsum(equation, *arrays, optimize=optimize)

    return Tensor(new_data, output_inds)


def contract_path(*tensors: "Tensor", optimize: str | bool = "auto"):
    """
    Return the einsum contraction path information without executing the contraction.
    Useful for debugging or benchmarking.
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided")

    equation, _ = _build_einsum_eq(*tensors)
    arrays = [tensor.data for tensor in tensors]
    return jnp.einsum_path(equation, *arrays, optimize=optimize)






# import numpy as np
# i, j, k, l = Index(4, "i"), Index(5, "j"), Index(6, "k"), Index(7, "l")
# m, n, p, q = Index(4, "m"), Index(5, "n"), Index(2, "p"), Index(3, "q")
# A = Tensor(jnp.array(np.random.uniform(0, 1, (4, 5, 6, 7))), (i, j, k, l))
# B = Tensor(jnp.array(np.random.uniform(1, 2, (4, 5, 2, 3))), (i, j, p, q))
# D = Tensor(jnp.array(np.random.uniform(-1, 0, (6, 7, 2, 3))), (k, l, p, q))
# C = contract(A, B, D)

# print(jnp.einsum('ijkl,ijpq,klpq',A.data, B.data, D.data))
# print(C.data)

    
    
