"""
Basic examples demonstrating Index and Tensor usage in QTensor.

Run:
    python tests/basic_tensor_operations.py
"""

import jax
import jax.numpy as jnp

from qtensor import Index, Tensor


def example_create_indices():
    print("=== Create Indices ===")
    i = Index(3, name="i")
    j = Index(4, name="j")
    k = Index(2, name="k")

    print("i:", i)
    print("j:", j)
    print("k:", k)
    return i, j, k


def example_create_tensors(i, j, k):
    print("\n=== Create Tensors ===")
    # Random tensors with given indices
    A = Tensor(jax.random.normal(jax.random.PRNGKey(0), (i.dim, j.dim)), (i, j))
    B = Tensor(jax.random.normal(jax.random.PRNGKey(1), (j.dim, k.dim)), (j, k))

    print("A indices:", A.inds)
    print("B indices:", B.inds)
    print("A shape:", A.data.shape)
    print("B shape:", B.data.shape)

    return A, B


def example_permute(A):
    print("\n=== Permute Tensor ===")
    A_perm = A.permute(A.inds[::-1])  # reverse index order
    print("Original inds:", A.inds)
    print("Permuted inds:", A_perm.inds)
    print("Permuted shape:", A_perm.data.shape)
    return A_perm


def example_contract(A, B):
    print("\n=== Contract Tensors ===")
    # Contract along shared index j
    C = A @ B
    print("Result indices:", C.inds)
    print("Result shape:", C.data.shape)
    return C


def example_reshape(A):
    print("\n=== Reshape Tensor ===")
    # Example: reshape into a matrix
    mat = A.to_array(A.inds[0], A.inds[1])
    print("Matrix shape:", mat.shape)
    return mat


def main():
    i, j, k = example_create_indices()
    A, B = example_create_tensors(i, j, k)
    A_perm = example_permute(A)
    C = example_contract(A, B)
    mat = example_reshape(A)


if __name__ == "__main__":
    main()