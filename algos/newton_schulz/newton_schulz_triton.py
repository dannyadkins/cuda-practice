import torch
import triton
import triton.language as tl

@triton.jit
def newton_schulz_kernel(
    G_ptr, X_ptr, N, M, steps,
    a, b, c,
    stride_G0, stride_G1,
    stride_X0, stride_X1,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE
    col_start = col_idx * BLOCK_SIZE

    G = tl.load(
        G_ptr + row_start * stride_G0 + col_start * stride_G1,
        mask=(row_start + tl.arange(0, BLOCK_SIZE)[:, None] < N) & (col_start + tl.arange(0, BLOCK_SIZE)[None, :] < M),
        other=0.0
    )

    X = G.clone()

    for _ in range(steps):
        # Compute A = X @ X^T
        X_T = tl.trans(X)
        A = tl.dot(X, X_T)

        # Compute B = b * A + c * A @ A
        A_sq = tl.dot(A, A)
        B = b * A + c * A_sq

        # Update X = a * X + B @ X
        X = a * X + tl.dot(B, X)

    # Write back to output
    tl.store(
        X_ptr + row_start * stride_X0 + col_start * stride_X1,
        X,
        mask=(row_start + tl.arange(0, BLOCK_SIZE)[:, None] < N) & (col_start + tl.arange(0, BLOCK_SIZE)[None, :] < M)
    )

def zeropower_via_newtonschulz5_triton(G, steps=10, eps=1e-7):
    """
    Triton-accelerated Newton-Schulz iteration for orthogonalization.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    N, M = G.shape

    # Normalize G
    G_norm = G.norm() + eps
    G_normalized = G / G_norm

    # Prepare output tensor X
    X = torch.empty_like(G_normalized)

    # Define block size
    BLOCK_SIZE = 32  # Adjust based on your hardware

    # Launch the Triton kernel
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, (M + BLOCK_SIZE - 1) // BLOCK_SIZE )

    newton_schulz_kernel[grid](
        G_normalized, X, N, M, steps,
        a, b, c,
        G_normalized.stride(0), G_normalized.stride(1),
        X.stride(0), X.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Rescale X back
    return X * G_norm
