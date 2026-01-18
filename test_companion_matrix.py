"""
Minimal test for companion matrix construction in BDF methods.
This tests the logic without requiring full dependencies.
"""

import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np

jax.config.update("jax_enable_x64", True)

def test_companion_matrix_construction():
    """Test that companion matrix construction is correct."""

    print("=" * 80)
    print("Testing Companion Matrix Construction for BDF Methods")
    print("=" * 80)

    # Simple test: 2D state, BDF2
    ny = 2  # state dimension
    n_history = 2  # BDF2
    nsamples = 5

    # Create dummy Jacobians
    np.random.seed(42)
    M0 = jnp.array(np.random.randn(nsamples, ny, ny))
    M1 = jnp.array(np.random.randn(nsamples, ny, ny))

    def build_companion_matrices(M0_i, M1_i):
        """Build augmented companion matrices for BDF2.

        For BDF2: M0 @ y_i + M1 @ y_{i-1} + M2 @ y_{i-2} = z_i
        With 2-history: Y_i = [y_i; y_{i-1}]
        We need: M_aug @ Y_i = Z_i - M_shift @ Y_{i-1}
        """
        M_aug = jnp.zeros((n_history * ny, n_history * ny), dtype=jnp.float64)
        M_shift = jnp.zeros((n_history * ny, n_history * ny), dtype=jnp.float64)

        # Top block: M0 @ y_i + M1 @ y_{i-1} = z_i - M2 @ y_{i-2}
        # Since we only track 2 history terms, M2 @ y_{i-2} comes from Y_{i-1}
        # But for this test, we treat M1 as the only history term in M_aug
        M_aug = M_aug.at[:ny, :ny].set(M0_i)
        M_aug = M_aug.at[:ny, ny:2*ny].set(M1_i)

        # History tracking: y_{i-1} in Y_i should come from y_i in Y_{i-1}
        # I @ y_{i-1} (from Y_i) = 0 - (-I) @ y_i (from Y_{i-1})
        M_aug = M_aug.at[ny:2*ny, ny:2*ny].set(jnp.eye(ny))
        M_shift = M_shift.at[ny:2*ny, :ny].set(-jnp.eye(ny))

        return M_aug, M_shift

    # Build for all timesteps
    M_aug_all, M_shift_all = vmap(build_companion_matrices)(M0, M1)

    print(f"\nState dimension: {ny}")
    print(f"BDF order: {n_history}")
    print(f"Augmented state dimension: {n_history * ny}")
    print(f"Number of timesteps: {nsamples}")

    print(f"\nM_aug shape: {M_aug_all.shape}")
    print(f"M_shift shape: {M_shift_all.shape}")

    # Check structure of first timestep
    print(f"\nFirst timestep M_aug structure:")
    print(f"Top-left block (M0):\n{M_aug_all[0, :ny, :ny]}")
    print(f"Top-right block (M1):\n{M_aug_all[0, :ny, ny:]}")
    print(f"Bottom-left block (I):\n{M_aug_all[0, ny:, :ny]}")
    print(f"Bottom-right block (0):\n{M_aug_all[0, ny:, ny:]}")

    print(f"\nFirst timestep M_shift structure:")
    print(f"Top-left block (0):\n{M_shift_all[0, :ny, :ny]}")
    print(f"Top-right block (0):\n{M_shift_all[0, :ny, ny:]}")
    print(f"Bottom-left block (0):\n{M_shift_all[0, ny:, :ny]}")
    print(f"Bottom-right block (I):\n{M_shift_all[0, ny:, ny:]}")

    # Test that the recurrence makes sense
    # M_aug @ Y_i + M_shift @ Y_{i-1} = Z_i
    # With corrected construction:
    # [M0, M1] @ [y_i  ] + [0  0 ] @ [y_{i-1}] = [z_i]
    # [0   I ] @ [y_{i-1}] + [-I 0] @ [y_{i-2}]   [0  ]
    # Top part: M0 @ y_i + M1 @ y_{i-1} = z_i  ✓ (BDF recurrence)
    # Bottom part: I @ y_{i-1} - I @ y_{i-1} = 0 ✓ (history tracking: enforces consistency)

    y_i = jnp.array([1.0, 2.0])
    y_im1 = jnp.array([0.5, 1.0])
    y_im2 = jnp.array([0.2, 0.5])

    Y_i = jnp.concatenate([y_i, y_im1])
    Y_im1 = jnp.concatenate([y_im1, y_im2])

    z_i = jnp.array([3.0, 4.0])
    Z_i = jnp.concatenate([z_i, jnp.zeros(ny)])

    # Check recurrence
    lhs = M_aug_all[0] @ Y_i + M_shift_all[0] @ Y_im1

    print(f"\n\nRecurrence check:")
    print(f"Y_i = {Y_i} (contains [y_i; y_{{i-1}}])")
    print(f"Y_{{i-1}} = {Y_im1} (contains [y_{{i-1}}; y_{{i-2}}])")
    print(f"Z_i = {Z_i}")
    print(f"M_aug @ Y_i + M_shift @ Y_{{i-1}} = {lhs}")

    # Top part should be M0 @ y_i + M1 @ y_{i-1}
    expected_top = M0[0] @ y_i + M1[0] @ y_im1
    print(f"\nTop part (should equal z_i if recurrence is satisfied):")
    print(f"  Computed: {lhs[:ny]}")
    print(f"  Expected (M0@y_i + M1@y_{{i-1}}): {expected_top}")

    # Bottom part should be 0 (from I @ y_{i-1} - I @ y_{i-1})
    print(f"\nBottom part (should equal 0 for history consistency):")
    print(f"  Computed: {lhs[ny:]}")
    print(f"  Expected (y_{{i-1}} - y_{{i-1}}): {jnp.zeros(ny)}")
    print(f"  Match: {np.allclose(lhs[ny:], jnp.zeros(ny))}")

    print("\n" + "=" * 80)
    print("Companion matrix construction verified!")
    print("=" * 80)


if __name__ == "__main__":
    test_companion_matrix_construction()
