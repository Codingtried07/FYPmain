import numpy as np
import torch


def simulate_densification(sparse_points, target_n=256):
    """
    Simulate mmPoint densification by upsampling sparse points.
    
    Args:
        sparse_points: (M, 4) — x, y, z, doppler
        target_n: number of dense points to generate
    
    Returns:
        dense_points: (target_n, 4) — x, y, z, propagated_doppler
    """
    M = len(sparse_points)
    if M == 0:
        return np.zeros((target_n, 4))

    sparse_xyz = sparse_points[:, :3]
    sparse_doppler = sparse_points[:, 3]

    # Step 1: Generate dense xyz points by interpolation
    # Sample random pairs of existing points and interpolate between them
    dense_xyz = []

    # First keep all original points
    dense_xyz.append(sparse_xyz)

    # Then generate new points by interpolating between random pairs
    needed = target_n - M
    if needed > 0:
        idx1 = np.random.randint(0, M, needed)
        idx2 = np.random.randint(0, M, needed)
        t = np.random.uniform(0, 1, (needed, 1))
        interpolated = sparse_xyz[idx1] * t + sparse_xyz[idx2] * (1 - t)
        # Add small jitter to avoid exact duplicates
        interpolated += np.random.normal(0, 0.01, interpolated.shape)
        dense_xyz.append(interpolated)

    dense_xyz = np.vstack(dense_xyz)[:target_n]  # (target_n, 3)

    # Step 2: Propagate Doppler via nearest neighbour
    dists = np.sum(
        (dense_xyz[:, None, :] - sparse_xyz[None, :, :]) ** 2,
        axis=-1
    )  # (target_n, M)
    nearest = np.argmin(dists, axis=-1)  # (target_n,)
    propagated_doppler = sparse_doppler[nearest]  # (target_n,)

    # Step 3: Combine
    dense_points = np.concatenate(
        [dense_xyz, propagated_doppler[:, None]], axis=-1
    )  # (target_n, 4)

    return dense_points


if __name__ == '__main__':
    # Test
    sparse = np.random.randn(30, 4)
    sparse[:, 3] = np.random.uniform(-1, 1, 30)  # doppler values

    dense = simulate_densification(sparse, target_n=256)
    print('Sparse shape:', sparse.shape)
    print('Dense shape:', dense.shape)
    print('Doppler range sparse:', sparse[:, 3].min(), sparse[:, 3].max())
    print('Doppler range dense:', dense[:, 3].min(), dense[:, 3].max())