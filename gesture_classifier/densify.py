import numpy as np


def simulate_densification(sparse_points, target_n=256):
    """
    Upsample sparse radar points via pairwise interpolation + NN doppler propagation.
    sparse_points: (M, 4) — x, y, z, doppler
    Returns: (target_n, 4)
    """
    M = len(sparse_points)
    if M == 0:
        return np.zeros((target_n, 4))

    sparse_xyz = sparse_points[:, :3]
    sparse_doppler = sparse_points[:, 3]

    dense_xyz = [sparse_xyz]

    needed = target_n - M
    if needed > 0:
        idx1 = np.random.randint(0, M, needed)
        idx2 = np.random.randint(0, M, needed)
        t = np.random.uniform(0, 1, (needed, 1))
        interpolated = sparse_xyz[idx1] * t + sparse_xyz[idx2] * (1 - t)
        interpolated += np.random.normal(0, 0.01, interpolated.shape)
        dense_xyz.append(interpolated)

    dense_xyz = np.vstack(dense_xyz)[:target_n]

    dists = np.sum((dense_xyz[:, None, :] - sparse_xyz[None, :, :]) ** 2, axis=-1)
    nearest = np.argmin(dists, axis=-1)
    propagated_doppler = sparse_doppler[nearest]

    dense_points = np.concatenate([dense_xyz, propagated_doppler[:, None]], axis=-1)
    return dense_points


if __name__ == '__main__':
    sparse = np.random.randn(30, 4)
    sparse[:, 3] = np.random.uniform(-1, 1, 30)
    dense = simulate_densification(sparse, target_n=256)
    print('Sparse shape:', sparse.shape)
    print('Dense shape:', dense.shape)