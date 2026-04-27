import torch


def velocity_aware_fps(points, num_samples, alpha=0.0):
    """
    Velocity-Aware Farthest Point Sampling.
    points: (B, N, 4) — x, y, z, doppler
    alpha: weight for doppler in distance metric
    Returns: (B, num_samples) indices
    """
    B, N, C = points.shape
    device = points.device

    sampled_indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distances = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(num_samples):
        sampled_indices[:, i] = farthest
        centroid = points[torch.arange(B), farthest, :].unsqueeze(1)

        spatial_dist = torch.sum((points[:, :, :3] - centroid[:, :, :3]) ** 2, dim=-1)
        velocity_dist = alpha * (points[:, :, 3] - centroid[:, :, 3]) ** 2
        dist = spatial_dist + velocity_dist

        distances = torch.min(distances, dist)
        farthest = torch.max(distances, dim=-1)[1]

    return sampled_indices


def index_points(points, idx):
    """Index into points tensor. points: (B,N,C), idx: (B,S) -> (B,S,C)"""
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, 1, idx_expanded)


if __name__ == '__main__':
    B, N, C = 2, 100, 4
    points = torch.randn(B, N, C)
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        idx = velocity_aware_fps(points, num_samples=32, alpha=alpha)
        sampled = index_points(points, idx)
        print(f'alpha={alpha}: sampled shape={sampled.shape}')