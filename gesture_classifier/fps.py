import torch


def velocity_aware_fps(points, num_samples, alpha=0.0):
    """
    Velocity-Aware Farthest Point Sampling.
    
    Args:
        points: (B, N, 4) tensor — x, y, z, doppler
        num_samples: number of points to sample
        alpha: weight for doppler velocity in distance metric
    
    Returns:
        sampled_indices: (B, num_samples) indices of sampled points
    """
    B, N, C = points.shape
    device = points.device

    sampled_indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distances = torch.ones(B, N, device=device) * 1e10

    # Start from a random point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(num_samples):
        sampled_indices[:, i] = farthest

        # Get current farthest point coordinates
        centroid = points[torch.arange(B), farthest, :]  # (B, 4)
        centroid = centroid.unsqueeze(1)  # (B, 1, 4)

        # Spatial distance: Δx² + Δy² + Δz²
        spatial_dist = torch.sum((points[:, :, :3] - centroid[:, :, :3]) ** 2, dim=-1)  # (B, N)

        # Velocity distance: α · Δv²
        velocity_dist = alpha * (points[:, :, 3] - centroid[:, :, 3]) ** 2  # (B, N)

        # Combined distance
        dist = spatial_dist + velocity_dist  # (B, N)

        # Update distances — keep minimum
        distances = torch.min(distances, dist)

        # Select farthest point
        farthest = torch.max(distances, dim=-1)[1]

    return sampled_indices


def index_points(points, idx):
    """
    Index into points tensor using indices.
    
    Args:
        points: (B, N, C)
        idx: (B, S)
    
    Returns:
        (B, S, C)
    """
    B = points.shape[0]
    device = points.device
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, 1, idx_expanded)


if __name__ == '__main__':
    # Test with a batch of random points
    B, N, C = 2, 100, 4
    points = torch.randn(B, N, C)

    for alpha in [0.0, 0.5, 1.0, 2.0]:
        idx = velocity_aware_fps(points, num_samples=32, alpha=alpha)
        sampled = index_points(points, idx)
        print(f'alpha={alpha}: sampled shape={sampled.shape}')