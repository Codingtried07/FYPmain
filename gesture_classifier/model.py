import torch
import torch.nn as nn
import torch.nn.functional as F
from fps import velocity_aware_fps, index_points


def ball_query(radius, num_samples, xyz, new_xyz):
    """
    Vectorized ball query.
    Args:
        xyz: (B, N, 3)
        new_xyz: (B, S, 3)
    Returns:
        idx: (B, S, num_samples)
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]

    # Compute all pairwise distances at once — no Python loops
    dists = torch.cdist(new_xyz, xyz)  # (B, S, N)

    # Sort and take top num_samples
    idx = dists.argsort(dim=-1)[:, :, :num_samples]  # (B, S, num_samples)

    return idx


class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction layer with Velocity-Aware FPS.
    """
    def __init__(self, num_points, radius, num_samples, in_channels, mlp_channels, alpha=0.0):
        super().__init__()
        self.num_points = num_points
        self.radius = radius
        self.num_samples = num_samples
        self.alpha = alpha

        layers = []
        last_ch = in_channels + 3  # +3 for xyz
        for out_ch in mlp_channels:
            layers += [
                nn.Conv2d(last_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ]
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = last_ch

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) — spatial coordinates
            features: (B, N, C) — point features including doppler

        Returns:
            new_xyz: (B, num_points, 3)
            new_features: (B, num_points, mlp_out)
        """
        B, N, _ = xyz.shape

        # Velocity-Aware FPS — use full 4D points (xyz + doppler)
        if features is not None:
            points_4d = torch.cat([xyz, features[:, :, :1]], dim=-1)  # use doppler as 4th dim
        else:
            points_4d = torch.cat([xyz, torch.zeros(B, N, 1, device=xyz.device)], dim=-1)

        fps_idx = velocity_aware_fps(points_4d, self.num_points, alpha=self.alpha)
        new_xyz = index_points(xyz, fps_idx)  # (B, num_points, 3)

        # Ball query grouping
        idx = ball_query(self.radius, self.num_samples, xyz, new_xyz)

        grouped_xyz = index_points(xyz, idx.reshape(B, -1)).reshape(B, self.num_points, self.num_samples, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_features = index_points(features, idx.reshape(B, -1)).reshape(B, self.num_points, self.num_samples, -1)
            grouped = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            grouped = grouped_xyz

        # MLP
        grouped = grouped.permute(0, 3, 2, 1)  # (B, C, num_samples, num_points)
        grouped = self.mlp(grouped)
        new_features = grouped.max(dim=2)[0].permute(0, 2, 1)  # (B, num_points, mlp_out)

        return new_xyz, new_features


class GestureClassifier(nn.Module):
    def __init__(self, num_classes=4, alpha=0.0):
        super().__init__()
        self.alpha = alpha

        self.sa1 = SetAbstraction(
            num_points=32, radius=0.5, num_samples=16,
            in_channels=1, mlp_channels=[32, 32, 64], alpha=alpha
        )
        self.sa2 = SetAbstraction(
            num_points=16, radius=1.0, num_samples=8,
            in_channels=64, mlp_channels=[64, 64, 128], alpha=alpha
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, 4) — x, y, z, doppler
        """
        xyz = x[:, :, :3]
        features = x[:, :, 3:]  # doppler

        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)

        # Global max pooling
        x = features.max(dim=1)[0]  # (B, 128)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = GestureClassifier(num_classes=4, alpha=1.0)
    x = torch.randn(4, 64, 4)
    out = model(x)
    print('Output shape:', out.shape)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))