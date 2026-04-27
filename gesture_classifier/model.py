import torch
import torch.nn as nn
from fps import velocity_aware_fps, index_points


def knn_query(num_samples, xyz, new_xyz):
    """K-nearest neighbour grouping. Returns (B, S, num_samples) indices."""
    dists = torch.cdist(new_xyz, xyz)
    return dists.argsort(dim=-1)[:, :, :num_samples]


class SetAbstraction(nn.Module):
    """PointNet++ Set Abstraction with Velocity-Aware FPS."""

    def __init__(self, num_points, radius, num_samples, in_channels, mlp_channels, alpha=0.0):
        super().__init__()
        self.num_points = num_points
        self.radius = radius
        self.num_samples = num_samples
        self.alpha = alpha

        layers = []
        last_ch = in_channels + 3
        for out_ch in mlp_channels:
            layers += [nn.Conv2d(last_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU()]
            last_ch = out_ch
        self.mlp = nn.Sequential(*layers)
        self.out_channels = last_ch

    def forward(self, xyz, features):
        """xyz: (B,N,3), features: (B,N,C) -> new_xyz: (B,S,3), new_features: (B,S,D)"""
        B, N, _ = xyz.shape

        if features is not None:
            points_4d = torch.cat([xyz, features[:, :, :1]], dim=-1)
        else:
            points_4d = torch.cat([xyz, torch.zeros(B, N, 1, device=xyz.device)], dim=-1)

        fps_idx = velocity_aware_fps(points_4d, self.num_points, alpha=self.alpha)
        new_xyz = index_points(xyz, fps_idx)

        idx = knn_query(self.num_samples, xyz, new_xyz)

        grouped_xyz = index_points(xyz, idx.reshape(B, -1)).reshape(B, self.num_points, self.num_samples, 3)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_features = index_points(features, idx.reshape(B, -1)).reshape(B, self.num_points, self.num_samples, -1)
            grouped = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            grouped = grouped_xyz

        grouped = grouped.permute(0, 3, 2, 1)
        grouped = self.mlp(grouped)
        new_features = grouped.max(dim=2)[0].permute(0, 2, 1)

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
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """x: (B, N, 4) — x, y, z, doppler"""
        xyz = x[:, :, :3]
        features = x[:, :, 3:]

        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)

        x = features.max(dim=1)[0]
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = GestureClassifier(num_classes=4, alpha=1.0)
    x = torch.randn(4, 64, 4)
    out = model(x)
    print('Output shape:', out.shape)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))