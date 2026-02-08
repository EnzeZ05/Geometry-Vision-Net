import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class EpisodeCfg:
    ways: int = 5
    shots: int = 5
    queries: int = 15


def make_grid(H, W, device):
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1)


def angular_sort(pts):
    center = pts.mean(1, keepdim=True)
    ang = torch.atan2(pts[..., 1] - center[..., 1], pts[..., 0] - center[..., 0])
    return torch.gather(pts, 1, ang.argsort(1)[..., None].expand(-1, -1, 2))


def points_to_mask(points, grid, tau):
    pts = angular_sort(points)
    x, y = pts[..., 0], pts[..., 1]
    signed = (x * torch.roll(y, -1, 1) - y * torch.roll(x, -1, 1)).sum(1)
    flip_mask = signed < 0
    if flip_mask.any():
        pts = pts.clone()
        pts[flip_mask] = torch.flip(pts[flip_mask], [1])

    v1 = pts
    v2 = torch.roll(pts, -1, 1)
    edge = v2 - v1
    diff = grid.view(-1, 2)[None, None] - v1[:, :, None]
    cross = edge[..., 0, None] * diff[..., 1] - edge[..., 1, None] * diff[..., 0]
    inside = torch.sigmoid((-tau * torch.logsumexp(-cross / tau, 1)) / tau)
    return inside.view(points.shape[0], grid.shape[0], grid.shape[1])


def soft_iou(m1, m2, eps=1e-6):
    inter = (m1 * m2).sum((-2, -1))
    union = (m1 + m2 - m1 * m2).sum((-2, -1))
    return inter / (union + eps)


def point_repulsion(pts, sigma=0.15):
    d2 = torch.cdist(pts, pts) + torch.eye(pts.shape[1], device=pts.device) * 1e9
    return torch.exp(-d2 / (2 * sigma**2)).mean()


def centroid_repulsion_multi(centroids, margin=0.9):
    if centroids.dim() == 2:
        dists = torch.cdist(centroids, centroids)
        ways = centroids.shape[0]
        mask = ~torch.eye(ways, dtype=torch.bool, device=centroids.device)
        return F.relu(margin - dists[mask]).mean()

    dists = torch.cdist(centroids, centroids)
    ways = centroids.shape[1]
    mask = ~torch.eye(ways, dtype=torch.bool, device=centroids.device)
    return F.relu(margin - dists[:, mask]).mean()


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.0, use_pool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(
            out_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        )
        self.pool = nn.MaxPool2d(2) if use_pool else nn.Identity()
        self.drop_rate = drop_rate

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.1)
        out = self.pool(F.leaky_relu(self.bn3(self.conv3(out)) + self.shortcut(x), 0.1))
        return F.dropout(out, self.drop_rate, self.training) if self.drop_rate > 0 else out


class ResNet12(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), drop_rate=0.1, num_pools=3):
        super().__init__()
        blocks, ch = [], 3
        for i, c in enumerate(channels):
            blocks.append(BasicBlock(ch, c, drop_rate=drop_rate, use_pool=(i < num_pools)))
            ch = c
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x):
        return self.blocks(x)


class GVNet(nn.Module):
    def __init__(
        self,
        n_points=8,
        grid_hw=32,
        tau=0.15,
        backbone_channels=(64, 128, 256, 512),
        drop_rate=0.1,
        num_pools=3,
        n_heads=4,
        **kwargs
    ):
        super().__init__()
        self.n_points = n_points
        self.n_heads = n_heads
        self.grid_hw = grid_hw
        self.tau = tau

        self.backbone = ResNet12(backbone_channels, drop_rate=drop_rate, num_pools=num_pools)
        C = self.backbone.out_channels

        self.kp_proj = nn.Sequential(
            nn.Conv2d(C, C // 2, 1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(C // 2, n_heads * n_points, 1),
        )

    def encode(self, x):
        feat = self.kp_proj(self.backbone(x))
        B, HK, H, W = feat.shape
        Hh = self.n_heads
        K = self.n_points
        feat = feat.view(B, Hh, K, -1)
        probs = F.softmax(feat, dim=-1)

        grid = make_grid(H, W, x.device).view(-1, 2)
        pts = torch.matmul(probs, grid)
        return pts

    def forward_episode(self, sx, sy, qx, qy, cfg: EpisodeCfg):
        ways = cfg.ways

        s_pts = self.encode(sx)
        q_pts = self.encode(qx)

        grid = make_grid(self.grid_hw, self.grid_hw, sx.device)

        Bs = s_pts.shape[0]
        Bq = q_pts.shape[0]
        Hh = self.n_heads

        s_pts_flat = s_pts.reshape(Bs * Hh, self.n_points, 2)
        q_pts_flat = q_pts.reshape(Bq * Hh, self.n_points, 2)

        s_mask_flat = points_to_mask(s_pts_flat, grid, self.tau)
        q_mask_flat = points_to_mask(q_pts_flat, grid, self.tau)

        s_mask = s_mask_flat.view(Bs, Hh, self.grid_hw, self.grid_hw)
        q_mask = q_mask_flat.view(Bq, Hh, self.grid_hw, self.grid_hw)

        proto_mask = torch.stack([s_mask[sy == c].mean(0) for c in range(ways)])  # [ways, Hh, H, W]

        head_iou = soft_iou(q_mask[:, None], proto_mask[None])  # [Bq, ways, Hh]
        logits = head_iou.mean(-1)  # pure IoU: average over heads -> [Bq, ways]

        loss = F.cross_entropy(logits, qy)
        acc = (logits.argmax(-1) == qy).float().mean()
        return loss, acc, logits
