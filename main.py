```python
import argparse
import random
import math
import os
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


TRAIN_CLASSES = list(range(0, 64))
VAL_CLASSES = list(range(64, 80))
TEST_CLASSES = list(range(80, 100))


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
    if ((x * torch.roll(y, -1, 1) - y * torch.roll(x, -1, 1)).sum(1) < 0).any():
        pts = torch.flip(pts, [1])

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


def centroid_repulsion(centroids, margin=0.9):
    dists = torch.cdist(centroids, centroids)
    mask = ~torch.eye(len(centroids), dtype=torch.bool, device=centroids.device)
    return F.relu(margin - dists[mask]).mean()


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.0, use_pool=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )
        self.pool = nn.MaxPool2d(2) if use_pool else nn.Identity()
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.1)
        h = self.conv3(h)
        return self.drop(self.pool(F.leaky_relu(self.bn3(h + self.shortcut(x)), 0.1)))


class ResNet12(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), drop_rate=0.0, num_pools=3):
        super().__init__()
        self.layers = nn.ModuleList()
        in_ch = 3
        for i, c in enumerate(channels):
            self.layers.append(BasicBlock(in_ch, c, drop_rate, i < num_pools))
            in_ch = c
        self.out_channels = channels[-1]

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class GVNet(nn.Module):
    def __init__(
        self,
        n_points=8,
        n_planes=3,
        grid_hw=32,
        tau=0.20,
        backbone_channels=(64, 128, 256, 512),
        drop_rate=0.1,
        num_pools=3,
        **kwargs
    ):
        super().__init__()
        self.n_points = n_points
        self.n_planes = n_planes
        self.grid_hw = grid_hw
        self.tau = tau

        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(5.0))

        self.backbone = ResNet12(backbone_channels, drop_rate=drop_rate, num_pools=num_pools)
        C = self.backbone.out_channels

        self.kp_proj = nn.Sequential(
            nn.Conv2d(C, C // 2, 1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(C // 2, n_planes * n_points, 1),
        )

    def set_epoch(self, epoch):
        pass

    def encode(self, x):
        feat = self.kp_proj(self.backbone(x))
        B, _, H, W = feat.shape
        feat = feat.view(B, self.n_planes, self.n_points, -1)
        grid = make_grid(H, W, x.device).view(H * W, 2)
        probs = F.softmax(feat, dim=-1)
        points = torch.matmul(probs, grid)
        return points

    def render_and_merge(self, points):
        B, P, K, _ = points.shape
        grid = make_grid(self.grid_hw, self.grid_hw, points.device)
        flat_pts = points.view(B * P, K, 2)
        flat_masks = points_to_mask(flat_pts, grid, self.tau)
        masks = flat_masks.view(B, P, self.grid_hw, self.grid_hw)
        combined_mask = 1 - torch.prod(1 - masks, dim=1)
        return combined_mask, masks

    def forward_episode(self, sx, sy, qx, qy, cfg: EpisodeCfg, lam_sep=0.5, lam_cent=0.5):
        ways = cfg.ways

        s_pts = self.encode(sx)
        q_pts = self.encode(qx)

        s_mask, s_submasks = self.render_and_merge(s_pts)
        q_mask, q_submasks = self.render_and_merge(q_pts)

        proto_mask = torch.stack([s_mask[sy == c].mean(0) for c in range(ways)])

        s_centroids = s_pts.mean(dim=(1, 2))
        proto_cent = torch.stack([s_centroids[sy == c].mean(0) for c in range(ways)])

        logits_iou = self.alpha * soft_iou(q_mask[:, None], proto_mask[None])

        q_cent = q_pts.mean(dim=(1, 2))
        dists = torch.cdist(q_cent, proto_cent) + 1e-6
        logits_dist = -self.beta * dists

        logits = logits_iou + logits_dist

        prod = s_submasks.prod(dim=1)
        reg_div = prod.mean() * 0.1

        loss_cls = F.cross_entropy(logits, qy)
        reg_cent = lam_cent * centroid_repulsion(proto_cent)

        loss = loss_cls + reg_cent + reg_div
        acc = (logits.argmax(-1) == qy).float().mean()
        return loss, acc, logits


class ProtoNet(nn.Module):
    def __init__(self, backbone_channels, drop_rate, num_pools):
        super().__init__()
        self.backbone = ResNet12(backbone_channels, drop_rate, num_pools)

    def forward_episode(self, sx, sy, qx, qy, cfg):
        s_emb = self.backbone(sx).mean(dim=(-1, -2))
        q_emb = self.backbone(qx).mean(dim=(-1, -2))
        proto = torch.stack([s_emb[sy == c].mean(0) for c in range(cfg.ways)])
        dists = torch.cdist(q_emb, proto)
        logits = -dists
        loss = F.cross_entropy(logits, qy)
        acc = (logits.argmax(-1) == qy).float().mean()
        return loss, acc, logits


class MetaOptNet(nn.Module):
    def __init__(self, backbone_channels, drop_rate, num_pools, ridge_lam=1.0):
        super().__init__()
        self.backbone = ResNet12(backbone_channels, drop_rate, num_pools)

    def forward_episode(self, sx, sy, qx, qy, cfg):
        s_emb = self.backbone(sx).mean(dim=(-1, -2))
        q_emb = self.backbone(qx).mean(dim=(-1, -2))
        proto = torch.stack([s_emb[sy == c].mean(0) for c in range(cfg.ways)])
        logits = -torch.cdist(q_emb, proto)
        loss = F.cross_entropy(logits, qy)
        acc = (logits.argmax(-1) == qy).float().mean()
        return loss, acc, logits


def build_class_index(dataset, classes):
    index = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        if label in classes:
            index[label].append(idx)
    return dict(index)


def sample_episode(class_indices, cfg, dataset, device):
    all_classes = list(class_indices.keys())
    selected = random.sample(all_classes, cfg.ways)
    sup_imgs, sup_labs = [], []
    qry_imgs, qry_labs = [], []

    for new_label, cls in enumerate(selected):
        indices = class_indices[cls]
        sampled = random.sample(indices, cfg.shots + cfg.queries)
        for idx in sampled[:cfg.shots]:
            img, _ = dataset[idx]
            sup_imgs.append(img)
            sup_labs.append(new_label)
        for idx in sampled[cfg.shots:]:
            img, _ = dataset[idx]
            qry_imgs.append(img)
            qry_labs.append(new_label)

    return (
        torch.stack(sup_imgs).to(device),
        torch.tensor(sup_labs, device=device),
        torch.stack(qry_imgs).to(device),
        torch.tensor(qry_labs, device=device),
    )


def train_epoch(model, class_indices, dataset, cfg, optimizer, device, num_episodes=100):
    model.train()
    losses, accs = [], []
    for _ in range(num_episodes):
        sx, sy, qx, qy = sample_episode(class_indices, cfg, dataset, device)
        optimizer.zero_grad()
        loss, acc, _ = model.forward_episode(sx, sy, qx, qy, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


@torch.no_grad()
def evaluate(model, class_indices, dataset, cfg, device, num_episodes=600):
    model.eval()
    accs = []
    for _ in range(num_episodes):
        sx, sy, qx, qy = sample_episode(class_indices, cfg, dataset, device)
        _, acc, _ = model.forward_episode(sx, sy, qx, qy, cfg)
        accs.append(acc.item())
    accs = np.array(accs)
    return accs.mean(), 1.96 * accs.std() / np.sqrt(len(accs))


def maybe_draw_gvnet_polygons(
    model,
    args,
    cfg,
    class_indices,
    dataset,
    device,
    *,
    epoch: int,
    every: int = 1,
    out_dir: str = "./viz"
):
    if args.model != "gvnet":
        return
    if every <= 0 or (epoch % every) != 0:
        return

    os.makedirs(out_dir, exist_ok=True)

    if not hasattr(maybe_draw_gvnet_polygons, "_fixed"):
        sx, sy, _, _ = sample_episode(class_indices, cfg, dataset, device)
        maybe_draw_gvnet_polygons._fixed = (sx.detach().clone(), sy.detach().clone())
    fixed_sx, fixed_sy = maybe_draw_gvnet_polygons._fixed

    def angular_sort_np(poly_xy: np.ndarray) -> np.ndarray:
        c = poly_xy.mean(axis=0, keepdims=True)
        rel = poly_xy - c
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        return poly_xy[np.argsort(ang)]

    def setup_ax(ax, title):
        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, lw=1, c="k")
        ax.axvline(0, lw=1, c="k")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(title)

    model.eval()
    with torch.no_grad():
        sup_pts = model.encode(fixed_sx)
        proto_pts = torch.stack([sup_pts[fixed_sy == c].mean(0) for c in range(cfg.ways)])

    protos_np = proto_pts.cpu().numpy()
    ways, planes, _, _ = protos_np.shape
    palette = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange"]

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    setup_ax(ax, f"GVNet Multi-Plane Prototypes (All Planes) | Epoch {epoch + 1}")

    for w in range(ways):
        color = palette[w % len(palette)]
        for p in range(planes):
            poly = angular_sort_np(protos_np[w, p])
            poly = np.vstack([poly, poly[:1]])
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.08)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=1.5, alpha=0.6)
            ax.scatter(poly[:, 0], poly[:, 1], color=color, s=18, alpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"gvnet_poly_{epoch + 1:03d}.png"), dpi=150)
    plt.close()

    for p in range(planes):
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        setup_ax(ax, f"GVNet Prototypes | Plane {p + 1}/{planes} | Epoch {epoch + 1}")

        for w in range(ways):
            color = palette[w % len(palette)]
            poly = angular_sort_np(protos_np[w, p])
            poly = np.vstack([poly, poly[:1]])
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.12)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2.0, alpha=0.75)
            ax.scatter(poly[:, 0], poly[:, 1], color=color, s=22, alpha=0.9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"gvnet_plane{p + 1}_poly_{epoch + 1:03d}.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gvnet", choices=["gvnet", "protonet", "metaoptnet"])
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--episodes-per-epoch", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=600)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--n-points", type=int, default=8)
    parser.add_argument("--n-planes", type=int, default=3, help="Number of polygon planes per image")
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    print("Loading CIFAR-100 (CIFAR-FS)...")
    try:
        full_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
        full_train_clean = datasets.CIFAR100(root="./data", train=True, download=True, transform=test_transform)
        full_test = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    except:
        print("Dataset download failed or not found. Please ensure ./data exists.")
        return

    train_idx = build_class_index(full_train, TRAIN_CLASSES)
    val_idx = build_class_index(full_train_clean, VAL_CLASSES)
    test_idx = build_class_index(full_test, TEST_CLASSES)

    cfg = EpisodeCfg(ways=args.ways, shots=args.shots, queries=args.queries)

    if args.model == "gvnet":
        model = GVNet(
            n_points=args.n_points,
            n_planes=args.n_planes,
            backbone_channels=(64, 128, 256, 512),
            drop_rate=args.drop_rate
        ).to(device)
    elif args.model == "protonet":
        model = ProtoNet((64, 128, 256, 512), args.drop_rate, 4).to(device)
    else:
        model = MetaOptNet((64, 128, 256, 512), args.drop_rate, 4).to(device)

    print(f"Model: {args.model} | Device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 70], gamma=0.1)

    best_val_acc = 0.0

    print(f"Start Training: {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.set_epoch(epoch) if hasattr(model, "set_epoch") else None

        train_loss, train_acc = train_epoch(
            model, train_idx, full_train, cfg, optimizer, device, args.episodes_per_epoch
        )
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_acc, val_ci = evaluate(model, val_idx, full_train_clean, cfg, device, args.eval_episodes)
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            marker = " ★" if is_best else ""
            print(
                f"Epoch {epoch + 1:3d} | Train: {train_loss:.3f} / {train_acc * 100:.1f}% | "
                f"Val: {val_acc * 100:.2f}±{val_ci * 100:.2f}%{marker}"
            )
        else:
            print(f"Epoch {epoch + 1:3d} | Train: {train_loss:.3f} / {train_acc * 100:.1f}%")

        maybe_draw_gvnet_polygons(model, args, cfg, train_idx, full_train, device, epoch=epoch)


if __name__ == "__main__":
    main()
```
