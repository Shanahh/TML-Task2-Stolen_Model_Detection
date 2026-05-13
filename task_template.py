"""
Methods implemented:
  1. Layerwise parameter cosine similarity
  2. BatchNorm statistic similarity
  3. Clean functional similarity on CIFAR-100 test/non-target/train-main subsets
  4. Target-train-subset memorization/alignment signal
  5. Augmentation/transform sensitivity similarity
  6. Linear CKA representation similarity on selected layers
  7. Optional FGSM boundary/adversarial functional similarity

"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


# -----------------------------
# Reproducibility / device
# -----------------------------

def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


# -----------------------------
# Model
# -----------------------------

def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def load_resnet18_cifar100(path: Path, device: torch.device) -> nn.Module:
    state_dict = load_file(str(path), device="cpu")
    model = make_model()
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model


# -----------------------------
# Data
# -----------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def build_datasets(data_root: Path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    test = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=transform)
    return train, test


def make_subset_loader(dataset, indices: List[int], batch_size: int, workers: int, shuffle: bool = False):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_train_main_indices(path: Path) -> List[int]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ["indices", "idx", "train_main_idx", "train_idx"]:
            if key in data:
                return list(map(int, data[key]))
        for value in data.values():
            if isinstance(value, list):
                return list(map(int, value))
        raise ValueError(f"Could not find an index list in {path}")
    return list(map(int, data))


# -----------------------------
# Numeric helpers
# -----------------------------

def safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().float().flatten().cpu()
    b = b.detach().float().flatten().cpu()
    denom = (a.norm() * b.norm()).item()
    if denom < eps:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def sign_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().cpu()
    b = b.detach().flatten().cpu()
    return float((torch.sign(a) == torch.sign(b)).float().mean().item())


def rank01(values: np.ndarray, higher_is_more_stolen: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=np.nanmedian(values), posinf=np.nanmax(values), neginf=np.nanmin(values))
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) > 1:
        ranks = ranks / (len(values) - 1)
    if not higher_is_more_stolen:
        ranks = 1.0 - ranks
    return ranks


def robust_minmax(values: np.ndarray, lo_q: float = 1, hi_q: float = 99) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(values, [lo_q, hi_q])
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0, 1)


# -----------------------------
# Weight / BN similarity
# -----------------------------

def weight_features(target_sd: Dict[str, torch.Tensor], suspect_sd: Dict[str, torch.Tensor]) -> Dict[str, float]:
    cosines = []
    signs = []
    bn_cos = []
    conv_cos = []
    fc_cos = []
    early_cos = []
    late_cos = []

    for name, tw in target_sd.items():
        if name not in suspect_sd:
            continue
        sw = suspect_sd[name]
        if not torch.is_floating_point(tw):
            continue
        if tw.shape != sw.shape:
            continue

        c = safe_cosine(tw, sw)
        cosines.append(c)

        if tw.numel() > 100:
            signs.append(sign_agreement(tw, sw))

        if "bn" in name or "downsample.1" in name:
            bn_cos.append(c)
        if "conv" in name:
            conv_cos.append(c)
        if name.startswith("fc"):
            fc_cos.append(c)
        if name.startswith("conv1") or name.startswith("bn1") or name.startswith("layer1"):
            early_cos.append(c)
        if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
            late_cos.append(c)

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    return {
        "w_cos_all": mean(cosines),
        "w_cos_conv": mean(conv_cos),
        "w_cos_bn": mean(bn_cos),
        "w_cos_fc": mean(fc_cos),
        "w_cos_early": mean(early_cos),
        "w_cos_late": mean(late_cos),
        "w_sign": mean(signs),
    }


# -----------------------------
# Functional similarity
# -----------------------------

@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_all, y_all = [], []
    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu()
        logits_all.append(logits)
        y_all.append(y.cpu())
    return torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0)


def logit_similarity_features(t_logits: torch.Tensor, s_logits: torch.Tensor, y: torch.Tensor, prefix: str) -> Dict[str, float]:
    t = t_logits.float()
    s = s_logits.float()
    y = y.long()

    t_prob = F.softmax(t, dim=1)
    s_prob = F.softmax(s, dim=1)
    t_logp = F.log_softmax(t, dim=1)
    s_logp = F.log_softmax(s, dim=1)

    t_pred = t.argmax(dim=1)
    s_pred = s.argmax(dim=1)

    tc = t - t.mean(dim=1, keepdim=True)
    sc = s - s.mean(dim=1, keepdim=True)

    logit_cos = F.cosine_similarity(tc, sc, dim=1).mean().item()
    top1_agree = (t_pred == s_pred).float().mean().item()
    same_wrong = ((t_pred == s_pred) & (t_pred != y)).float().mean().item()

    kl_ts = F.kl_div(s_logp, t_prob, reduction="batchmean").item()  # KL(target || suspect), lower is closer
    kl_st = F.kl_div(t_logp, s_prob, reduction="batchmean").item()
    js = 0.5 * (kl_ts + kl_st)

    t_conf = t_prob.max(dim=1).values
    s_conf = s_prob.max(dim=1).values
    conf_corr = np.corrcoef(t_conf.numpy(), s_conf.numpy())[0, 1]
    if not np.isfinite(conf_corr):
        conf_corr = 0.0

    t_rank = torch.argsort(t, dim=1, descending=True)[:, :5]
    s_rank = torch.argsort(s, dim=1, descending=True)[:, :5]
    top5_overlap = []
    for a, b in zip(t_rank, s_rank):
        top5_overlap.append(len(set(a.tolist()).intersection(set(b.tolist()))) / 5.0)

    return {
        f"{prefix}_logit_cos": float(logit_cos),
        f"{prefix}_top1_agree": float(top1_agree),
        f"{prefix}_same_wrong": float(same_wrong),
        f"{prefix}_neg_js": float(-js),
        f"{prefix}_conf_corr": float(conf_corr),
        f"{prefix}_top5_overlap": float(np.mean(top5_overlap)),
    }


def loss_confidence_stats(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float]:
    loss = F.cross_entropy(logits.float(), y.long(), reduction="mean").item()
    prob = F.softmax(logits.float(), dim=1)
    conf = prob.max(dim=1).values.mean().item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return float(loss), float(conf), float(acc)


# -----------------------------
# Representation similarity: linear CKA
# -----------------------------

class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.handles = []
        modules = dict(model.named_modules())
        for name in layer_names:
            if name not in modules:
                raise KeyError(f"Layer not found: {name}")
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, inp, out):
            z = out.detach()
            if z.ndim == 4:
                z = F.adaptive_avg_pool2d(z, 1).flatten(1)
            else:
                z = z.flatten(1)
            self.features[name] = z.cpu()
        return fn

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def collect_features(model: nn.Module, loader: DataLoader, device: torch.device, layer_names: List[str], max_batches: int = None) -> Dict[str, torch.Tensor]:
    ext = FeatureExtractor(model, layer_names)
    collected = {name: [] for name in layer_names}
    try:
        for bi, (x, _) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            _ = model(x)
            for name in layer_names:
                collected[name].append(ext.features[name])
    finally:
        ext.close()
    return {name: torch.cat(parts, dim=0).float() for name, parts in collected.items()}


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    # x: [n, dx], y: [n, dy]
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.T @ y
    hsic = (xty ** 2).sum()
    x_norm = ((x.T @ x) ** 2).sum().sqrt()
    y_norm = ((y.T @ y) ** 2).sum().sqrt()
    denom = x_norm * y_norm
    if denom.item() < eps:
        return 0.0
    return float((hsic / denom).item())


def cka_features(target_feats: Dict[str, torch.Tensor], suspect_feats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    vals = []
    for name in target_feats.keys():
        val = linear_cka(target_feats[name], suspect_feats[name])
        out[f"cka_{name.replace('.', '_')}"] = val
        vals.append(val)
    out["cka_mean"] = float(np.mean(vals)) if vals else 0.0
    return out


# -----------------------------
# Transform sensitivity
# -----------------------------

def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR100_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR100_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def transform_batch(x: torch.Tensor, mode: str) -> torch.Tensor:
    # x is normalized. Transform in normalized/image space depending on operation.
    if mode == "hflip":
        return torch.flip(x, dims=[3])
    if mode == "shift_right":
        return torch.roll(x, shifts=2, dims=3)
    if mode == "shift_down":
        return torch.roll(x, shifts=2, dims=2)
    if mode == "noise":
        return x + 0.08 * torch.randn_like(x)
    if mode == "brightness":
        u = denormalize(x).clamp(0, 1)
        u = (u + 0.08).clamp(0, 1)
        return normalize(u)
    raise ValueError(mode)


@torch.no_grad()
def transform_sensitivity_features(target: nn.Module, suspect: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> Dict[str, float]:
    modes = ["hflip", "shift_right", "shift_down", "noise", "brightness"]
    sims = []
    agrees = []
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        t0 = target(x)
        s0 = suspect(x)
        for mode in modes:
            xt = transform_batch(x, mode)
            t1 = target(xt)
            s1 = suspect(xt)
            dt = (t1 - t0).float()
            ds = (s1 - s0).float()
            sims.append(F.cosine_similarity(dt, ds, dim=1).detach().cpu())
            agrees.append(((t1.argmax(dim=1) == t0.argmax(dim=1)) == (s1.argmax(dim=1) == s0.argmax(dim=1))).float().detach().cpu())
    if not sims:
        return {"transform_delta_cos": 0.0, "transform_stability_agree": 0.0}
    return {
        "transform_delta_cos": float(torch.cat(sims).mean().item()),
        "transform_stability_agree": float(torch.cat(agrees).mean().item()),
    }


# -----------------------------
# Optional FGSM boundary similarity
# -----------------------------

def fgsm_boundary_features(target: nn.Module, suspect: nn.Module, loader: DataLoader, device: torch.device, max_batches: int, eps: float) -> Dict[str, float]:
    target.eval()
    suspect.eval()
    logit_cos, top1_agree, neg_js = [], [], []

    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        x.requires_grad_(True)
        with torch.enable_grad():
            logits = target(x)
            pseudo_y = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, pseudo_y)
            grad = torch.autograd.grad(loss, x)[0]
            x_adv = (x + eps * grad.sign()).detach()

        with torch.no_grad():
            t = target(x_adv).float().cpu()
            s = suspect(x_adv).float().cpu()
            y_dummy = t.argmax(dim=1)
            feats = logit_similarity_features(t, s, y_dummy, "fgsm")
            logit_cos.append(feats["fgsm_logit_cos"])
            top1_agree.append(feats["fgsm_top1_agree"])
            neg_js.append(feats["fgsm_neg_js"])

    return {
        "fgsm_logit_cos": float(np.mean(logit_cos)) if logit_cos else 0.0,
        "fgsm_top1_agree": float(np.mean(top1_agree)) if top1_agree else 0.0,
        "fgsm_neg_js": float(np.mean(neg_js)) if neg_js else 0.0,
    }


# -----------------------------
# Main scoring
# -----------------------------

def build_final_scores(df: pd.DataFrame) -> np.ndarray:
    # Rank-aggregate. Every selected feature is oriented so higher = more stolen-like.
    groups = {
        "weight": [
            "w_cos_all", "w_cos_conv", "w_cos_bn", "w_cos_fc", "w_cos_early", "w_cos_late", "w_sign",
        ],
        "repr": [
            "cka_mean", "cka_layer1", "cka_layer2", "cka_layer3", "cka_layer4", "cka_avgpool",
        ],
        "functional": [
            "test_logit_cos", "test_top1_agree", "test_same_wrong", "test_neg_js", "test_conf_corr", "test_top5_overlap",
            "main_logit_cos", "main_top1_agree", "main_same_wrong", "main_neg_js", "main_conf_corr", "main_top5_overlap",
            "nonmain_logit_cos", "nonmain_top1_agree", "nonmain_same_wrong", "nonmain_neg_js", "nonmain_conf_corr", "nonmain_top5_overlap",
        ],
        "memorization": [
            "mem_loss_gap_similarity", "mem_conf_gap_similarity", "mem_acc_gap_similarity",
        ],
        "transform": [
            "transform_delta_cos", "transform_stability_agree",
        ],
        "fgsm": [
            "fgsm_logit_cos", "fgsm_top1_agree", "fgsm_neg_js",
        ],
    }

    group_weights = {
        "weight": 0.25,
        "repr": 0.25,
        "functional": 0.22,
        "memorization": 0.12,
        "transform": 0.10,
        "fgsm": 0.06,
    }

    n = len(df)
    final = np.zeros(n, dtype=np.float64)
    used_weight = 0.0

    for group, cols in groups.items():
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue
        group_score = np.zeros(n, dtype=np.float64)
        for c in valid_cols:
            group_score += rank01(df[c].values, higher_is_more_stolen=True)
        group_score /= len(valid_cols)
        final += group_weights[group] * group_score
        used_weight += group_weights[group]

    if used_weight > 0:
        final /= used_weight

    high_weight = rank01(df.get("w_cos_all", pd.Series(np.zeros(n))).values)
    high_cka = rank01(df.get("cka_mean", pd.Series(np.zeros(n))).values)
    high_func = rank01(df.get("test_neg_js", pd.Series(np.zeros(n))).values)
    consensus = (high_weight * high_cka * high_func) ** (1.0 / 3.0)
    final = 0.85 * final + 0.15 * consensus

    return robust_minmax(final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="Project root containing target_model/ and suspect_models/.")
    parser.add_argument("--data-root", type=str, default="./dataset/cifar100")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num-test", type=int, default=2048)
    parser.add_argument("--num-main", type=int, default=2048)
    parser.add_argument("--num-nonmain", type=int, default=2048)
    parser.add_argument("--cka-samples", type=int, default=512)
    parser.add_argument("--transform-batches", type=int, default=4)
    parser.add_argument("--fgsm-batches", type=int, default=2)
    parser.add_argument("--no-fgsm", action="store_true", help="Disable FGSM similarity for speed.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    root = Path(args.root)
    target_path = root / "target_model" / "weights.safetensors"
    train_idx_path = root / "target_model" / "train_main_idx.json"
    suspects_dir = root / "suspect_models"

    if not target_path.exists():
        raise FileNotFoundError(target_path)
    if not train_idx_path.exists():
        raise FileNotFoundError(train_idx_path)
    if not suspects_dir.exists():
        raise FileNotFoundError(suspects_dir)

    print(f"Using device: {device}")
    print("Loading CIFAR-100...")
    train_ds, test_ds = build_datasets(Path(args.data_root))

    main_indices = load_train_main_indices(train_idx_path)
    main_set = set(main_indices)
    all_train_indices = list(range(len(train_ds)))
    nonmain_indices = [i for i in all_train_indices if i not in main_set]

    rng = np.random.default_rng(args.seed)
    test_indices = rng.choice(len(test_ds), size=min(args.num_test, len(test_ds)), replace=False).tolist()
    main_probe = rng.choice(main_indices, size=min(args.num_main, len(main_indices)), replace=False).tolist()
    nonmain_probe = rng.choice(nonmain_indices, size=min(args.num_nonmain, len(nonmain_indices)), replace=False).tolist()
    cka_probe = rng.choice(test_indices, size=min(args.cka_samples, len(test_indices)), replace=False).tolist()

    test_loader = make_subset_loader(test_ds, test_indices, args.batch_size, args.workers)
    main_loader = make_subset_loader(train_ds, main_probe, args.batch_size, args.workers)
    nonmain_loader = make_subset_loader(train_ds, nonmain_probe, args.batch_size, args.workers)
    cka_loader = make_subset_loader(test_ds, cka_probe, args.batch_size, args.workers)

    print("Loading target model...")
    target = load_resnet18_cifar100(target_path, device)
    target_sd = load_file(str(target_path), device="cpu")

    print("Precomputing target logits/features...")
    target_test_logits, y_test = collect_logits(target, test_loader, device)
    target_main_logits, y_main = collect_logits(target, main_loader, device)
    target_nonmain_logits, y_nonmain = collect_logits(target, nonmain_loader, device)

    t_main_loss, t_main_conf, t_main_acc = loss_confidence_stats(target_main_logits, y_main)
    t_nonmain_loss, t_nonmain_conf, t_nonmain_acc = loss_confidence_stats(target_nonmain_logits, y_nonmain)
    target_loss_gap = t_nonmain_loss - t_main_loss
    target_conf_gap = t_main_conf - t_nonmain_conf
    target_acc_gap = t_main_acc - t_nonmain_acc

    layer_names = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    target_feats = collect_features(target, cka_loader, device, layer_names)

    rows = []
    suspect_paths = []
    for i in range(360):
        p = suspects_dir / f"suspect_{i:03d}.safetensors"
        if not p.exists():
            raise FileNotFoundError(f"Missing suspect model: {p}")
        suspect_paths.append(p)

    for model_id, suspect_path in enumerate(suspect_paths):
        print(f"[{model_id:03d}/359] scoring {suspect_path.name}")
        row = {"id": model_id}

        suspect_sd = load_file(str(suspect_path), device="cpu")
        row.update(weight_features(target_sd, suspect_sd))

        suspect = make_model()
        suspect.load_state_dict(suspect_sd, strict=True)
        suspect.eval().to(device)

        s_test_logits, _ = collect_logits(suspect, test_loader, device)
        s_main_logits, _ = collect_logits(suspect, main_loader, device)
        s_nonmain_logits, _ = collect_logits(suspect, nonmain_loader, device)

        row.update(logit_similarity_features(target_test_logits, s_test_logits, y_test, "test"))
        row.update(logit_similarity_features(target_main_logits, s_main_logits, y_main, "main"))
        row.update(logit_similarity_features(target_nonmain_logits, s_nonmain_logits, y_nonmain, "nonmain"))

        s_main_loss, s_main_conf, s_main_acc = loss_confidence_stats(s_main_logits, y_main)
        s_nonmain_loss, s_nonmain_conf, s_nonmain_acc = loss_confidence_stats(s_nonmain_logits, y_nonmain)
        suspect_loss_gap = s_nonmain_loss - s_main_loss
        suspect_conf_gap = s_main_conf - s_nonmain_conf
        suspect_acc_gap = s_main_acc - s_nonmain_acc

        # Similarity to target's exact train-subset trace. Higher is closer.
        row["mem_loss_gap_similarity"] = -abs(suspect_loss_gap - target_loss_gap)
        row["mem_conf_gap_similarity"] = -abs(suspect_conf_gap - target_conf_gap)
        row["mem_acc_gap_similarity"] = -abs(suspect_acc_gap - target_acc_gap)

        suspect_feats = collect_features(suspect, cka_loader, device, layer_names)
        row.update(cka_features(target_feats, suspect_feats))

        row.update(transform_sensitivity_features(
            target, suspect, test_loader, device, max_batches=args.transform_batches
        ))

        if args.no_fgsm:
            row.update({"fgsm_logit_cos": 0.0, "fgsm_top1_agree": 0.0, "fgsm_neg_js": 0.0})
        else:
            row.update(fgsm_boundary_features(
                target, suspect, test_loader, device, max_batches=args.fgsm_batches, eps=0.03
            ))

        rows.append(row)

        del suspect
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    features_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    scores = build_final_scores(features_df)
    features_df["score"] = scores

    out_path = Path(args.output)
    submission = features_df[["id", "score"]]
    submission.to_csv(out_path, index=False)

    debug_path = out_path.with_name(out_path.stem + "_features.csv")
    features_df.to_csv(debug_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote debug features to {debug_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
