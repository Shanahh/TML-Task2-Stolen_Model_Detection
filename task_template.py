"""
Stolen Model Detection - low-FPR optimized version.

Methods implemented:
  1. Layerwise parameter cosine/sign similarity
  2. BatchNorm statistic similarity
  3. Clean functional similarity on CIFAR-100 test/non-target/train-main subsets
  4. Target-train-subset memorization/alignment signal
  5. Same confident target mistake signal
  6. Target high-leakage probes: low-margin + high-entropy samples
  7. Synthetic/OOD soft-label similarity for knockoff/data-free extraction
  8. Augmentation/transform sensitivity similarity
  9. MixMatch-style augmentation averaging + sharpening signal
 10. Linear CKA representation similarity on selected layers
 11. Optional FGSM boundary/adversarial functional similarity
 12. Optional Jacobian/input-gradient fingerprinting

Writes:
  submission.csv
  submission_features.csv
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset, TensorDataset
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
    finite = np.isfinite(values)
    if not finite.any():
        values = np.zeros_like(values)
    else:
        med = np.nanmedian(values[finite])
        values = np.nan_to_num(values, nan=med, posinf=np.nanmax(values[finite]), neginf=np.nanmin(values[finite]))
    order = np.argsort(values, kind="mergesort")
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


def sharpen_probs(p: torch.Tensor, temperature: float = 0.5, eps: float = 1e-12) -> torch.Tensor:
    q = torch.clamp(p, eps, 1.0) ** (1.0 / temperature)
    return q / q.sum(dim=1, keepdim=True).clamp_min(eps)


# -----------------------------
# Weight / BN similarity
# -----------------------------

def weight_features(target_sd: Dict[str, torch.Tensor], suspect_sd: Dict[str, torch.Tensor]) -> Dict[str, float]:
    cosines, signs, bn_cos, conv_cos, fc_cos, early_cos, mid_cos, late_cos = [], [], [], [], [], [], [], []

    for name, tw in target_sd.items():
        if name not in suspect_sd:
            continue
        sw = suspect_sd[name]
        if not torch.is_floating_point(tw) or tw.shape != sw.shape:
            continue

        c = safe_cosine(tw, sw)
        cosines.append(c)
        if tw.numel() > 100:
            signs.append(sign_agreement(tw, sw))
        if "bn" in name or "downsample.1" in name or "running_mean" in name or "running_var" in name:
            bn_cos.append(c)
        if "conv" in name:
            conv_cos.append(c)
        if name.startswith("fc"):
            fc_cos.append(c)
        if name.startswith("conv1") or name.startswith("bn1") or name.startswith("layer1"):
            early_cos.append(c)
        if name.startswith("layer2") or name.startswith("layer3"):
            mid_cos.append(c)
        if name.startswith("layer4") or name.startswith("fc"):
            late_cos.append(c)

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    return {
        "w_cos_all": mean(cosines),
        "w_cos_conv": mean(conv_cos),
        "w_cos_bn": mean(bn_cos),
        "w_cos_fc": mean(fc_cos),
        "w_cos_early": mean(early_cos),
        "w_cos_mid": mean(mid_cos),
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
        logits_all.append(model(x).detach().cpu())
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

    kl_ts = F.kl_div(s_logp, t_prob, reduction="batchmean").item()
    kl_st = F.kl_div(t_logp, s_prob, reduction="batchmean").item()
    js = 0.5 * (kl_ts + kl_st)

    t_conf = t_prob.max(dim=1).values
    s_conf = s_prob.max(dim=1).values
    conf_corr = np.corrcoef(t_conf.numpy(), s_conf.numpy())[0, 1]
    if not np.isfinite(conf_corr):
        conf_corr = 0.0

    t_rank = torch.argsort(t, dim=1, descending=True)[:, :5]
    s_rank = torch.argsort(s, dim=1, descending=True)[:, :5]
    top5_overlap = [len(set(a.tolist()).intersection(set(b.tolist()))) / 5.0 for a, b in zip(t_rank, s_rank)]

    return {
        f"{prefix}_logit_cos": float(logit_cos),
        f"{prefix}_top1_agree": float(top1_agree),
        f"{prefix}_same_wrong": float(same_wrong),
        f"{prefix}_neg_js": float(-js),
        f"{prefix}_conf_corr": float(conf_corr),
        f"{prefix}_top5_overlap": float(np.mean(top5_overlap)),
    }


def masked_logit_features(t_logits: torch.Tensor, s_logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, prefix: str) -> Dict[str, float]:
    mask = mask.bool().cpu()
    if mask.sum().item() < 8:
        return {
            f"{prefix}_logit_cos": 0.0,
            f"{prefix}_top1_agree": 0.0,
            f"{prefix}_same_wrong": 0.0,
            f"{prefix}_neg_js": -100.0,
            f"{prefix}_conf_corr": 0.0,
            f"{prefix}_top5_overlap": 0.0,
            f"{prefix}_n": float(mask.sum().item()),
        }
    out = logit_similarity_features(t_logits[mask], s_logits[mask], y[mask], prefix)
    out[f"{prefix}_n"] = float(mask.sum().item())
    return out


def target_probe_masks(t_logits: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    prob = F.softmax(t_logits.float(), dim=1)
    conf, pred = prob.max(dim=1)
    top2 = torch.topk(prob, k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=1)

    wrong = pred != y.long()
    confident_wrong = wrong & (conf >= torch.quantile(conf, 0.70))
    very_conf_wrong = wrong & (conf >= torch.quantile(conf, 0.85))
    low_margin = margin <= torch.quantile(margin, 0.25)
    high_entropy = entropy >= torch.quantile(entropy, 0.75)
    high_leakage = low_margin | high_entropy

    return {
        "confwrong": confident_wrong.cpu(),
        "vconfwrong": very_conf_wrong.cpu(),
        "lowmargin": low_margin.cpu(),
        "highentropy": high_entropy.cpu(),
        "leakage": high_leakage.cpu(),
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
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    xty = x.T @ y
    hsic = (xty ** 2).sum()
    x_norm = ((x.T @ x) ** 2).sum().sqrt()
    y_norm = ((y.T @ y) ** 2).sum().sqrt()
    denom = x_norm * y_norm
    if denom.item() < eps:
        return 0.0
    return float((hsic / denom).item())


def cka_features(target_feats: Dict[str, torch.Tensor], suspect_feats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out, vals, early, mid, late = {}, [], [], [], []
    for name in target_feats.keys():
        val = linear_cka(target_feats[name], suspect_feats[name])
        key = f"cka_{name.replace('.', '_')}"
        out[key] = val
        vals.append(val)
        if name in ["layer1", "layer2"]:
            early.append(val)
        if name in ["layer2", "layer3"]:
            mid.append(val)
        if name in ["layer4", "avgpool"]:
            late.append(val)
    out["cka_mean"] = float(np.mean(vals)) if vals else 0.0
    out["cka_early"] = float(np.mean(early)) if early else 0.0
    out["cka_mid"] = float(np.mean(mid)) if mid else 0.0
    out["cka_late"] = float(np.mean(late)) if late else 0.0
    return out


# -----------------------------
# Transform / MixMatch / OOD
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
    if mode == "cutout":
        out = x.clone()
        h0, w0 = 10, 10
        out[:, :, h0:h0 + 10, w0:w0 + 10] = 0.0
        return out
    raise ValueError(mode)


@torch.no_grad()
def transform_sensitivity_features(target: nn.Module, suspect: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> Dict[str, float]:
    modes = ["hflip", "shift_right", "shift_down", "noise", "brightness", "cutout"]
    sims, agrees = [], []
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


@torch.no_grad()
def mixmatch_style_features(target: nn.Module, suspect: nn.Module, loader: DataLoader, device: torch.device, max_batches: int, temperature: float) -> Dict[str, float]:
    cos_vals, neg_js_vals, top1_vals = [], [], []
    modes = ["hflip", "shift_right", "noise"]
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        t_probs, s_probs = [], []
        for mode in ["identity"] + modes:
            xx = x if mode == "identity" else transform_batch(x, mode)
            t_probs.append(F.softmax(target(xx).float(), dim=1))
            s_probs.append(F.softmax(suspect(xx).float(), dim=1))
        tp = sharpen_probs(torch.stack(t_probs).mean(dim=0), temperature=temperature)
        sp = sharpen_probs(torch.stack(s_probs).mean(dim=0), temperature=temperature)
        cos_vals.append(F.cosine_similarity(tp, sp, dim=1).cpu())
        js = 0.5 * (
            F.kl_div(torch.log(sp.clamp_min(1e-12)), tp, reduction="none").sum(dim=1)
            + F.kl_div(torch.log(tp.clamp_min(1e-12)), sp, reduction="none").sum(dim=1)
        )
        neg_js_vals.append((-js).cpu())
        top1_vals.append((tp.argmax(dim=1) == sp.argmax(dim=1)).float().cpu())
    if not cos_vals:
        return {"mixmatch_cos": 0.0, "mixmatch_neg_js": 0.0, "mixmatch_top1": 0.0}
    return {
        "mixmatch_cos": float(torch.cat(cos_vals).mean().item()),
        "mixmatch_neg_js": float(torch.cat(neg_js_vals).mean().item()),
        "mixmatch_top1": float(torch.cat(top1_vals).mean().item()),
    }


def make_ood_loader(num: int, batch_size: int, workers: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    # Three synthetic probe families in normalized space: uniform images, gaussian noise, and coarse color blocks.
    n1 = num // 3
    n2 = num // 3
    n3 = num - n1 - n2
    u = torch.rand(n1, 3, 32, 32, generator=g)
    gnoise = torch.rand(n2, 3, 32, 32, generator=g).normal_(0.5, 0.25).clamp(0, 1)
    blocks = torch.rand(n3, 3, 4, 4, generator=g)
    blocks = F.interpolate(blocks, size=(32, 32), mode="nearest")
    x = torch.cat([u, gnoise, blocks], dim=0)
    x = normalize(x)
    y = torch.zeros(len(x), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


# -----------------------------
# Optional FGSM / Jacobian
# -----------------------------

def fgsm_boundary_features(target: nn.Module, suspect: nn.Module, loader: DataLoader, device: torch.device, max_batches: int, eps: float) -> Dict[str, float]:
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


def jacobian_features(target: nn.Module, suspect: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> Dict[str, float]:
    cos_vals, sign_vals = [], []
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            pseudo = target(x).argmax(dim=1)

        xt = x.detach().clone().requires_grad_(True)
        xs = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            lt = F.cross_entropy(target(xt), pseudo)
            ls = F.cross_entropy(suspect(xs), pseudo)
            gt = torch.autograd.grad(lt, xt)[0].flatten(1)
            gs = torch.autograd.grad(ls, xs)[0].flatten(1)
        cos_vals.append(F.cosine_similarity(gt, gs, dim=1).detach().cpu())
        sign_vals.append((torch.sign(gt) == torch.sign(gs)).float().mean(dim=1).detach().cpu())
    if not cos_vals:
        return {"jacobian_cos": 0.0, "jacobian_sign": 0.0}
    return {
        "jacobian_cos": float(torch.cat(cos_vals).mean().item()),
        "jacobian_sign": float(torch.cat(sign_vals).mean().item()),
    }


# -----------------------------
# Low-FPR specialist scoring
# -----------------------------

def mean_rank(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    valid = [c for c in cols if c in df.columns]
    if not valid:
        return np.zeros(len(df), dtype=np.float64)
    acc = np.zeros(len(df), dtype=np.float64)
    for c in valid:
        acc += rank01(df[c].values, higher_is_more_stolen=True)
    return acc / len(valid)


def build_final_scores(df: pd.DataFrame) -> np.ndarray:
    """Low-FPR specialist ranking tuned for TPR@5%FPR."""
    n = len(df)
    zeros = pd.Series(np.zeros(n))

    weight_rank = mean_rank(df, [
        "w_cos_all", "w_cos_conv", "w_cos_bn", "w_cos_early", "w_cos_mid", "w_cos_late", "w_sign"
    ])
    bn_rank = mean_rank(df, ["w_cos_bn"])
    early_mid_cka = mean_rank(df, ["cka_early", "cka_mid", "cka_layer1", "cka_layer2", "cka_layer3"])
    late_cka = mean_rank(df, ["cka_late", "cka_layer4", "cka_avgpool"])
    cka_rank = mean_rank(df, ["cka_mean", "cka_early", "cka_mid", "cka_late"])

    clean_dark = mean_rank(df, [
        "test_logit_cos", "test_neg_js", "test_top5_overlap", "main_neg_js", "nonmain_neg_js"
    ])
    mistake_rank = mean_rank(df, [
        "test_same_wrong", "main_same_wrong", "nonmain_same_wrong",
        "confwrong_top1_agree", "vconfwrong_top1_agree", "confwrong_logit_cos", "confwrong_neg_js"
    ])
    leakage_rank = mean_rank(df, [
        "lowmargin_logit_cos", "lowmargin_neg_js", "lowmargin_top5_overlap",
        "highentropy_logit_cos", "highentropy_neg_js", "leakage_logit_cos", "leakage_neg_js"
    ])
    ood_rank = mean_rank(df, ["ood_logit_cos", "ood_neg_js", "ood_top5_overlap", "ood_conf_corr"])
    transform_rank = mean_rank(df, [
        "transform_delta_cos", "transform_stability_agree",
        "mixmatch_cos", "mixmatch_neg_js", "mixmatch_top1"
    ])
    mem_rank = mean_rank(df, [
        "mem_loss_gap_similarity", "mem_conf_gap_similarity",
        "mem_acc_gap_similarity", "mem_agree_gap_similarity"
    ])
    fgsm_rank = mean_rank(df, ["fgsm_logit_cos", "fgsm_top1_agree", "fgsm_neg_js"])
    jac_rank = mean_rank(df, ["jacobian_cos", "jacobian_sign"])

    # More specialized, less averaged.
    direct_score = (
        0.50 * weight_rank
      + 0.35 * bn_rank
      + 0.15 * cka_rank
    )

    finetune_score = (
        0.35 * early_mid_cka
      + 0.25 * bn_rank
      + 0.25 * mem_rank
      + 0.15 * transform_rank
    )

    distilled_score = (
        0.28 * leakage_rank
      + 0.22 * ood_rank
      + 0.22 * mistake_rank
      + 0.18 * clean_dark
      + 0.10 * late_cka
    )

    boundary_score = (
        0.35 * leakage_rank
      + 0.25 * fgsm_rank
      + 0.20 * jac_rank
      + 0.20 * mistake_rank
    )

    dataset_score = (
        0.60 * mem_rank
      + 0.25 * early_mid_cka
      + 0.15 * transform_rank
    )

    # Conservative consensus terms. Helps low-FPR ranking.
    copy_consensus = np.sqrt(np.maximum(direct_score, 1e-9) * np.maximum(finetune_score, 1e-9))
    behavior_consensus = np.sqrt(np.maximum(distilled_score, 1e-9) * np.maximum(boundary_score, 1e-9))
    dataset_consensus = np.sqrt(np.maximum(dataset_score, 1e-9) * np.maximum(early_mid_cka, 1e-9))

    final = np.maximum.reduce([
        direct_score,
        finetune_score,
        0.90 * distilled_score + 0.10 * behavior_consensus,
        0.90 * boundary_score + 0.10 * behavior_consensus,
        0.90 * dataset_score + 0.10 * dataset_consensus,
        0.70 * copy_consensus + 0.30 * direct_score,
    ])

    # Hard override only for genuinely obvious descendants.
    w_all = df.get("w_cos_all", zeros).values
    w_bn = df.get("w_cos_bn", zeros).values
    w_early = df.get("w_cos_early", zeros).values
    w_late = df.get("w_cos_late", zeros).values
    cka_mean_raw = df.get("cka_mean", zeros).values
    transform_raw = df.get("transform_delta_cos", zeros).values

    obvious_direct = (
        (w_all > 0.985) |
        ((w_bn > 0.985) & (cka_mean_raw > 0.90)) |
        ((w_early > 0.975) & (w_late > 0.925))
    )

    obvious_finetune = (
        (w_early > 0.945) &
        (cka_mean_raw > 0.86) &
        (transform_raw > 0.65)
    )

    final[obvious_direct] = np.maximum(final[obvious_direct], 0.995)
    final[obvious_finetune] = np.maximum(final[obvious_finetune], 0.970)

    # Penalize clean-agreement-only models:
    # likely independent same-distribution models.
    clean_top1 = rank01(df.get("test_top1_agree", zeros).values)
    clean_top5 = rank01(df.get("test_top5_overlap", zeros).values)
    clean_signal = 0.5 * clean_top1 + 0.5 * clean_top5

    weak_lineage = 1.0 - np.maximum.reduce([
        weight_rank,
        bn_rank,
        cka_rank,
        mistake_rank,
        mem_rank,
    ])

    clean_only_penalty = clean_signal * weak_lineage
    final = final - 0.12 * clean_only_penalty

    # Penalize OOD-only spikes unless supported by leakage/mistake/CKA.
    ood_only_penalty = ood_rank * (1.0 - np.maximum.reduce([
        leakage_rank,
        mistake_rank,
        late_cka,
        cka_rank,
    ]))
    final = final - 0.06 * ood_only_penalty

    # Small boost when multiple independent families agree.
    multi_signal = np.mean(np.stack([
        direct_score,
        finetune_score,
        distilled_score,
        boundary_score,
        dataset_score,
    ], axis=0), axis=0)

    final = 0.90 * final + 0.10 * multi_signal

    return rank01(final, higher_is_more_stolen=True)


# -----------------------------
# Main
# -----------------------------

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
    parser.add_argument("--ood-samples", type=int, default=768)
    parser.add_argument("--transform-batches", type=int, default=4)
    parser.add_argument("--mixmatch-batches", type=int, default=4)
    parser.add_argument("--fgsm-batches", type=int, default=2)
    parser.add_argument("--jacobian-batches", type=int, default=1)
    parser.add_argument("--mixmatch-temperature", type=float, default=0.5)
    parser.add_argument("--no-fgsm", action="store_true", help="Disable FGSM similarity for speed.")
    parser.add_argument("--no-jacobian", action="store_true", help="Disable Jacobian similarity for speed.")
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
    ood_loader = make_ood_loader(args.ood_samples, args.batch_size, args.workers, args.seed + 999)

    print("Loading target model...")
    target = load_resnet18_cifar100(target_path, device)
    target_sd = load_file(str(target_path), device="cpu")

    print("Precomputing target logits/features...")
    target_test_logits, y_test = collect_logits(target, test_loader, device)
    target_main_logits, y_main = collect_logits(target, main_loader, device)
    target_nonmain_logits, y_nonmain = collect_logits(target, nonmain_loader, device)
    target_ood_logits, y_ood = collect_logits(target, ood_loader, device)

    target_masks = target_probe_masks(target_test_logits, y_test)

    t_main_loss, t_main_conf, t_main_acc = loss_confidence_stats(target_main_logits, y_main)
    t_nonmain_loss, t_nonmain_conf, t_nonmain_acc = loss_confidence_stats(target_nonmain_logits, y_nonmain)
    target_loss_gap = t_nonmain_loss - t_main_loss
    target_conf_gap = t_main_conf - t_nonmain_conf
    target_acc_gap = t_main_acc - t_nonmain_acc

    target_agree_gap_ref = 0.0  # target agrees with itself equally; suspects get measured relative to target below.

    layer_names = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    target_feats = collect_features(target, cka_loader, device, layer_names)

    suspect_paths = []
    for i in range(360):
        p = suspects_dir / f"suspect_{i:03d}.safetensors"
        if not p.exists():
            raise FileNotFoundError(f"Missing suspect model: {p}")
        suspect_paths.append(p)

    rows = []
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
        s_ood_logits, _ = collect_logits(suspect, ood_loader, device)

        row.update(logit_similarity_features(target_test_logits, s_test_logits, y_test, "test"))
        row.update(logit_similarity_features(target_main_logits, s_main_logits, y_main, "main"))
        row.update(logit_similarity_features(target_nonmain_logits, s_nonmain_logits, y_nonmain, "nonmain"))
        row.update(logit_similarity_features(target_ood_logits, s_ood_logits, y_ood, "ood"))

        for name, mask in target_masks.items():
            row.update(masked_logit_features(target_test_logits, s_test_logits, y_test, mask, name))

        s_main_loss, s_main_conf, s_main_acc = loss_confidence_stats(s_main_logits, y_main)
        s_nonmain_loss, s_nonmain_conf, s_nonmain_acc = loss_confidence_stats(s_nonmain_logits, y_nonmain)
        suspect_loss_gap = s_nonmain_loss - s_main_loss
        suspect_conf_gap = s_main_conf - s_nonmain_conf
        suspect_acc_gap = s_main_acc - s_nonmain_acc

        row["mem_loss_gap_similarity"] = -abs(suspect_loss_gap - target_loss_gap)
        row["mem_conf_gap_similarity"] = -abs(suspect_conf_gap - target_conf_gap)
        row["mem_acc_gap_similarity"] = -abs(suspect_acc_gap - target_acc_gap)

        target_main_pred = target_main_logits.argmax(dim=1)
        target_nonmain_pred = target_nonmain_logits.argmax(dim=1)
        s_main_pred = s_main_logits.argmax(dim=1)
        s_nonmain_pred = s_nonmain_logits.argmax(dim=1)
        agree_main = (target_main_pred == s_main_pred).float().mean().item()
        agree_nonmain = (target_nonmain_pred == s_nonmain_pred).float().mean().item()
        # Higher means unusually aligned on the victim's exact train subset.
        row["mem_agree_gap_similarity"] = agree_main - agree_nonmain - target_agree_gap_ref

        suspect_feats = collect_features(suspect, cka_loader, device, layer_names)
        row.update(cka_features(target_feats, suspect_feats))

        row.update(transform_sensitivity_features(target, suspect, test_loader, device, max_batches=args.transform_batches))
        row.update(mixmatch_style_features(target, suspect, test_loader, device, max_batches=args.mixmatch_batches, temperature=args.mixmatch_temperature))

        if args.no_fgsm:
            row.update({"fgsm_logit_cos": 0.0, "fgsm_top1_agree": 0.0, "fgsm_neg_js": 0.0})
        else:
            row.update(fgsm_boundary_features(target, suspect, test_loader, device, max_batches=args.fgsm_batches, eps=0.03))

        if args.no_jacobian:
            row.update({"jacobian_cos": 0.0, "jacobian_sign": 0.0})
        else:
            row.update(jacobian_features(target, suspect, test_loader, device, max_batches=args.jacobian_batches))

        rows.append(row)
        del suspect
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    features_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    features_df["score"] = build_final_scores(features_df)

    out_path = Path(args.output)
    submission = features_df[["id", "score"]]
    submission.to_csv(out_path, index=False)
    debug_path = out_path.with_name(out_path.stem + "_features.csv")
    features_df.to_csv(debug_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote debug features to {debug_path}")
    print(submission.head())
    print("Top 20 candidates:")
    print(features_df.sort_values("score", ascending=False)[["id", "score", "w_cos_all", "w_cos_bn", "cka_mean", "test_same_wrong", "confwrong_top1_agree", "leakage_neg_js", "ood_neg_js"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
