# -*- coding: utf-8 -*-
"""
Model-internal PID control ("layers predicting layers").

Implements Sec. 3.7 and Supplementary Tables S6 / S7 of the paper, using the same PID
pipeline as the neural analysis (Sec. 2.3-2.5) except that, as stated in the paper, the
second MP-PCA pass on the concatenated sources is dropped for model sources.

Pipeline (one source pair  A, B  and one external target  Y):
  1. Features: spatially (global-average) pooled layer outputs for 6,000 random THINGS images.
  2. Sources:  PCA to min(50, d-1) components  ->  MP-PCA per source (Eq. 1-2)
               ->  rank-based Gaussianisation (probit of (rank-0.5)/n).
  3. Target:   PCA to 50 components (standard PCA, no whitening) -> Gaussianisation.
  4. Joint representation Z = raw column-wise concatenation [S_A, S_B]   (no 2nd MP-PCA).
  5. Gaussian plug-in MI (Eq. 3), signed log-determinants, clipped to 0 on a non-positive sign.
  6. Shuffle bias (Sec. 2.4): Y row-permuted B = 500 times on a subsample of 5,000 rows;
     separate bias for I(S_A;Y), I(S_B;Y) and I(Z;Y);  I_corr = max(0, I - b).
  7. MMI-PID (Eq. 4-6):  R = min(I_A, I_B),  U_r = max(0, I_r - R),
                         Syn = max(0, I_Z - R - U_A - U_B).
  8. Marginal balance = min(I_A, I_B) / max(I_A, I_B)   (Supp. Table S7).
  9. CorNet-S: 30 bootstrap resamples per target; count how often V1-IT is the largest of the
     three pairs (Sec. 3.7).

Targets: VGG-16 fc7, DINOv2 CLS (never the source model itself); VGG-16 sources use
ResNet-18 avgpool instead of fc7 (Supp. Table S6 footnote); DINOv2 sources use fc7 only.

Run
    python internal_pid.py --selftest                 # numpy-only checks of the statistics, ~1 min
    python internal_pid.py --smoke                    # tiny end-to-end run (needs torch + THINGS images)
    python internal_pid.py --img-root <THINGS images> --out-dir <output dir>      # full run

The script prints the reproduced numbers next to the values published in the paper
(Tables S6 / S7 and the rebuttal tables) and writes everything to CSV.

Choices that the paper does not pin down (kept as explicit settings so they can be changed):
  * which 6,000 images (random sample, SEED),
  * layer numbering: DINOv2 blockN = blocks[N] and MobileNetV2 blockN = features[N] (0-based);
    VGG-16 blockN = N-th max-pool output (1-based); SqueezeNet uses the paper's numbering
    (fire2 is the first Fire module, so fire3 = features[4] and fire9 = features[12]),
  * DINOv2 variant (ViT-S/14) and the CLS token as the per-block representation,
  * SqueezeNet version, fc7 taken after its ReLU,
  * the 5,000-row bias subsample is drawn from the 6,000 images.
"""
import os
import sys
import time
import zlib
import argparse
import itertools
import subprocess
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------------------
# Settings (values stated in the paper)
# --------------------------------------------------------------------------------------
IMG_ROOT = "/content/drive/MyDrive/THINGS_images_unzipped"
OUT_DIR = "/content/drive/MyDrive/model_internal_pid"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
N_IMAGES = 6000          # Sec. 3.7 / Supp. Implementation
SEED = 0
N_PCS = 50               # PCA components (31 for MobileNetV2 block4, which has 32 channels)
B_SHUFFLES = 500         # Sec. 2.4
N_SUB = 5000             # bias subsample, Sec. 2.4
N_BOOT = 30              # Sec. 3.7 (CorNet-S bootstrap)
GAUSSIANIZE_TARGET = True
SQUEEZENET_VERSION = "1_0"
BATCH = 128
TOL_NATS = 0.05          # only used to flag reproduced-vs-published agreement in the printout

# DINOv2 blocks used as sources in Supp. Table S6 (0-based block indices)
DINOV2_PAIRS = [(2, 6), (2, 8), (2, 10), (4, 8), (4, 10), (6, 10)]

T_FC7, T_CLS, T_R18 = "VGG16-fc7", "DINOv2-CLS", "ResNet18-avgpool"

# Supp. Table S6: (model, layer A, layer B, targets)
S6_EXPERIMENTS = (
    [("ResNet-18", "layer1", "layer3", [T_FC7, T_CLS]),
     ("VGG-16", "block2", "block4", [T_R18]),
     ("MobileNetV2", "block4", "block10", [T_FC7, T_CLS]),
     ("SqueezeNet", "fire3", "fire9", [T_FC7, T_CLS])]
    + [("DINOv2", f"block{a}", f"block{b}", [T_FC7]) for a, b in DINOV2_PAIRS]
)

# Supp. Table S7: CorNet-S pairs (source names are produced by extract_cornet_s)
S7_PAIRS = [
    ("V1-IT (final)", "V1", "IT_t2"),
    ("V1-V4 (final)", "V1", "V4_t4"),
    ("V4-IT (final)", "V4_t4", "IT_t2"),
    ("V1-V4, t1", "V1", "V4_t1"),
    ("V1-V4, t2", "V1", "V4_t2"),
    ("V1-V4, t3", "V1", "V4_t3"),
    ("V1-V4, t4", "V1", "V4_t4"),
    ("V1-IT, t1", "V1", "IT_t1"),
    ("V1-IT, t2", "V1", "IT_t2"),
]
BOOT_PAIRS = {"V1-IT": ("V1", "IT_t2"), "V1-V4": ("V1", "V4_t4"), "V4-IT": ("V4_t4", "IT_t2")}

# Values published in the paper / rebuttal: (synergy, total joint MI or None).
PUBLISHED_S6 = {
    ("ResNet-18", "layer1", "layer3", T_FC7): (0.447, 7.745),
    ("ResNet-18", "layer1", "layer3", T_CLS): (0.132, 4.552),
    ("VGG-16", "block2", "block4", T_R18): (0.419, 5.351),
    ("MobileNetV2", "block4", "block10", T_FC7): (0.308, 5.035),
    ("MobileNetV2", "block4", "block10", T_CLS): (0.152, 2.973),
    ("SqueezeNet", "fire3", "fire9", T_FC7): (0.420, 6.649),
    ("SqueezeNet", "fire3", "fire9", T_CLS): (0.111, 3.815),
    ("DINOv2", "block2", "block6", T_FC7): (0.673, 6.631),
    ("DINOv2", "block2", "block8", T_FC7): (1.259, None),
    ("DINOv2", "block2", "block10", T_FC7): (1.269, 10.133),
    ("DINOv2", "block4", "block8", T_FC7): (1.638, 8.112),
    ("DINOv2", "block4", "block10", T_FC7): (1.593, None),
    ("DINOv2", "block6", "block10", T_FC7): (1.916, 10.779),
}
# label -> {target: (synergy, total MI, marginal balance)}
PUBLISHED_S7 = {
    "V1-IT (final)": {T_FC7: (0.757, 16.208, 0.180), T_CLS: (0.268, 8.611, 0.192)},
    "V1-V4 (final)": {T_FC7: (0.432, 9.244, 0.337), T_CLS: (0.161, 5.994, 0.290)},
    "V4-IT (final)": {T_FC7: (2.575, 17.970, 0.535), T_CLS: (1.596, 9.887, 0.662)},
    "V1-V4, t1": {T_FC7: (0.263, 6.247, 0.541), T_CLS: (0.010, 4.134, 0.476)},
    "V1-V4, t2": {T_FC7: (0.235, 7.333, 0.436), T_CLS: (0.045, 4.732, 0.390)},
    "V1-V4, t3": {T_FC7: (0.423, 8.172, 0.389), T_CLS: (0.068, 5.236, 0.343)},
    "V1-V4, t4": {T_FC7: (0.432, 9.244, 0.337), T_CLS: (0.161, 5.994, 0.290)},
    "V1-IT, t1": {T_FC7: (0.470, 12.293, 0.241), T_CLS: (0.076, 7.442, 0.222)},
    "V1-IT, t2": {T_FC7: (0.759, 16.208, 0.180), T_CLS: (0.268, 8.611, 0.192)},
}


def rng_for(*tags):
    """Deterministic RNG per experiment (python's hash() is salted per process, so it is not used)."""
    return np.random.default_rng([SEED] + [zlib.crc32(str(t).encode()) for t in tags])


def tqdm_(it, **kw):
    try:
        from tqdm.auto import tqdm
        return tqdm(it, **kw)
    except Exception:
        return it


# --------------------------------------------------------------------------------------
# Statistics (numpy only)
# --------------------------------------------------------------------------------------
def pca_reduce(X, n_max=N_PCS):
    """Standard PCA (centred, not whitened) to min(n_max, d-1) components."""
    X = np.asarray(X, dtype=np.float64)
    k = int(min(n_max, X.shape[1] - 1))
    return PCA(n_components=k, svd_solver="full", random_state=SEED).fit_transform(X)


def mp_pca(X):
    """Marchenko-Pastur PCA, Eq. (1)-(2): sigma^2 = median positive eigenvalue,
    lambda+ = sigma^2 (1 + sqrt(gamma))^2 with gamma = p/n, k = max(1, #{lambda_i > lambda+})."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(0)
    n, p = Xc.shape
    lam, vec = np.linalg.eigh((Xc.T @ Xc) / (n - 1))
    lam, vec = lam[::-1], vec[:, ::-1]
    pos = lam[lam > 0]
    sigma2 = float(np.median(pos)) if pos.size else 1.0
    lam_plus = sigma2 * (1.0 + np.sqrt(p / n)) ** 2
    k = max(1, int(np.sum(lam > lam_plus)))
    return Xc @ vec[:, :k], k


def rank_probit(X):
    """Marginal Gaussianisation (Supp. Eq. S1): probit((rank - 0.5) / n), column-wise."""
    X = np.asarray(X, dtype=np.float64)
    r = rankdata(X, axis=0)          # 1..n, ties averaged
    return ndtri((r - 0.5) / X.shape[0])


def _cov(Z):
    Z = Z - Z.mean(0)
    return (Z.T @ Z) / (len(Z) - 1)


def _mi_from_logdets(sx, ldx, sy, ldy, sc, ldc):
    if sx <= 0 or sy <= 0 or sc <= 0:
        return 0.0
    return max(0.0, 0.5 * (ldx + ldy - ldc))


def gaussian_mi(X, Y):
    """Gaussian plug-in MI, Eq. (3), signed log-determinants of sample covariances,
    clipped to zero if any determinant sign is non-positive."""
    dx = X.shape[1]
    C = _cov(np.hstack([X, Y]))
    sx, ldx = np.linalg.slogdet(C[:dx, :dx])
    sy, ldy = np.linalg.slogdet(C[dx:, dx:])
    sc, ldc = np.linalg.slogdet(C)
    return _mi_from_logdets(sx, ldx, sy, ldy, sc, ldc)


def shuffle_bias(X, Y, perms):
    """Mean MI over row-permutations of Y (Sec. 2.4).  Permuting the rows of Y leaves the
    covariances of X and of Y unchanged; only the cross-covariance block changes, so only
    that block and the joint log-determinant are recomputed per permutation (exactly equal to
    recomputing the full covariance)."""
    n, dx = X.shape
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Cx = (Xc.T @ Xc) / (n - 1)
    Cy = (Yc.T @ Yc) / (n - 1)
    sx, ldx = np.linalg.slogdet(Cx)
    sy, ldy = np.linalg.slogdet(Cy)
    vals = np.empty(len(perms))
    for i, p in enumerate(perms):
        cross = (Xc.T @ Yc[p]) / (n - 1)
        C = np.block([[Cx, cross], [cross.T, Cy]])
        sc, ldc = np.linalg.slogdet(C)
        vals[i] = _mi_from_logdets(sx, ldx, sy, ldy, sc, ldc)
    return float(vals.mean())


def pid_two_sources(S1, S2, Y, rng, B=B_SHUFFLES, n_sub=N_SUB):
    """MMI-PID of two sources about Y with shuffle bias correction; joint = [S1, S2] (no 2nd MP-PCA)."""
    n = len(Y)
    J = np.hstack([S1, S2])
    sub = rng.choice(n, n_sub, replace=False) if n > n_sub else np.arange(n)
    perms = [rng.permutation(len(sub)) for _ in range(B)]       # same permutations for all three terms
    out = {}
    for name, S in (("i1", S1), ("i2", S2), ("ij", J)):
        raw = gaussian_mi(S, Y)
        bias = shuffle_bias(S[sub], Y[sub], perms)
        out[name] = max(0.0, raw - bias)
        out[name + "_raw"], out[name + "_bias"] = raw, bias
    R = min(out["i1"], out["i2"])
    u1, u2 = max(0.0, out["i1"] - R), max(0.0, out["i2"] - R)
    syn = max(0.0, out["ij"] - R - u1 - u2)
    lo, hi = sorted([out["i1"], out["i2"]])
    out.update(red=R, u1=u1, u2=u2, syn=syn, mi_joint=out["ij"],
               balance=(lo / hi if hi > 1e-9 else np.nan))
    return out


def prepare_source(X):
    """PCA(50) -> MP-PCA -> rank-probit.  Returns (S, k_mp)."""
    S, k = mp_pca(pca_reduce(X))
    return rank_probit(S), k


def prepare_target(X):
    Y = pca_reduce(X)
    return rank_probit(Y) if GAUSSIANIZE_TARGET else Y


def bootstrap_pair_ranking(sources, pairs, Y, rng, n_boot=N_BOOT, B=B_SHUFFLES, n_sub=N_SUB):
    """Resample images with replacement; recompute synergy (with bias correction) for each pair;
    record the largest pair (Sec. 3.7)."""
    n = len(Y)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        syn = {name: pid_two_sources(sources[a][idx], sources[c][idx], Y[idx], rng, B, n_sub)["syn"]
               for name, (a, c) in pairs.items()}
        rows.append(dict(replicate=b, winner=max(syn, key=syn.get), **{f"syn_{k}": v for k, v in syn.items()}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Image loading and feature extraction (needs torch / torchvision)
# --------------------------------------------------------------------------------------
def load_paths(img_root, n_images):
    paths = []
    for root, _, files in os.walk(img_root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(root, fn))
    paths.sort()                                   # os.walk order is filesystem dependent
    if not paths:
        raise FileNotFoundError(f"no images found under {img_root}")
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(paths), min(n_images, len(paths)), replace=False)
    return [paths[i] for i in sorted(sel)]


def preload_images(paths, size=224):
    """Resize(256) -> CenterCrop(224), stored as uint8 (N, 224, 224, 3); converted per batch."""
    import torchvision.transforms as T
    from PIL import Image
    from multiprocessing.pool import ThreadPool
    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(size)])

    def load_one(p):
        try:
            with Image.open(p) as im:
                return np.asarray(tf(im.convert("RGB")), dtype=np.uint8)
        except Exception:
            return None

    with ThreadPool(16) as pool:
        arrs = list(tqdm_(pool.imap(load_one, paths), total=len(paths), desc="decode images"))
    bad = [p for p, a in zip(paths, arrs) if a is None]
    if bad:
        print(f"WARNING: {len(bad)} unreadable images dropped, e.g. {bad[:3]}")
    return np.stack([a for a in arrs if a is not None])


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _batches(images_u8, norm, bs=BATCH):
    import torch
    dev = _device()
    for s in range(0, len(images_u8), bs):
        x = torch.from_numpy(images_u8[s:s + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        yield norm(x)


def _imagenet_norm():
    import torchvision.transforms as T
    return T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _gap(o):
    return o.mean(dim=(2, 3)) if o.dim() == 4 else o.flatten(1)


def _collect(model, images_u8, hooks, desc, bs=BATCH, forward=None):
    """hooks: {key: (module, fn)}.  Returns {key: float64 array (N, d)}."""
    import torch
    buf = defaultdict(list)
    handles = [mod.register_forward_hook(lambda m, i, o, key=key, fn=fn: buf[key].append(fn(o).detach().float().cpu().clone()))
               for key, (mod, fn) in hooks.items()]
    out = []
    with torch.no_grad():
        for x in tqdm_(_batches(images_u8, _imagenet_norm(), bs), total=(len(images_u8) + bs - 1) // bs, desc=desc):
            r = (forward or model)(x)
            if forward is not None:
                out.append(r.detach().float().cpu())
    for h in handles:
        h.remove()
    res = {k: torch.cat(v, 0).numpy().astype(np.float64) for k, v in buf.items()}
    if out:
        res["__output__"] = torch.cat(out, 0).numpy().astype(np.float64)
    return res


def extract_resnet18(images):
    import torchvision.models as tvm
    m = tvm.resnet18(weights="DEFAULT").to(_device()).eval()
    f = _collect(m, images, {f"layer{k}": (getattr(m, f"layer{k}"), _gap) for k in (1, 2, 3, 4)}, "ResNet-18")
    f["avgpool"] = f["layer4"]          # global-average-pooled layer4 == avgpool output
    return f


def extract_vgg16(images):
    import torch.nn as nn
    import torchvision.models as tvm
    m = tvm.vgg16(weights="DEFAULT").to(_device()).eval()
    pools = [mod for mod in m.features if isinstance(mod, nn.MaxPool2d)]        # block1..block5 outputs
    hooks = {f"block{k + 1}": (p, _gap) for k, p in enumerate(pools)}
    hooks["fc7"] = (m.classifier[4], lambda o: o.flatten(1))                    # ReLU after the second FC layer
    return _collect(m, images, hooks, "VGG-16")


def extract_mobilenetv2(images):
    import torchvision.models as tvm
    m = tvm.mobilenet_v2(weights="DEFAULT").to(_device()).eval()
    return _collect(m, images, {f"block{k}": (m.features[k], _gap) for k in (4, 10)}, "MobileNetV2")


def extract_squeezenet(images):
    import torchvision.models as tvm
    m = getattr(tvm, f"squeezenet{SQUEEZENET_VERSION}")(weights="DEFAULT").to(_device()).eval()
    # paper numbering: conv1, maxpool, fire2, fire3, fire4, [maxpool], fire5..fire8, [maxpool], fire9
    # -> features[4] is fire3 and features[12] is fire9 in both squeezenet1_0 and squeezenet1_1
    return _collect(m, images, {"fire3": (m.features[4], _gap), "fire9": (m.features[12], _gap)}, "SqueezeNet")


def extract_dinov2(images):
    import torch
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True).to(_device()).eval()
    cls = lambda o: (o[:, 0, :] if o.dim() == 3 else o)
    blocks = sorted({b for pair in DINOV2_PAIRS for b in pair})
    f = _collect(m, images, {f"block{b}": (m.blocks[b], cls) for b in blocks}, "DINOv2", forward=m)
    f["cls_final"] = f.pop("__output__")
    return f


def extract_cornet_s(images):
    """V1, V4 at its 4 recurrent timesteps, IT at its 2.  Per-timestep outputs are read from each
    CORblock_S's `output` (Identity) module, which fires once per timestep."""
    import torch
    try:
        from cornet import cornet_s
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/dicarlolab/CORnet"])
        from cornet import cornet_s
    dev = _device()
    net = cornet_s(pretrained=True, map_location=dev)
    core = (net.module if hasattr(net, "module") else net).to(dev).eval()
    times = {"V4": 4, "IT": 2}
    calls = defaultdict(list)
    buf = defaultdict(list)

    def mk(name):
        def h(m, i, o):
            calls[name].append(_gap(o).detach().float().cpu().clone())
        return h

    handles = [core.V1.register_forward_hook(mk("V1")),
               core.V4.output.register_forward_hook(mk("V4")),
               core.IT.output.register_forward_hook(mk("IT"))]
    with torch.no_grad():
        for x in tqdm_(_batches(images, _imagenet_norm(), 128), total=(len(images) + 127) // 128, desc="CORnet-S"):
            calls.clear()
            core(x)
            assert len(calls["V4"]) == times["V4"] and len(calls["IT"]) == times["IT"], \
                f"unexpected timestep counts: V4={len(calls['V4'])}, IT={len(calls['IT'])}"
            buf["V1"].append(calls["V1"][0])
            for area, T_ in times.items():
                for t in range(T_):
                    buf[f"{area}_t{t + 1}"].append(calls[area][t])
    for h in handles:
        h.remove()
    return {k: torch.cat(v, 0).numpy().astype(np.float64) for k, v in buf.items()}


EXTRACTORS = {"ResNet-18": extract_resnet18, "VGG-16": extract_vgg16, "MobileNetV2": extract_mobilenetv2,
              "SqueezeNet": extract_squeezenet, "DINOv2": extract_dinov2, "CorNet-S": extract_cornet_s}


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------
def run_all(args):
    os.makedirs(args.out_dir, exist_ok=True)
    feats = {}
    images = None
    for name, fn in EXTRACTORS.items():
        cache = os.path.join(args.out_dir, f"features_{name}_{args.n_images}.npz")
        if os.path.exists(cache):
            feats[name] = dict(np.load(cache))
            print(f"loaded cached features: {cache}")
            continue
        if images is None:
            images = preload_images(load_paths(args.img_root, args.n_images))
            print(f"images: {images.shape}")
        print(f"=== extracting {name} ===")
        feats[name] = fn(images)
        np.savez(cache, **feats[name])
    n = len(next(iter(feats["ResNet-18"].values())))
    print(f"N = {n} images")

    targets = {T_FC7: feats["VGG-16"]["fc7"], T_CLS: feats["DINOv2"]["cls_final"], T_R18: feats["ResNet-18"]["avgpool"]}
    Y = {k: prepare_target(v) for k, v in targets.items()}

    src_cache = {}

    def source(model, layer):
        key = (model, layer)
        if key not in src_cache:
            src_cache[key] = prepare_source(feats[model][layer])
        return src_cache[key]

    rows_s6 = []
    for model, la, lb, tgts in S6_EXPERIMENTS:
        (SA, kA), (SB, kB) = source(model, la), source(model, lb)
        for tg in tgts:
            r = pid_two_sources(SA, SB, Y[tg], rng_for("S6", model, la, lb, tg), args.shuffles, args.n_sub)
            pub = PUBLISHED_S6.get((model, la, lb, tg), (None, None))
            rows_s6.append(dict(model=model, layer_A=la, layer_B=lb, target=tg, n=n, k_A=kA, k_B=kB,
                                MI_A=r["i1"], MI_B=r["i2"], MI_joint=r["mi_joint"], redundancy=r["red"],
                                unique_A=r["u1"], unique_B=r["u2"], synergy=r["syn"], marginal_balance=r["balance"],
                                published_synergy=pub[0], published_total_MI=pub[1]))
            print(f"S6 {model:12s} {la:>7s}-{lb:<7s} {tg:16s} syn={r['syn']:.3f} (paper {pub[0]})  MI={r['mi_joint']:.3f} (paper {pub[1]})")
    df6 = pd.DataFrame(rows_s6)
    df6.to_csv(os.path.join(args.out_dir, "table_S6_reproduced.csv"), index=False)

    cor = {k: source("CorNet-S", k) for k in ("V1", "V4_t1", "V4_t2", "V4_t3", "V4_t4", "IT_t1", "IT_t2")}
    rows_s7 = []
    for label, a, b in S7_PAIRS:
        for tg in (T_FC7, T_CLS):
            r = pid_two_sources(cor[a][0], cor[b][0], Y[tg], rng_for("S7", label, tg), args.shuffles, args.n_sub)
            ps, pm, pb = PUBLISHED_S7[label][tg]
            rows_s7.append(dict(config=label, target=tg, n=n, k_A=cor[a][1], k_B=cor[b][1], MI_A=r["i1"], MI_B=r["i2"],
                                MI_joint=r["mi_joint"], synergy=r["syn"], marginal_balance=r["balance"],
                                published_synergy=ps, published_total_MI=pm, published_balance=pb))
            print(f"S7 {label:14s} {tg:11s} syn={r['syn']:.3f} (paper {ps})  MI={r['mi_joint']:.3f} (paper {pm})  bal={r['balance']:.3f} (paper {pb})")
    df7 = pd.DataFrame(rows_s7)
    df7.to_csv(os.path.join(args.out_dir, "table_S7_reproduced.csv"), index=False)

    boot_rows = []
    for tg in (T_FC7, T_CLS):
        cs = {k: v[0] for k, v in cor.items()}
        bt = bootstrap_pair_ranking(cs, BOOT_PAIRS, Y[tg], rng_for("boot", tg), args.boot, args.shuffles, args.n_sub)
        bt.insert(0, "target", tg)
        boot_rows.append(bt)
        wins = bt["winner"].value_counts().to_dict()
        print(f"bootstrap [{tg}]: largest pair over {len(bt)} resamples -> {wins}; V1-IT largest in {wins.get('V1-IT', 0)}/{len(bt)}")
    pd.concat(boot_rows).to_csv(os.path.join(args.out_dir, "cornet_bootstrap.csv"), index=False)

    # ---- reproduced vs published ----
    chk = []
    for d, key in ((df6, "S6"), (df7, "S7")):
        for _, r in d.iterrows():
            if pd.notna(r["published_synergy"]):
                chk.append((key, r.get("model", "CorNet-S"), r.get("config", f"{r.get('layer_A', '')}-{r.get('layer_B', '')}"),
                            r["target"], "synergy", r["synergy"], r["published_synergy"]))
            if pd.notna(r["published_total_MI"]):
                chk.append((key, r.get("model", "CorNet-S"), r.get("config", f"{r.get('layer_A', '')}-{r.get('layer_B', '')}"),
                            r["target"], "total_MI", r["MI_joint"], r["published_total_MI"]))
    cmp_df = pd.DataFrame(chk, columns=["table", "model", "config", "target", "quantity", "reproduced", "published"])
    cmp_df["abs_diff"] = (cmp_df["reproduced"] - cmp_df["published"]).abs()
    cmp_df.to_csv(os.path.join(args.out_dir, "comparison_with_paper.csv"), index=False)
    syn = cmp_df[cmp_df.quantity == "synergy"]
    print("\n=== reproduced vs published ===")
    print(cmp_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"synergy within {TOL_NATS} nats of the published value: {(syn.abs_diff <= TOL_NATS).sum()}/{len(syn)}; "
          f"median |diff| = {syn.abs_diff.median():.3f} nats")
    print(f"outputs in {args.out_dir}")


# --------------------------------------------------------------------------------------
# Self-test (numpy only)
# --------------------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(1)
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

    # 1. rank-probit gives N(0,1) marginals and matches the formula
    X = rng.gamma(2.0, size=(4000, 3))
    Z = rank_probit(X)
    check("rank_probit marginals ~ N(0,1)", np.all(np.abs(Z.mean(0)) < 0.02) and np.all(np.abs(Z.std(0) - 1) < 0.03))
    r0 = rankdata(X[:, 0])
    check("rank_probit equals probit((r-0.5)/n)", np.allclose(Z[:, 0], ndtri((r0 - 0.5) / 4000)))

    # 2. MP-PCA recovers planted components
    n, p, k_true = 6000, 50, 6
    L = rng.normal(size=(k_true, p)) * 3.0
    Xs = rng.normal(size=(n, k_true)) @ L + rng.normal(size=(n, p))
    _, k = mp_pca(Xs)
    check("mp_pca recovers planted rank", k == k_true, f"(k={k}, planted={k_true})")
    _, k1 = mp_pca(rng.normal(size=(n, p)))
    check("mp_pca floor of 1 on pure noise", k1 >= 1, f"(k={k1})")

    # 3. fast shuffle bias == brute force with the same permutations
    S = rng.normal(size=(800, 7)); Yb = S[:, :3] + rng.normal(size=(800, 3))
    perms = [rng.permutation(800) for _ in range(20)]
    fast = shuffle_bias(S, Yb, perms)
    brute = float(np.mean([gaussian_mi(S, Yb[p_]) for p_ in perms]))
    check("shuffle_bias equals brute-force recomputation", abs(fast - brute) < 1e-9, f"({fast:.6f} vs {brute:.6f})")

    # 4. plug-in MI against the analytic Gaussian value
    rho = 0.6
    n = 200000
    a = rng.normal(size=n); b = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    check("gaussian_mi matches -0.5 log(1-rho^2)", abs(gaussian_mi(a[:, None], b[:, None]) + 0.5 * np.log(1 - rho ** 2)) < 5e-3)

    # 5. shuffle bias is ~ d_x d_y / (2 n) for independent data and removes most of the plug-in bias
    n, dx, dy = 600, 10, 10
    Xi, Yi = rng.normal(size=(n, dx)), rng.normal(size=(n, dy))
    raw = gaussian_mi(Xi, Yi)
    bias = shuffle_bias(Xi, Yi, [rng.permutation(n) for _ in range(200)])
    check("bias estimate ~ raw MI of independent data", abs(raw - bias) < 0.5 * bias, f"(raw={raw:.3f}, bias={bias:.3f}, dx*dy/2n={dx * dy / (2 * n):.3f})")

    # 6. MMI-PID: Y = S1 + S2 + eps, independent unit-variance sources, eps variance 1
    #    I(S1;Y) = 0.5 ln(3/2), I(S1,S2;Y) = 0.5 ln 3  =>  R = I1, U = 0, Syn = 0.5 ln 3 - 0.5 ln 1.5
    n = 400000
    s1, s2 = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    yy = s1 + s2 + rng.normal(size=(n, 1))
    r = pid_two_sources(s1, s2, yy, rng, B=20, n_sub=20000)
    exp_syn = 0.5 * np.log(3.0) - 0.5 * np.log(1.5)
    check("PID synergy of additive-Gaussian system", abs(r["syn"] - exp_syn) < 0.01 and r["u1"] < 0.01 and r["u2"] < 0.01,
          f"(syn={r['syn']:.3f}, analytic={exp_syn:.3f})")

    # 7. redundant copies carry no synergy
    c = rng.normal(size=(n, 1)); yc = c + rng.normal(size=(n, 1))
    r = pid_two_sources(c, c + 1e-6 * rng.normal(size=(n, 1)), yc, rng, B=20, n_sub=20000)
    check("identical sources -> ~0 synergy, ~0 unique", r["syn"] < 0.01 and r["u1"] < 0.01 and r["u2"] < 0.01, f"(syn={r['syn']:.4f})")

    # 8. 31-component rule for a 32-channel layer (MobileNetV2 block4)
    check("PCA keeps min(50, d-1): 32-d -> 31, 64-d -> 50", pca_reduce(rng.normal(size=(500, 32))).shape[1] == 31
          and pca_reduce(rng.normal(size=(500, 64))).shape[1] == 50)

    print("\nALL SELF-TESTS PASSED" if ok else "\nSELF-TEST FAILURES")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img-root", default=IMG_ROOT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument("--shuffles", type=int, default=B_SHUFFLES)
    ap.add_argument("--n-sub", type=int, default=N_SUB)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--smoke", action="store_true", help="300 images, 20 shuffles, 3 bootstrap resamples (plumbing check only)")
    ap.add_argument("--selftest", action="store_true", help="numpy-only checks of the statistics")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.smoke:
        args.n_images, args.shuffles, args.n_sub, args.boot = 300, 20, 300, 3
        args.out_dir = os.path.join(args.out_dir, "smoke")
    t0 = time.time()
    run_all(args)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
