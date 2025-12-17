import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ----------------------------------------------------------
# ✅ Temporal Augmentation
# ----------------------------------------------------------
def temporal_augment(seq, p=0.5):
    T, _, H, W = seq.shape

    # Frame dropout
    if random.random() < p:
        drop_ratio = random.uniform(0.05, 0.15)
        keep = max(1, int(T * (1 - drop_ratio)))
        idx = sorted(random.sample(range(T), keep))
        seq = seq[idx]

    # Speed jitter (slow/fast motion)
    if random.random() < p:
        factor = random.uniform(0.9, 1.1)
        new_T = max(1, int(T * factor))
        seq = F.interpolate(
            seq.unsqueeze(0),
            size=(new_T, H, W),
            mode="trilinear",
            align_corners=False
        ).squeeze(0)

    # Random temporal crop
    T = seq.shape[0]
    if random.random() < p and T > 8:
        crop_len = random.randint(int(0.7*T), T)
        start = random.randint(0, T - crop_len)
        seq = seq[start:start + crop_len]

    # Reverse sequence (ED ↔ ES)
    if random.random() < 0.3:
        seq = torch.flip(seq, dims=[0])

    return seq

# ----------------------------------------------------------
# ✅ Spatial Augmentation
# ----------------------------------------------------------
def spatial_augment(seq, p=0.5):
    T, _, H, W = seq.shape

    # Random crop
    if random.random() < p:
        r = random.uniform(0.85, 0.95)
        new_H, new_W = int(H*r), int(W*r)
        top = random.randint(0, H-new_H)
        left = random.randint(0, W-new_W)
        seq = seq[:, :, top:top+new_H, left:left+new_W]

    # Rotation
    if random.random() < p:
        angle = random.uniform(-10, 10)
        seq = torch.stack([TF.rotate(f, angle) for f in seq])

    # Gaussian noise
    if random.random() < p:
        seq += torch.randn_like(seq) * 0.03

    return seq
