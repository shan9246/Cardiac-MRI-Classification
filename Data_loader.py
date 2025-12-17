import os, re
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from augmentation import temporal_augment, spatial_augment


# ----------------------------------------------------------
# ✅ ACDC Dataset Loader
# ----------------------------------------------------------
class ACDC_Dataset(Dataset):
    def __init__(self, root, size=(224,224), augment=False):
        self.size = size
        self.augment = augment
        self.data = []

        valid_labels = {"NOR","MINF","DCM","HCM","RV"}  # Allowed ACDC classes
        cases = sorted([d for d in os.listdir(root) if d.startswith("patient")])

        for d in cases:
            patient_dir = os.path.join(root, d)
            info_file = os.path.join(patient_dir, "Info.cfg")
            label = "Unknown"

            # ✅ Read diagnostic group
            if os.path.exists(info_file):
                for line in open(info_file):
                    match = re.match(r"Group:(\w+)", line.replace(" ", ""))
                    if match:
                        label = match.group(1)
                        break

            # ✅ Only include valid cardiac labels
            if label not in valid_labels:
                print(f"[WARN] Skipping {patient_dir} — label={label}")
                continue

            # ✅ Path to 4D MRI
            nii_file = os.path.join(patient_dir, f"{d}_4d.nii.gz")
            if os.path.exists(nii_file):
                self.data.append((nii_file, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        img = nib.load(path).get_fdata()  # H, W, Z, T
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # ✅ Middle slice sequence
        mid = img.shape[2] // 2
        seq = img[:, :, mid, :]  # H,W,T
        seq = np.moveaxis(seq, -1, 0)[:, None, :, :]  # T,1,H,W
        seq = torch.tensor(seq, dtype=torch.float32)

        # ✅ Apply augmentations only during training
        if self.augment:
            seq = temporal_augment(seq)
            seq = spatial_augment(seq)

        # ✅ Resize frames
        seq = F.interpolate(seq, size=self.size, mode='bilinear', align_corners=False)

        return seq, label


# ----------------------------------------------------------
# ✅ Pad Collate — Handles variable sequence lengths
# ----------------------------------------------------------
def pad_collate(batch):
    seqs, labels = zip(*batch)
    maxT = max([s.shape[0] for s in seqs])
    padded = []

    for s in seqs:
        if s.shape[0] < maxT:
            pad = (0,0,0,0,0,0,0, maxT - s.shape[0])
            s = F.pad(s, pad)
        padded.append(s)

    return torch.stack(padded), list(labels)
