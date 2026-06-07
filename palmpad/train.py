"""
Train PalmPad on a 4090 GPU.

Training protocol (following the CHI 2025 paper):
  - Leave-one-user-out cross-validation  OR  random 80/20 split
  - Optimizer : AdamW
  - LR        : 1e-4 with cosine annealing
  - Loss       : cross-entropy (balanced class weights for imbalanced touch/no-touch)
  - Mixed precision : bf16 (4090 native support)
  - torch.compile   : enabled for extra throughput

Usage:
  python train.py --processed_root processed/ --epochs 50 --batch_size 256
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

from model import PalmPadModel
from dataset import PalmPadDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    full = PalmPadDataset(
        args.processed_root,
        time_steps=args.time_steps,
        frame_interval=args.frame_interval,
        augment=True,
    )
    n = len(full)
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Disable augment for val split (Dataset wraps same object — use a second loader)
    val_ds_clean = PalmPadDataset(
        args.processed_root,
        time_steps=args.time_steps,
        frame_interval=args.frame_interval,
        augment=False,
    )
    val_indices = val_ds.indices
    val_subset = torch.utils.data.Subset(val_ds_clean, val_indices)

    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_subset, batch_size=args.batch_size * 2,
                               shuffle=False, **loader_kwargs)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for palm, index, flow, labels in loader:
        palm  = palm.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)
        flow  = flow.to(device, non_blocking=True)
        logits = model(palm, index, flow)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds) * 100
    f1  = f1_score(all_labels, all_preds, average="macro") * 100
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--time_steps",     type=int,   default=2)
    parser.add_argument("--frame_interval", type=int,   default=2,
                        help="Frame gap between time steps (2=1/60s at 120fps)")
    parser.add_argument("--lstm_hidden",    type=int,   default=512)
    parser.add_argument("--workers",        type=int,   default=8)
    parser.add_argument("--ckpt_dir",       default="checkpoints")
    parser.add_argument("--log_dir",        default="runs")
    parser.add_argument("--compile",        action="store_true", default=True,
                        help="Use torch.compile (requires PyTorch 2+)")
    args = parser.parse_args()

    set_seed()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    writer = SummaryWriter(args.log_dir)

    train_loader, val_loader = build_loaders(args)
    print(f"Train samples: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    model = PalmPadModel(
        time_steps=args.time_steps,
        lstm_hidden=args.lstm_hidden,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    # Compute class weights from training labels for imbalanced touch/no-touch
    labels_all = np.array([
        int(train_loader.dataset[i][-1]) for i in range(len(train_loader.dataset))
    ])
    counts = np.bincount(labels_all, minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()   # AMP scaler (bf16 doesn't need it but kept for fp16 compat)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (palm, index, flow, labels) in enumerate(train_loader):
            palm   = palm.to(device, non_blocking=True)
            index  = index.to(device, non_blocking=True)
            flow   = flow.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(palm, index, flow)
                loss   = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        acc, f1 = evaluate(model, val_loader, device)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Acc/val",    acc,       epoch)
        writer.add_scalar("F1/val",     f1,        epoch)

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"val_acc={acc:.1f}%  val_f1={f1:.1f}%")

        if f1 > best_f1:
            best_f1 = f1
            ckpt = os.path.join(args.ckpt_dir, "best.pt")
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_f1": f1, "args": vars(args)}, ckpt)
            print(f"  ✓ Saved best checkpoint (f1={f1:.1f}%)")

    writer.close()
    print(f"\nTraining done. Best val F1: {best_f1:.1f}%")


if __name__ == "__main__":
    main()
