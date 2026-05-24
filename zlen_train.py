import os
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from zlen_core import (
    DEVICE, IMG_SIZE, LR, N_ITER,
    W_SPA, W_EXP, W_COL, W_NOL, W_TVA,
    ZLENModel,
    zlen_total_loss,
    tensor_to_pil,
    plot_loss_curve,
)

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

class LowLightDataset(Dataset):
    def __init__(self, root_dir, img_size=IMG_SIZE, augment=True):
        self.paths = sorted([
            p for p in Path(root_dir).rglob('*')
            if p.suffix.lower() in IMG_EXTENSIONS
        ])

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"Tidak ada gambar ditemukan di '{root_dir}'.\n"
                f"Ekstensi yang didukung: {IMG_EXTENSIONS}"
            )

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

        print(f"[Dataset] Ditemukan {len(self.paths)} gambar di '{root_dir}'")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


def train_with_dataset(args):
    dataset = LowLightDataset(
        root_dir  = args.data_dir,
        img_size  = args.img_size,
        augment   = True,
    )

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.workers,
        pin_memory  = True,
        drop_last   = True,
    )

    print(f"[INFO] Batch size : {args.batch_size}")
    print(f"[INFO] Jumlah batch per epoch: {len(loader)}")

    model = ZLENModel(n_iter=N_ITER).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameter trainable: {n_params:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = 1e-5,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs,
        eta_min = 1e-6,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, 'zlen_best.pth')
    history_path    = os.path.join(args.save_dir, 'zlen_loss.png')

    loss_history = []
    best_loss    = float('inf')

    print(f"\n{'='*65}")
    print(f"  ZLEN Dataset Training")
    print(f"  Epochs={args.epochs} | LR={args.lr} | BatchSize={args.batch_size}")
    print(f"  Wspa={W_SPA} Wexp={W_EXP} Wcol={W_COL} Wnol={W_NOL} WtvA={W_TVA}")
    print(f"{'='*65}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_total = 0.0
        epoch_parts = {'spa': 0.0, 'exp': 0.0, 'col': 0.0,
                       'nol': 0.0, 'tva': 0.0}

        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(DEVICE)

            optimizer.zero_grad()
            enhanced, alpha_maps, _ = model(imgs)

            total_loss, loss_dict = zlen_total_loss(enhanced, imgs, alpha_maps)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            epoch_total += loss_dict['total']
            for k in epoch_parts:
                epoch_parts[k] += loss_dict[k]

        scheduler.step()

        n_batch = len(loader)
        avg_total = epoch_total / n_batch
        avg_parts = {k: v / n_batch for k, v in epoch_parts.items()}
        loss_history.append(avg_total)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch:4d}/{args.epochs}] "
                  f"Loss:{avg_total:8.4f} | "
                  f"Exp:{avg_parts['exp']:.4f} | "
                  f"Col:{avg_parts['col']:.4f} | "
                  f"Spa:{avg_parts['spa']:.4f} | "
                  f"Nol:{avg_parts['nol']:.4f} | "
                  f"TV:{avg_parts['tva']:.4f} | "
                  f"LR:{lr_now:.2e}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'loss'       : best_loss,
                'args'       : vars(args),
            }, checkpoint_path)
            if epoch % 10 == 0 or epoch == 1:
                print(f"    → Checkpoint tersimpan (loss={best_loss:.4f})")

        if args.save_every > 0 and epoch % args.save_every == 0:
            periodic_path = os.path.join(args.save_dir, f'zlen_epoch{epoch:04d}.pth')
            torch.save({'epoch': epoch, 'model_state': model.state_dict()}, periodic_path)

    print(f"{'='*65}")
    print(f"  Training selesai! Best loss: {best_loss:.4f}")
    print(f"  Model tersimpan: {checkpoint_path}")
    print(f"{'='*65}\n")

    plot_loss_curve(loss_history, history_path)

    return checkpoint_path


def parse_args():
    p = argparse.ArgumentParser(
        description='ZLEN — Training dengan dataset gambar low-light'
    )
    p.add_argument('--data_dir',   type=str,   required=True,
                   help='Folder berisi gambar-gambar low-light (rekursif)')
    p.add_argument('--save_dir',   type=str,   default='zlen_checkpoints',
                   help='Folder untuk menyimpan checkpoint (default: zlen_checkpoints)')
    p.add_argument('--epochs',     type=int,   default=200,
                   help='Jumlah epoch training (default: 200)')
    p.add_argument('--batch_size', type=int,   default=4,
                   help='Ukuran batch (default: 4, turunkan jika VRAM kurang)')
    p.add_argument('--img_size',   type=int,   default=IMG_SIZE,
                   help=f'Ukuran resize gambar (default: {IMG_SIZE})')
    p.add_argument('--lr',         type=float, default=LR,
                   help=f'Learning rate (default: {LR})')
    p.add_argument('--workers',    type=int,   default=2,
                   help='Jumlah worker DataLoader (default: 2)')
    p.add_argument('--save_every', type=int,   default=50,
                   help='Simpan checkpoint tiap N epoch (0=nonaktif, default: 50)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f"[INFO] Device   : {DEVICE}")
    print(f"[INFO] Data dir : {args.data_dir}")
    print(f"[INFO] Save dir : {args.save_dir}")

    checkpoint_path = train_with_dataset(args)

    print(f"""
Langkah selanjutnya — inferensi gambar baru:
  python zlen_infer.py --checkpoint {checkpoint_path} \\
                       --input foto_baru.jpg \\
                       --output hasil_enhanced.jpg
""")
