import os
import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from zlen_core import (
    DEVICE, IMG_SIZE, N_ITER,
    ZLENModel,
    tensor_to_pil,
    save_comparison,
)

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_checkpoint(checkpoint_path):
    print(f"[INFO] Memuat checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    saved_args = ckpt.get('args', {})
    n_iter = saved_args.get('n_iter', N_ITER)

    model = ZLENModel(n_iter=n_iter).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    loss  = ckpt.get('loss',  '?')
    print(f"[INFO] Checkpoint: epoch={epoch}, best_loss={loss:.4f}"
          if isinstance(loss, float) else f"[INFO] Checkpoint: epoch={epoch}")

    return model


def enhance_single(model, img_path, img_size=IMG_SIZE, save_comparison_img=False):
    img = Image.open(img_path).convert('RGB')
    original_size = img.size

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img_tensor = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enhanced, _, noise_map = model(img_tensor)

    enhanced_pil = tensor_to_pil(enhanced)
    enhanced_pil = enhanced_pil.resize(original_size, Image.LANCZOS)

    if save_comparison_img:
        cmp_path = str(img_path).rsplit('.', 1)[0] + '_comparison.png'
        save_comparison(img_tensor, enhanced, noise_map, cmp_path)

    return enhanced_pil


def run(args):
    model      = load_checkpoint(args.checkpoint)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = enhance_single(
            model, input_path,
            img_size           = args.img_size,
            save_comparison_img = args.comparison,
        )
        result.save(str(output_path))
        print(f"[INFO] Tersimpan: {output_path}")

    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        image_files = sorted([
            p for p in input_path.rglob('*')
            if p.suffix.lower() in IMG_EXTENSIONS
        ])

        if not image_files:
            print(f"[WARN] Tidak ada gambar ditemukan di '{input_path}'")
            return

        print(f"[INFO] Memproses {len(image_files)} gambar...")
        for i, img_file in enumerate(image_files, 1):
            rel_path  = img_file.relative_to(input_path)
            out_file  = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

            result = enhance_single(
                model, img_file,
                img_size            = args.img_size,
                save_comparison_img = False,
            )
            result.save(str(out_file))

            if args.comparison:
                tf = transforms.Compose([
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ToTensor(),
                ])
                img_tensor = tf(Image.open(img_file).convert('RGB')).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    enhanced_t, _, noise_map = model(img_tensor)
                cmp_path = str(out_file.with_suffix('')) + '_comparison.png'
                save_comparison(img_tensor, enhanced_t, noise_map, cmp_path)

            print(f"  [{i:3d}/{len(image_files)}] {img_file.name} -> {out_file}")

        print(f"\n[INFO] Selesai! Hasil tersimpan di '{output_path}'")

    else:
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description='ZLEN — Inferensi gambar menggunakan checkpoint terlatih'
    )
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path ke file .pth hasil zlen_train.py')
    p.add_argument('--input',      type=str, required=True,
                   help='Gambar atau folder input')
    p.add_argument('--output',     type=str, required=True,
                   help='Gambar atau folder output')
    p.add_argument('--img_size',   type=int, default=IMG_SIZE,
                   help=f'Ukuran resize saat inferensi (default: {IMG_SIZE})')
    p.add_argument('--comparison', action='store_true',
                   help='Simpan 3-panel perbandingan (input|noise|output) per gambar')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
