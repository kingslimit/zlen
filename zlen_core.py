import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 256
BATCH_SIZE = 5
LR         = 3e-4
NUM_EPOCHS = 500
N_ITER     = 8

W_SPA  = 1.0
W_EXP  = 10.0
W_COL  = 20.0
W_NOL  = 0.5
W_TVA  = 200.0

EXPOSURE_TARGET = 0.5

SNR_TARGET_W = 0.5

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] N_ITER (iterasi kurva): {N_ITER}")


class NoiseEstimationModule(nn.Module):
    def __init__(self):
        super(NoiseEstimationModule, self).__init__()

        grad_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        grad_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("grad_x", grad_x)
        self.register_buffer("grad_y", grad_y)

    def forward(self, x):
        mean_c    = x.mean(dim=1, keepdim=True)
        mean_c    = mean_c.expand_as(x)
        color_map = x / (mean_c + 1e-7)
        color_map = torch.clamp(color_map, 0.0, 3.0)

        cm_gray = (0.299 * color_map[:, 0:1, :, :] +
                   0.587 * color_map[:, 1:2, :, :] +
                   0.114 * color_map[:, 2:3, :, :])

        gx = F.conv2d(cm_gray, self.grad_x, padding=1)
        gy = F.conv2d(cm_gray, self.grad_y, padding=1)
        noise_1ch = torch.max(gx.abs(), gy.abs())

        n_max     = noise_1ch.amax(dim=[2, 3], keepdim=True) + 1e-8
        noise_1ch = noise_1ch / n_max

        noise_map = noise_1ch.expand(-1, 3, -1, -1)

        return noise_map


class SemanticAwareAttentionModule(nn.Module):
    def __init__(self, channels=3):
        super(SemanticAwareAttentionModule, self).__init__()

        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)

        self.feedforward = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, noise_map):
        B, C, H, W = noise_map.shape

        k = self.conv_k(noise_map)
        q = self.conv_q(noise_map)
        v = self.conv_v(noise_map)

        k_gap = k.mean(dim=[2, 3])
        q_gap = q.mean(dim=[2, 3])

        scale = float(C) ** 0.5
        attn  = torch.bmm(
            k_gap.unsqueeze(2),
            q_gap.unsqueeze(1)
        ) / scale
        attn  = torch.softmax(attn, dim=-1)

        v_flat = v.view(B, C, -1)
        out    = torch.bmm(attn, v_flat).view(B, C, H, W)

        fnoise = self.feedforward(out + noise_map)

        return fnoise


class NoiseFusionModule(nn.Module):
    def __init__(self):
        super(NoiseFusionModule, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, image, fnoise):
        fused = image + self.beta * fnoise
        return torch.clamp(fused, 0.0, 1.0)


class DepthCurveEstimationNet(nn.Module):
    def __init__(self, n_iter=N_ITER):
        super(DepthCurveEstimationNet, self).__init__()
        self.n_iter = n_iter
        out_ch = 3 * n_iter

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
             nn.Dropout2d(p=0.1)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        f6 = self.conv6(f5)
        alpha = self.conv7(f6)
        return alpha


def apply_curve_enhancement(image, alpha_maps, noise_map, n_iter=N_ITER):
    LE = image
    noise_scaled = noise_map * 0.1

    for i in range(n_iter):
        alpha_n = alpha_maps[:, i*3 : (i+1)*3, :, :]
        LE = LE + alpha_n * LE * (1.0 - LE + noise_scaled)
        LE = torch.clamp(LE, 0.0, 1.0)

    return LE


def exposure_loss(enhanced, E=EXPOSURE_TARGET):
    gray = (0.299 * enhanced[:, 0] +
            0.587 * enhanced[:, 1] +
            0.114 * enhanced[:, 2]).unsqueeze(1)

    patch_mean = F.avg_pool2d(gray, kernel_size=16)
    return torch.mean(torch.abs(patch_mean - E))


def color_consistency_loss(enhanced):
    r = torch.mean(enhanced[:, 0])
    g = torch.mean(enhanced[:, 1])
    b = torch.mean(enhanced[:, 2])
    return (r - g)**2 + (r - b)**2 + (g - b)**2


_KERNEL_RIGHT = torch.FloatTensor(
    [[0, 0, 0], [0, -1, 1], [0, 0, 0]]
).view(1, 1, 3, 3)

_KERNEL_DOWN = torch.FloatTensor(
    [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
).view(1, 1, 3, 3)


def spatial_consistency_loss(enhanced, original):
    kernel_right = _KERNEL_RIGHT.to(enhanced.device)
    kernel_down  = _KERNEL_DOWN.to(enhanced.device)

    def to_gray(img):
        return (0.299 * img[:, 0] +
                0.587 * img[:, 1] +
                0.114 * img[:, 2]).unsqueeze(1)

    enh_g  = to_gray(enhanced)
    orig_g = to_gray(original)

    d_er = F.conv2d(enh_g,  kernel_right, padding=1)
    d_ed = F.conv2d(enh_g,  kernel_down,  padding=1)
    d_or = F.conv2d(orig_g, kernel_right, padding=1)
    d_od = F.conv2d(orig_g, kernel_down,  padding=1)

    return torch.mean((d_er - d_or)**2 + (d_ed - d_od)**2)


def luminance_smoothness_loss(alpha_maps):
    dx = torch.abs(alpha_maps[:, :, :, 1:] - alpha_maps[:, :, :, :-1])
    dy = torch.abs(alpha_maps[:, :, 1:, :] - alpha_maps[:, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)


def noise_estimation_loss(enhanced, W=SNR_TARGET_W):
    gray = (0.299 * enhanced[:, 0] +
            0.587 * enhanced[:, 1] +
            0.114 * enhanced[:, 2]).unsqueeze(1)

    local_mean = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
    local_sq   = F.avg_pool2d(gray**2, kernel_size=5, stride=1, padding=2)
    local_var  = torch.clamp(local_sq - local_mean**2, min=1e-8)
    local_std  = torch.sqrt(local_var)

    snr_local = local_mean / (local_std + 1e-8)
    snr_norm  = torch.clamp(snr_local / 10.0, 0.0, 1.0)

    S    = gray.shape[2] * gray.shape[3]
    loss = (1.0 / S) * torch.sum((snr_norm - W)**2)
    return loss


def zlen_total_loss(enhanced, original, alpha_maps,
                    w_spa=W_SPA, w_exp=W_EXP, w_col=W_COL,
                    w_nol=W_NOL, w_tva=W_TVA):
    l_spa = spatial_consistency_loss(enhanced, original)
    l_exp = exposure_loss(enhanced)
    l_col = color_consistency_loss(enhanced)
    l_nol = noise_estimation_loss(enhanced)
    l_tva = luminance_smoothness_loss(alpha_maps)

    total = (w_spa * l_spa +
             w_exp * l_exp +
             w_col * l_col +
             w_nol * l_nol +
             w_tva * l_tva)

    return total, {
        "spa"  : l_spa.item(),
        "exp"  : l_exp.item(),
        "col"  : l_col.item(),
        "nol"  : l_nol.item(),
        "tva"  : l_tva.item(),
        "total": total.item()
    }


class ZLENModel(nn.Module):
    def __init__(self, n_iter=N_ITER):
        super(ZLENModel, self).__init__()
        self.n_iter          = n_iter
        self.noise_estimator = NoiseEstimationModule()
        self.saam            = SemanticAwareAttentionModule(channels=3)
        self.fusion          = NoiseFusionModule()
        self.dce_net         = DepthCurveEstimationNet(n_iter=n_iter)

    def forward(self, x):
        noise_map  = self.noise_estimator(x)
        fnoise     = self.saam(noise_map)
        f_fusion   = self.fusion(x, fnoise)
        alpha_maps = self.dce_net(f_fusion)
        enhanced   = apply_curve_enhancement(x, alpha_maps, noise_map, self.n_iter)
        return enhanced, alpha_maps, noise_map


def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    tf  = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0).to(DEVICE)


def tensor_to_pil(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))


def save_comparison(orig, enh, noise, save_path="zlen_result.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (orig,  "Input (Low-Light)"),
        (noise, "Noise Map N(x)"),
        (enh,   "Output (ZLEN Enhanced)"),
    ]
    for ax, (t, title) in zip(axes, panels):
        arr = t.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        ax.imshow(np.clip(arr, 0, 1))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    plt.suptitle("ZLEN -- Zero-Reference Low-Light Enhancement", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Hasil disimpan: {save_path}")


def plot_loss_curve(loss_history, save_path="zlen_loss.png"):
    plt.figure(figsize=(9, 4))
    plt.plot(loss_history, color="steelblue", linewidth=1.5)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("ZLEN -- Training Loss Curve", fontsize=13)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Grafik loss disimpan: {save_path}")
