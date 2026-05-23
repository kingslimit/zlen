# =============================================================================
# ZLEN: Zero-Reference Low-Light Image Enhancement Based on Noise Estimation
# Implementasi PyTorch — Sesuai jurnal:
#   Cao, P.; Niu, Q.; Zhu, Y.; Li, T.
#   "A Zero-Reference Low-Light Image-Enhancement Approach Based on Noise Estimation"
#   Applied Sciences 2024, 14, 2846. https://doi.org/10.3390/app14072846
# =============================================================================
# Spesifikasi hardware:
#   GPU  : NVIDIA RTX 3050 6GB
#   Image: 256x256
#   Batch: 1
#
# CHANGELOG (bugfix):
#   v2 - Fix color cast & edge artifacts:
#        1. NoiseEstimationModule: color_map clamp 10.0 -> 3.0, grayscale noise
#           (1-channel) menggantikan per-channel noise untuk cegah channel imbalance
#        2. apply_curve_enhancement: noise_map dikonversi ke grayscale sebelum
#           dimasukkan ke Eq.1 agar tidak mendistorsi tiap channel R/G/B secara berbeda
#        3. W_COL dinaikkan 5.0 -> 20.0 untuk menekan color cast lebih agresif
#        4. W_NOL diturunkan 1.0 -> 0.5 agar noise loss tidak override color loss
# =============================================================================

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

# -----------------------------------------------------------------------------
# 0. KONFIGURASI GLOBAL
#    Semua hyperparameter dikumpulkan di sini agar mudah diubah.
#    Bobot loss sesuai paper: Wspa=1, Wexp=10, Wcol=5, Wnol=1, WtvA=200
# -----------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 256      # resolusi gambar (paper asli 512, diturunkan agar ringan)
BATCH_SIZE = 5
LR         = 3e-4     # dinaikkan dari 1e-4 agar konvergensi lebih cepat
NUM_EPOCHS = 500      # dinaikkan dari 200, foto gelap butuh lebih banyak iterasi
N_ITER     = 8        # jumlah iterasi curve enhancement (n pada Eq.1 jurnal)

# Bobot loss function (Eq. 10 jurnal)
W_SPA  = 1.0    # Wspa  -- spatial consistency
W_EXP  = 10.0   # Wexp  -- exposure control (perlu dorongan kuat untuk foto sangat gelap)
W_COL  = 20.0   # Wcol  -- color consistency [FIX v2: dinaikkan 5->20 untuk tekan color cast]
W_NOL  = 0.5    # Wnol  -- noise estimation  [FIX v2: diturunkan 1->0.5 agar tidak override Wcol]
W_TVA  = 200.0  # WtvA  -- luminance smoothness (TV loss)

# Target exposure (nilai rata-rata yang diinginkan, antara 0-1)
EXPOSURE_TARGET = 0.5   # diturunkan dari 0.6, target moderat untuk foto malam

# Konstanta W pada noise loss (Eq. 9 jurnal): target SNR yang diinginkan
SNR_TARGET_W = 0.5   # bisa dieksperimen antara 0.3-0.7

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] N_ITER (iterasi kurva): {N_ITER}")


# =============================================================================
# 1. NOISE ESTIMATION MODULE
#    Implementasi sesuai Section 2.2.2 jurnal (Eq. 2 & 3):
#
#    Langkah 1 -- Hitung Color Map:
#      C(x) = x / mean_c(x)
#      di mana mean_c(x) = rata-rata nilai RGB tiap piksel
#
#    Langkah 2 -- Estimasi Noise dari gradien Color Map:
#      N(x) = max( |grad_x C(x)|, |grad_y C(x)| )
#
#    Intuisi: daerah content-rich (detail) punya gradien tinggi -> noise rendah
#             daerah flat (langit, dinding) punya gradien rendah -> noise tinggi
# =============================================================================
class NoiseEstimationModule(nn.Module):
    """
    Mengestimasi noise map berdasarkan Color Map (Retinex-inspired).
    Sesuai Persamaan (2) dan (3) jurnal ZLEN.

    Input : x -- gambar low-light (B, 3, H, W), range [0, 1]
    Output: noise_map             (B, 3, H, W), range [0, 1]
    """

    def __init__(self):
        super(NoiseEstimationModule, self).__init__()

        # Kernel gradien untuk arah-x dan arah-y (fixed, tidak dilatih)
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

        # register_buffer: ikut .to(device) tapi bukan parameter trainable
        self.register_buffer("grad_x", grad_x)
        self.register_buffer("grad_y", grad_y)

    def forward(self, x):
        """x: (B, 3, H, W) gambar input, range [0, 1]"""

        # Langkah 1: Hitung Color Map C(x) = x / mean_c(x)  -- Eq. 2
        # mean_c(x): rata-rata nilai piksel di seluruh channel RGB
        mean_c    = x.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        mean_c    = mean_c.expand_as(x)               # (B, 3, H, W)
        color_map = x / (mean_c + 1e-7)               # (B, 3, H, W)  ◆ Eq.2: C(x) = x / mean_c(x)

        # [FIX v2] Clamp lebih ketat: 10.0 -> 3.0
        # Di area sangat gelap (mean_c ~ 0), color_map bisa meledak hingga 10x.
        # Gradient Sobel dari nilai 10x ini menghasilkan edge artifacts neon.
        # Clamp 3.0 cukup untuk representasi rasio warna tanpa artefak.
        color_map = torch.clamp(color_map, 0.0, 3.0)

        # [FIX v2] Gunakan luminance grayscale color_map untuk hitung gradient,
        # bukan per-channel. Gradient per-channel (R/G/B) menghasilkan noise_map
        # yang berbeda per channel -> noise term di Eq.1 mendistorsi balance warna.
        # Grayscale noise = satu nilai per piksel yang sama untuk R,G,B -> netral.
        cm_gray = (0.299 * color_map[:, 0:1, :, :] +
                   0.587 * color_map[:, 1:2, :, :] +
                   0.114 * color_map[:, 2:3, :, :])  # (B, 1, H, W)

        gx = F.conv2d(cm_gray, self.grad_x, padding=1)
        gy = F.conv2d(cm_gray, self.grad_y, padding=1)
        noise_1ch = torch.max(gx.abs(), gy.abs())    # (B, 1, H, W)  ◆ Eq.3

        # Normalisasi ke [0, 1] per sample
        n_max     = noise_1ch.amax(dim=[2, 3], keepdim=True) + 1e-8
        noise_1ch = noise_1ch / n_max

        # Expand ke 3 channel (nilai sama di R,G,B -> tidak ada channel imbalance)
        noise_map = noise_1ch.expand(-1, 3, -1, -1)  # (B, 3, H, W)

        return noise_map   # (B, 3, H, W)


# =============================================================================
# 2. SEMANTIC-AWARE ATTENTION MODULE (SAAM)
#    Sesuai Section 2.2.2 jurnal (Eq. 4-7):
#    Mengekstrak fitur dari noise map menggunakan mekanisme attention.
#
#    Eq. 4: M(x,y) = sum_k wk * fk(x,y)   -- activation mapping
#    Eq. 5: Fs = weighted sum of noise pixels
#    Eq. 6: A = Softmax(Lk(Fi) x Lq(Fs) / sqrt(C))  -- attention map
#    Eq. 7: Fnoise = FN(L(Fi) x A + F)   -- noise feature output
# =============================================================================
class SemanticAwareAttentionModule(nn.Module):
    """
    Mengekstrak semantic-aware features dari noise map.
    Sesuai Persamaan (4)-(7) jurnal ZLEN.

    Input : noise_map (B, 3, H, W)
    Output: Fnoise    (B, 3, H, W) -- noise feature yang sudah di-attend

    CATATAN IMPLEMENTASI:
    Attention asli di jurnal mengacu pada spatial attention (H*W x H*W).
    Pada resolusi 256x256 hal itu membutuhkan matrix (65536 x 65536) = ~16 GB
    VRAM -- tidak mungkin dijalankan di GPU consumer.
    Solusi: gunakan CHANNEL-WISE attention (C x C, di sini hanya 3x3) yang
    secara semantik setara (menangkap inter-channel relationship dari noise map)
    dan menggunakan memori O(C^2) bukan O((H*W)^2).
    """

    def __init__(self, channels=3):
        super(SemanticAwareAttentionModule, self).__init__()

        # Lk, Lq, Lv: projection conv (kernel 1x1 = linear per piksel)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)

        # FN: feedforward network (Eq. 7)
        self.feedforward = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, noise_map):
        """noise_map: (B, 3, H, W)"""
        B, C, H, W = noise_map.shape

        k = self.conv_k(noise_map)   # (B, C, H, W)  ◆ Eq.4: fk(x,y) — key projection (Lk)
        q = self.conv_q(noise_map)   # (B, C, H, W)  ◆ Eq.4: fk(x,y) — query projection (Lq)
        v = self.conv_v(noise_map)   # (B, C, H, W)  ◆ Eq.5: Fs — value/semantic feature

        # --- Channel-wise attention: O(C^2) bukan O((H*W)^2) ---
        # Global average pooling: rangkum informasi spasial -> (B, C, 1, 1)
        k_gap = k.mean(dim=[2, 3])   # (B, C)
        q_gap = q.mean(dim=[2, 3])   # (B, C)

        # Attention matrix di ruang channel -- Eq. 6: A = softmax(Lk . Lq^T / sqrt(C))
        # Ukuran: (B, C, C) -- maksimal 3x3 untuk RGB, sangat ringan
        scale = float(C) ** 0.5                      # ◆ Eq.6: sqrt(C) — scaling factor
        attn  = torch.bmm(
            k_gap.unsqueeze(2),          # (B, C, 1)
            q_gap.unsqueeze(1)           # (B, 1, C)
        ) / scale                        # (B, C, C)  ◆ Eq.6: Lk(Fi) × Lq(Fs) / sqrt(C)
        attn  = torch.softmax(attn, dim=-1)  # (B, C, C)  ◆ Eq.6: A = Softmax(...)

        # Terapkan attention ke value: (B, C, C) x (B, C, H*W) -> (B, C, H*W)
        v_flat = v.view(B, C, -1)                         # (B, C, H*W)
        out    = torch.bmm(attn, v_flat).view(B, C, H, W) # (B, C, H, W)

        # FN feedforward + residual -- Eq. 7: Fnoise = FN(L(Fi) x A + F)
        fnoise = self.feedforward(out + noise_map)     # ◆ Eq.7: Fnoise = FN(L(Fi) × A + F)

        return fnoise   # (B, 3, H, W)


# =============================================================================
# 3. NOISE FUSION MODULE
#    Sesuai Section 2.2.2 jurnal (Eq. 8):
#
#    F_fusion = Fi + beta * F_noise
#
#    beta adalah koefisien learnable yang mengontrol kekuatan noise fusion.
# =============================================================================
class NoiseFusionModule(nn.Module):
    """
    Fusi gambar input dengan noise feature map.
    Sesuai Persamaan (8): F_fusion = Fi + beta * F_noise
    """

    def __init__(self):
        super(NoiseFusionModule, self).__init__()
        # beta: learnable scalar, diinisialisasi kecil
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, image, fnoise):
        """
        image  : (B, 3, H, W) gambar low-light asli
        fnoise : (B, 3, H, W) noise feature dari SAAM
        return : (B, 3, H, W) fused image
        """
        fused = image + self.beta * fnoise              # ◆ Eq.8: F_fusion = Fi + β·Fnoise
        return torch.clamp(fused, 0.0, 1.0)


# =============================================================================
# 4. DEPTH CURVE ESTIMATION NETWORK (DCE-Net)
#    Sesuai Section 2.2.3 jurnal dan Figure 3:
#
#    Arsitektur 7 Conv layer:
#      Conv1: 3 -> 32   + ReLU
#      Conv2: 32 -> 64  + ReLU
#      Conv3: 64 -> 64  + ReLU
#      Conv4: 64 -> 64  + ReLU
#      Conv5: 64 -> 64  + ReLU + DROPOUT  <-- dropout pertama (jurnal 3.3.3)
#      Conv6: 64 -> 64  + ReLU + DROPOUT  <-- dropout kedua
#      Conv7: 64 -> n   + Tanh            <-- output layer
#
#    n = 3 x N_ITER (satu set alpha RGB per iterasi)
# =============================================================================
class DepthCurveEstimationNet(nn.Module):
    """
    CNN 7 layer untuk mengestimasi parameter kurva alpha_n.
    Sesuai Figure 3 jurnal ZLEN (dengan 2 dropout setelah conv5 dan conv6).

    Input : fused image (B, 3, H, W)
    Output: alpha maps  (B, 3*N_ITER, H, W), range [-1, 1]
    """

    def __init__(self, n_iter=N_ITER):
        super(DepthCurveEstimationNet, self).__init__()
        self.n_iter = n_iter
        out_ch = 3 * n_iter  # satu set alpha (R,G,B) per iterasi

        # Conv1: 3 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Conv2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Conv3: 64 -> 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Conv4: 64 -> 64
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Conv5: 64 -> 64 + Dropout (pertama, sesuai ablation study 3.3.3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
             nn.Dropout2d(p=0.1)
        )
        # Conv6: 64 -> 64 + Dropout (kedua)
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        # Conv7: 64 -> (3 x N_ITER) + Tanh
        # Tanh menghasilkan alpha_n dalam [-1, 1] sesuai Eq.1
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """x: (B, 3, H, W) fused image"""
        f1 = self.conv1(x)    # (B, 32, H, W)
        f2 = self.conv2(f1)   # (B, 64, H, W)
        f3 = self.conv3(f2)   # (B, 64, H, W)
        f4 = self.conv4(f3)   # (B, 64, H, W)
        f5 = self.conv5(f4)   # (B, 64, H, W) + dropout
        f6 = self.conv6(f5)   # (B, 64, H, W) + dropout
        alpha = self.conv7(f6) # (B, 3*N_ITER, H, W)
        return alpha


# =============================================================================
# 5. IMPROVED HIGHER-ORDER CURVE ENHANCEMENT
#    Sesuai Section 2.2.1 jurnal (Eq. 1):
#
#    LE_n(x) = LE_{n-1}(x) + alpha_n * LE_{n-1}(x) * (1 - LE_{n-1}(x) + N(x))
#
#    Perbedaan dengan Zero-DCE original: ditambahkan suku N(x) di kurva
#    untuk menekan noise secara langsung dalam proses enhancement.
# =============================================================================
def apply_curve_enhancement(image, alpha_maps, noise_map, n_iter=N_ITER):
    """
    Iterative curve enhancement sesuai Persamaan (1) jurnal ZLEN.

    Args:
        image     : (B, 3, H, W) gambar asli, range [0, 1]
        alpha_maps: (B, 3*N_ITER, H, W) dari DCE-Net, range [-1, 1]
        noise_map : (B, 3, H, W) dari NoiseEstimationModule, range [0, 1]
        n_iter    : jumlah iterasi

    Returns:
        LE : (B, 3, H, W) gambar enhanced, range [0, 1]

    [FIX v2] noise_map di-scale dengan faktor 0.1 sebelum masuk ke Eq.1.
    Kontribusi N(x) di paper dimaksudkan sebagai koreksi kecil terhadap suku
    (1 - LE_{n-1}), bukan sebagai term dominan. Tanpa scaling, N(x) ~ [0,1]
    bisa sebesar (1 - LE) dan menyebabkan overshoot / color distortion parah.
    """
    LE = image   # LE_0 = x

    # [FIX v2] Scale noise contribution: N(x) seharusnya koreksi minor
    noise_scaled = noise_map * 0.1

    for i in range(n_iter):
        # Alpha untuk iterasi ke-i
        alpha_n = alpha_maps[:, i*3 : (i+1)*3, :, :]  # (B, 3, H, W)

        # ◆ Eq.1: LE_n = LE_{n-1} + α_n · LE_{n-1} · (1 − LE_{n-1} + N(x))
        LE = LE + alpha_n * LE * (1.0 - LE + noise_scaled)
        LE = torch.clamp(LE, 0.0, 1.0)

    return LE


# =============================================================================
# 6. ZERO-REFERENCE LOSS FUNCTIONS
#    Sesuai Section 2.2.4 jurnal (Eq. 9 & 10).
#
#    L_total = Wspa*Lspa + Wexp*Lexp + Wcol*Lcol + Wnol*Lnol + WtvA*LtvA
# =============================================================================

def exposure_loss(enhanced, E=EXPOSURE_TARGET):
    """
    Exposure Control Loss (Lexp):
    Mendorong rata-rata luminansi tiap patch mendekati nilai target E.
    Menggunakan patch 16x16.
    """
    gray = (0.299 * enhanced[:, 0] +
            0.587 * enhanced[:, 1] +
            0.114 * enhanced[:, 2]).unsqueeze(1)

    patch_mean = F.avg_pool2d(gray, kernel_size=16)
    return torch.mean(torch.abs(patch_mean - E))


def color_consistency_loss(enhanced):
    """
    Color Consistency Loss (Lcol):
    Mendorong keseimbangan rata-rata antar channel R, G, B.
    Mencegah color cast.
    """
    r = torch.mean(enhanced[:, 0])
    g = torch.mean(enhanced[:, 1])
    b = torch.mean(enhanced[:, 2])
    return (r - g)**2 + (r - b)**2 + (g - b)**2


# Kernel untuk spatial consistency loss -- didefinisikan sekali di level modul
# (menghindari alokasi ulang tiap forward pass)
_KERNEL_RIGHT = torch.FloatTensor(
    [[0, 0, 0], [0, -1, 1], [0, 0, 0]]
).view(1, 1, 3, 3)

_KERNEL_DOWN = torch.FloatTensor(
    [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
).view(1, 1, 3, 3)


def spatial_consistency_loss(enhanced, original):
    """
    Spatial Consistency Loss (Lspa):
    Menjaga perbedaan spasial antar piksel tetap konsisten
    antara gambar asli dan gambar enhanced.
    """
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
    """
    Luminance Smoothness / TV Loss (LtvA):
    Mendorong alpha map agar spasially halus -> enhancement lebih natural.
    Total Variation loss.
    """
    dx = torch.abs(alpha_maps[:, :, :, 1:] - alpha_maps[:, :, :, :-1])
    dy = torch.abs(alpha_maps[:, :, 1:, :] - alpha_maps[:, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)


def noise_estimation_loss(enhanced, W=SNR_TARGET_W):
    """
    Noise Estimation Loss (Lnol) -- sesuai Persamaan (9) jurnal:

    L_nol = (1/S) * sum( (SNR - W)^2 )

    di mana:
      SNR = Signal-to-Noise Ratio lokal (local mean / local std)
      W   = konstanta target SNR
      S   = ukuran gambar (H x W)
    """
    gray = (0.299 * enhanced[:, 0] +
            0.587 * enhanced[:, 1] +
            0.114 * enhanced[:, 2]).unsqueeze(1)

    # Local mean dan std menggunakan avg pooling
    local_mean = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
    local_sq   = F.avg_pool2d(gray**2, kernel_size=5, stride=1, padding=2)
    local_var  = torch.clamp(local_sq - local_mean**2, min=1e-8)
    local_std  = torch.sqrt(local_var)

    # SNR lokal, dinormalisasi ke [0, 1] dengan divisor tetap (stabil antar epoch)
    # Menggunakan amax() dinamis menyebabkan target W bergeser implisit tiap epoch
    snr_local = local_mean / (local_std + 1e-8)         # ◆ Eq.9: SNR = mean / std
    snr_norm  = torch.clamp(snr_local / 10.0, 0.0, 1.0)

    # ◆ Eq.9: L_nol = (1/S) · Σ(SNR − W)²
    S    = gray.shape[2] * gray.shape[3]
    loss = (1.0 / S) * torch.sum((snr_norm - W)**2)
    return loss


def zlen_total_loss(enhanced, original, alpha_maps,
                    w_spa=W_SPA, w_exp=W_EXP, w_col=W_COL,
                    w_nol=W_NOL, w_tva=W_TVA):
    """
    Total Loss ZLEN sesuai Persamaan (10) jurnal:
    L_total = Wspa*Lspa + Wexp*Lexp + Wcol*Lcol + Wnol*Lnol + WtvA*LtvA
    """
    l_spa = spatial_consistency_loss(enhanced, original)
    l_exp = exposure_loss(enhanced)
    l_col = color_consistency_loss(enhanced)
    l_nol = noise_estimation_loss(enhanced)
    l_tva = luminance_smoothness_loss(alpha_maps)

    # ◆ Eq.10: L_total = Wspa·Lspa + Wexp·Lexp + Wcol·Lcol + Wnol·Lnol + WtvA·LtvA
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


# =============================================================================
# 7. MODEL ZLEN LENGKAP
#    Pipeline sesuai Figure 2 jurnal:
#
#    Input -> NoiseEstimation -> N(x)
#          -> SAAM(N(x))     -> Fnoise
#          -> Fusion(I, Fnoise) -> F_fusion
#          -> DCE-Net(F_fusion) -> alpha_maps
#          -> CurveEnhancement(I, alpha, N(x)) -> Enhanced
# =============================================================================
class ZLENModel(nn.Module):
    """Model ZLEN lengkap sesuai Figure 2 jurnal."""

    def __init__(self, n_iter=N_ITER):
        super(ZLENModel, self).__init__()
        self.n_iter          = n_iter
        self.noise_estimator = NoiseEstimationModule()
        self.saam            = SemanticAwareAttentionModule(channels=3)
        self.fusion          = NoiseFusionModule()
        self.dce_net         = DepthCurveEstimationNet(n_iter=n_iter)

    def forward(self, x):
        """
        x: (B, 3, H, W) gambar low-light, range [0, 1]
        Returns:
            enhanced  : (B, 3, H, W)
            alpha_maps: (B, 3*N_ITER, H, W)
            noise_map : (B, 3, H, W)
        """
        # Step 1: Estimasi noise map dari color map -- Eq. 2 & 3
        noise_map = self.noise_estimator(x)

        # Step 2: Semantic-aware attention untuk ekstrak noise features -- Eq. 4-7
        fnoise    = self.saam(noise_map)

        # Step 3: Fusi gambar + noise features -- Eq. 8
        f_fusion  = self.fusion(x, fnoise)

        # Step 4: Estimasi alpha map dari fused image
        alpha_maps = self.dce_net(f_fusion)

        # Step 5: Iterative curve enhancement dengan noise term -- Eq. 1
        enhanced = apply_curve_enhancement(x, alpha_maps, noise_map, self.n_iter)

        return enhanced, alpha_maps, noise_map


# =============================================================================
# 8. UTILITAS: LOAD & SIMPAN GAMBAR
# =============================================================================
def load_image(path, size=IMG_SIZE):
    """Memuat gambar dari file -> tensor (1, 3, H, W) di [0, 1]."""
    img = Image.open(path).convert("RGB")
    tf  = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0).to(DEVICE)


def tensor_to_pil(tensor):
    """Tensor (1, 3, H, W) -> PIL Image."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))


def save_comparison(orig, enh, noise, save_path="zlen_result.png"):
    """Simpan 3-panel: input | noise map | enhanced."""
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
    """Gambar kurva total loss selama training."""
    plt.figure(figsize=(9, 4))
    plt.plot(loss_history, color="steelblue", linewidth=1.5)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("ZLEN -- Training Loss Curve", fontsize=13)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Grafik loss disimpan: {save_path}")


# =============================================================================
# File ini adalah library — tidak dijalankan langsung.
# Gunakan:
#   zlen_train.py  -> training pada dataset gambar
#   zlen_infer.py  -> inferensi gambar baru dari checkpoint
# =============================================================================