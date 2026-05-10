
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# AdaLN & NAFBlock
# ---------------------------------------------------------------------------

class AdaLayerNorm2d(nn.Module):
    """Adaptive Layer Normalization: LN(x) * (1 + scale) + shift,
    where (scale, shift) = Linear(SiLU(t_emb))."""
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, channels * 2))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, emb):
        x = self.norm(x)
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return x * (1 + scale[..., None, None]) + shift[..., None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class AdaNAFBlock(nn.Module):
    """NAFBlock with SGM + SCA, using AdaLN for timestep conditioning."""
    def __init__(self, c, emb_dim, kernel_size=3, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_ch = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size,
                               padding=(kernel_size - 1) // 2, groups=dw_ch)
        self.conv3 = nn.Conv2d(dw_ch // 2, c, 1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.sg = SimpleGate()

        ffn_ch = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)

        self.norm1 = AdaLayerNorm2d(c, emb_dim)
        self.norm2 = AdaLayerNorm2d(c, emb_dim)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp, t_emb):
        x = self.norm1(inp, t_emb)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y, t_emb))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


# ---------------------------------------------------------------------------
# Score network (NAFNet U-Net)
# ---------------------------------------------------------------------------

class ScoreUNet(nn.Module):
    """NAFNet-based U-Net score network s_theta(x_t, t) with AdaLN."""
    def __init__(self, in_ch=1, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 4], dec_blk_nums=[1, 1, 1, 1],
                 kernel_size=3):
        super().__init__()

        emb_dim = width * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(width),
            nn.Linear(width, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.ending = nn.Conv2d(width, in_ch, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.ModuleList([AdaNAFBlock(chan, emb_dim, kernel_size) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.ModuleList(
            [AdaNAFBlock(chan, emb_dim, kernel_size) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(
                nn.ModuleList([AdaNAFBlock(chan, emb_dim, kernel_size) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, x, t):
        """
        Args:
            x: [B, C, H, W] noisy input
            t: [B] continuous timestep in [0, 1] (or integer index)
        """
        t_emb = self.time_embed(t)
        B, C, H, W = x.shape
        x = self._pad(x)
        x = self.intro(x)

        encs = []
        for blocks, down in zip(self.encoders, self.downs):
            for blk in blocks:
                x = blk(x, t_emb)
            encs.append(x)
            x = down(x)

        for blk in self.middle_blks:
            x = blk(x, t_emb)

        for blocks, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + skip
            for blk in blocks:
                x = blk(x, t_emb)

        x = self.ending(x)
        return x[:, :, :H, :W]

    def _pad(self, x):
        _, _, h, w = x.size()
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph))


# ---------------------------------------------------------------------------
# VP-SDE utilities
# ---------------------------------------------------------------------------

class VPSDE:
    """Variance Preserving SDE: dx = -0.5 beta(t) x dt + sqrt(beta(t)) dw.

    Continuous-time formulation with linear beta schedule beta(t) in [beta_min, beta_max].
    """
    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_params(self, t):
        """Mean coefficient and std of q(x_t | x_0)."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean_coeff, std

    def perturb(self, x0, t, noise=None):
        """Forward SDE: sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        mean_coeff, std = self.marginal_params(t)
        mean_coeff = mean_coeff[:, None, None, None]
        std = std[:, None, None, None]
        return mean_coeff * x0 + std * noise, noise

    def reverse_step(self, x, t, score, dt):
        """One step of the reverse-time SDE (Euler-Maruyama).
        dx = [-0.5 beta(t) x - beta(t) score] dt + sqrt(beta(t)) dw_bar
        """
        beta_t = self.beta(t)
        drift = -0.5 * beta_t * x - beta_t * score
        diffusion = math.sqrt(beta_t)
        noise = torch.randn_like(x)
        return x - drift * dt + diffusion * math.sqrt(dt) * noise

    def reverse_ode_step(self, x, t, score, dt):
        """One step of the probability flow ODE (deterministic).
        dx/dt = -0.5 beta(t) [x + score]
        """
        beta_t = self.beta(t)
        drift = -0.5 * beta_t * (x + score)
        return x - drift * dt


# ---------------------------------------------------------------------------
# Shadow generator wrapper
# ---------------------------------------------------------------------------

class DiffusionShadowGenerator(nn.Module):
    """VP-SDE shadow mask generator.

    Trains a score network to model the distribution of shadow masks,
    then samples via reverse-time SDE or probability flow ODE.
    """
    def __init__(self, beta_min=0.1, beta_max=20.0, N=1000,
                 in_ch=1, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 4], dec_blk_nums=[1, 1, 1, 1],
                 kernel_size=3):
        super().__init__()
        self.sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=N)
        self.N = N
        self.model = ScoreUNet(
            in_ch=in_ch, width=width, middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums,
            kernel_size=kernel_size,
        )

    def compute_loss(self, x0):
        """Denoising score matching loss.
        L = E_{t, x_0, eps} [ || s_theta(x_t, t) + eps / std ||^2 ]
        Parameterised as noise prediction (equivalent to score matching).
        """
        B = x0.shape[0]
        t = torch.rand(B, device=x0.device) * (1.0 - 1e-5) + 1e-5
        xt, noise = self.sde.perturb(x0, t)
        pred = self.model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, shape, device, num_steps=50, method='sde'):
        """Reverse-time sampling.

        Args:
            shape: (B, C, H, W)
            device: torch device
            num_steps: number of discretisation steps
            method: 'sde' (stochastic) or 'ode' (deterministic)
        """
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps

        timesteps = torch.linspace(1.0, 1e-5, num_steps, device=device)
        for t_scalar in timesteps:
            t_batch = torch.full((shape[0],), t_scalar.item(), device=device)
            _, std = self.sde.marginal_params(t_batch)

            # network predicts noise -> convert to score: s = -noise / std
            noise_pred = self.model(x, t_batch)
            score = -noise_pred / std[:, None, None, None]

            if method == 'sde':
                x = self.sde.reverse_step(x, t_scalar, score, dt)
            else:
                x = self.sde.reverse_ode_step(x, t_scalar, score, dt)

        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Shadow overlay
# ---------------------------------------------------------------------------

def apply_shadow_mask(clean_img, shadow_mask, intensity_range=(0.3, 0.8)):
    """Overlay shadow mask onto clean image to create shadowed image.

    Args:
        clean_img: [B, 3, H, W] clean/restored image
        shadow_mask: [B, 1, H, W] shadow mask from diffusion model
        intensity_range: (min, max) shadow darkening intensity
    Returns:
        shadowed image [B, 3, H, W]
    """
    B = clean_img.shape[0]
    intensity = torch.rand(B, 1, 1, 1, device=clean_img.device)
    intensity = intensity * (intensity_range[1] - intensity_range[0]) + intensity_range[0]

    if shadow_mask.shape[2:] != clean_img.shape[2:]:
        shadow_mask = F.interpolate(shadow_mask, size=clean_img.shape[2:],
                                    mode='bilinear', align_corners=False)

    shadow_factor = 1.0 - shadow_mask * intensity
    return torch.clamp(clean_img * shadow_factor, 0., 1.)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cpu')

    gen = DiffusionShadowGenerator(N=100, width=16,
                                   enc_blk_nums=[1, 1, 1, 2], dec_blk_nums=[1, 1, 1, 1])
    print(f'score network params: {sum(p.numel() for p in gen.model.parameters())/1e6:.2f}M')

    x0 = torch.rand(2, 1, 64, 64)
    loss = gen.compute_loss(x0)
    print(f'score matching loss: {loss.item():.4f}')

    masks = gen.sample((2, 1, 64, 64), device, num_steps=10, method='sde')
    print(f'SDE sample: {masks.shape}, range [{masks.min():.3f}, {masks.max():.3f}]')

    masks_ode = gen.sample((2, 1, 64, 64), device, num_steps=10, method='ode')
    print(f'ODE sample: {masks_ode.shape}, range [{masks_ode.min():.3f}, {masks_ode.max():.3f}]')

    clean = torch.rand(2, 3, 64, 64)
    shadowed = apply_shadow_mask(clean, masks)
    print(f'shadow overlay: clean{clean.shape} + mask{masks.shape} -> {shadowed.shape}')
