import torch
import torch.nn as nn


class RadialSpectralDropout(nn.Module):
    """Stochastic Radial Spectral Dropout (SRSD).
    Drops random ring-shaped frequency bands of a complex spectrum during training.
    Identity at eval time or when p == 0. Adds no parameters / buffers."""

    def __init__(self, num_bands: int = 8, p: float = 0.2, protect_dc: bool = True, shifted: bool = False):
        super().__init__()
        self.K, self.p, self.protect_dc, self.shifted = num_bands, p, protect_dc, shifted

    def _band_index(self, H, W, device):
        fy = torch.fft.fftfreq(H, device=device)
        fx = torch.fft.fftfreq(W, device=device)
        r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        r = r / r.max()
        band = torch.clamp((r * self.K).long(), max=self.K - 1)       # (H, W)
        if self.shifted:                                                # match fftshift-ed spectra
            band = torch.fft.fftshift(band, dim=(-2, -1))
        return band

    def make_mask(self, B, H, W, device):
        band = self._band_index(H, W, device)
        keep = (torch.rand(B, self.K, device=device) > self.p).float() / (1.0 - self.p)
        if self.protect_dc:
            keep[:, 0] = 1.0
        return keep[:, band].unsqueeze(1)                               # (B, 1, H, W), shared over channels

    def forward(self, spec: torch.Tensor) -> torch.Tensor:              # spec: complex (B, C, H, W)
        if not self.training or self.p <= 0:
            return spec
        B, _, H, W = spec.shape
        return spec * self.make_mask(B, H, W, spec.device)
