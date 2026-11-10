"""Unit checks for SRSD. Run: .\.venv\Scripts\python.exe scripts\check_srsd.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from networks.srsd import RadialSpectralDropout
from networks.freqnet import freqnet

# 1. eval mode -> identity
m = RadialSpectralDropout(8, 0.2, shifted=True).eval()
s = torch.fft.fft2(torch.randn(4, 3, 32, 32))
print('1 eval identity      :', torch.equal(m(s), s))

# 2. train mode: mean mask ~ 1 (unbiased), DC band protected
m.train()
masks = torch.stack([m.make_mask(1, 32, 32, 'cpu')[0, 0] for _ in range(20000)])
print('2 mean mask          : %.4f (expect ~1)' % masks.mean().item())

# 3. real input -> imaginary part of iFFT of masked spectrum negligible
x = torch.randn(8, 4, 32, 32)
spec = torch.fft.ifftshift(m(torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))), dim=(-2, -1))
y = torch.fft.ifft2(spec)
print('3 rel. imag part     : %.2e (expect < 1e-5)' % (y.imag.abs().max() / y.real.abs().max()).item())

# 4. shifted band map: centre (DC) pixel is band 0
b = m._band_index(32, 32, 'cpu')
print('4 DC band at centre  :', b[16, 16].item() == 0, '| bands present:', sorted(b.unique().tolist()))

# 5. model: params unchanged, p=0 bit-identical, p>0 differs in train / equal in eval
if torch.cuda.is_available():
    torch.manual_seed(0); a = freqnet(srsd_p=0.0).cuda()
    torch.manual_seed(0); b0 = freqnet().cuda()
    torch.manual_seed(0); c = freqnet(srsd_p=0.2).cuda()
    c.load_state_dict(a.state_dict())
    n = lambda mm: sum(p.numel() for p in mm.parameters())
    print('5 params p0/default/p0.2:', n(a), n(b0), n(c))
    xin = torch.randn(4, 3, 128, 128).cuda()
    a.train(); b0.load_state_dict(a.state_dict()); b0.train()
    print('  p=0 identical to default:', torch.equal(a(xin), b0(xin)))
    c.load_state_dict(a.state_dict())   # re-sync BN running stats changed by the train-mode passes above
    a.eval(); c.eval()
    print('  p=0.2 eval == baseline  :', torch.equal(a(xin), c(xin)))
    c.train(); a.train()
    print('  p=0.2 train != baseline :', not torch.equal(a(xin), c(xin)))
