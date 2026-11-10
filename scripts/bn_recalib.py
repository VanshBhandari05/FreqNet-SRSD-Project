"""Re-estimate BatchNorm running statistics WITHOUT SRSD (no weights are trained).

Usage (same data flags as train.py, plus --src / --dst):
  .\.venv\Scripts\python.exe scripts\bn_recalib.py --src checkpoints/srsd_p0.1_ft_seed42.pth --dst checkpoints/srsd_p0.1_ft_seed42_bnrecal.pth --name bn_recalib --dataroot ./data --classes <20 classes> --batch_size 32 --cropSize 128 --num_threads 0
Apply the same procedure to the baseline checkpoint for a fair comparison.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
args, rest = ap.parse_known_args()
sys.argv = [sys.argv[0]] + rest

from options.train_options import TrainOptions
from data import create_dataloader
from networks.freqnet import freqnet

opt = TrainOptions().parse()
opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)

model = freqnet(srsd_p=0.0).cuda()                      # SRSD off: clean features
model.load_state_dict(torch.load(args.src, map_location='cpu'), strict=True)

for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()
        m.momentum = None                               # cumulative average over all batches

model.train()                                           # BN uses batch stats and accumulates running stats
loader = create_dataloader(opt)
with torch.no_grad():                                   # no gradients, no weight updates
    for img, _ in loader:
        model(img.cuda())

os.makedirs(os.path.dirname(args.dst) or '.', exist_ok=True)
torch.save(model.state_dict(), args.dst)
print('saved', args.dst)
