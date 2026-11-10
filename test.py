import os, csv, time
import numpy as np
import torch
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions

opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

model = freqnet(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)  # raw state_dict
model.cuda()
model.eval()

rows = []


def run(name, root, classes, no_resize, no_crop):
    opt.dataroot, opt.classes, opt.no_resize, opt.no_crop = root, classes, no_resize, no_crop
    acc, ap, r_acc, f_acc, _, _ = validate(model, opt)
    rows.append([name, acc * 100, ap * 100, r_acc * 100, f_acc * 100])
    print("({:12}) acc: {:.1f}; ap: {:.1f}; real_acc: {:.1f}; fake_acc: {:.1f}".format(name, *rows[-1][1:]))


print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
# in-domain val (20 ProGAN classes; classes=[] -> every sub-folder)
if opt.val_root and os.path.isdir(opt.val_root):
    run('in_domain_val', opt.val_root, [''] if '0_real' in os.listdir(opt.val_root) else [], False, False)
# unseen GANs (all 128x128, flat <GAN>/{0_real,1_fake})
gan_rows = []
for gan in sorted(os.listdir(opt.test_root)):
    run(gan, os.path.join(opt.test_root, gan), [''], True, True)
    gan_rows.append(rows[-1])
if gan_rows:
    m = np.mean([r[1:] for r in gan_rows], axis=0)
    rows.append(['Mean_unseen'] + m.tolist())
    print("({:12}) acc: {:.1f}; ap: {:.1f}; real_acc: {:.1f}; fake_acc: {:.1f}".format('Mean_unseen', *m))

os.makedirs(os.path.dirname(opt.out) or '.', exist_ok=True)
with open(opt.out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['set', 'acc', 'ap', 'real_acc', 'fake_acc'])
    w.writerows([[r[0]] + [f'{v:.2f}' for v in r[1:]] for r in rows])
print('wrote', opt.out)
