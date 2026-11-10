import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger

import random, shutil


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# test config
# [PATCH] test sets are discovered from <dataroot>/test/<GAN>/{0_real,1_fake}


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    set_seed(opt.seed)  # [PATCH]
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    
    def testmodel():
        print('*'*25);accs = [];aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        names = sorted(os.listdir(Testdataroot)) if os.path.isdir(Testdataroot) else []
        for v_id, val in enumerate(names):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = ['']
            Testopt.no_resize = True
            Testopt.no_crop = True
            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc);aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
        if accs:
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(len(accs),'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100))
        print('*'*25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    # model.eval();testmodel();
    model.eval()
    acc, ap = validate(model.model, val_opt)[:2]
    print("(Val @ start, before training) acc: {}; ap: {}".format(acc, ap))
    model.train()
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
            

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        # testmodel()
        model.train()

    model.save_networks('last')  # [PATCH] save first, so a failing test loop can never lose the model
    os.makedirs('checkpoints', exist_ok=True); os.makedirs('logs', exist_ok=True)
    shutil.copy(os.path.join(model.save_dir, 'model_epoch_last.pth'), os.path.join('checkpoints', opt.name + '.pth'))
    model.eval();testmodel()
    sys.stdout.flush()
    shutil.copy(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'), os.path.join('logs', opt.name + '.log'))
    
