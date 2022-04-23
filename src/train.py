from __future__ import print_function

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from mylibs.losses import L1Loss, PerceptualLoss, SmoothDispLoss, SSLCriterion
from mylibs.trainer import DisparityTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str)
parser.add_argument('--l1_weight', type=float)
parser.add_argument('--smooth_weight', type=float)
parser.add_argument('--per_weight', type=float)
parser.add_argument('--per_layers', type=int, nargs='+')
parser.add_argument('--agg', type=str, choices=['sum', 'prod'])
parser.add_argument('--gpu', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--num_workers', type=int)

args = parser.parse_args()
print(args)
print()

# Main training loop
EXP_NAME = args.exp
CKPT_DIR = Path(f'ckpts/{EXP_NAME}')
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(f'results/{EXP_NAME}')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRETRAIN_CHECKPOINT = None
AGG_PROD = args.agg == 'prod'

LR = 0.0002
DEVICE = f'cuda:{args.gpu}'
DATA_DIR = "/data"
# TRAIN_LIST = 'IRS/lists/Restaurant_TRAIN_small.list'
TRAIN_LIST = '/project/extlibs/IRS/lists/Restaurant_TRAIN.list'
TEST_METAL_LIST = '/project/extlibs/IRS/lists/IRS_restaurant_metal_test.list'
BATCH_SIZE = args.batch_size
MAX_DISP = 194
NUM_WORKERS = args.num_workers
EPOCHS = args.epochs
BATCH_NORM = True

loss_list = []
weight_list = []

if(args.l1_weight > 0):
    l1_loss = L1Loss()
    loss_list.append(l1_loss)
    weight_list.append(args.l1_weight)

if(args.smooth_weight > 0):
    smooth_loss = SmoothDispLoss()
    loss_list.append(smooth_loss)
    weight_list.append(args.smooth_weight)

if(args.per_weight > 0):
    vgg_layers = [PerceptualLoss.L1, PerceptualLoss.L2, PerceptualLoss.L3, PerceptualLoss.L4]
    per_layer_list = [vgg_layers[i-1] for i in args.per_layers]
    per_loss = PerceptualLoss(per_layer_list, DEVICE, AGG_PROD)
    loss_list.append(per_loss)
    weight_list.append(args.per_weight)

criterion = SSLCriterion(loss_list, weight_list)
trainer = DisparityTrainer(
    lr=LR,
    device=DEVICE,
    trainlist=TRAIN_LIST,
    vallist=TEST_METAL_LIST,
    datapath=DATA_DIR,
    batch_size=BATCH_SIZE,
    maxdisp=MAX_DISP,
    criterion=criterion,
    pretrain=PRETRAIN_CHECKPOINT,
    num_workers=NUM_WORKERS,
    batch_norm=BATCH_NORM
)


best_EPE = 10000
train_losses, train_EPEs, val_EPEs = [], [], []
for epoch in range(EPOCHS):
    train_loss, train_EPE = trainer.train_one_epoch(epoch)
    val_EPE = trainer.validate()
    train_losses += train_loss
    train_EPEs += train_EPE
    val_EPEs.append(val_EPE)
    ckpt = {
        'epoch': epoch,
        'epe': val_EPE,
        'model': trainer.get_model(),
    }
    if val_EPE < best_EPE:
        save_file = f"dispnet_epoch_{epoch}_BEST"
    else:
        save_file = f"dispnet_epoch_{epoch}"
    best_EPE = min(best_EPE, val_EPE)
    torch.save(ckpt, CKPT_DIR / save_file)
    np.save(RESULTS_DIR / "train_losses", np.array(train_losses))
    np.save(RESULTS_DIR / "train_EPEs", np.array(train_EPEs))
    np.save(RESULTS_DIR / "val_EPEs", np.array(val_EPEs))
    