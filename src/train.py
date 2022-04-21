from __future__ import print_function
import time
from typing import Callable, Optional
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from IRS.dataloader.IRSLoader import IRSDataset
from IRS.utils.AverageMeter import AverageMeter
from IRS.utils.common import logger
from IRS.losses.multiscaleloss import EPE
from IRS.dataloader.IRSLoader import IRSDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IRS.networks.DispNetC import DispNetC
from torch.utils.data import DataLoader
import torchvision.models as models
from typing import List
from tqdm import tqdm
import argparse

class DisparityTrainer(object):
    def __init__(
        self,
        lr: float,
        device: str,
        trainlist: str,
        vallist: str,
        datapath: str,
        batch_size: int,
        maxdisp: int,
        criterion: Callable,
        pretrain: Optional[str]=None,
        num_workers=4,
        batch_norm=False,
    ):
        super(DisparityTrainer, self).__init__()
        self.lr = lr
        self.current_lr = lr
        self.device = device
        self.trainlist = trainlist
        self.vallist = vallist
        self.datapath = datapath
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.maxdisp = maxdisp
        self.num_workers = num_workers
        self.criterion = criterion
        self.epe = EPE
        self.batch_norm = batch_norm

        self.initialize()

    def _prepare_dataset(self):
        train_dataset = IRSDataset(txt_file=self.trainlist, root_dir=self.datapath, phase='train', load_norm=False)
        test_dataset = IRSDataset(txt_file=self.vallist, root_dir=self.datapath, phase='test', load_norm=False)
        self.img_size = train_dataset.get_img_size()
        self.scale_size = train_dataset.get_scale_size()
        self.focal_length = train_dataset.get_focal_length()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.num_batches_per_epoch = len(self.train_loader)

    def _build_net(self):
        self.net = DispNetC(batchNorm=self.batch_norm, input_channel=3, maxdisp=self.maxdisp).to(self.device)

        if self.pretrain is not None:
            assert Path(self.pretrain).exists(), f"{self.pretrain} does not exist"
            model_data = torch.load(self.pretrain)
            assert "model" in model_data.keys(), f"{self.pretrain} does not contain model"
            self.net.load_state_dict(model_data["model"])

    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            self.lr,
            betas=(momentum, beta),
            amsgrad=True,
        )

    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()

    def adjust_learning_rate(self, epoch):
        cur_lr = self.lr / (2 ** (epoch // 10))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = cur_lr
        self.current_lr = cur_lr
        return cur_lr

    def train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        EPEs = AverageMeter()
        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))

        for i_batch, sample_batched in enumerate(tqdm(self.train_loader)):

            left_input = sample_batched["img_left"].to(self.device)
            right_input = sample_batched["img_right"].to(self.device)
            input = torch.cat((left_input, right_input), 1)
            target_disp = sample_batched["gt_disp"].to(self.device)
            data_time.update(time.time() - end)

            self.optimizer.zero_grad()

            disps = self.net(input)
            loss = self.criterion(left_input, right_input, disps[0])
            epe = self.epe(disps[0], target_disp)
            # record loss and EPE
            losses.update(loss.data.item(), input.size(0))
            EPEs.update(epe.data.item(), input.size(0))
            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i_batch % 10 == 0:
                logger.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                    "EPE {EPE.val:.3f} ({EPE.avg:.3f})\t".format(
                        epoch,
                        i_batch,
                        self.num_batches_per_epoch,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        EPE=EPEs,
                    )
                )

        return losses.hist, EPEs.hist

    def validate(self):
        batch_time = AverageMeter()
        EPEs = AverageMeter()
        # switch to evaluate mode
        end = time.time()
        self.net.eval()
        for i, sample_batched in enumerate(tqdm(self.test_loader)):

            left_input = sample_batched["img_left"].to(self.device)
            right_input = sample_batched["img_right"].to(self.device)
            left_input = F.interpolate(left_input, self.scale_size, mode="bilinear")
            right_input = F.interpolate(right_input, self.scale_size, mode="bilinear")

            input = torch.cat((left_input, right_input), 1)

            target_disp = sample_batched["gt_disp"].to(self.device)
            
            with torch.no_grad():
                disp = self.net(input)[0]

            # upsampling the predicted disparity map
            disp = nn.Upsample(size=target_disp.shape[2:], mode='bilinear')(disp)
            epe = self.epe(disp, target_disp)

            # record loss and EPE
            EPEs.update(epe.data.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            logger.info(
                "Test: [{0}/{1}]\t Time {2}\t EPE {3}".format(
                    i,
                    len(self.test_loader),
                    batch_time.val,
                    EPEs.val,
                )
            )

        logger.info(" * EPE {:.3f}".format(EPEs.avg))
        return EPEs.avg

    def get_model(self):
        return self.net.state_dict()


def occ_mask(disp: torch.Tensor):
    '''
    disp: [N, C(1), H, W]
    '''
    N, _, H, W = disp.shape
    # [N, H, W]
    x_base = torch.linspace(0, W-1, W).repeat(N,
                H, 1).type_as(disp)
    # [N, H, W]
    x_query = (x_base - disp.squeeze(1)).round().long()
    # [N, H, W]
    disp_order = disp.squeeze(1).sort(dim=-1, descending=True).indices
    x_query = x_query.gather(-1, disp_order)
    x_base = x_base.gather(-1, disp_order)
    x_query_order = torch.sort(x_query, stable=True, dim=-1).indices
    x_query = x_query.gather(-1, x_query_order)
    x_base = x_base.gather(-1, x_query_order)
    # [N, H, W]
    occ_mask = (x_query.roll(shifts=1, dims=-1) - x_query) == 0
    return occ_mask.gather(-1, x_base.sort(dim=-1).indices)



def estimate_left(im_l: torch.Tensor, im_r: torch.Tensor, disp: torch.Tensor):
    '''
    im_l, im_r: [N, C(3), H, W]
    disp: [N, C(1), H, W]
    '''
    N, _, H, W = disp.shape
    x_base = torch.linspace(0, 1, W).repeat(N, 1,
                H, 1).type_as(im_l)
    y_base = torch.linspace(0, 1, H).repeat(N, 1,
                W, 1).transpose(2, 3).type_as(im_l)
    x_query = x_base - (disp / W)
    # [1, H, W]
    valid_mask = (x_query >= 0) & ~occ_mask(disp).unsqueeze(1)
    flow_field = 2 * torch.stack((x_query, y_base), dim=4) - 1
    est_l = F.grid_sample(im_r, flow_field.squeeze(1), mode='bilinear', padding_mode='zeros')
    return torch.where(valid_mask, est_l, im_l)


class PerceptualLoss(nn.Module):
    L1 = 4
    L2 = 9
    L3 = 16
    L4 = 30

    def __init__(self, layers: List[int], device: str, agg_prod: Optional[bool]=True):
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        vgg16 = models.vgg16(pretrained=True)
        self.feat_seq = list(vgg16.named_children())[0][1].to(device)
        self.agg_prod = agg_prod

    def _compute_l1_diff(self, inputs: torch.Tensor, targets: torch.Tensor):
        '''
        Returns [N, M, H, W] where M is len(self.layers)
        '''
        diffs = []
        H, W = inputs.shape[2:]
        if self.agg_prod:
            up = torch.nn.Upsample(size=(H,W))
        for layer in self.layers:
            input_map = self.feat_seq[0:layer](inputs)
            target_map = self.feat_seq[0:layer](targets)
            if self.agg_prod:
                diffs.append(up(torch.norm(input_map - target_map, p=1, dim=1).unsqueeze(1)).squeeze(1))
            else:
                diffs.append(torch.norm(input_map - target_map, p=1, dim=1))
        if self.agg_prod:
            return torch.stack(diffs, dim=1)
        else:
            return diffs

    def _compare_images(self, est: torch.Tensor, gt: torch.Tensor):
        '''
        est, gt: [N, C(3), H, W]
        '''
        if self.agg_prod:
            return torch.prod(self._compute_l1_diff(est, gt), dim=1).mean()
        else:
            return sum([diff.mean() for diff in self._compute_l1_diff(est, gt)])

    def forward(self, im_l: torch.Tensor, im_r: torch.Tensor, disp: torch.Tensor):
        '''
        im_l, im_r: [N, C(3), H, W]
        disp: [N, C(1), H, W]
        '''
        return self._compare_images(estimate_left(im_l, im_r, disp), im_l)

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    
    def forward(self, im_l: torch.Tensor, im_r: torch.Tensor, disp: torch.Tensor):
        est_l = estimate_left(im_l, im_r, disp)
        return nn.L1Loss()(est_l, im_l)

class SmoothDispLoss(nn.Module):
    def __init__(self):
        super(SmoothDispLoss, self).__init__()

    def _gradient_x(self, img):
        '''
        img: [N, C, H, W]
        '''
        return img[:,:,:,:-1] - img[:,:,:,1:]

    def _gradient_y(self, img):
        '''
        img: [N, C, H, W]
        '''
        return img[:,:,:-1,:] - img[:,:,1:,:]

    def forward(self, im_l: torch.Tensor, im_r: torch.Tensor, disp: torch.Tensor):
        disp_grad_x = self._gradient_x(disp)
        disp_grad_y = self._gradient_y(disp)
        im_grad_x = self._gradient_x(im_l)
        im_grad_y = self._gradient_y(im_l)
        weight_x = torch.exp(-torch.mean(torch.abs(im_grad_x), 1, keepdim=True))
        weight_y = torch.exp(-torch.mean(torch.abs(im_grad_y), 1, keepdim=True))
        smooth_pen_x = disp_grad_x * weight_x
        smooth_pen_y = disp_grad_y * weight_y
        return smooth_pen_x.abs().mean() + smooth_pen_y.abs().mean()

class SSLCriterion(nn.Module):
    def __init__(self, modules: nn.ModuleList, weights: List[float]):
        super(SSLCriterion, self).__init__()
        self.modules = modules
        assert sum(weights) == 1
        self.weights = weights

    def forward(self, im_l: torch.Tensor, im_r: torch.Tensor, disp: torch.Tensor):
        '''
        im_l, im_r: [N, C(3), H, W]
        disp: [N, C(1), H, W]
        '''
        return sum([w * mod(im_l, im_r, disp) for mod, w in zip(self.modules, self.weights)])


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
TRAIN_LIST = 'IRS/lists/Restaurant_TRAIN.list'
TEST_METAL_LIST = 'IRS/lists/IRS_restaurant_metal_test.list'
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
    