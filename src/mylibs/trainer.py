from __future__ import print_function

import time
from pathlib import Path
import sys
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('extlibs')

from IRS.dataloader.IRSLoader import IRSDataset
from IRS.losses.multiscaleloss import EPE
from IRS.networks.DispNetC import DispNetC
from IRS.utils.AverageMeter import AverageMeter
from IRS.utils.common import logger


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
