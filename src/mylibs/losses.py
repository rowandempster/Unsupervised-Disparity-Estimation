
from __future__ import print_function
from typing import Optional, List

import torch
import torch.nn as nn

import torchvision.models as models

from mylibs.reconstruction import estimate_left

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