import torch
import torch.nn.functional as F


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
