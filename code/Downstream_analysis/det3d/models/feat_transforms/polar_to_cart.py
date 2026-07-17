import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..registry import FEAT_TRANSFORMS

deg2rad = np.pi / 180.0

# @FEAT_TRANSFORMS.register_module
class PolarToCart(nn.Module):
    def __init__(self, cart_ROI, voxel_size, polar_range, dimension):
        # dimension: 2 (BEV) or 3(3D)
        super(PolarToCart, self).__init__()
        # TODO: initialize the output grid sampling location given range-azimuth and x, y, grid size
        z_min, z_max = cart_ROI['z']
        y_min, y_max = cart_ROI['y']
        x_min, x_max = cart_ROI['x']
        r_min, r_max, a_min, a_max, e_min, e_max = polar_range
        a_min, a_max, e_min, e_max = np.array([a_min, a_max, e_min, e_max]) * deg2rad
        x_coord = torch.arange(x_min, x_max, voxel_size)
        y_coord = torch.arange(y_min, y_max, voxel_size)
        if dimension == '2':
            y_coord, x_coord = torch.meshgrid(y_coord, x_coord)
            yx_coord = torch.stack([y_coord, x_coord], dim=-1)
            r_ratio = (torch.norm(yx_coord, dim=-1) - r_min) / (r_max - r_min)
            a_ratio = (torch.atan2(y_coord, x_coord) - a_min) / (a_max - a_min)
            grid = torch.stack([a_ratio, r_ratio], dim=-1).unsqueeze(0) # 1xHxWx2
        else:
            z_coord = torch.arange(z_min, z_max, voxel_size)
            z_coord, y_coord, x_coord = torch.meshgrid(z_coord, y_coord, x_coord)
            zyx_coord = torch.stack([z_coord, y_coord, x_coord], dim=-1)
            r_ratio = (torch.norm(zyx_coord, dim=-1) - r_min) / (r_max - r_min)
            a_ratio = (torch.atan2(y_coord, x_coord) - a_min) / (a_max - a_min)
            e_ratio = (torch.atan2(z_coord, x_coord) - e_min) / (e_max - e_min)
            grid = torch.stack([e_ratio, a_ratio, r_ratio], dim=-1).unsqueeze(0) # 1xDxHxWx3
        self.register_buffer('grid', grid)
        self.transform_dim = dimension
        # TODO: compare fixed with additional learnable offset


    def forward(self, polar_feature):
        # TODO: implement the 2D polar to cart feature transformation
        grid_expand_dim = (np.ones(len(self.grid.shape) - 1, dtype=int) * -1).tolist()
        cart_feature = F.grid_sample(polar_feature, self.grid.expand(polar_feature.shape[0], *grid_expand_dim), mode='bilinear', align_corners=False)

        return cart_feature
    

if __name__ == '__main__':
    raise SystemExit(
        "PolarToCart is a library module. Import it from the training code or "
        "create a local example with project-relative input tensors."
    )

    dar = dear.mean(axis=1)
    polar_to_cart = PolarToCart(cart_ROI, voxel_size, polar_range, dimension='2')
    polar_to_cart = polar_to_cart.cuda()
    dyx = polar_to_cart(dar)
    print(dyx.shape)
