import os
import numpy as np
import torch
from functools import reduce, partial
import operator
import scipy


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        self.data = scipy.io.loadmat(self.file_path)
        self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float():
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


def torch1dgrid(num, bot=0, top=1):
    arr = torch.linspace(bot, top, steps=num)
    mesh = torch.stack([arr], dim=1)
    return mesh


def torch2dgrid(num_x, num_y, bot=(0, 0), top=(1, 1)):
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing='ij')
    mesh = torch.stack([xx, yy], dim=2)
    return mesh


def torch3dgrid(num_x, num_y, num_z, bot=(0, 0, 0), top=(1, 1, 1)):
    x_bot, y_bot, z_bot = bot
    x_top, y_top, z_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y)
    z_arr = torch.linspace(z_bot, z_top, steps=num_z)
    xx, yy, zz = torch.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
    mesh = torch.stack([xx, yy, zz], dim=3)
    return mesh


def get_mgrid_from_tensors(tensors):
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid


def generate_skipped_lat_lon_mask(coords, base_jump=0):
    lons = coords[:, 0, 0].numpy()
    lats = coords[0, :, 1].numpy()
    n_lon = lons.size
    delta_dis_equator = 2 * np.pi * 1 / n_lon
    mask_list = []
    for lat in lats:
        delta_dis_lat = 2 * np.pi * np.sin(lat) / n_lon
        ratio = delta_dis_lat / delta_dis_equator
        n = int(np.ceil(np.log(ratio) / np.log(2 / 5)))
        mask = torch.zeros(n_lon)
        mask[::2 ** (n - 1 + base_jump)] = 1
        mask_list.append(mask)

    mask = torch.stack(mask_list, dim=-1)
    return mask


