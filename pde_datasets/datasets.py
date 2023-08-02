import scipy.io
import numpy as np
import os
import h5py

import torch
from torch.utils.data import Dataset
from einops import rearrange

try:
    from pde_datasets.data_utils import *
    from utils import *
except:
    from data_utils import *
import xarray as xr
import os

os.environ["OMP_NUM_THREADS"] = "1"


class Burgers(Dataset):
    def __init__(
            self, datapath, nx, sub, n_train=None, n_test=None):
        self.S = int(nx // sub)
        data = scipy.io.loadmat(datapath)
        a = data['a']
        u = data['u']
        if n_train:
            self.a = torch.tensor(a[:n_train, ::sub], dtype=torch.float)
            self.u = torch.tensor(u[:n_train, ::sub], dtype=torch.float)
        if n_test:
            self.a = torch.tensor(a[-n_test:, ::sub], dtype=torch.float)
            self.u = torch.tensor(u[-n_test:, ::sub], dtype=torch.float)
        if n_train and n_test:
            raise ValueError
        if not n_train and not n_test:
            raise ValueError

        self.mesh = torch1dgrid(self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        a = self.a[idx]
        return torch.cat([a.unsqueeze(1), self.mesh], dim=1), self.u[idx]


class DarcyFlow(Dataset):
    def __init__(
            self, datapath, nx, sub, offset=0, num=1):
        if sub == 1:
            self.S = int(nx)
        else:
            self.S = int(nx // sub) + 1
        data = scipy.io.loadmat(datapath)
        a = data['coeff']
        u = data['sol']
        self.a = torch.tensor(a[offset: offset + num, ::sub, ::sub], dtype=torch.float)
        self.u = torch.tensor(u[offset: offset + num, ::sub, ::sub], dtype=torch.float)

        self.mesh = torch2dgrid(self.S, self.S)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        a = self.a[idx]
        return torch.cat([a.unsqueeze(2), self.mesh], dim=2), self.u[idx]


class Airfoil(Dataset):
    def __init__(self, input1_path, input2_path, output_path, n_train, n_test=None):
        input1 = np.load(input1_path)
        input2 = np.load(input2_path)
        input = np.stack([input1, input2], axis=-1)

        output = np.load(output_path)[:, 4]

        s1 = int(((221 - 1) / 1) + 1)
        s2 = int(((51 - 1) / 1) + 1)

        self.mesh = torch2dgrid(221, 51)

        if not n_train:
            raise ValueError
        if not n_test:
            self.input = torch.tensor(input[:n_train, :s1, :s2], dtype=torch.float)
            self.output = torch.tensor(output[:n_train, :s1, :s2], dtype=torch.float)
        if n_test:
            self.input = torch.tensor(input[n_train:n_train + n_test, :s1, :s2], dtype=torch.float)
            self.output = torch.tensor(output[n_train:n_train + n_test, :s1, :s2], dtype=torch.float)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        input = self.input[idx]
        return torch.cat([input, self.mesh], dim=2), self.output[idx]


class Elasticity(Dataset):
    def __init__(self, input1_path, input2_path, output_path, n_train=None, n_test=None):
        input_rr = np.load(input1_path)
        input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1, 0)

        input_xy = np.load(input2_path)
        input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

        output = np.load(output_path)
        output = torch.tensor(output, dtype=torch.float).permute(1, 0)

        # some feature engineering
        self.center = torch.tensor([0.0001, 0.0001]).reshape(1, 1, 2)
        angle = torch.atan2(input_xy[:, :, 1] - self.center[:, :, 1], input_xy[:, :, 0] - self.center[:, :, 0])
        radius = torch.norm(input_xy - self.center, dim=-1, p=2)
        input_xy = torch.stack([input_xy[:, :, 0], input_xy[:, :, 1], angle, radius], dim=-1)

        self.mesh = input_xy

        input_rr = input_rr.unsqueeze(1).repeat(1, input_xy.shape[1], 1)
        input = torch.cat([input_rr, input_xy], dim=-1)
        print(input_rr.shape, input_xy.shape, input.shape)

        if not n_train:
            raise ValueError
        if not n_test:
            self.input = input[:n_train]
            self.mesh = self.mesh[:n_train]
            self.output = output[:n_train]
        if n_test:
            self.input = input[n_train: n_train + n_test]
            self.mesh = self.mesh[n_train: n_train + n_test]
            self.output = output[n_train: n_train + n_test]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.mesh[idx], self.output[idx]


class Plasticity(Dataset):
    def __init__(self, datapath, s1, s2, t, n_train=None, n_test=None):
        data = scipy.io.loadmat(datapath)

        input = data['input']
        output = data['output']

        if n_train:
            self.input = torch.tensor(input[:n_train], dtype=torch.float).reshape(n_train, s1, 1, 1, 1).repeat(1, 1, s2,
                                                                                                               t, 1)
            self.output = torch.tensor(output[:n_train], dtype=torch.float)
        if n_test:
            self.input = torch.tensor(input[-n_test:], dtype=torch.float).reshape(n_test, s1, 1, 1, 1).repeat(1, 1, s2,
                                                                                                              t, 1)
            self.output = torch.tensor(output[-n_test:], dtype=torch.float)
        if n_train and n_test:
            raise ValueError
        if not n_train and not n_test:
            raise ValueError

        self.mesh = torch3dgrid(s1, s2, t)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        input = self.input[idx]
        return torch.cat([input, self.mesh], dim=3), self.output[idx]


class NavierStokes(Dataset):
    def __init__(self, datapath, nx, sub, T_start=0, T_in=10, T_out=40, n_train=None, n_test=None, is_train=True):
        self.T_start = T_start
        self.T_in = T_in
        self.T_out = T_out
        self.sub = sub
        self.n_train = n_train
        self.n_test = n_test
        self.is_train = is_train
        self.S = nx // sub

        data = h5py.File(datapath)['u']

        if self.is_train:
            self.a = torch.tensor(data[T_start:T_in, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
            self.u = torch.tensor(data[T_in:T_in + T_out, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
        else:
            self.a = torch.tensor(data[T_start:T_in, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3)
            self.u = torch.tensor(data[T_in:T_in + T_out, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3)

        self.mesh = torch2dgrid(self.S, self.S)

        print(self.a.shape, self.u.shape, self.mesh.shape)

    def __len__(self):
        if self.is_train:
            return self.n_train
        else:
            return self.n_test

    def __getitem__(self, idx):
        return torch.cat((self.a[idx], self.mesh), dim=-1), self.u[idx]


class ERA5_temperature(Dataset):
    def __init__(self, datapath, sub, n_train, n_test, is_train):
        self.n_train = n_train
        self.n_test = n_test
        self.is_train = is_train
        np.random.seed(0)

        ds = xr.open_dataset(datapath, engine='cfgrib')
        self.is_train = is_train
        data = np.array(ds["t2m"])

        h = int(((721 - 1) / sub))
        s = int(((1441 - 1) / sub))

        Tn = 7 * int(data.shape[0] / 7)
        data = data[:, :720, :]
        data = data[:, ::sub, ::sub]

        x_data, y_data = [], []

        for i in range(0, data.shape[0] - 14, 7):
            x_data.append(data[i:i + 7])
            y_data.append(data[i + 7:i + 14])

        x_data = np.array(x_data).transpose(0, 2, 3, 1)
        y_data = np.array(y_data).transpose(0, 2, 3, 1)

        x_train = x_data[:n_train]
        y_train = y_data[:n_train]

        x_test = x_data[n_train:]
        y_test = y_data[n_train:]

        self.x_train = torch.tensor(x_train, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.float)

        self.x_test = torch.tensor(x_test, dtype=torch.float)
        self.y_test = torch.tensor(y_test, dtype=torch.float)

        self.mesh = torch2dgrid(h, s, bot=(-0.5, 0), top=(0.5, 2))

        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape, self.mesh.shape)

    def __len__(self):
        if self.is_train:
            return self.n_train
        else:
            return self.n_test

    def __getitem__(self, idx):
        if self.is_train:
            return torch.cat((self.x_train[idx], self.mesh), dim=-1), self.y_train[idx]
        else:
            return torch.cat((self.x_test[idx], self.mesh), dim=-1), self.y_test[idx]

