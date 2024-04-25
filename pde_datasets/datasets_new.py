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
    

class Kdv(Dataset):
    def __init__(
        self, datapath, nx, sub, n_train=None, n_test=None):
        self.S = int(nx // sub)
        data = scipy.io.loadmat(datapath)
        a = data['input']
        u = data['output']
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


class NavierStokes(Dataset):
    def __init__(
        self, datapath, nx, sub=1, T=49, t_interval=1, n_train=None, n_test=None):
        self.S = nx // sub
        self.T = T
        self.sub = sub
        self.t_interval = t_interval
        self.n_train = n_train
        self.n_test = n_test
        #### new #################################
        try:
            data = h5py.File(datapath)['u']
        except:
            data = scipy.io.loadmat(datapath)['u']
            data = np.array(data).transpose(3, 1, 2, 0)
        print(data.shape)
        ###########################################
        if n_train:
            self.a = torch.tensor(data[9:9 + self.T, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
            self.u = torch.tensor(data[9 + 1:9 + self.T + 1, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
            self.a = rearrange(self.a, 'b m n t -> (b t) m n 1')
            self.u = rearrange(self.u, 'b m n t -> (b t) m n 1')
        if n_test:
            self.a = torch.tensor(data[9:9 + self.T, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3)      # channel dimension = 1
            self.u = torch.tensor(data[9 + 1:9 + self.T + 1, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3) # channel dimension = 1
        
        if n_train and n_test:
            raise ValueError
        if not n_train and not n_test:
            raise ValueError
        
        # geometry locations (x, y)
        mesh1 = torch.tensor(np.linspace(0, 1, self.S), dtype=torch.float)
        mesh2 = torch.tensor(np.linspace(0, 1, self.S), dtype=torch.float)
        mesh1 = mesh1.reshape(self.S, 1, 1).repeat([1, self.S, 1])
        mesh2 = mesh2.reshape(1, self.S, 1).repeat([self.S, 1, 1])
        self.mesh = torch.cat((mesh1, mesh2), dim=-1)                      # (S x S, 2)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        if self.n_train:
            a = self.a[idx]
            return torch.cat([a, self.mesh], dim=-1), self.u[idx]
        if self.n_test:
            return self.a[idx], self.u[idx]
        

class NavierStokes3D(Dataset):
    def __init__(self, datapath, nx, sub=1, T_in=10, T=40, n_train=None, n_test=None):
        self.S = nx // sub
        self.T_in = T_in
        self.T = T
        self.sub = sub
        self.n_train = n_train
        self.n_test = n_test
        data = h5py.File(datapath)['u']
        if n_train:
            self.a = torch.tensor(data[:T_in, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3).reshape(n_train, self.S, self.S, 1, T_in).repeat([1, 1, 1, T, 1])
            self.u = torch.tensor(data[T_in:T_in + T, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
        if n_test:
            self.a = torch.tensor(data[:T_in, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3).reshape(n_test, self.S, self.S, 1, T_in).repeat([1, 1, 1, T, 1])
            self.u = torch.tensor(data[T_in:T_in + T, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3)
        
        if n_train and n_test:
            raise ValueError
        if not n_train and not n_test:
            raise ValueError
        
        self.mesh = torch3dgrid(self.S, self.S, T)

    def __len__(self):
        return self.a.shape[0]
    
    def __getitem__(self, idx):
        a = self.a[idx]
        return torch.cat([a, self.mesh], dim=3), self.u[idx]


class ShallowWater(Dataset):
    def __init__(
        self, datapath, sub=2, T=20, t_interval=1, n_train=None, n_test=None):
        self.datapath = datapath
        self.T = T
        self.sub = sub
        self.t_interval = t_interval
        self.n_train = n_train
        self.n_test = n_test

        data_train, data_test = [], []
        if n_train:
            for i in range(28):
                f = h5py.File(os.path.join(self.datapath, f"traj_{i:04d}.h5"), mode="r")
                if i == 0:
                    data_train = torch.stack([
                        torch.from_numpy(f['tasks/height'][...]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][...] * 2),
                    ], dim=0)
                    data_train = rearrange(data_train, 'c (n_traj n_seq) h w -> c n_traj n_seq h w', n_traj=8)
                else:
                    data = torch.stack([
                        torch.from_numpy(f['tasks/height'][...]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][...] * 2),
                    ], dim=0)
                    data = rearrange(data, 'c (n_traj n_seq) h w -> c n_traj n_seq h w', n_traj=8)
                    data_train = torch.cat((data_train, data), dim=1)
            print(data_train.shape)
            self.a = data_train[:, :, :self.T, ::sub, ::sub].float().permute(1, 2, 3, 4, 0)
            self.u = data_train[:, :, 1:self.T + 1, ::sub, ::sub].float().permute(1, 2, 3, 4, 0)
            self.a = rearrange(self.a, 'b t m n c -> (b t) m n c')
            self.u = rearrange(self.u, 'b t m n c -> (b t) m n c')
        
        if n_test:
            for i in range(2):
                f = h5py.File(os.path.join(self.datapath, f"traj_{i:04d}.h5"), mode="r")
                if i == 0:
                    data_test = torch.stack([
                        torch.from_numpy(f['tasks/height'][...]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][...] * 2),
                    ], dim=0)
                    data_test = rearrange(data_test, 'c (n_traj n_seq) h w -> c n_traj n_seq h w', n_traj=8)
                else:
                    data = torch.stack([
                        torch.from_numpy(f['tasks/height'][...]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][...] * 2),
                    ], dim=0)
                    data = rearrange(data, 'c (n_traj n_seq) h w -> c n_traj n_seq h w', n_traj=8)
                    data_test = torch.cat((data_test, data), dim=1)

            # Total sequence length
            self.a = data_test[:, :, :self.T, ::sub, ::sub].float().permute(1, 2, 3, 4, 0)
            self.u = data_test[:, :, 1:self.T + 1, ::sub, ::sub].float().permute(1, 2, 3, 4, 0)

            self.a = rearrange(self.a, 'b t m n c -> b m n c t')
            self.u = rearrange(self.u, 'b t m n c -> b m n c t')
        
        if n_train and n_test:
            raise ValueError
        if not n_train and not n_test:
            raise ValueError

        # geometry locations (x, y, z)
        coords_list = []
        print(self.a.shape, self.u.shape)
        phi = torch.tensor(f['tasks/vorticity'].dims[1][0][:].ravel()[::sub])
        theta = torch.tensor(f['tasks/vorticity'].dims[2][0][:].ravel()[::sub])

        spherical = get_mgrid_from_tensors([phi, theta])
        phi_vert = spherical[..., 0]
        theta_vert = spherical[..., 1]
        r = 1
        x = torch.cos(phi_vert) * torch.sin(theta_vert) * r
        y = torch.sin(phi_vert) * torch.sin(theta_vert) * r
        z = torch.cos(theta_vert) * r
        coords_list.append(torch.stack([x, y, z], dim=-1))

        self.coords_ang = get_mgrid_from_tensors([phi, theta])
        self.mesh = torch.cat(coords_list, dim=-1).float()
        self.coord_dim = self.mesh.shape[-1]                 # (h x w, 3)
        print(self.a.shape, self.u.shape, self.mesh.shape)

        self.mask = generate_skipped_lat_lon_mask(self.coords_ang)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        if self.n_train:
            a = self.a[idx]
            return torch.cat([a, self.mesh], dim=-1), self.u[idx]
        if self.n_test:
            return self.a[idx], self.u[idx]
        

class ShallowWater_new(Dataset):
    def __init__(
        self, datapath, sub=2, T=20, t_interval=1, n_train=None, n_test=None):
        self.datapath = datapath
        self.T = T
        self.sub = sub
        self.t_interval = t_interval
        self.n_train = n_train
        self.n_test = n_test

        data_train, data_test = [], []
        if n_train:
            for i in range(28):
                f = h5py.File(os.path.join(self.datapath, f"traj_{i:04d}.h5"), mode="r")
                if i == 0:
                    data_train = torch.stack([
                        torch.from_numpy(f['tasks/height'][...]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][...] * 2),
                    ], dim=0)
                    data_train = rearrange(data_train, 'c (n_traj n_seq) h w -> c n_traj n_seq h w', n_traj=8)
                else:
                    data = torch.stack([
                        torch.from_numpy(f['tasks/height'][...]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][...] * 2),
                    ], dim=0)
                    data = rearrange(data, 'c (n_traj n_seq) h w -> c n_traj n_seq h w', n_traj=8)
                    data_train = torch.cat((data_train, data), dim=1)
            print(data_train.shape)
            self.a = data_train[:, :, :self.T, ::sub, ::sub].float().permute(1, 2, 3, 4, 0)
            self.u = data_train[:, :, 1:self.T + 1, ::sub, ::sub].float().permute(1, 2, 3, 4, 0)
            self.a = rearrange(self.a, 'b t m n c -> (b t) m n c')
            self.u = rearrange(self.u, 'b t m n c -> (b t) m n c')
        
        if n_test:
            for i in range(2):
                f = h5py.File(os.path.join(self.datapath, f"traj_{i:04d}.h5"), mode="r")
                if i == 0:
                    data_test = torch.stack([
                        torch.from_numpy(f['tasks/height'][...]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][...] * 2),
                    ], dim=0)
                    data_test = rearrange(data_test, 'c (n_traj n_seq) h w -> c n_traj n_seq h w', n_traj=4)
                else:
                    data = torch.stack([
                        torch.from_numpy(f['tasks/height'][...]) * 3000.,
                        torch.from_numpy(f['tasks/vorticity'][...] * 2),
                    ], dim=0)
                    data = rearrange(data, 'c (n_traj n_seq) h w -> c n_traj n_seq h w', n_traj=4)
                    data_test = torch.cat((data_test, data), dim=1)

            print(data_test.shape)
            # Total sequence length
            self.a = data_test[:, :, :self.T, ::sub, ::sub].float().permute(1, 2, 3, 4, 0)
            self.u = data_test[:, :, 1:self.T + 1, ::sub, ::sub].float().permute(1, 2, 3, 4, 0)

            self.a = rearrange(self.a, 'b t m n c -> b m n c t')
            self.u = rearrange(self.u, 'b t m n c -> b m n c t')
        
        if n_train and n_test:
            raise ValueError
        if not n_train and not n_test:
            raise ValueError

        # geometry locations (x, y, z)
        coords_list = []
        print(self.a.shape, self.u.shape)
        phi = torch.tensor(f['tasks/vorticity'].dims[1][0][:].ravel()[::sub])
        theta = torch.tensor(f['tasks/vorticity'].dims[2][0][:].ravel()[::sub])

        spherical = get_mgrid_from_tensors([phi, theta])
        phi_vert = spherical[..., 0]
        theta_vert = spherical[..., 1]
        r = 1
        x = torch.cos(phi_vert) * torch.sin(theta_vert) * r
        y = torch.sin(phi_vert) * torch.sin(theta_vert) * r
        z = torch.cos(theta_vert) * r
        coords_list.append(torch.stack([x, y, z], dim=-1))

        self.coords_ang = get_mgrid_from_tensors([phi, theta])
        self.mesh = torch.cat(coords_list, dim=-1).float()
        self.coord_dim = self.mesh.shape[-1]                 # (h x w, 3)
        print(self.a.shape, self.u.shape, self.mesh.shape)

        self.mask = generate_skipped_lat_lon_mask(self.coords_ang)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        if self.n_train:
            a = self.a[idx]
            return torch.cat([a, self.mesh], dim=-1), self.u[idx]
        if self.n_test:
            return self.a[idx], self.u[idx]


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
            self.input = input[n_train : n_train + n_test]
            self.mesh = self.mesh[n_train : n_train + n_test]
            self.output = output[n_train : n_train + n_test]

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
            self.input = torch.tensor(input[:n_train], dtype=torch.float).reshape(n_train, s1, 1, 1, 1).repeat(1, 1, s2, t, 1)
            self.output = torch.tensor(output[:n_train], dtype=torch.float)
        if n_test:
            self.input = torch.tensor(input[-n_test:], dtype=torch.float).reshape(n_test, s1, 1, 1, 1).repeat(1, 1, s2, t, 1)
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


class ERA5_temp(Dataset):
    def __init__(self, datapath, sub, n_train, n_test, is_train):
        ds = xr.open_dataset(datapath, engine='cfgrib')
        self.is_train = is_train
        data = np.array(ds["t2m"])
        data = torch.tensor(data)
        data = data[:, :720, :]

        h = int(((721 - 1) / sub))
        s = int(((1441 - 1) / sub))
        print(data.shape)

        x_train = data[:-1][:n_train, ::sub, ::sub]
        y_train = data[1:][:n_train, ::sub, ::sub]

        x_test = data[:-1][-n_test:, ::sub, ::sub]
        self.y_test = data[1:][-n_test:, ::sub, ::sub]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        self.y_normalizer = UnitGaussianNormalizer(y_train)
        self.y_train = self.y_normalizer.encode(y_train)

        self.x_train = x_train.reshape(n_train, h, s, 1)
        self.x_test = x_test.reshape(n_test, h, s, 1)

        self.mesh = torch2dgrid(h, s, bot=(-0.5, 0), top=(0.5, 2))
        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape, self.mesh.shape)

    def __len__(self):
        if self.is_train:
            return self.x_train.shape[0]
        else:
            return self.x_test.shape[0]
        
    def __getitem__(self, idx):
        if self.is_train:
            x = self.x_train[idx]
            return torch.cat([x, self.mesh], dim=2), self.y_train[idx]
        else:
            x = self.x_test[idx]
            return torch.cat([x, self.mesh], dim=2), self.y_test[idx]


# NEW: Time-dependent PDEs
class NavierStokes_time(Dataset):
    def __init__(self, datapath, nx, sub, T_start=0, T_in=10, T_out=40, n_train=None, n_test=None, is_train=True):
        self.T_start = T_start
        self.T_in = T_in
        self.T_out = T_out
        self.sub = sub
        self.n_train = n_train
        self.n_test = n_test
        self.is_train = is_train
        self.S = nx // sub
        
        #### new #################################
        try:
            data = h5py.File(datapath)['u']
        except:
            data = scipy.io.loadmat(datapath)['u']
            data = np.array(data).transpose(3, 1, 2, 0)
        print(data.shape)
        ###########################################

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


class ERA5_day_5years_new(Dataset):
    def __init__(self, datapath, sub, n_train, n_test, is_train, version=4):
        self.n_train = n_train
        self.n_test = n_test
        self.is_train = is_train

        r = sub # 2**4 for 4^o, # 2**3 for 2^o
        h = int(((721 - 1)/r))
        s = int(((1441 - 1)/r))

        np.random.seed(0)

        ds = xr.open_dataset(datapath, engine='cfgrib')
        # data = np.array(ds["t2m"]) - 273
        # print(np.max(data[:]), np.min(data[:]))
        data = data / 300
        data = torch.tensor(data)
        data = data[:,:720,:]

        Tn = 7*int(1937/7)
        x_data = data[:-1, :, :]
        y_data = data[1:, :, :]

        x_data = x_data[:Tn, :, :]
        y_data = y_data[:Tn, :, :]

        x_data = x_data.reshape(1932,720,1440,1)
        x_data = list(torch.split(x_data, int(1932/7), dim=0))
        x_data = torch.cat((x_data), dim=3)

        y_data = y_data.reshape(1932,720,1440,1)
        y_data = list(torch.split(y_data, int(1932/7), dim=0))
        y_data = torch.cat((y_data), dim=3)

        x_train = x_data[:n_train, ::r, ::r]
        y_train = y_data[:n_train, ::r, ::r]

        x_test = y_data[-n_test:, ::r, ::r]
        y_test = y_data[-n_test:, ::r, ::r]

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


class ERA5_day_5years(Dataset):
    def __init__(self, datapath, sub, n_train, n_test, is_train, version=4):
        self.n_train = n_train
        self.n_test = n_test
        self.is_train = is_train
        np.random.seed(0)

        ds = xr.open_dataset(datapath, engine='cfgrib')
        self.is_train = is_train
        # data = np.array(ds["t2m"]) / 300
        data = np.array(ds["t2m"]) - 273
        data = data / 30
        print(np.max(data), np.min(data))

        # data = np.array(ds["2m"])
        
        h = int(((721 - 1) / sub))
        s = int(((1441 - 1) / sub))

        Tn = 7 * int(1937 / 7)
        data = data[:, :720, :]
        data = data[:, ::sub, ::sub]

        x_train, x_test = [], []
        y_train, y_test = [], []
        x_data, y_data = [], []

        if version == 0:
            # 7-to-1 naive mapping
            for i in range(data.shape[0] - 7):
                if i < data.shape[0] - 14:
                    x_train.append(data[i:i + 7])
                    y_train.append(data[i + 7:i + 8])
                else:
                    x_test.append(data[i:i + 7])
                    y_test.append(data[i + 7:i + 8])

        elif version == 1:
            # 7-to-1 mapping time marching in physical space
            for i in range(data.shape[0] - 13):
                if i < data.shape[0] - 14:
                    x_train.append(data[i:i + 7])
                    y_train.append(data[i + 7:i + 8])
                else:
                    x_test.append(data[i:i + 7])
                    y_test.append(data[i + 7:i + 14])
        
        elif version == 2:
            # 7-to-latent-to-1 with latent marching making 7-to-7
            for i in range(data.shape[0] - 13):
                if i < data.shape[0] - 14:
                    x_train.append(data[i:i + 7])
                    y_train.append(data[i + 7:i + 14])
                else:
                    x_test.append(data[i:i + 7])
                    y_test.append(data[i + 7:i + 14])

        elif version == 3:
            # 7-to-latent-to-1 with latent marching making 7-to-7, 1924 random split
            for i in range(data.shape[0] - 13):
                x_data.append(data[i:i + 7])
                y_data.append(data[i + 7:i + 14])

            x_data = np.array(x_data).transpose(0, 2, 3, 1)
            y_data = np.array(y_data).transpose(0, 2, 3, 1)

            indices = np.arange(x_data.shape[0])
            np.random.shuffle(indices)
            train_idx, test_idx = indices[:1824], indices[1824:]

            x_train = x_data[train_idx]
            y_train = y_data[train_idx]

            x_test = x_data[test_idx]
            y_test = y_data[test_idx]

        elif version == 4:
            # 7-to-latent-to-1 with latent marching making 7-to-7, 275 random split
            print(data.shape[0])
            for i in range(0, data.shape[0] - 14, 7):
                x_data.append(data[i:i + 7])
                y_data.append(data[i + 7:i + 14])
            
            x_data = np.array(x_data).transpose(0, 2, 3, 1)
            y_data = np.array(y_data).transpose(0, 2, 3, 1)

            indices = np.arange(x_data.shape[0])
            # np.random.shuffle(indices)
            train_idx, test_idx = indices[:n_train], indices[n_train:]

            x_train = x_data[train_idx]
            y_train = y_data[train_idx]

            x_test = x_data[test_idx]
            y_test = y_data[test_idx]

        else:
            raise NotImplementedError

        if version == 0 or version == 1 or version == 2:
            x_train = np.array(x_train).transpose(0, 2, 3, 1)
            y_train = np.array(y_train).transpose(0, 2, 3, 1)

            x_test = np.array(x_test).transpose(0, 2, 3, 1)
            y_test = np.array(y_test).transpose(0, 2, 3, 1)
        
        print(indices)
        # np.save('indices.npy', indices)

        self.x_train = torch.tensor(x_train, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.float)

        self.x_test = torch.tensor(x_test, dtype=torch.float)
        self.y_test = torch.tensor(y_test, dtype=torch.float)

        self.mesh = torch2dgrid(h, s, bot=(-0.5, 0), top=(0.5, 2))

        self.T = 7
        self.t_interval = 1

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


class ERA5_day_5years_per(Dataset):
    def __init__(self, datapath, sub, n_train=None, n_test=None):
        self.n_train = n_train
        self.n_test = n_test
        np.random.seed(0)

        ds = xr.open_dataset(datapath, engine='cfgrib')
        # data = np.array(ds["t2m"]) / 300
        data = np.array(ds["t2m"]) - 273
        data = data / 30
        print(np.max(data), np.min(data))

        # data = np.array(ds["2m"])
        
        h = int(((721 - 1) / sub))
        s = int(((1441 - 1) / sub))

        Tn = 7 * int(1937 / 7)
        data = data[:, :720, :]
        # data = data[:, ::sub, ::sub]

        # x_train, x_test = [], []
        # y_train, y_test = [], []
        # x_data, y_data = [], []

        # # 7-to-latent-to-1 with latent marching making 7-to-7, 275 random split
        # print(data.shape[0])
        # for i in range(0, data.shape[0] - 14, 7):
        #     x_data.append(data[i:i + 7])
        #     y_data.append(data[i + 7:i + 14])

        if n_train:
            self.a = torch.tensor(data[:self.n_train * 7, ::sub, ::sub], dtype=torch.float)
            self.u = torch.tensor(data[1:self.n_train * 7 + 1, ::sub, ::sub], dtype=torch.float)
 
        if n_test:
            self.a = torch.tensor(data[-self.n_test * 7:-1, ::sub, ::sub], dtype=torch.float)    # channel dimension = 1
            self.u = torch.tensor(data[-self.n_test * 7 + 1:, ::sub, ::sub], dtype=torch.float) # channel dimension = 1
        
        # x_data = np.array(x_data).transpose(0, 2, 3, 1)
        # y_data = np.array(y_data).transpose(0, 2, 3, 1)

        # indices = np.arange(x_data.shape[0])
        # train_idx, test_idx = indices[:n_train], indices[n_train:]

        # x_train = x_data[train_idx]
        # y_train = y_data[train_idx]

        # x_test = x_data[test_idx]
        # y_test = y_data[test_idx]
        
        # print(indices)
        # np.save('indices.npy', indices)

        # self.x_train = torch.tensor(x_train, dtype=torch.float)
        # self.y_train = torch.tensor(y_train, dtype=torch.float)

        # self.x_train = rearrange(self.x_train, 'b m n t -> (b t) m n 1')
        # self.y_train = rearrange(self.y_train, 'b m n t -> (b t) m n 1')

        # self.x_test = torch.tensor(x_test, dtype=torch.float)
        # self.y_test = torch.tensor(y_test, dtype=torch.float)

        self.mesh = torch2dgrid(h, s, bot=(-0.5, 0), top=(0.5, 2))

        print(self.a.shape, self.u.shape)

        # print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape, self.mesh.shape)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        if self.n_train:
            a = self.a[idx].unsqueeze(-1)
            # print(a.shape, self.mesh.shape)
            return torch.cat([a, self.mesh], dim=-1), self.u[idx]
        if self.n_test:
            a = self.a[idx].unsqueeze(-1)
            return torch.cat([a, self.mesh], dim=-1), self.u[idx]
    

# if __name__ == "__main__":
#     naca_dir = '/shared_data/ai_lab/seungjun.lee/pde_data/airfoil/NACA_Cylinder_'
#     air = Airfoil(
#         input1_path=naca_dir + 'X.npy',
#         input2_path=naca_dir + 'Y.npy',
#         output_path=naca_dir + 'Q.npy',
#         n_train=1000, n_test=100)
#     print(air.input.shape, air.output.shape)

# if __name__ == "__main__":
#     ela_dir = '/shared_data/ai_lab/seungjun.lee/pde_data/elasticity/Random_UnitCell_'
#     ela = Elasticity(
#         input1_path=ela_dir + 'rr_10.npy',
#         input2_path=ela_dir + 'XY_10.npy',
#         output_path=ela_dir + 'sigma_10.npy',
#         n_train=100, n_test=100)
#     print(ela.input.shape, ela.mesh.shape, ela.output.shape)

# if __name__ == "__main__":
#     pla_dir = '/shared_data/ai_lab/seungjun.lee/pde_data/plasticity/plas_N987_T20.mat'
#     pla = Plasticity(datapath=pla_dir, s1=101, s2=31, t=20, n_train=10)
#     print(pla.input.shape, pla.output.shape)

# if __name__ == "__main__":
#     sha_dir = '/shared_data/ai_lab/seungjun.lee/pde_data/shallowwater/shallowwater_train'
#     sha = ShallowWater(
#         datapath=sha_dir, sub=2, T=20, t_interval=1, n_test=2)
#     print(sha.a.shape, sha.u.shape, sha.mesh.shape)

# if __name__ == "__main__":
#     ns_dir = 'data/navierstokes/ns_V1e-3_N5000_T50.mat'
#     ns = NavierStokes3D(datapath=ns_dir, nx=64, sub=1, T_in=10, T=40, n_test=10)
#     input, output = ns.__getitem__(1)
#     print(input.shape, output.shape)

# if __name__ == "__main__":
#     datapath = '/shared_data/ai_lab/seungjun.lee/pde_data/era5/ERA5_day_5years.grib'
#     era5 = ERA5_day_5years(datapath=datapath, sub=8, n_train=250, n_test=25, is_train=True, version=3)
#     input, output = era5.__getitem__(12)
#     print(input.shape, output.shape)


# if __name__ == "__main__":
#     datapath = '/shared_data/ai_lab/seungjun.lee/pde_data/shallowwater/shallowwater_test'
#     sw = ShallowWater_new(datapath=datapath, sub=1, T=39, t_interval=1, n_test=2)
#     print(sw.a.shape, sw.u.shape)
