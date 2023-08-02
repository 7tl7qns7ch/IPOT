import yaml
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
import os
import math

from helpers import *
from models import *
from pde_datasets.datasets import *


torch.manual_seed(123)


def main(args, train_dataset, test_dataset):
    ############################################################################
    # Device
    ############################################################################
    cuda_num = 'cuda:' + args.gpu_num
    device = torch.device(cuda_num if torch.cuda.is_available() else 'cpu')

    ############################################################################
    # Get model - Encoder-Processor-Decoder
    ############################################################################
    if args.model_type == "ipot":
        input_channel = args.input_channel
        pos_channel = args.pos_channel
        num_bands = args.num_bands
        max_resolution = args.max_resolution
        num_latents = args.num_latents
        latent_channel = args.latent_channel
        self_per_cross_attn = args.self_per_cross_attn
        cross_heads_num = args.cross_heads_num
        self_heads_num = args.self_heads_num
        cross_heads_channel = args.cross_heads_channel
        self_heads_channel = args.self_heads_channel
        ff_mult = args.ff_mult
        latent_init_scale = args.latent_init_scale
        output_scale = args.output_scale
        output_channel = args.output_channel
        position_encoding_type = args.position_encoding_type

        # Preprocessor - positional encoding / flatten
        ipot_input_preprocessor = IPOTBasicPreprocessor(
            position_encoding_type=position_encoding_type,
            in_channel=input_channel,
            pos_channel=pos_channel,
            pos2fourier_position_encoding_kwargs=dict(
                num_bands=num_bands,
                max_resolution=max_resolution,
            )
        )
        # Encoder
        ipot_encoder = IPOTEncoder(
            input_channel=input_channel + (2 * sum(num_bands) + len(num_bands)),  # pos2fourier
            num_latents=num_latents,
            latent_channel=latent_channel,
            cross_heads_num=cross_heads_num,
            cross_heads_channel=cross_heads_channel,
            latent_init_scale=latent_init_scale
        )
        # Processor
        ipot_processor = IPOTProcessor(
            self_per_cross_attn=self_per_cross_attn,
            self_heads_channel=self_heads_channel,
            latent_channel=latent_channel,
            self_heads_num=self_heads_num,
            ff_mult=ff_mult,
        )
        # Decoder
        ipot_decoder = IPOTDecoder(
            output_channel=output_channel,
            query_channel=2 * sum(num_bands) + len(num_bands),  # pos2fourier
            latent_channel=latent_channel,
            cross_heads_num=cross_heads_num,
            cross_heads_channel=cross_heads_channel,
            ff_mult=ff_mult,
            output_scale=output_scale,
            position_encoding_type=position_encoding_type,
            pos2fourier_position_encoding_kwargs=dict(
                num_bands=num_bands,
                max_resolution=max_resolution, )
        )
        model = EncoderProcessorDecoder(
            encoder=ipot_encoder,
            processor=ipot_processor,
            decoder=ipot_decoder,
            input_preprocessor=ipot_input_preprocessor
        )
    else:
        raise NotImplementedError

    ############################################################################
    # Optimizer and scheduler
    ############################################################################
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)

    ############################################################################
    # Dataloader and Trainer/Tester
    ############################################################################
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.data_name == 'navierstokes_Ve-3' or args.data_name == 'navierstokes_Ve-4' or args.data_name == 'navierstokes_Ve-5' or args.data_name == 'shallowwater' or args.data_name == 'era5_temperature':
        training_time(
            model, train_loader, test_loader, optimizer, scheduler, args, device, permute_obs=False, rand_keep_ratio=1)
    else:
        training(
            model, train_loader, test_loader, optimizer, scheduler, args, device, permute_obs=False, rand_keep_ratio=1)


########################################################
# Configs
########################################################
parser = ArgumentParser(description='Basic parse')

# Device
parser.add_argument('--gpu_num', type=str, default='0')

# Data
parser.add_argument('--data_name', type=str, default='burgers')
parser.add_argument('--n_train', type=int, default=1000)
parser.add_argument('--n_test', type=int, default=100)
parser.add_argument('--nx', type=int, default=8192)
parser.add_argument('--sub', type=int, default=8)

# Model
parser.add_argument('--model_type', type=str, default='ipot')
parser.add_argument('--num_bands', type=int, default=[64], nargs='+')
parser.add_argument('--max_resolution', type=int, default=[64], nargs='+')
parser.add_argument('--num_latents', type=int, default=128)
parser.add_argument('--latent_channel', type=int, default=64)
parser.add_argument('--self_per_cross_attn', type=int, default=4)
parser.add_argument('--cross_heads_num', type=int, default=1)
parser.add_argument('--self_heads_num', type=int, default=4)
parser.add_argument('--cross_heads_channel', type=int, default=64)
parser.add_argument('--self_heads_channel', type=int, default=None)
parser.add_argument('--ff_mult', type=int, default=2)
parser.add_argument('--latent_init_scale', type=float, default=0.02)
parser.add_argument('--output_scale', type=float, default=0.1)
parser.add_argument('--position_encoding_type', type=str, default="pos2fourier")

# Train
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--milestones', type=int, default=[100, 200, 300, 400], nargs='+')
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--scheduler_gamma', type=float, default=0.5)
parser.add_argument('--is_interp', type=bool, default=True)

args = parser.parse_args()

data_dir = '/data/'


if args.data_name == 'burgers':
    args.datapath = data_dir + 'burgers/burgers_data_R10.mat'
    args.input_channel = 1
    args.pos_channel = 1
    args.output_channel = 1

elif args.data_name == 'darcyflow':
    args.datapath_train = data_dir + 'darcyflow/piececonst_r421_N1024_smooth1.mat'
    args.datapath_test = data_dir + 'darcyflow/piececonst_r421_N1024_smooth2.mat'
    args.input_channel = 1
    args.pos_channel = 2
    args.output_channel = 1

elif args.data_name == 'navierstokes_Ve-3':
    args.datapath = data_dir + 'navierstokes/ns_V1e-3_N5000_T50.mat'
    args.input_channel = 10
    args.pos_channel = 2
    args.output_channel = 1
    args.T_start = 0
    args.T_in = 10
    args.T_out = 40
    args.step = 1

elif args.data_name == 'airfoil':
    args.input1_path = data_dir + 'airfoil/NACA_Cylinder_X.npy'
    args.input2_path = data_dir + 'airfoil/NACA_Cylinder_Y.npy'
    args.output_path = data_dir + 'airfoil/NACA_Cylinder_Q.npy'
    args.input_channel = 2
    args.pos_channel = 2
    args.output_channel = 1

elif args.data_name == 'elasticity':
    args.input1_path = data_dir + 'elasticity/Random_UnitCell_rr_10.npy'
    args.input2_path = data_dir + 'elasticity/Random_UnitCell_XY_10.npy'
    args.output_path = data_dir + 'elasticity/Random_UnitCell_sigma_10.npy'
    args.input_channel = 42
    args.pos_channel = 4
    args.output_channel = 1

elif args.data_name == 'plasticity':
    args.datapath = data_dir + 'plasticity/plas_N987_T20.mat'
    args.input_channel = 1
    args.pos_channel = 3
    args.output_channel = 4

elif args.data_name == 'era5_temperature':
    args.datapath = data_dir + 'era5_temperature/era5_temperature.grib'
    args.input_channel = 7
    args.pos_channel = 2
    args.output_channel = 1
    args.T_in = 7
    args.T_out = 7
    args.step = 1

else:
    raise NotImplementedError

args.save_dir = args.data_name
args.save_name = args.model_type + '_' + str(args.num_bands[0]) + '_' + str(args.max_resolution[0]) + '_' + str(
    args.num_latents) + '_' + str(args.latent_channel) + '_' + str(args.self_per_cross_attn) + '_' + str(
    args.cross_heads_num) + '_' + str(args.self_heads_num) + '_' + str(args.cross_heads_channel) + '_' + str(
    args.self_heads_channel) + '_' + str(args.ff_mult) + '_' + str(args.latent_init_scale) + '_' + str(
    args.output_scale) + '_' + str(args.epochs) + '.pt'

# Test
args.ckpt = 'checkpoints/' + args.save_dir + '/' + args.save_name

############################################################################
# Load dataset
############################################################################
if args.data_name == 'burgers':
    train_dataset = Burgers(
        args.datapath, nx=args.nx, sub=args.sub, n_train=args.n_train)
    test_dataset = Burgers(
        args.datapath, nx=args.nx, sub=args.sub, n_test=args.n_test)

elif args.data_name == 'darcyflow':
    train_dataset = DarcyFlow(
        args.datapath_train, nx=args.nx, sub=args.sub, num=args.n_train)
    test_dataset = DarcyFlow(
        args.datapath_test, nx=args.nx, sub=args.sub, num=args.n_test)

elif args.data_name == 'navierstokes_Ve-3' or args.data_name == 'navierstokes_Ve-4' or args.data_name == 'navierstokes_Ve-5':
    train_dataset = NavierStokes(
        args.datapath, nx=args.nx, sub=args.sub,
        T_start=args.T_start, T_in=args.T_in, T_out=args.T_out, n_train=args.n_train, is_train=True)
    test_dataset = NavierStokes(
        args.datapath, nx=args.nx, sub=args.sub,
        T_start=args.T_start, T_in=args.T_in, T_out=args.T_out, n_test=args.n_test, is_train=False)

elif args.data_name == 'airfoil':
    train_dataset = Airfoil(
        args.input1_path, args.input2_path, args.output_path, n_train=args.n_train)
    test_dataset = Airfoil(
        args.input1_path, args.input2_path, args.output_path, n_train=args.n_train, n_test=args.n_test)

elif args.data_name == 'elasticity':
    train_dataset = Elasticity(
        args.input1_path, args.input2_path, args.output_path, n_train=args.n_train)
    test_dataset = Elasticity(
        args.input1_path, args.input2_path, args.output_path, n_train=args.n_train, n_test=args.n_test)

elif args.data_name == 'plasticity':
    train_dataset = Plasticity(
        datapath=args.datapath, s1=101, s2=31, t=20, n_train=args.n_train)
    test_dataset = Plasticity(
        datapath=args.datapath, s1=101, s2=31, t=20, n_test=args.n_test)

elif args.data_name == 'era5_temperature':
    train_dataset = ERA5_temperature(
        datapath=args.datapath, sub=args.sub, n_train=args.n_train, n_test=args.n_test, is_train=True)
    test_dataset = ERA5_temperature(
        datapath=args.datapath, sub=args.sub, n_train=args.n_train, n_test=args.n_test, is_train=False)

else:
    raise NotImplementedError

############################################################################
# Train and Test
############################################################################
print(args)
print('Training...')
main(args, train_dataset, test_dataset)
print('Training finished')

