from tqdm import tqdm
import numpy as np
import torch
import math

from utils import save_checkpoint, count_params, LpLoss
import torch.nn.functional as F


def training(model, train_loader, test_loader, optimizer, scheduler, args, device, permute_obs=True, rand_keep_ratio=1,
             use_tqdm=True):
    pbar = range(args.epochs)
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    if use_tqdm:
        pbar_test = tqdm(test_loader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar_test = test_loader

    myloss = LpLoss(size_average=False)

    model = model.to(device)
    print(count_params(model))
    model.train()

    best_rel_err = math.inf

    for e in pbar:
        loss_dict = {'train_rel': 0.0, 'test_rel': 0.0, 'train_l2': 0.0, 'test_l2': 0.0}

        for x, y in train_loader:
            mesh = train_loader.dataset.mesh.to(device)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            pred = model(x, mesh)
            pred = pred.reshape(y.shape)

            rel_loss = myloss(
                pred.view(args.batch_size, -1),
                y.view(args.batch_size, -1))

            l2_loss = F.mse_loss(
                pred.view(args.batch_size, -1),
                y.view(args.batch_size, -1), reduction='mean')

            rel_loss.backward()  # l2_loss.backward()
            optimizer.step()

            loss_dict['train_rel'] += rel_loss.item()
            loss_dict['train_l2'] += l2_loss.item()

        scheduler.step()
        train_rel_loss = loss_dict['train_rel'] / len(train_loader.dataset)
        train_l2_loss = loss_dict['train_l2'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, rel_loss: {train_rel_loss:.5f}, l2_loss: {train_l2_loss:.7f}'
                )
            )
        if e % 10 == 0:
            model.eval()

            # Mesh
            mesh = test_loader.dataset.mesh.to(device)
            index_dims = mesh.shape[:-1]
            indices = np.prod(index_dims)
            perm_ix = np.arange(indices)
            np.random.seed(0)

            if permute_obs:
                np.random.shuffle(perm_ix)

            num_keep = int(indices * rand_keep_ratio)
            ix_perm = perm_ix[:num_keep]

            test_rel_err, test_l2_err = [], []
            test_in_lr_err, test_out_lr_err = [], []

            with torch.no_grad():
                for x, y in pbar_test:  # make sure batch size be 1.
                    if permute_obs:
                        x = x.reshape(-1, x.shape[-1])[ix_perm].reshape(1, -1, x.shape[-1])

                    x = x.to(device)
                    pred = model(x, mesh)
                    pred = pred.reshape(y.shape)

                    y = y.to(device)
                    rel_loss = myloss(pred.reshape(1, -1), y.reshape(1, -1))
                    l2_loss = F.mse_loss(pred.reshape(1, -1), y.reshape(1, -1), reduction='mean')

                    test_rel_err.append(rel_loss.item())
                    test_l2_err.append(l2_loss.item())

                mean_rel_err = np.mean(test_rel_err)
                std_rel_err = np.std(test_rel_err, ddof=1) / np.sqrt(len(test_rel_err))

                mean_l2_err = np.mean(test_l2_err)
                std_l2_err = np.std(test_l2_err, ddof=1) / np.sqrt(len(test_l2_err))

                print(f'Test rel error mean: {mean_rel_err:.5f}, Test l2 err mean: {mean_l2_err:.7f}')

                if mean_rel_err < best_rel_err:
                    best_rel_err = mean_rel_err
                    print(f"Beat the record: {best_rel_err:.5f}")
                    best_l2_err = mean_l2_err
                    save_checkpoint(args.save_dir, args.save_name, model, optimizer)

    print('Done!')
    print(f'Test rel error: {best_rel_err}')
    print(f'Test l2 err: {best_l2_err}')


def training_time(model, train_loader, test_loader, optimizer, scheduler, args, device, permute_obs=True,
                  rand_keep_ratio=1, use_tqdm=True):
    pbar = range(args.epochs)
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    if use_tqdm:
        pbar_test = tqdm(test_loader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar_test = test_loader

    myloss = LpLoss(size_average=False)

    model = model.to(device)
    print(count_params(model))
    model.train()

    best_rel_err = math.inf

    for e in pbar:
        loss_dict = {'train_rel': 0.0, 'test_rel': 0.0, 'train_l2': 0.0, 'test_l2': 0.0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mesh = train_loader.dataset.mesh

            optimizer.zero_grad()

            pred = model(x, mesh, args.T_out)  # assume that step size = 1
            pred = pred.reshape(y.shape)

            rel_loss = myloss(
                pred.view(args.batch_size, -1),
                y.view(args.batch_size, -1))

            l2_loss = F.mse_loss(
                pred.view(args.batch_size, -1),
                y.view(args.batch_size, -1), reduction='mean')

            rel_loss.backward()  # l2_loss.backward()
            optimizer.step()

            loss_dict['train_rel'] += rel_loss.item()
            loss_dict['train_l2'] += l2_loss.item()

        scheduler.step()
        train_rel_loss = loss_dict['train_rel'] / len(train_loader.dataset)
        train_l2_loss = loss_dict['train_l2'] / len(train_loader.dataset)

        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch: {e}, rel_loss: {train_rel_loss:.5f}, l2_loss: {train_l2_loss:.7f}'
                )
            )
        if e % 10 == 0:
            model.eval()

            # Mesh
            mesh = test_loader.dataset.mesh
            index_dims = mesh.shape[:-1]
            indices = np.prod(index_dims)
            perm_ix = np.arange(indices)
            np.random.seed(0)

            if permute_obs:
                np.random.shuffle(perm_ix)

            num_keep = int(indices * rand_keep_ratio)
            ix_perm = perm_ix[:num_keep]

            test_rel_err, test_l2_err = [], []
            test_in_lr_err, test_out_lr_err = [], []

            with torch.no_grad():
                for x, y in pbar_test:  # make sure batch size be 1.
                    x, y = x.to(device), y.to(device)
                    if permute_obs:
                        x = x.reshape(-1, x.shape[-1])[ix_perm].reshape(1, -1, x.shape[-1])

                    pred = model(x, mesh, args.T_out)  # assume that step size = 1
                    pred = pred.reshape(y.shape)

                    rel_loss = myloss(
                        pred.view(test_loader.batch_size, -1),
                        y.view(test_loader.batch_size, -1))

                    l2_loss = F.mse_loss(
                        pred.view(test_loader.batch_size, -1),
                        y.view(test_loader.batch_size, -1), reduction='mean')

                    rel_loss = myloss(pred.reshape(1, -1), y.reshape(1, -1))
                    l2_loss = F.mse_loss(pred.reshape(1, -1), y.reshape(1, -1), reduction='mean')

                    test_rel_err.append(rel_loss.item())
                    test_l2_err.append(l2_loss.item())

                mean_rel_err = np.mean(test_rel_err)
                std_rel_err = np.std(test_rel_err, ddof=1) / np.sqrt(len(test_rel_err))

                mean_l2_err = np.mean(test_l2_err)
                std_l2_err = np.std(test_l2_err, ddof=1) / np.sqrt(len(test_l2_err))

                # print(f'Epoch: {e}, rel_loss: {train_rel_loss:.5f}, l2_loss: {train_l2_loss:.7f}')
                print(f'Test rel error mean: {mean_rel_err:.5f}, Test l2 err mean: {mean_l2_err:.7f}')

                if mean_rel_err < best_rel_err:
                    best_rel_err = mean_rel_err
                    print(f"Beat the record: {best_rel_err:.5f}")
                    best_l2_err = mean_l2_err

                    save_checkpoint(args.save_dir, args.save_name, model, optimizer)

    print('Done!')
    print(f'Test rel error: {best_rel_err}')
    print(f'Test l2 err: {best_l2_err}')
