import sys
sys.path.append('..')
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from models.utils import fps_subsample as fps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from io import BytesIO
from PIL import Image
import os

def build_lambda_sche(opti, config, last_epoch=-1):
    if config.get('decay_step') is not None:
        lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd, last_epoch=last_epoch)
    else:
        raise NotImplementedError()
    return scheduler


def build_lambda_bnsche(model, config):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)


def seprate_point_cloud(xyz, num_points, crop, inp_n_points=2048, fixed_points=None, padding_zeros=False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1)  # 1 1 2048

        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]  # 2048
        # print('idx.shape', idx.shape)
        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps(input_data, 2048))
            CROP.append(fps(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    input_data = fps(input_data.contiguous(), inp_n_points)
    return input_data, crop_data.contiguous()



def rescale_feats(xs, ys, zs, config):
    '''
    Undo's min-max scaling
    Author: Ben Wagner
    '''

    xs = xs * (config.RANGES.MAX_X - config.RANGES.MIN_X) + config.RANGES.MIN_X
    ys = ys * (config.RANGES.MAX_Y - config.RANGES.MIN_Y) + config.RANGES.MIN_Y
    zs = zs * (config.RANGES.MAX_Z - config.RANGES.MIN_Z) + config.RANGES.MIN_Z

    return xs, ys, zs


def val_img(event, config):

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(projection='3d')

    xs = event[:, 0]
    ys = event[:, 1]
    zs = event[:, 2]
    if config.include_q:
        qs = event[:, 3]
    else:
        qs = None

    xs, ys, zs, qs = rescale_feats(xs, ys, zs, qs, config)

    if qs is not None:
        cb = ax.scatter(xs, zs, ys, c=qs, s=1)
        fig.colorbar(cb)
    else:
        ax.scatter(xs, zs, ys, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.set_xlim(xmin=config.RANGES.MIN_X, xmax=config.RANGES.MAX_X)
    ax.set_ylim(ymin=config.RANGES.MIN_Z, ymax=config.RANGES.MAX_Z)
    ax.set_zlim(zmin=config.RANGES.MIN_Y, zmax=config.RANGES.MAX_Y)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    img = Image.open(buf)
    img = np.array(img)[:, :, :3]
    img = 255 - img

    return torch.tensor(img, dtype=torch.uint8).permute(2, 0, 1)

def test_images(gt_pc, in_pc, out_pc, config):

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='3d'))

    ax[2].scatter(gt_pc[:, 0], gt_pc[:, 2], gt_pc[:, 1], s=1)
    ax[0].scatter(in_pc[:, 0], in_pc[:, 2], in_pc[:, 1], s=1)
    ax[1].scatter(out_pc[:, 0], out_pc[:, 2], out_pc[:, 1], s=1)

    for a in ax:
        a.set_xlim(xmin=0, xmax=1)
        a.set_ylim(ymin=0, ymax=1)
        a.set_zlim(zmin=0, zmax=1)

    plt.savefig("/home/DAVIDSON/bewagner/TPC-SnowflakeNet/val_img_test.png")
    plt.close()
    return


def load_model(base_model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print(f'Loading weights from {ckpt_path}...')

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})')
    return 


def triplet_img(input_pc, output_pc, gt_pc, idx, path, cfg):

    fig, (input_ax, output_ax, gt_ax) = plt.subplots(1, 3, figsize=(18,6), subplot_kw=dict(projection='3d'))

    input_xs, input_ys, input_zs = rescale_feats(input_pc[:, 0], input_pc[:, 1], input_pc[:, 2], cfg)
    output_xs, output_ys, output_zs = rescale_feats(output_pc[:, 0], output_pc[:, 1], output_pc[:, 2], cfg)
    gt_xs, gt_ys, gt_zs = rescale_feats(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2], cfg)

    input_ax.scatter(input_xs, input_zs, input_ys, s=1)
    output_ax.scatter(output_xs, output_zs, output_ys, s=1)
    gt_ax.scatter(gt_xs, gt_zs, gt_ys, s=1)

    input_ax.set_xlabel('X')
    input_ax.set_ylabel('Z')
    input_ax.set_zlabel('Y')

    output_ax.set_xlabel('X')
    output_ax.set_ylabel('Z')
    output_ax.set_zlabel('Y')

    gt_ax.set_xlabel('X')
    gt_ax.set_ylabel('Z')
    gt_ax.set_zlabel('Y')

    input_ax.set_xlim(xmin=cfg.RANGES.MIN_X, xmax=cfg.RANGES.MAX_X)
    input_ax.set_ylim(ymin=cfg.RANGES.MIN_Z, ymax=cfg.RANGES.MAX_Z)
    input_ax.set_zlim(zmin=cfg.RANGES.MIN_Y, zmax=cfg.RANGES.MAX_Y)

    output_ax.set_xlim(xmin=cfg.RANGES.MIN_X, xmax=cfg.RANGES.MAX_X)
    output_ax.set_ylim(ymin=cfg.RANGES.MIN_Z, ymax=cfg.RANGES.MAX_Z)
    output_ax.set_zlim(zmin=cfg.RANGES.MIN_Y, zmax=cfg.RANGES.MAX_Y)

    gt_ax.set_xlim(xmin=cfg.RANGES.MIN_X, xmax=cfg.RANGES.MAX_X)
    gt_ax.set_ylim(ymin=cfg.RANGES.MIN_Z, ymax=cfg.RANGES.MAX_Z)
    gt_ax.set_zlim(zmin=cfg.RANGES.MIN_Y, zmax=cfg.RANGES.MAX_Y)

    input_ax.set_title('Input')
    output_ax.set_title('Output')
    gt_ax.set_title('Ground Truth')

    fig.suptitle('Event '+str(idx).zfill(4))

    if path == '':
        path = '/'.join(cfg.dataset.test.partial.path.split('/')[:-1]) + '/imgs/'

    plt.savefig(path+'event'+str(idx).zfill(4)+'.png')
    plt.close()


def experimental_img(input_pc, output_pc, idx, path, cfg):

    fig, (input_ax, output_ax) = plt.subplots(1, 2, figsize=(12,6), subplot_kw=dict(projection='3d'))

    input_xs, input_ys, input_zs = rescale_feats(input_pc[:, 0], input_pc[:, 1], input_pc[:, 2])
    output_xs, output_ys, output_zs = rescale_feats(output_pc[:, 0], output_pc[:, 1], output_pc[:, 2])

    input_ax.scatter(input_xs, input_zs, input_ys, s=1)
    output_ax.scatter(output_xs, output_zs, output_ys, s=1)

    input_ax.set_xlabel('X')
    input_ax.set_ylabel('Z')
    input_ax.set_zlabel('Y')

    output_ax.set_xlabel('X')
    output_ax.set_ylabel('Z')
    output_ax.set_zlabel('Y')

    input_ax.set_xlim(xmin=cfg.RANGES.MIN_X, xmax=cfg.RANGES.MAX_X)
    input_ax.set_ylim(ymin=cfg.RANGES.MIN_Z, ymax=cfg.RANGES.MAX_Z)
    input_ax.set_zlim(zmin=cfg.RANGES.MIN_Y, zmax=cfg.RANGES.MAX_Y)

    output_ax.set_xlim(xmin=cfg.RANGES.MIN_X, xmax=cfg.RANGES.MAX_X)
    output_ax.set_ylim(ymin=cfg.RANGES.MIN_Z, ymax=cfg.RANGES.MAX_Z)
    output_ax.set_zlim(zmin=cfg.RANGES.MIN_Y, zmax=cfg.RANGES.MAX_Y)

    input_ax.set_title('Input')
    output_ax.set_title('Output')

    fig.suptitle('Event '+str(idx).zfill(4))

    if path == '':
        path = '/'.join(cfg.dataset.test.partial.path.split('/')[:-1]) + '/imgs/'

    plt.savefig(path+'event'+str(idx).zfill(4)+'.png')
    plt.close()

def normed_img(input_pc, output_pc, gt_pc, idx, path, cfg):

    fig, (input_ax, output_ax, gt_ax) = plt.subplots(1, 3, figsize=(18,6), subplot_kw=dict(projection='3d'))

    input_xs, input_ys, input_zs = input_pc[:, 0], input_pc[:, 1], input_pc[:, 2]
    output_xs, output_ys, output_zs = output_pc[:, 0], output_pc[:, 1], output_pc[:, 2]
    gt_xs, gt_ys, gt_zs = gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2]

    input_ax.scatter(input_xs, input_zs, input_ys, s=1)
    output_ax.scatter(output_xs, output_zs, output_ys, s=1)
    gt_ax.scatter(gt_xs, gt_zs, gt_ys, s=1)

    input_ax.set_xlabel('X')
    input_ax.set_ylabel('Z')
    input_ax.set_zlabel('Y')

    output_ax.set_xlabel('X')
    output_ax.set_ylabel('Z')
    output_ax.set_zlabel('Y')

    gt_ax.set_xlabel('X')
    gt_ax.set_ylabel('Z')
    gt_ax.set_zlabel('Y')

    input_ax.set_xlim(xmin=0, xmax=1)
    input_ax.set_ylim(ymin=0, ymax=1)
    input_ax.set_zlim(zmin=0, zmax=1)

    output_ax.set_xlim(xmin=0, xmax=1)
    output_ax.set_ylim(ymin=0, ymax=1)
    output_ax.set_zlim(zmin=0, zmax=1)

    gt_ax.set_xlim(xmin=0, xmax=1)
    gt_ax.set_ylim(ymin=0, ymax=1)
    gt_ax.set_zlim(zmin=0, zmax=1)

    input_ax.set_title('Input')
    output_ax.set_title('Output')
    gt_ax.set_title('Ground Truth')

    fig.suptitle('Event '+str(idx).zfill(4))

    if path == '':
        path = '/'.join(cfg.dataset.test.partial.split('/')[:-1]) + '/imgs/'

    plt.savefig(path+'event'+str(idx).zfill(4)+'.png')
    plt.close()


def check_grads(module, grad_in, grad_out):
    in_flags = np.ndarray((len(grad_in)*2), dtype=bool)
    for i in range(len(grad_in)):
        if grad_in[i] is None:
            in_flags[i*2] = False
            in_flags[i*2+1] = False
        else:
            in_flags[i*2] = np.any(np.isnan(grad_in[i].detach().cpu().numpy()))
            in_flags[i*2+1] = np.any(np.isinf(grad_in[i].detach().cpu().numpy()))
    out_flags = np.ndarray((len(grad_out)*2), dtype=bool)
    for i in range(len(grad_out)):
        if grad_out[i] is None:
            out_flags[i*2] = False
            out_flags[i*2+1] = False
        else:
            out_flags[i*2] = np.any(np.isnan(grad_out[i].detach().cpu().numpy()))
            out_flags[i*2+1] = np.any(np.isinf(grad_out[i].detach().cpu().numpy()))
    
    if np.any(in_flags):
        print("NaN or Inf found in input gradient:")
        print(grad_in)
        print("to module:")
        print(type(module))
        print(module)
        raise Exception(f"NaN of Inf found in input gradient to module {type(module)}")
    if np.any(out_flags):
        print("NaN or Inf found in output gradient:")
        print(grad_out)
        print("from module:")
        print(type(module))
        print(module)
        raise Exception(f"NaN of Inf found in output gradient from module {type(module)}")
    
def xy_plane_img(input_pc, output_pc, gt_pc, idx, path, cfg):

    fig, (input_ax, output_ax, gt_ax) = plt.subplots(1, 3, figsize=(18,6))

    input_xs, input_ys, _ = rescale_feats(input_pc[:, 0], input_pc[:, 1], input_pc[:, 2], cfg)
    output_xs, output_ys, _ = rescale_feats(output_pc[:, 0], output_pc[:, 1], output_pc[:, 2], cfg)
    gt_xs, gt_ys, _ = rescale_feats(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2], cfg)

    input_ax.scatter(input_xs, input_ys, s=1)
    output_ax.scatter(output_xs, output_ys, s=1)
    gt_ax.scatter(gt_xs, gt_ys, s=1)

    input_ax.set_xlabel('X (mm)')
    input_ax.set_ylabel('Y (mm)')

    output_ax.set_xlabel('X (mm)')
    output_ax.set_ylabel('Y (mm)')

    gt_ax.set_xlabel('X (mm)')
    gt_ax.set_ylabel('Y (mm)')

    input_ax.set_xlim(xmin=cfg.RANGES.MIN_X, xmax=cfg.RANGES.MAX_X)
    input_ax.set_ylim(ymin=cfg.RANGES.MIN_Y, ymax=cfg.RANGES.MAX_Y)

    output_ax.set_xlim(xmin=cfg.RANGES.MIN_X, xmax=cfg.RANGES.MAX_X)
    output_ax.set_ylim(ymin=cfg.RANGES.MIN_Y, ymax=cfg.RANGES.MAX_Y)

    gt_ax.set_xlim(xmin=cfg.RANGES.MIN_X, xmax=cfg.RANGES.MAX_X)
    gt_ax.set_ylim(ymin=cfg.RANGES.MIN_Y, ymax=cfg.RANGES.MAX_Y)

    input_ax.set_title('Input')
    output_ax.set_title('Output')
    gt_ax.set_title('Ground Truth')

    fig.suptitle('Event '+str(idx).zfill(4))

    if path == '':
        path = '/'.join(cfg.dataset.test.partial.path.split('/')[:-1]) + '/imgs/'

    plt.savefig(path+'event'+str(idx).zfill(4)+'_xy_plane.png')
    plt.close()



def pad_plane_w_threeD(input_pc, output_pc, gt_pc, idx, config, args):

    R = 250.0
    thetas = np.linspace(0, 2*np.pi, 1000)
    xs = R * np.cos(thetas)
    ys = R * np.sin(thetas)
    txs = xs[ys >= 0]
    tys = ys[ys >= 0]
    bxs = xs[ys < 0]
    bys = ys[ys < 0]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Event "+str(idx).zfill(4))

    gs = GridSpec(2, 3)

    iThreeD, oThreeD, gThreeD = plt.subplot(gs[0, 0], projection='3d'), plt.subplot(gs[0, 1], projection='3d'), plt.subplot(gs[0, 2], projection='3d')

    ixs, iys, izs = rescale_feats(input_pc[:, 0], input_pc[:, 1], input_pc[:, 2], config)
    oxs, oys, ozs = rescale_feats(output_pc[:, 0], output_pc[:, 1], output_pc[:, 2], config)
    gxs, gys, gzs = rescale_feats(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2], config)

    iThreeD.scatter(ixs, izs, iys, s=1)
    oThreeD.scatter(oxs, ozs, oys, s=1)
    gThreeD.scatter(gxs, gzs, gys, s=1)

    iThreeD.set_title("Input Cloud 3D View")
    oThreeD.set_title("Output Cloud 3D View")
    gThreeD.set_title("Ground Truth Cloud 3D View")

    for ax in [iThreeD, oThreeD, gThreeD]:
        ax.set_xlabel("X (mm)")
        ax.set_xlim((config.RANGES.MIN_X, config.RANGES.MAX_X))
        ax.set_ylabel("Z (mm)")
        ax.set_ylim((config.RANGES.MIN_Z, config.RANGES.MAX_Z))
        ax.set_zlabel("Y (mm)")
        ax.set_zlim((config.RANGES.MIN_Y, config.RANGES.MAX_Y))
        fig.add_subplot(ax)

    iTwoD, oTwoD, gTwoD = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])

    iTwoD.scatter(ixs, iys, s=1)
    oTwoD.scatter(oxs, oys, s=1)
    gTwoD.scatter(gxs, gys, s=1)

    for ax in [iTwoD, oTwoD, gTwoD]:
        ax.plot(txs, tys, color='grey')
        ax.plot(bxs, bys, color='grey')
        ax.fill_between(txs, tys, config.RANGES.MAX_Y, color='grey')
        ax.fill_between(bxs, bys, config.RANGES.MIN_Y, color='grey')
        if not args.normed:
            ax.fill_between(np.linspace(-270, -250, 40), config.RANGES.MAX_Y, config.RANGES.MIN_Y, color='grey')
            ax.fill_between(np.linspace(250, 270, 40), config.RANGES.MAX_Y, config.RANGES.MIN_Y, color='grey')
        ax.grid(True)

    iTwoD.set_title("Input Cloud Pad Plane View")
    oTwoD.set_title("Output Cloud Pad Plane View")
    gTwoD.set_title("Ground Truth Cloud Pad Plane View")

    for ax in [iTwoD, oTwoD, gTwoD]:
        ax.set_xlabel("X (mm)")
        ax.set_xlim((config.RANGES.MIN_X, config.RANGES.MAX_X))
        ax.set_ylabel("Y (mm)")
        ax.set_ylim((config.RANGES.MIN_Y, config.RANGES.MAX_Y))
        fig.add_subplot(ax)

    path = args.save_img_path
    if path == '':
        path = '/'.join(config.dataset.test.partial.path.split('/')[:-1]) + '/imgs/'
    plt.savefig(path+"event"+str(idx).zfill(4)+"_pp_3d.png")
    plt.close()


def experimental_pad_plane_w_threeD(input_pc, output_pc, idx, path, config):

    R = 250.0
    thetas = np.linspace(0, 2*np.pi, 1000)
    xs = R * np.cos(thetas)
    ys = R * np.sin(thetas)
    txs = xs[ys >= 0]
    tys = ys[ys >= 0]
    bxs = xs[ys < 0]
    bys = ys[ys < 0]

    ixs, iys, izs = rescale_feats(input_pc[:, 0], input_pc[:, 1], input_pc[:, 2], config)
    oxs, oys, ozs = rescale_feats(output_pc[:, 0], output_pc[:, 1], output_pc[:, 2], config)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Event "+str(idx).zfill(5))

    gs = GridSpec(2, 2)
    iThreeD, oThreeD = plt.subplot(gs[0, 0], projection='3d'), plt.subplot(gs[0, 1], projection='3d')

    iThreeD.scatter(ixs, izs, iys, s=1)
    oThreeD.scatter(oxs, ozs, oys, s=1)

    iThreeD.set_title("Input Cloud 3D View")
    oThreeD.set_title("Output Cloud 3D View")

    for ax in [iThreeD, oThreeD]:
        ax.set_xlim((config.RANGES.MIN_X, config.RANGES.MAX_X))
        ax.set_xlabel("X (mm)")
        ax.set_ylim((config.RANGES.MIN_Z, config.RANGES.MAX_Z))
        ax.set_ylabel("Z (mm)")
        ax.set_zlim((config.RANGES.MIN_Y, config.RANGES.MAX_Y))
        ax.set_zlabel("Y (mm)")
        fig.add_subplot(ax)

    iTwoD, oTwoD = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])

    iTwoD.scatter(ixs, iys, s=1)
    oTwoD.scatter(oxs, oys, s=1)

    iTwoD.set_title("Input Cloud Pad Plane View")
    oTwoD.set_title("Output Cloud Pad Plane View")

    for ax in [iTwoD, oTwoD]:
        ax.set_xlim((config.RANGES.MIN_X, config.RANGES.MAX_X))
        ax.set_xlabel("X (mm)")
        ax.set_ylim((config.RANGES.MIN_Y, config.RANGES.MAX_Y))
        ax.set_ylabel("Y (mm)")
        ax.plot(txs, tys, color='grey')
        ax.plot(bxs, bys, color='grey')
        ax.fill_between(txs, tys, config.RANGES.MAX_Y, color='grey')
        ax.fill_between(bxs, bys, config.RANGES.MIN_Y, color='grey')
        ax.fill_between(np.linspace(-270, -250, 40), config.RANGES.MAX_Y, config.RANGES.MIN_Y, color='grey')
        ax.fill_between(np.linspace(250, 270, 40), config.RANGES.MAX_Y, config.RANGES.MIN_Y, color='grey')
        ax.grid(True)
        fig.add_subplot(ax)

    if path == '':
        path = '/'.join(config.dataset.test.partial.path.split('/')[:-1]) + '/imgs/'
    plt.savefig(path+"event"+str(idx).zfill(4)+"_exp_pp_3d.png")
    plt.close()