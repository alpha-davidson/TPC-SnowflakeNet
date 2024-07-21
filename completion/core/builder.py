import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR

sys.path.append('../..')
from utils.misc import build_lambda_sche, build_lambda_bnsche
from models.model_completion import SnowflakeNet
from utils.scheduler import GradualWarmupScheduler


def make_dataloader(config, split, args=None):
    '''
    Author: Ben Wagner
    '''
    if split == "train":
        subset_cfgs = config.dataset.train
    elif split == "val":
        subset_cfgs = config.dataset.val
    elif split == "test":
        subset_cfgs = config.dataset.test
    else:
        raise ValueError(f"{split} split unrecognized")
    
    feats = np.load(subset_cfgs.partial)
    labels = np.load(subset_cfgs.complete)

    if not config.include_q:
        feats = feats[:, :, :3]
        labels = labels[:, :, :3]

    feats = torch.Tensor(feats)
    labels = torch.Tensor(labels)

    dataset = TensorDataset(feats, labels)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1 if args is None else config.batch_size,
                                              shuffle=False,
                                              drop_last= split=="train",
                                              num_workers=int(config.num_workers))
    
    return data_loader


def make_model(config):
    model = SnowflakeNet(
        dim_feat=config.model.dim_feat,
        num_pc=config.model.num_pc,
        num_p0=config.model.num_p0,
        radius=config.model.radius,
        bounding=config.model.bounding,
        up_factors=config.model.up_factors,
    )

    # if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model).cuda()

    return model


def make_optimizer(config, model):
    if config.type == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **config.kwargs)
    elif config.type == 'AdamW':
        optimizer = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': config.kwargs.lr}],
                                      **config.kwargs)
    elif config.type == 'NAdam':
        optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), **config.kwargs)
    else:
        raise Exception('optimizer {} not supported yet!'.format(config.type))
    
    return optimizer


def make_scheduler(config, optimizer, last_epoch=-1):
    if config.type == 'StepLR':
        scheduler = StepLR(optimizer, **config.kwargs, last_epoch=last_epoch)
    elif config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, config.kwargs)  # misc.py
    elif config.type == "GradualWarmup":
        scheduler_steplr = MultiStepLR(optimizer, last_epoch=last_epoch, **config.kwargs_1)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr, **config.kwargs_2)
    else:
        raise Exception('scheduler {} not supported yet!'.format(config.type))

    return scheduler