# -*- coding: utf-8 -*-
# @Author: XP

import os
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
import torch.autograd
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import helpers, average_meter, yaml_reader, loss_util, misc
from core import builder
from test import test
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default=None, help='Configuration File')
    parser.add_argument('--start_checkpoint', type=str, default=None, help='Start training from checkpoint')
    parser.add_argument('--exp_name', type=str, default=None, help="Path to store tensorboard logs and model checkpoints")
    parser.add_argument('--resume', action='store_true', default=False, help="If resuming model from a checkpoint")
    args = parser.parse_args()
    if (args.resume and args.start_checkpoint is None) or (not args.resume and args.start_checkpoint is not None):
        raise AttributeError("A starting checkpoint and resume flag must be provided if resuming model")
    return args

def train(config, args):


    # dataloaders
    train_dataloader = builder.make_dataloader(config, "train", args)
    val_dataloader = builder.make_dataloader(config, "val", args)

    model = builder.make_model(config)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()



    # out folders
    if args.exp_name is None:
        output_dir = os.path.join("./exp", '%s', datetime.now().isoformat())
    else:
        output_dir = os.path.join('./exp', '%s', args.exp_name)
    config.dataset.train.path_checkpoints = output_dir % 'checkpoints'
    config.dataset.train.path_logs = output_dir % 'logs'
    if not os.path.exists(config.dataset.train.path_checkpoints):
        os.makedirs(config.dataset.train.path_checkpoints)

    # log writers
    train_writer = SummaryWriter(os.path.join(config.dataset.train.path_logs, 'train'))
    val_writer = SummaryWriter(os.path.join(config.dataset.train.path_logs, 'val'))

    init_epoch = 1
    best_metric = float('inf')

    if args.start_checkpoint is not None:
        if not os.path.exists(args.start_checkpoint):
            raise Exception('checkpoints does not exists: {}'.format(args.start_checkpoint))

        print('Recovering from %s ...' % (args.start_checkpoint), end='')
        checkpoint = torch.load(args.start_checkpoint)
        model.load_state_dict(checkpoint['model'])
        print('recovered!')

        init_epoch = checkpoint['epoch_index']
        best_metric = checkpoint['best_metric']

    optimizer = builder.make_optimizer(config.optimizer, model)
    # scheduler = builder.make_scheduler(config.scheduler, optimizer, last_epoch=init_epoch if args.resume else -1)

    multiplier = 1.0
    if config.loss_func == 'cd_l1':
        multiplier = 1e3
    elif config.loss_func == 'cd_l2':
        multiplier = 1e4
    elif config.loss_func == 'emd':
        multiplier = 1e2

    completion_loss = loss_util.Completionloss(loss_func=config.loss_func)

    n_batches = len(train_dataloader)
    avg_meter_loss = average_meter.AverageMeter(['loss_partial', 'loss_pc', 'loss_p1', 'loss_p2', 'loss_p3', 'loss_total'])
    torch.autograd.set_detect_anomaly(True)
    model.zero_grad()
    for epoch_idx in range(init_epoch, config.epochs+1):
        avg_meter_loss.reset()
        model.train()
        print([l for l in ['partial_matching', 'cdc', 'cd1', 'cd2', 'cd3', 'total_loss']])
        for batch_idx, (feats, labels) in enumerate(train_dataloader):
            
            partial = feats.cuda()
            gt = labels.cuda()


            pcds_pred = model(partial)
            loss_total, losses = completion_loss.get_loss(pcds_pred, partial, gt)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses = [ls*multiplier for ls in losses]
            avg_meter_loss.update(losses + [loss_total*multiplier])

            print('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, config.epochs, batch_idx + 1, n_batches), end='', flush=True)
            print('%s' % ['%.4f' % l for l in losses], flush=True)

        # if type(scheduler) != ReduceLROnPlateau:
        #     scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        train_writer.add_scalar('Loss/Epoch/partial_matching', avg_meter_loss.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/loss_c', avg_meter_loss.avg(1), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/loss_1', avg_meter_loss.avg(2), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/loss_2', avg_meter_loss.avg(3), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/loss_3', avg_meter_loss.avg(4), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/loss_total', avg_meter_loss.avg(5), epoch_idx)


        cd_eval, total_test_loss = test(config, model=model, test_dataloader=val_dataloader, validation=True,
                                        epoch_idx=epoch_idx, test_writer=val_writer, completion_loss=completion_loss)
        
        # if type(scheduler) == ReduceLROnPlateau:
        #     scheduler.step(total_test_loss)

        # Save checkpoints
        if epoch_idx % config.save_freq == 0 or cd_eval < best_metric:
            file_name = 'ckpt-best.pth' if cd_eval < best_metric else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(config.dataset.train.path_checkpoints, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metric': best_metric,
                'model': model.state_dict()
            }, output_path)
            print("Saved new checkpoint to %s ..." % output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metric:
                best_metric = cd_eval

    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    args = get_args_from_command_line()
    args.n_imgs = None
    if args.config is None:
        raise ValueError("No config file provided")

    config = yaml_reader.read_yaml(args.config)


    set_seed(config.seed)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.gpu)
    train(config, args)
