import numpy as np
import torch
import matplotlib.pyplot as plt
import core.builder as builder
from utils import misc, yaml_reader, helpers
import argparse
import os


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default=None, help='Configuration File')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--gt_save_path', type=str, default=None, help='Path to store gt clouds')
    parser.add_argument('--pred_save_path', type=str, default=None, help='Path to store pred clouds')
    args = parser.parse_args()
    return args


def predict(model, args, config):


    data_loader = builder.get_dataloader(config, "test", args)
    gt_clouds = np.ndarray((len(data_loader), config.dataset.complete_points, 4), dtype=np.float32)
    pred_clouds = np.ndarray((len(data_loader), config.dataset.complete_points, 4), dtype=np.float32)

    with torch.no_grad():

        for i, (experiment, data) in enumerate(data_loader):

            for k, v in data.items():
                data[k] = helpers.var_or_cuda(v)

            partial = data['partial_cloud']
            gt = data['gt_cloud']

            ret = model(partial)

            gt_clouds[i] = gt.squeeze().detach().cpu().numpy()
            pred_clouds[i] = ret[-1].squeeze().detach().cpu().numpy()

    np.save(args.gt_save_path, gt_clouds)
    np.save(args.pred_save_path, pred_clouds)

    return


if __name__ == '__main__':

    args = get_args_from_command_line()
    if args.config is None:
        raise ValueError("No config file provided")
    config = yaml_reader.read_yaml(args.config)

    model = builder.make_model(config)
    if args.model is None:
        raise ValueError("No model path provided")
    misc.load_model(model, args.model)
    model.to("cuda:0".lower())
    model.eval()

    predict(model, args, config)
    print("Done")