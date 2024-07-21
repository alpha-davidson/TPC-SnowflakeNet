import numpy as np
import torch
import matplotlib.pyplot as plt
import core.builder as builder
from utils import misc, yaml_reader
import argparse

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default=None, help='Configuration File')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--n_imgs', type=str, default="10", help='Number of images to save: default is 10, if \"all\" is passed all images will be saved')
    parser.add_argument('--save_img_path', type=str, default="", help='Where to save images')
    parser.add_argument('--experimental', action="store_true", default=False, help='No ground truth provided')
    parser.add_argument('--normed', action="store_true", default=False, help='Keep scaling of images to 0-1')
    args = parser.parse_args()
    return args

def my_inference(model, args, config):

    if args.n_imgs == "all":
        n_imgs_flag = -1
    else:
        n_imgs_flag == int(args.n_imgs)

    with torch.no_grad():
        
        data_loader = builder.make_dataloader(config, "test")

        for idx, (feats, labels) in enumerate(data_loader):

            if idx == n_imgs_flag:
                break

            partial = feats.cuda()
            gt = labels.cuda()

            ret = model(partial)

            input_pc = partial.squeeze().detach().cpu().numpy()
            output_pc = ret[-1].squeeze().detach().cpu().numpy()
            gt_pc = gt.squeeze().detach().cpu().numpy()

            # misc.better_img(input_pc, idx)
            # misc.better_img(output_pc, idx, out=True)
            # misc.better_img(gt_pc, idx, gt=True)

            if args.experimental:
                misc.experimental_img(input_pc, output_pc, idx, args.save_img_path, config)
            elif args.normed:
                misc.normed_img(input_pc, output_pc, gt_pc, idx, args.save_img_path, config)
            else:
                misc.triplet_img(input_pc, output_pc, gt_pc, idx, args.save_img_path, config)

    return



if __name__ == "__main__":

    args = get_args_from_command_line()
    if args.config is None:
        raise ValueError("No config file provided")
    config = yaml_reader.read_yaml(args.config)

    model = builder.make_model(config)
    if args.checkpoint_path is None:
        raise ValueError("No model path provided")
    misc.load_model(model, args.checkpoint_path)
    model.to("cuda:0".lower())
    model.eval()
    my_inference(model, args, config)
    print("Done")
