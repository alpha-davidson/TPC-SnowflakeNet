import numpy as np
import torch
import matplotlib.pyplot as plt
import core.builder as builder
from utils import misc, yaml_reader
import argparse

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default=None, help='Configuration File')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--n_imgs', type=str, default="10", help='Number of images to save: default is 10, if \"all\" is passed all images will be saved')
    parser.add_argument('--save_img_path', type=str, default="", help='Where to save images')
    parser.add_argument('--experimental', action="store_true", default=False, help='No ground truth provided')
    parser.add_argument('--debug', action="store_true", default=False, help='Save all model outputs in seperate images')
    args = parser.parse_args()
    return args

def my_inference(model, args, config):

    n_imgs_flag = -1
    if args.n_imgs != "all":
        n_imgs_flag = int(args.n_imgs)

    with torch.no_grad():
        
        data_loader = builder.make_dataloader(config, "test")

        for idx, (feats, labels) in enumerate(data_loader):

            if idx == n_imgs_flag:
                break

            partial = feats.cuda()
            gt = labels.cuda()

            ret = model(partial, return_P0=True)

            input_pc = partial.squeeze().detach().cpu().numpy()
            output_pc = ret[-1].squeeze().detach().cpu().numpy()
            gt_pc = gt.squeeze().detach().cpu().numpy()

            # Output checking
            assert not (np.any(np.isnan(output_pc)) or np.any(np.isinf(output_pc))), "NaNs or Infs in pred cloud"
            assert max(np.amax(output_pc[:, :2]), np.abs(np.amin(output_pc[:, :2]))) < config.RANGES.MAX_X, "Predicted point out of bounds in XY plane"
            assert np.amax(output_pc[:, 2]) < config.RANGES.MAX_Z and np.amin(output_pc) > config.RANGES.MIN_Z, "Predicted point out of bounds in Z dimension"

            if args.experimental:
                misc.experimental_pad_plane_w_threeD(input_pc, output_pc, idx, args.save_img_path, config)
            elif args.debug:
                clouds = []
                for c in ret:
                    clouds.append(c.squeeze().detach().cpu().numpy())
                misc.debug_img(clouds, idx, args.save_img_path, config)
            else:
                misc.pad_plane_w_threeD(input_pc, output_pc, gt_pc, idx, config, args)

    return



if __name__ == "__main__":

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
    my_inference(model, args, config)
    print("Done")
