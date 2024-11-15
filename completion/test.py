import os
import torch
import argparse
import numpy as np
from utils import average_meter, yaml_reader, loss_util, helpers
from core import builder

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./configs/pcn_cd1.yaml', help='Configuration File')
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint to test')
    args = parser.parse_args()
    return args

def test(config, model=None, test_dataloader=None, epoch_idx=-1, validation=False, test_writer=None, completion_loss=None, args=None):
    if test_dataloader is None:
        test_dataloader = builder.make_dataloader(config, "test", args)

    if model is None:
        model = builder.make_model(config)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        if not os.path.exists(args.model):
            raise Exception('checkpoints does not exists: {}'.format(args.model))

        print('Recovering from %s ...' % (args.model), end='')
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model'])
        print('recovered!')

    model.eval()

    n_samples = len(test_dataloader)
    test_losses = average_meter.AverageMeter(['loss_partial', 'loss_pc', 'loss_p1', 'loss_p2', 'loss_output', 'loss_total'])
    test_metrics = average_meter.AverageMeter([config.loss_func])
    category_metrics = dict()

    multiplier = 1.0
    if config.loss_func == 'cd_l1':
        multiplier = 1e3
    elif config.loss_func == 'cd_l2':
        multiplier = 1e4
    elif config.loss_func == 'emd':
        multiplier = 1e2

    loss_names = ["pc", "p1", "p2", "p3"]
    if completion_loss is None:
        completion_loss = loss_util.Completionloss(loss_func=config.loss_func)

    with torch.no_grad():
        for idx, (experiment, data) in enumerate(test_dataloader):

            for k, v in data.items():
                data[k] = helpers.var_or_cuda(v)

            partial = data['partial_cloud']
            gt = data['gt_cloud']
            pcds_pred = model(partial)

            # for pc_idx, pc in enumerate(pcds_pred):
            #     assert not np.any(np.isnan(pc.cpu().detach().numpy())) and not np.any(np.isinf(pc.cpu().detach().numpy())), f"NaN or Inf found in {loss_names[pc_idx]} cloud at epoch {epoch_idx}, batch {idx} during validation"
            # for tag, param in model.named_parameters():
            #     assert not np.any(np.isnan(param.cpu().detach().numpy())) and not np.any(np.isinf(param.cpu().detach().numpy())), f"NaN or Inf found in model parameter {tag} at epoch {epoch_idx}, batch {idx}"

            loss_total, losses = completion_loss.get_loss(pcds_pred, partial, gt)

            partial_matching = losses[0].item() * multiplier
            loss_c = losses[1].item() * multiplier
            loss_1 = losses[2].item() * multiplier
            loss_2 = losses[3].item() * multiplier
            loss_3 = losses[4].item() * multiplier
            _metrics = [loss_3]

            test_losses.update([partial_matching, loss_c, loss_1, loss_2, loss_3, loss_total.item() * multiplier])
            test_metrics.update(_metrics)
            if experiment not in category_metrics:
                category_metrics[experiment] = average_meter.AverageMeter([config.loss_func])
            category_metrics[experiment].update(_metrics)

            if idx % 100 == 0 or not validation:
                print('Test[%d/%d] Losses = %s' % (idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()]), flush=True)

    print('============================ TEST RESULTS ============================')
    print("      \t", end='')
    print([l for l in ['loss_partial', 'loss_pc', 'loss_output', 'loss_total']])
    print('Epoch ', epoch_idx, end='\t', flush=True)
    for value in test_losses.avg():
        print('%.4f' % value, end='\t', flush=True)
    print('\n', flush=True)

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/loss_partial', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_pc', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_1', test_losses.avg(2), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_2', test_losses.avg(3), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_output', test_losses.avg(4), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_total', test_losses.avg(5), epoch_idx)

    return test_losses.avg(3), test_losses.avg(4)


if __name__ == '__main__':
    args = get_args_from_command_line()

    config = yaml_reader.read_yaml(args.config)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.gpu)
    test(config, args=args)