import os
import torch
import argparse
from tqdm import tqdm
from utils import average_meter, yaml_reader, loss_util, misc
from core import builder

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./configs/pcn_cd1.yaml', help='Configuration File')
    args = parser.parse_args()
    return args

def test(config, model=None, test_dataloader=None, epoch_idx=-1, validation=False, test_writer=None, completion_loss=None):
    if test_dataloader is None:
        test_dataloader = builder.make_dataloader(config, "test", args)

    if model is None:
        model = builder.make_model(config)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        if not os.path.exists(config.test.model_path):
            raise Exception('checkpoints does not exists: {}'.format(config.test.model_path))

        print('Recovering from %s ...' % (config.test.model_path), end='')
        checkpoint = torch.load(config.test.model_path)
        model.load_state_dict(checkpoint['model'])
        print('recovered!')

    model.eval()

    n_samples = len(test_dataloader)
    test_losses = average_meter.AverageMeter(['partial_matching', 'cdc', 'cd1', 'cd2', 'cd3', 'total_loss'])

    multiplier = 1.0
    if config.loss_func == 'cd_l1':
        multiplier = 1e3
    elif config.loss_func == 'cd_l2':
        multiplier = 1e4
    elif config.loss_func == 'emd':
        multiplier = 1e2

    if completion_loss is None:
        completion_loss = loss_util.Completionloss(loss_func=config.loss_func)

    with torch.no_grad():
        for idx, (feats, labels) in enumerate(test_dataloader):

            partial = feats.cuda()
            gt = labels.cuda()
            pcds_pred = model(partial)

            loss_total, losses = completion_loss.get_loss(pcds_pred, partial, gt)

            partial_matching = losses[0].item() * multiplier
            loss_c = losses[1].item() * multiplier
            loss_1 = losses[2].item() * multiplier
            loss_2 = losses[3].item() * multiplier
            loss_3 = losses[4].item() * multiplier

            test_losses.update([partial_matching, loss_c, loss_1, loss_2, loss_3, loss_total.item() * multiplier])

            print('Test[%d/%d] Losses = %s' % (idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()]), flush=True)

    print('============================ TEST RESULTS ============================')
    print("      \t", end='')
    print([l for l in ['partial_matching', 'cdc', 'cd1', 'cd2', 'cd3', 'total_loss']])
    print('Epoch ', epoch_idx, end='\t', flush=True)
    for value in test_losses.avg():
        print('%.4f' % value, end='\t', flush=True)
    print('\n', flush=True)

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/partial_matching', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_c', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_1', test_losses.avg(2), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_2', test_losses.avg(3), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_3', test_losses.avg(4), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_total', test_losses.avg(5), epoch_idx)

    return test_losses.avg(4), test_losses.avg(5)


if __name__ == '__main__':
    args = get_args_from_command_line()

    config = yaml_reader.read_yaml(args.config)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.gpu)
    test(config)