import sys
import os

sys.path.append("..")
sys.path.append(".")

window = True

from util.log import build_log
from preprocess.load_data import read_yaml
from model import lasso, gbrt, mlp, stdann

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NO NAME')
    # data setting
    parser.add_argument('--source', type=str, default='beijing', required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='source city')
    parser.add_argument('--target', type=str, default='tianjing', required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='target city')
    # model setting
    parser.add_argument('--model', type=str, default="cdp_cnn", required=True,
                        choices=['lasso', 'gbrt', 'ranknet', 'lambdamart', 'mlp', 'cnn', 'convdann', 'cdp_cnn',
                                 'cdp_sacnn', 'cdp_res_sacnn', 'stdann', 'stdann_wd', 'stdann_single_task'], help='which model')
    parser.add_argument('--grid', type=int, default=1, help='grid search or one pass')
    # model training
    parser.add_argument('--gpu', type=str, default="0", help='use which gpu 0, 1 or 2')
    parser.add_argument('--epoch', type=int, default=60, help='how many epoch to train')
    parser.add_argument('--repeat', type=int, default=3, help='one hyper parameters combination running time')
    parser.add_argument('--mode', type=str, default='source', required=True,
                        choices=['source', 'combine', 'target'], help='only source or source & target, target for test')
    args = parser.parse_args()

    files = read_yaml(windows=window)
    logger = build_log(args.model, files['log']['model_log'] + os.sep + '{}_{}_{}_{}'
                       .format(args.model, args.mode, args.source, args.target), need_console=False)

    if args.mode == 'combine' and args.model in ['lasso', 'gbrt', 'ranknet', 'mlp']:
        logger.error('{} can only use source and target model'.format(args.model))


    if args.model == 'lasso_full':
        if args.grid == 1:
            lasso.grid_search_cdp(logger, args.model, args.source, args.target, args.mode,
                                  gpu=args.gpu, epoch=args.epoch, repeat=args.repeat, window=window)
        else:
            param = {'alpha': 0.1, 'neigh': 9, 'bs': 64}
            map_metric, pred_metric = \
                lasso.one_pass_cdp(logger, args.model, args.source_city, args.target_city, args.mode, args.epoch,
                                   args.gpu, window=window, **param)

    if args.model == 'gbrt_full':
        if args.grid == 1:
            gbrt.grid_search_cdp(logger, args.model, args.source, args.target, args.mode,
                                 gpu=args.gpu, epoch=args.epoch, repeat=args.repeat, window=window)
        else:
            param = {'lr': 0.1, 'n_estimators': 50, 'max_depth': 3, 'neigh': 1, 'bs': 64}
            map_metric, pred_metric = \
                gbrt.one_pass_cdp(logger, args.model, args.source_city, args.target_city, args.mode, args.epoch,
                                  args.gpu, window=window, **param)

    if args.model == 'mlp_full':
        if args.grid == 1:
            mlp.grid_search_cdp(logger, args.model, args.source, args.target, args.mode,
                                gpu=args.gpu, epoch=args.epoch, repeat=args.repeat, window=window)
        else:
            param = {'alpha': 0.5, 'neigh': 5, 'bs': 64, 'lr': 0.001}
            feature_dim = param['neigh'] * 52
            map_metric, pred_metric, map_path, pred_path = \
                mlp.one_pass_cdp(logger, args.model, args.source_city, args.target_city, args.mode, args.epoch,
                                 args.gpu, feature_dim, window=window, **param)

    if args.model == 'stdann':
        if args.grid == 1:
            stdann.grid_search_cdp(logger, args.model, args.source, args.target, args.mode,
                                   gpu=args.gpu, epoch=args.epoch, repeat=args.repeat, window=window)
        else:
            param = {'alpha': 0.5, 'beta': 1, 'neigh': 5, 'bs': 64, 'lr': 0.001, 'dropout': 0.2}
            train_progress = {'param_cnt': 0, 'param_cur': 0, 'repeat_cnt': 1, 'repeat_cur': 1}
            map_metric, pred_metric = \
                stdann.one_pass_cdp(logger, args.model, args.source, args.target, args.mode,
                                    args.epoch, args.gpu, train_progress, window=window, **param)
