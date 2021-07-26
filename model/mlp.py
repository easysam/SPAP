import sys
sys.path.append("..")
sys.path.append(".")
from torch import nn
import torch
import torchsnooper

import numpy as np
import copy
import time
import os
import argparse

from preprocess.load_data import read_yaml
from util.dataloader_full import get_data_loader
from util.metric import evaluate_rank_metric, CdpLoss, PredLoss
from util.param import get_param_combination
from util.log import log_result, build_log
from model.run import window

files = read_yaml(windows=window)
loss_ratio = nn.MSELoss()
loss_domain = nn.NLLLoss()

"""
train mode: source and target
"""

class Mlp(nn.Module):
    def __init__(self, feature_dim):
        super(Mlp, self).__init__()
        self.feature_dim = 57 #feature_dim

        # build domain net
        self.predict_net = nn.Sequential()
        self.predict_net.add_module('p_fc1', nn.Linear(self.feature_dim, 8))
        self.predict_net.add_module('p_fc2', nn.Linear(8, 4))
        self.predict_net.add_module('p_fc3', nn.Linear(4, 1))
        self.time_dense = nn.Linear(1, 1)
        self.pred_dense = nn.Linear(2, 3)
        self.relu = nn.ReLU()

    # @torchsnooper.snoop()
    def forward(self, x, t):
        feat = self.predict_net(x)
        time = self.time_dense(t)
        ratio = self.relu(self.pred_dense(torch.cat((feat, time), 1)))
        return ratio

def change_mode(cdp, criterion, mode):
    cdp.set_mode(mode)
    criterion.set_mode(mode)


def evaluate(logger, device, model, data_loaders, batch_size):
    # load model & get pred and target
    model.eval()
    # running_loss = 0.0

    with torch.no_grad():
        predicts = []
        target = []
        # change_mode(cdp, criterion, 'source')
        for test_cx, test_ex, test_t, test_y in data_loaders['test']:
            cx = test_cx.view(batch_size, -1).float().to(device)
            ex = test_ex.view(batch_size, -1).float().to(device)
            x = torch.cat((cx, ex), 1)
            t = test_t.float().to(device)
            y = test_y.float().to(device)

            # _, pred_demand, _, domain_logits = cdp(x)
            # loss = criterion(pred_demand, y, domain_logits, domain)
            ratio_output = model(x, t)
            # ratio_error = loss_ratio(ratio_output, y, alpha)
            # running_loss += ratio_error.item() * x.size(0)
            predicts.append(ratio_output.cpu())
            target.append(y.cpu())
        # test_loss = running_loss / len(data_loaders['test'])
        pred_target = np.vstack(predicts)
        target = np.vstack(target)

    # evaluate rank metric[function]
    rank_metric = evaluate_rank_metric(logger, pred_target, target)
    return rank_metric


def train_cdp(logger, model_name, model, data_loaders, train_mode, batch_size, model_file_name, epoch, device,
              optimizer, scheduler, train_progress):
    """
    train model by using a parameter combination, record train data and restore model dict with best validation
    :return: source_acc, val_acc, target_acc, domain_acc, embed feature, source pred, target pred, save path
    """
    since = time.time()

    model.to(device)
    best_map_metric = None
    best_map_score = 0.0

    best_pred_metric = None
    best_pred_loss = 1000.0

    # train and val
    for i in range(epoch):
        running_ratio_loss = 0.0

        # train
        model.train()

        # change_mode(cdp, criterion, 'source')
        for train_cx, train_ex, train_t, train_y in data_loaders['train']:
            cx = train_cx.view(batch_size, -1).float().to(device)
            ex = train_ex.view(batch_size, -1).float().to(device)
            x = torch.cat((cx, ex), 1)
            t = train_t.float().to(device)
            y = train_y.float().to(device)

            optimizer.zero_grad()
            source_ratio_out = model(x, t)
            source_ratio_error = torch.sqrt(loss_ratio(source_ratio_out, y))

            source_ratio_error.backward()
            optimizer.step()

            running_ratio_loss += source_ratio_error.item()

        scheduler.step()
        logger.info('mode {} progress {}/{} repeat {}/{} epoch {}/{} train  ratio loss: {:.4f}'
                    .format(train_mode,
                            train_progress['param_cur'], train_progress['param_cnt'],
                            train_progress['repeat_cur'], train_progress['repeat_cnt'],
                            i, epoch, running_ratio_loss))

        rank_metric = evaluate(logger, device, model, data_loaders, batch_size)
        logger.info('mode {} progress {}/{} repeat {}/{} epoch {}/{} test  pred  loss: {:.4f} map@10: {:.4f}'
                    .format(train_mode,
                            train_progress['param_cur'], train_progress['param_cnt'],
                            train_progress['repeat_cur'], train_progress['repeat_cnt'],
                            i, epoch, rank_metric['rmse'], rank_metric['map@10']))

        if best_map_metric == None or (rank_metric['map@10'] > best_map_score and rank_metric['map@10'] < 1):
            logger.info('update map@10 score in epoch: {}'.format(rank_metric['map@10']))
            best_map_score = rank_metric['map@10']
            best_map_metric = rank_metric.copy()
        if best_pred_metric == None or rank_metric['rmse'] < best_pred_loss:
            logger.info('update pred loss in epoch: {}'.format(rank_metric['rmse']))
            best_pred_loss = rank_metric['rmse']
            best_pred_metric = rank_metric.copy()

    time_elapsed = time.time() - since
    logger.info('Training and val complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return best_map_metric, best_pred_metric


def one_pass_cdp(logger, model_name, source_city, target_city, train_mode,
                 epoch, gpu, feature_dim, train_progress, window, **param):
    """
    build model && train && save checkpoint, feature, res... && return metric
    :return: rank metric
    """

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    data_loaders = get_data_loader(source_city, target_city, param['bs'], param['neigh'], train_mode, window)
    # cdp, criterion = build_model(device, train_mode, **param)
    model = Mlp(feature_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=param['lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_file_name = '{}_{}_{}_bs_{}_alpha_{}_neigh_{}_lr_{}' \
        .format(model_name, source_city, target_city,
                param['bs'], param['alpha'], param['neigh'], param['lr'])

    best_map_metric, best_pred_metric = \
        train_cdp(logger, model_name, model, data_loaders, train_mode, param['bs'], model_file_name, epoch, device,
              optimizer, scheduler, train_progress)
    return best_map_metric, best_pred_metric


def check_path(model_name):
    model_path = files['inter_data']['cache_model'] + os.sep + model_name
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    for dir in ['tmp', 'map', 'pred']:
        if not os.path.exists(model_path + os.sep + dir):
            os.mkdir(model_path + os.sep + dir)

def grid_search_cdp(logger, model_name, source_city="beijing", target_city="tianjing",
                    train_mode="source", gpu="0", epoch=150, repeat=1, window=False):
    logger.info('model: {} source city: {} target city: {} train mode: {}'.format(model_name, source_city, target_city, train_mode))
    check_path(model_name)

    best_pred_loss = 1000.0

    best_map_score = 0

    header = {'neigh': 5, 'bs': 5, 'lr': 8,
              'repeat': 8, 'type': 5, 'time': 12, 'rmse': 8, 'mae': 8, 'map@10': 8, 'map@15': 8, 'map@20': 8, 'map@25': 8,
              'ndcg@10': 8, 'ndcg@15': 8, 'ndcg@20': 8, 'ndcg@25': 8}
    result_file_path = files['inter_data']['result'] + os.sep + '{}_{}_{}_{}'.format(model_name, train_mode, source_city, target_city)

    # get param for grid search
    undone_combination = get_param_combination(source_city, target_city, model_name, train_mode, window=window)
    param_cnt = len(undone_combination)
    for cur, param in enumerate(undone_combination):
        # load data
        for k, v in param.items():
            if k == 'bs' or k == 'neigh':
                param[k] = int(v)
            else:
                param[k] = float(v)

        feature_dim = param['neigh'] * 52
        logger.info('use new param combination')
        for i in range(repeat):
            param['repeat'] = i + 1
            logger.info('{} {} train param with {}'.format(model_name, train_mode, str(param)))

            # build model with param, train and get metric[function]
            train_progress = {'param_cnt': param_cnt, 'param_cur': cur, 'repeat_cnt': repeat, 'repeat_cur': i}
            map_metric, pred_metric = \
                one_pass_cdp(logger, model_name, source_city, target_city, train_mode, epoch, gpu, feature_dim, train_progress, window, **param)

            map_metric['type'] = 'map'
            map_metric.update(param.copy())
            pred_metric['type'] = 'pred'
            pred_metric.update(param.copy())
            # param.update(rank_metrics)

            # check whether best val metric can update
            if pred_metric['rmse'] < best_pred_loss:
                logger.info('update best pred loss: {}'.format(pred_metric['rmse']))
                best_pred_loss = pred_metric['rmse']

            if map_metric['map@10'] > best_map_score and map_metric['map@10'] < 1:
                logger.info('update best map@10 score: {}'.format(map_metric['map@10']))
                best_map_score = map_metric['map@10']

            # record result
            logger.info('map@10: {:.4f} rmse: {:.4f}  best map@10: {:.4f} best rmse: {:.4f}'
                        .format(map_metric['map@10'], pred_metric['rmse'], best_map_score, best_pred_loss))
            log_result(header, map_metric, result_file_path)
            log_result(header, pred_metric, result_file_path)
    logger.info('finish')

if __name__ == '__main__':
    source_city, target_city, train_mode = "beijing", "tianjing", "source"
    logger = build_log('test', files['log']['model_log'] + os.sep + 'mlp')
    logger.info('=' * 50)
    model_name = 'mlp'
    grid_search_cdp(logger, model_name, source_city, target_city, train_mode, gpu="0", epoch=1, repeat=1, window=window)
