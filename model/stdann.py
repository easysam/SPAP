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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from preprocess.load_data import read_yaml
from util.dataloader_full import get_data_loader
from util.metric import full_evaluate_rank_metric, Full_PredLoss
from util.param import get_param_combination
from util.log import log_result, build_log
from model.run import window

files = read_yaml(windows=window)
loss_ratio = Full_PredLoss()
loss_domain = nn.NLLLoss()


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SAM(nn.Module):
    def __init__(self, channel):
        super(SAM, self).__init__()
        self.para_lambda = nn.Parameter(torch.zeros(1))
        self.query_conv = nn.Conv1d(channel, channel, 1)
        self.key_conv = nn.Conv1d(channel, channel, 1)
        self.value_conv = nn.Conv1d(channel, channel, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H = x.size()
        proj_query = self.query_conv(x).view(N, -1, H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(N, -1, H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(N, -1, H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = out.view(N, C, H)

        out = self.para_lambda*out + x
        return out


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.para_mu = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        N, C, H = x.size()
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key) # N,C,C
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy) # N,C,C
        proj_value = x.view(N, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H)

        out = self.para_mu*out + x
        return out


class St_Dann(nn.Module):
    def __init__(self, num_neigh, dropout=0.2):
        super(St_Dann, self).__init__()

        # build station feature net
        self.station_feature_net = nn.Sequential()
        self.station_feature_net.add_module('cf_fc1', nn.Linear(5, 5))
        self.station_feature_net.add_module('cf_relu1', nn.ReLU())
        self.station_feature_net.add_module('cf_fc2', nn.Linear(5, 4))
        self.station_feature_net.add_module('cf_relu2', nn.ReLU())

        # build external feature net
        self.external_feature_net = nn.Sequential()
        self.external_feature_net.add_module('ef_conv1', nn.Conv1d(num_neigh, 16, kernel_size=1))
        self.external_feature_net.add_module('ef_bn1', nn.BatchNorm1d(16))
        self.external_feature_net.add_module('ef_drop1', nn.Dropout(dropout))
        self.external_feature_net.add_module('ef_relu1', nn.ReLU())
        self.external_feature_net.add_module('ef_sam', SAM(16))
        self.external_feature_net.add_module('ef_conv2', nn.Conv1d(16, 16, kernel_size=3, padding=1))
        self.external_feature_net.add_module('ef_bn2', nn.BatchNorm1d(16))
        self.external_feature_net.add_module('ef_relu2', nn.ReLU())
        self.external_feature_net.add_module('ef_avg1', nn.AdaptiveAvgPool2d((16, 1)))

        # build predict net
        self.predict_net = nn.Sequential()
        self.predict_net.add_module('p_fc1', nn.Linear(20, 10))
        self.predict_net.add_module('p_bn1', nn.BatchNorm1d(10))
        self.predict_net.add_module('p_relu1', nn.ReLU())
        self.predict_net.add_module('p_fc2', nn.Linear(10, 6))

        self.time_dense = nn.Embedding(26, 2)  # nn.Linear(1, 2)
        self.predict_dense = nn.Linear(8, 3)
        self.predict_relu = nn.ReLU()

        # build domain net
        self.domain_net = nn.Sequential()
        self.domain_net.add_module('d_fc1', nn.Linear(20, 10))
        self.domain_net.add_module('d_relu1', nn.ReLU())
        self.domain_net.add_module('d_fc2', nn.Linear(10, 2))
        self.domain_net.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def set_mode(self, mode):
        self.train_mode = mode

    # @torchsnooper.snoop()
    def forward(self, cx, ex, t, beta):#, t):
        station_feature = self.station_feature_net(cx).view(-1, 4)
        external_feature = self.external_feature_net(ex).view(-1, 16)
        feature = torch.cat((station_feature, external_feature), 1)
        pred_out = self.predict_net(feature)
        time_out = self.time_dense(t).squeeze()
        ratio_out = self.predict_relu(self.predict_dense(torch.cat((pred_out, time_out), 1)))
        reverse_feature = ReverseLayerF.apply(feature, beta)
        domain_out = self.domain_net(reverse_feature)
        return ratio_out, domain_out, feature.clone().detach()


def change_mode(cdp, criterion, mode):
    cdp.set_mode(mode)
    criterion.set_mode(mode)


def evaluate(logger, device, cdp, data_loaders, beta, alpha):
    # load model & get pred and target
    cdp.eval()
    # running_loss = 0.0

    with torch.no_grad():
        predicts = []
        target = []
        # change_mode(cdp, criterion, 'source')
        for test_cx, test_ex, test_t, test_y in data_loaders['test']:
            cx = test_cx.float().to(device)
            ex = test_ex.float().to(device)
            t = test_t.long().to(device)
            y = test_y.float().to(device)

            # _, pred_demand, _, domain_logits = cdp(x)
            # loss = criterion(pred_demand, y, domain_logits, domain)
            ratio_output, _, _ = cdp(cx, ex, t, beta)
            # ratio_error = loss_ratio(ratio_output, y, alpha)
            # running_loss += ratio_error.item() * x.size(0)
            predicts.append(ratio_output.cpu())
            target.append(y.cpu())
        # test_loss = running_loss / len(data_loaders['test'])
        pred_target = np.vstack(predicts)
        target = np.vstack(target) # slow, fast, total

    # evaluate rank metric[function]
    rank_metric = full_evaluate_rank_metric(logger, pred_target, target)
    return rank_metric


def train_cdp(logger, model_name, cdp, train_mode, data_loaders, batch_size, model_file_name, epoch, device,
              optimizer, scheduler, beta, param_alpha, train_progress):
    """
    train model by using a parameter combination, record train data and restore model dict with best validation
    :return: source_acc, val_acc, target_acc, domain_acc, embed feature, source pred, target pred, save path
    """
    since = time.time()

    cdp.to(device)
    best_map_metric = None
    best_map_score = 0.0

    best_pred_metric = None
    best_pred_loss = 1000.0

    # train and val
    tensor_list = []
    for i in range(epoch):
        running_total_loss = 0.0
        running_ratio_loss = 0.0
        running_domain_loss = 0.0
        # running_corrects = 0
        p = float(i + epoch) / epoch
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 if param_alpha == -1 else param_alpha

        # train
        cdp.train()
        # if train mode is combine
        if train_mode == 'combine':
            # change_mode(cdp, criterion, 'combine')
            j = 0
            for (source_train_cx, source_train_ex, source_train_t, source_train_y), (target_train_cx, target_train_ex, target_train_t, target_train_y) in zip(
                    data_loaders['combine_source'], data_loaders['combine_target']):
                # x = torch.cat((source_train_x, target_train_x), 0).float().to(device)
                # t = torch.cat((source_train_t, target_train_t), 0).float().to(device)
                # y = torch.cat((source_train_y, target_train_y), 0).float().to(device)
                # domain = torch.cat((torch.zeros((batch_size // 2, 1)), torch.ones((batch_size // 2, 1))), 0).float().to(device)
                source_train_cx = source_train_cx.float().to(device)
                source_train_ex = source_train_ex.float().to(device)
                source_train_t = source_train_t.long().to(device)
                source_train_y = source_train_y.float().to(device)
                target_train_cx = target_train_cx.float().to(device)
                target_train_ex = target_train_ex.float().to(device)
                target_train_t = target_train_t.long().to(device)
                source_domain = torch.zeros(batch_size//2).long().to(device)
                target_domain = torch.ones(batch_size//2).long().to(device)

                optimizer.zero_grad()
                # _, pred_demand, pred_domain, domain_logits = cdp(x)
                # loss = criterion(pred_demand, y, domain_logits, domain)
                source_ratio_out, source_domain_out, source_feature = cdp(source_train_cx, source_train_ex, source_train_t, beta)
                source_ratio_error = loss_ratio(source_ratio_out, source_train_y, alpha)
                source_domain_error = loss_domain(source_domain_out, source_domain)
                _, target_domain_out, target_feature = cdp(target_train_cx, target_train_ex, target_train_t, beta)
                target_domain_error = loss_domain(target_domain_out, target_domain)
                total_err = source_ratio_error + source_domain_error + target_domain_error
                domain_err = source_domain_error + target_domain_error
                total_err.backward()

                # loss.backward()
                optimizer.step()

                # running_loss += loss.item() * x.size(0) / 2
                # running_corrects += torch.sum(domain == torch.round(pred_domain)).item()
                running_ratio_loss += source_ratio_error.item()
                running_domain_loss += domain_err.item()
                running_total_loss += total_err.item()

                # save embed feature
                j = j + 1
                if j == len(data_loaders['combine_source']) - 1:
                    embed_feature = torch.cat((source_feature, target_feature), dim=0)
                    tensor_list.append(embed_feature.cpu().numpy())

            scheduler.step()
            logger.info('mode {} repeat {}/{} progress {}/{} epoch {}/{} train  ratio loss: {:.4f} domain loss: {:.4f}  total loss: {:.4f}'
                        .format(train_mode,
                                train_progress['repeat_cur'], train_progress['repeat_cnt'],
                                train_progress['param_cur'], train_progress['param_cnt'],
                                i, epoch, running_ratio_loss, running_domain_loss, running_total_loss))
        # if train mode is source or target
        else:
            # change_mode(cdp, criterion, 'source')
            for train_cx, train_ex, train_t, train_y in data_loaders['train']:
                cx = train_cx.float().to(device)
                ex = train_ex.float().to(device)
                t = train_t.long().to(device)
                y = train_y.float().to(device)
                domain = torch.zeros((batch_size)).long().to(device) if train_mode == 'source' else torch.ones((batch_size)).long().to(device)

                optimizer.zero_grad()
                source_ratio_out, source_domain_out, _ = cdp(cx, ex, t, beta)
                source_ratio_error = loss_ratio(source_ratio_out, y, alpha)
                source_domain_error = loss_domain(source_domain_out, domain)
                # _, pred_demand, pred_domain, domain_logits = cdp(x)
                # loss = criterion(pred_demand, y, domain_logits, domain)
                # loss.backward()
                source_ratio_error.backward()
                optimizer.step()

                running_ratio_loss += source_ratio_error.item()
                running_domain_loss += source_domain_error.item()

            scheduler.step()
            logger.info('mode {} repeat {}/{} progress {}/{} epoch {}/{} train  ratio loss: {:.4f} domain loss: {:.4f}'
                        .format(train_mode,
                                train_progress['repeat_cur'], train_progress['repeat_cnt'],
                                train_progress['param_cur'], train_progress['param_cnt'],
                                i, epoch, running_ratio_loss, running_domain_loss))

        rank_metric = evaluate(logger, device, cdp, data_loaders, beta, alpha)
        logger.info('mode {} repeat {}/{} progress {}/{} epoch {}/{} test  pred  total_rmse: {:.4f} total_map@10: {:.4f}'
                    .format(train_mode,
                            train_progress['repeat_cur'], train_progress['repeat_cnt'],
                            train_progress['param_cur'], train_progress['param_cnt'],
                            i, epoch, rank_metric['total_rmse'], rank_metric['total_map@10']))

        if best_map_metric == None or (rank_metric['total_map@10'] > best_map_score and rank_metric['total_map@10'] < 1):
            logger.info('update total_map@10 score in epoch: {}'.format(rank_metric['total_map@10']))
            best_map_score = rank_metric['total_map@10']
            best_map_metric = rank_metric.copy()
        if best_pred_metric == None or rank_metric['total_rmse'] < best_pred_loss:
            logger.info('update pred loss in epoch: {}'.format(rank_metric['total_rmse']))
            best_pred_loss = rank_metric['total_rmse']
            best_pred_metric = rank_metric.copy()


    time_elapsed = time.time() - since
    logger.info('Training and val complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return best_map_metric, best_pred_metric, tensor_list


def one_pass_cdp(logger, model_name, source_city, target_city, train_mode,
                 epoch, gpu, train_progress, window=False, **param):
    """
    build model && train && save checkpoint, feature, res... && return metric
    :return: rank metric
    """

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    data_loaders = get_data_loader(source_city, target_city, param['bs'], param['neigh'], train_mode, window)
    # cdp, criterion = build_model(device, train_mode, **param)
    cdp = St_Dann(param['neigh'], param['dropout'])
    optimizer = torch.optim.SGD(cdp.parameters(), lr=param['lr'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_file_name = '{}_{}_{}_{}_bs_{}_alpha_{}_beta_{}_neigh_{}_lr_{}_dropout_{}' \
        .format(model_name, train_mode, source_city, target_city,
                param['bs'], param['alpha'], param['beta'], param['neigh'], param['lr'], param['dropout'])
    # train[function]
    # best_val_loss, domain_acc, embed_feature, cache_model_path = \
    #    train_cdp(logger, cdp, train_mode, data_loaders, param['bs'], epoch, device,
    #              optimizer, scheduler, model_name, param['beta'], param['alpha'])
    # evaluate
    # rank_metric, test_loss = evaluate(logger, device, cdp, cache_model_path, data_loaders,
    #                                  param['beta'], param['alpha'])
    #save embed feature
    # if saveFeature:
    #    feature_path = files['inter_data']['cache_data'] + os.sep + model_name
    #    np.save(feature_path, embed_feature)

    best_map_metric, best_pred_metric, tensor_list = \
        train_cdp(logger, model_name, cdp, train_mode, data_loaders, param['bs'], model_file_name, epoch, device,
              optimizer, scheduler, param['beta'], param['alpha'], train_progress)

    for i, t in enumerate(tensor_list):
        tensor_name = 'tensor_{}_{}.npy'.format(model_file_name, i)
        if not os.path.exists(files['inter_data']['cache_data'] + os.sep + model_name):
            os.mkdir(files['inter_data']['cache_data'] + os.sep + model_name)
        file_name = files['inter_data']['cache_data'] + os.sep + model_name + os.sep + tensor_name
        if os.path.exists(file_name):
            os.remove(file_name)
        np.save(file_name, t)
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
    logger.info('model: {} source city: {} target city: {} train mode: {}'.format(model_name, source_city, target_city,
                                                                                  train_mode))
    check_path(model_name)

    best_pred_loss = 1000.0

    best_map_score = 0

    s_header = {'alpha': 5, 'beta': 5, 'neigh': 5, 'bs': 5, 'lr': 8, 'dropout': 8,
              'repeat': 8, 'type': 5, 'time': 12,
              'slow_rmse': 9, 'slow_mae': 8, 'slow_map@10': 11, 'slow_map@15': 11, 'slow_map@20': 11, 'slow_map@25': 11,
              'slow_ndcg@10': 12, 'slow_ndcg@15': 12, 'slow_ndcg@20': 12, 'slow_ndcg@25': 12}
    f_header = {'alpha': 5, 'beta': 5, 'neigh': 5, 'bs': 5, 'lr': 8, 'dropout': 8,
              'repeat': 8, 'type': 5, 'time': 12,
              'fast_rmse': 9, 'fast_mae': 8, 'fast_map@10': 11, 'fast_map@15': 11, 'fast_map@20': 11, 'fast_map@25': 11,
              'fast_ndcg@10': 12, 'fast_ndcg@15': 12, 'fast_ndcg@20': 12, 'fast_ndcg@25': 12}
    t_header = {'alpha': 5, 'beta': 5, 'neigh': 5, 'bs': 5, 'lr': 8, 'dropout': 8,
              'repeat': 8, 'type': 5, 'time': 12,
              'total_rmse': 10, 'total_mae': 9, 'total_map@10': 12, 'total_map@15': 12, 'total_map@20': 12, 'total_map@25': 12,
              'total_ndcg@10': 13, 'total_ndcg@15': 13, 'total_ndcg@20': 13, 'total_ndcg@25': 13}

    # get param for grid search
    undone_combination = get_param_combination(source_city, target_city, model_name, train_mode, window=window)
    param_cnt = len(undone_combination)
    for i in range(repeat):
        for cur, param in enumerate(undone_combination):
            param['repeat'] = i + 1
            # load data
            for k, v in param.items():
                if k == 'bs' or k == 'neigh':
                    param[k] = int(v)
                else:
                    param[k] = float(v)

            logger.info('use new param combination')
            logger.info('{} {} train param with {}'.format(model_name, train_mode, str(param)))

            # build model with param, train and get metric[function]
            train_progress = {'param_cnt': param_cnt, 'param_cur': cur, 'repeat_cnt': repeat, 'repeat_cur': i}
            map_metric, pred_metric = \
                one_pass_cdp(logger, model_name, source_city, target_city, train_mode, epoch, gpu, train_progress, window, **param)

            map_metric['type'] = 'map'
            map_metric.update(param.copy())
            pred_metric['type'] = 'pred'
            pred_metric.update(param.copy())
            # param.update(rank_metrics)

            # check whether best val metric can update
            if pred_metric['total_rmse'] < best_pred_loss:
                logger.info('update best pred loss: {}'.format(pred_metric['total_rmse']))
                best_pred_loss = pred_metric['total_rmse']

            if map_metric['total_map@10'] > best_map_score and map_metric['total_map@10'] < 1:
                logger.info('update best total_map@10 score: {}'.format(map_metric['total_map@10']))
                best_map_score = map_metric['total_map@10']

            # record result
            logger.info('total_map@10: {:.4f} total_rmse: {:.4f}  best total_map@10: {:.4f} best total_rmse: {:.4f}'
                        .format(map_metric['total_map@10'], pred_metric['total_rmse'], best_map_score, best_pred_loss))
            # 这里还是按照总数需求的准确率作为考量，当需要快充慢充时，可以从日志文件里找
            for h, pattern in zip([s_header, f_header, t_header], ['slow', 'fast', 'total']):
                result_file_path = files['inter_data']['result'] + os.sep \
                                   + '{}_{}_{}_{}_{}'.format(model_name, train_mode, source_city, target_city, pattern)
                log_result(h, map_metric, result_file_path)
                log_result(h, pred_metric, result_file_path)
    logger.info('finish')


if __name__ == '__main__':
    source_city, target_city, train_mode = "beijing", "tianjing", "combine"
    logger = build_log('test', files['log']['model_log'] + os.sep + 'stdann')
    logger.info('=' * 50)
    mode_name = "stdann"
    grid_search_cdp(logger, mode_name, source_city, target_city, train_mode, gpu="0", epoch=1, repeat=1, window=window)
