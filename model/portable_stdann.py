import time
import numpy as np
import torch
# import torchsnooper
from torch import nn
from tqdm import tqdm
from util.metric import full_evaluate_rank_metric, Full_PredLoss
import preprocess.tio_process as tio_util
from util.dataloader_tio import get_data_loader, get_infer_dataloader

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

        out = self.para_lambda * out + x
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
    def forward(self, cx, ex, t, beta):  # , t):
        station_feature = self.station_feature_net(cx).view(-1, 4)
        external_feature = self.external_feature_net(ex).view(-1, 16)
        feature = torch.cat((station_feature, external_feature), 1)
        pred_out = self.predict_net(feature)
        time_out = self.time_dense(t).squeeze()
        ratio_out = self.predict_relu(self.predict_dense(torch.cat((pred_out, time_out), 1)))
        reverse_feature = ReverseLayerF.apply(feature, beta)
        domain_out = self.domain_net(reverse_feature)
        return ratio_out, domain_out, feature.clone().detach()


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
        target = np.vstack(target)  # slow, fast, total

    # evaluate rank metric[function]
    rank_metric = full_evaluate_rank_metric(logger, pred_target, target)
    return rank_metric


def train_cdp(logger, model_name, cdp, data_loaders, batch_size, epoch, device,
              optimizer, scheduler, beta, param_alpha):
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
    for i in tqdm(range(epoch)):
        running_total_loss = 0.0
        running_ratio_loss = 0.0
        running_domain_loss = 0.0
        # running_corrects = 0
        p = float(i + epoch) / epoch
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 if param_alpha == -1 else param_alpha

        cdp.train()
        j = 0
        for (source_train_cx, source_train_ex, source_train_t, source_train_y), (
        target_train_cx, target_train_ex, target_train_t, target_train_y) in zip(
                data_loaders['combine_source'], data_loaders['combine_target']):

            source_train_cx = source_train_cx.float().to(device)
            source_train_ex = source_train_ex.float().to(device)
            source_train_t = source_train_t.long().to(device)
            source_train_y = source_train_y.float().to(device)
            target_train_cx = target_train_cx.float().to(device)
            target_train_ex = target_train_ex.float().to(device)
            target_train_t = target_train_t.long().to(device)
            source_domain = torch.zeros(batch_size // 2).long().to(device)
            target_domain = torch.ones(batch_size // 2).long().to(device)

            optimizer.zero_grad()

            source_ratio_out, source_domain_out, source_feature = cdp(source_train_cx,
                                                                      source_train_ex,
                                                                      source_train_t,
                                                                      beta)

            source_ratio_error = loss_ratio(source_ratio_out,
                                            source_train_y,
                                            alpha)

            source_domain_error = loss_domain(source_domain_out,
                                              source_domain)

            _, target_domain_out, target_feature = cdp(target_train_cx,
                                                       target_train_ex,
                                                       target_train_t,
                                                       beta)

            target_domain_error = loss_domain(target_domain_out,
                                              target_domain)

            total_err = source_ratio_error + source_domain_error + target_domain_error

            domain_err = source_domain_error + target_domain_error

            total_err.backward()

            optimizer.step()

            running_ratio_loss += source_ratio_error.item()
            running_domain_loss += domain_err.item()
            running_total_loss += total_err.item()

            # save embed feature
            j = j + 1
            if j == len(data_loaders['combine_source']) - 1:
                embed_feature = torch.cat((source_feature, target_feature), dim=0)
                tensor_list.append(embed_feature.cpu().numpy())

        scheduler.step()

        logger.info(
            'epoch {}/{} train  ratio loss: {:.4f} domain loss: {:.4f}  total loss: {:.4f}'
            .format(i, epoch, running_ratio_loss, running_domain_loss, running_total_loss))

        rank_metric = evaluate(logger, device, cdp, data_loaders, beta, alpha)

        logger.info(
            'epoch {}/{} test  pred  total_rmse: {:.4f} total_map@10: {:.4f}'
            .format(i, epoch, rank_metric['total_rmse'], rank_metric['total_map@10']))

        if best_map_metric == None or (
                rank_metric['total_map@10'] > best_map_score and rank_metric['total_map@10'] < 1):
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


def inference(cdp, dataloader, beta, device):
    cdp.eval()
    with torch.no_grad():
        predicts = []
        dom_idxes = []
        for (infer_cx, infer_ex, infer_t, infer_y) in dataloader:
            cx = infer_cx.float().to(device)
            ex = infer_ex.float().to(device)
            t = infer_t.long().to(device)
            ratio, dom_idx, feature = cdp(cx, ex, t, beta)
            predicts.append(ratio.cpu())
            dom_idxes.append(dom_idx.cpu())
        pred_target = np.vstack(predicts)
        pred_dom_idxes = np.vstack(dom_idxes)
    return pred_target, pred_dom_idxes


class CityTransfer():
    def __init__(self, logger, source, target, C_t=None, gpu=0, param=None):
        self.logger = logger
        self.source = source
        self.target = target
        self.param = param
        self.param["device"] = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

        self.logger.info("## Initialize CityTransfer")

        self.s_dis = np.load("data/exp_data/similarity/similarity_distance_{}_numpy.npy".format(source), allow_pickle=True)
        self.t_dis = np.load("data/exp_data/similarity/similarity_distance_{}_numpy.npy".format(target), allow_pickle=True)

        self.logger.info("## Build static feature for {} and {}".format(self.source, self.target))
        self.source_static_feature, self.source_profiles = tio_util.static_feature(source)
        self.target_static_feature, self.target_profiles = tio_util.static_feature(target, profiles=C_t)
        self.source_object = tio_util.tio_data_set(self.logger, self.source, self.source_static_feature, self.source_profiles, self.s_dis)
        self.target_object = tio_util.tio_data_set(self.logger, self.target, self.target_static_feature, self.target_profiles, self.t_dis)

        self.cdp = St_Dann(param["neigh"], param["dropout"])
        self.optimizer = torch.optim.SGD(self.cdp.parameters(), lr=param["lr"], momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)


    def fit(self):
        self.logger.info("## Training the city transfer")
        data_loaders = get_data_loader(self.logger,
                                       self.source_object,
                                       self.target_object,
                                       self.param["bs"],
                                       self.param["neigh"])
        self.logger.info("## Start training.")
        train_cdp(self.logger, "TIO", self.cdp, data_loaders, self.param["bs"], self.param["epoch"],
                  self.param["device"], self.optimizer, self.scheduler, self.param["beta"], self.param["alpha"])

    def update_target_profiles(self, C_t):
        self.target_static_feature, self.target_profiles = tio_util.static_feature(self.target, profiles=C_t)
        self.target_object = tio_util.tio_data_set(self.logger, self.target, self.target_static_feature, self.target_profiles, self.t_dis)

    def target_city_infer(self, C_t_extend=None):
        # static feature is for every station, without time-variant demand
        static_feature, profiles = tio_util.static_feature(self.target, profiles=self.target_profiles)
        # demand object set is with daytime-variant day-invariant demand
        demand_object_set = tio_util.inference_data_set(self.target, static_feature, profiles, self.t_dis)
        if C_t_extend is not None:
            # The C_t_extend is for each staion, other than each time interval
            # Firstly, expand the C_t_extend to each time interval
            time_key_cnt = len(demand_object_set.index) // len(C_t_extend)
            C_t_extend = np.tile(C_t_extend, (time_key_cnt, 1))
            demand_object_set["slow_num"] = C_t_extend[:, 0]
            demand_object_set["fast_num"] = C_t_extend[:, 1]
            demand_object_set["total_num"] = C_t_extend[:, 0] + C_t_extend[:, 1]
        data_loaders = get_infer_dataloader(demand_object_set, self.param["bs"], self.param["neigh"])
        ratio, domain_idx = inference(self.cdp, data_loaders, self.param["beta"], self.param["device"])
        return ratio, domain_idx

    def predictor(self, C_t):
        ratio, domain_idx = self.target_city_infer(C_t_extend=C_t)
        return ratio

    def extend_predictor(self, C_t_extend):
        extend_ratio = []
        for i in tqdm(range(C_t_extend.shape[1])):
            ratio = self.predictor(C_t=C_t_extend[:, i, :])
            extend_ratio.append(ratio)

        self.logger.info("extend prediction result shape, extend_ratio: ({}, {})".format(len(extend_ratio), extend_ratio[0].shape))
        extend_ratio = np.hstack(extend_ratio).reshape(extend_ratio[0].shape[0], len(extend_ratio), -1)
        self.logger.info("after shape transform: {}".format(extend_ratio.shape))
        return extend_ratio

    def discriminator(self, C_t):
        ratio, domain_idx = self.target_city_infer(C_t)
        self.logger.warning("Discrimination result: {}".format(np.mean(domain_idx)))
        return np.mean(domain_idx)
