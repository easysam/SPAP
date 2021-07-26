import torch
from torch import nn
from util.dataloader_full import get_data_loader

class SAM(nn.Module):
    def __init__(self, channel):
        super(SAM, self).__init__()
        self.para_lambda = nn.Parameter(torch.zeros(1))
        self.query_conv = nn.Conv1d(channel, channel, 1)
        self.key_conv = nn.Conv1d(channel, channel, 1)
        self.value_conv = nn.Conv1d(channel, channel, 1)
        self.softmax = nn.Softmax(dim=-1)


class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


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
