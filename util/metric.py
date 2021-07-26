from torch import nn
import torch
import numpy as np
import pandas as pd
# import torchsnooper

class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()
        self.bce_rank = nn.BCEWithLogitsLoss()

    # @torchsnooper.snoop()
    def forward(self, ratio_out, ratio):

        # rank loss
        s_ij = ratio - ratio.permute(1, 0)
        real_score = torch.where(s_ij > 0, torch.full_like(s_ij, 1), torch.full_like(s_ij, 0))

        pairwise_score = ratio_out - ratio_out.permute(1, 0)
        rank_loss = self.bce_rank(pairwise_score, real_score)

        return rank_loss

class PredLoss(nn.Module):
    def __init__(self):
        super(PredLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce_rank = nn.BCEWithLogitsLoss()

    # @torchsnooper.snoop()
    def forward(self, ratio_out, ratio, alpha):
        pred_loss = torch.sqrt(self.mse(ratio_out, ratio))

        # rank loss
        s_ij = ratio - ratio.permute(1, 0)
        real_score = torch.where(s_ij > 0, torch.full_like(s_ij, 1), torch.full_like(s_ij, 0))

        pairwise_score = ratio_out - ratio_out.permute(1, 0)
        rank_loss = self.bce_rank(pairwise_score, real_score)

        regular_loss = (1 - alpha) * pred_loss + alpha * rank_loss
        return regular_loss


class Full_PredLoss(nn.Module):
    def __init__(self):
        super(Full_PredLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce_rank = nn.BCEWithLogitsLoss()

    # @torchsnooper.snoop()
    def forward(self, ratio_out, ratio, alpha):
        pred_loss = torch.sqrt(self.mse(ratio_out, ratio))

        # rank loss
        total_ratio = ratio[:, -1].unsqueeze(1)
        s_ij = total_ratio - total_ratio.permute(1, 0)
        real_score = torch.where(s_ij > 0, torch.full_like(s_ij, 1), torch.full_like(s_ij, 0))

        pairwise_score = total_ratio - total_ratio.permute(1, 0)
        rank_loss = self.bce_rank(pairwise_score, real_score)

        regular_loss = (1 - alpha) * pred_loss + alpha * rank_loss
        return regular_loss

class CdpLoss(nn.Module):
    def __init__(self, device, alpha=0.1, beta=0.5, train_mode='source'):
        super(CdpLoss, self).__init__()
        self.device = device
        self.train_mode = train_mode
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.bce_rank = nn.BCEWithLogitsLoss()
        self.bce_domain = nn.BCEWithLogitsLoss()

    def set_param(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def set_mode(self, mode):
        self.train_mode = mode

    # @torchsnooper.snoop()
    def forward(self, pred_demand, demand, domain_logits, domain):
        # pred loss
        if self.train_mode == 'combine':
            source_label = demand[:demand.size()[0] // 2]
        else:
            source_label = demand
        pred_loss = torch.sqrt(self.mse(pred_demand, source_label))

        # rank loss
        num_data = source_label.size()[0]
        s_ij = source_label - source_label.permute(1, 0)
        real_score = torch.where(s_ij > 0, torch.full_like(s_ij, 1), torch.full_like(s_ij, 0))
        pairwise_score = pred_demand - pred_demand.permute(1, 0)
        rank_loss = torch.mean((torch.ones(num_data, num_data) - torch.diag(torch.ones(num_data))).to(self.device)
                   * self.bce_rank(pairwise_score, real_score))

        # domain loss
        domain_loss = torch.mean(self.bce_domain(domain_logits, domain))
        print(pred_loss)
        print(rank_loss)
        print(domain_loss)

        # combine three loss
        regular_loss = (1 - self.alpha) * pred_loss + self.alpha * rank_loss
        total_loss = regular_loss + self.beta * domain_loss
        if self.train_mode == 'combine':
            return total_loss
        return regular_loss

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            tmp_precision = num_hits / (i + 1.0)
            score += tmp_precision
            # print('Position {}, precision {}'.format(i + 1, tmp_precision))

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)

def map_metric(logger, y_test, y_predict, top_k, percentile=90):
    y_test_df = pd.DataFrame({'id': np.arange(len(y_test)), 'value': y_test})
    y_test_df['hot'] = 0
    y_test_df.loc[y_test_df['value'] >= np.percentile(y_test, percentile), 'hot'] = 1

    y_pred_df = pd.DataFrame({'id': np.arange(len(y_predict)), 'value': y_predict})
    y_pred_df['hot'] = 0
    y_pred_df.loc[y_pred_df['value'] >= np.percentile(y_predict, percentile), 'hot'] = 1

    y_test_df.sort_values(by='value', ascending=False, inplace=True)
    y_pred_df.sort_values(by='value', ascending=False, inplace=True)
    ap_metric = apk(y_test_df[y_test_df['hot']==1].id.values, y_pred_df[y_pred_df['hot']==1].id.values, top_k)
    # logger.info('MAP@{} metric {}'.format(top_k, ap_metric))
    return ap_metric

def dcg(predicted_order):
    i = 1
    cumulative_dcg = 0
    for x in predicted_order:
        cumulative_dcg += (2 ** x - 1) / (np.log(1 + i))
        i += 1
    return cumulative_dcg

def ndcg(predicted_order, top_k):
    sorted_list = np.sort(predicted_order)
    sorted_list = sorted_list[::-1]
    our_dcg = dcg(predicted_order[:top_k]) # DCG
    if our_dcg == 0:
        return 0

    max_dcg = dcg(sorted_list[:top_k]) # 最优情况 IDCG
    ndcg_output = our_dcg / float(max_dcg) # DCG/IDCG = NDCG
    return ndcg_output

def ndcg_metric(logger, y_test, y_predict, top_k=10, percentile=90):
    y_pred_df = pd.DataFrame({'id': np.arange(len(y_predict)), 'value': y_predict})
    y_pred_df['rel'] = 0
    y_pred_df.loc[y_test >= np.percentile(y_test, percentile), 'rel'] = 1

    y_pred_df.sort_values(by='value', ascending=False, inplace=True)
    ndcg_val = ndcg(y_pred_df['rel'].values, top_k)

    # logger.info('NDCG@{} metric {}'.format(top_k, ndcg_val))
    return ndcg_val


def evaluate_rank_metric(logger, pred_target, target):
    """
    calculate metric includes rmse, map@{10, 20, 30}, ndcg@{10, 20, 30}
    :return: metric map
    """
    rank_metrics = {}
    # rmse
    rank_metrics['mae'] = np.sum(np.abs(pred_target - target)) / len(target)
    rank_metrics['rmse'] = np.sqrt(np.mean(np.square(pred_target.ravel() - target.ravel())))

    # map[function] && ndcg[function]
    for top_k in [10, 15, 20, 25]:
        rank_metrics['map@{}'.format(top_k)] = map_metric(logger, target.ravel(), pred_target.ravel(), top_k)
        rank_metrics['ndcg@{}'.format(top_k)] = ndcg_metric(logger, target.ravel(), pred_target.ravel(), top_k)

    for key, value in rank_metrics.items():
        rank_metrics[key] = round(value, 4)
    return rank_metrics

def full_evaluate_rank_metric(logger, pred_target, target):
    """
    calculate metric includes {}_rmse, {}_map@{10, 20, 30}, {}_ndcg@{10, 20, 30}
    :return: metric map
    """
    rank_metrics = {}
    for i, name in enumerate(['slow', 'fast', 'total']):
        # rmse
        rank_metrics['{}_mae'.format(name)] = np.sum(np.abs(pred_target[:, i] - target[:, i])) / len(target[:, i])
        rank_metrics['{}_rmse'.format(name)] = np.sqrt(np.mean(np.square(pred_target[:, i].ravel() - target[:, i].ravel())))

        # map[function] && ndcg[function]
        for top_k in [10, 15, 20, 25]:
            rank_metrics['{}_map@{}'.format(name, top_k)] = map_metric(logger, target[:, i].ravel(), pred_target[:, i].ravel(), top_k)
            rank_metrics['{}_ndcg@{}'.format(name, top_k)] = ndcg_metric(logger, target[:, i].ravel(), pred_target[:, i].ravel(), top_k)

        for key, value in rank_metrics.items():
            rank_metrics[key] = round(value, 4)
    return rank_metrics
