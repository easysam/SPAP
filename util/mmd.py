import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

from util.dataloader import get_data_loader
from preprocess.load_data import read_yaml
files = read_yaml(windows=True)

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 倍数
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # 将total复制（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    L2_distance = ((total0-total1)**2).sum(2) # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list] #高斯核函数
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 倍数
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0]) # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    print(kernels.shape)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def threshold(kernel_num, variance, sample_num, alpha, fix_sigma):
    '''
    计算拒绝零假设的检测量
    '''
    # version1:[A kernel two-sample test[J]] 
    # return math.sqrt(2 * kernel_num / sample_num) * (1 + math.sqrt(2 * math.log(20)))
    # version2:[Maximum Mean Discrepancy] return 2 * alpha * fix_sigma / math.sqrt(sample_num)
    # version3:[Optimal kernel choice for large-scale two-sample tests]
    import scipy.stats as st
    return (1 / math.sqrt(sample_num)) * variance * math.sqrt(2) * (1 / st.norm.cdf(1 - alpha, loc=0, scale=1))

def test():
    # 生产两种分布
    SAMPLE_SIZE = 30
    buckets = 50

    #第一种分布：对数正态分布，得到一个中值为mu，标准差为sigma的正态分布。mu可以取任何值，sigma必须大于零。
    plt.subplot(1,2,1)
    plt.xlabel("random.lognormalvariate")
    mu = -0.6
    sigma = 0.15#将输出数据限制到0-1之间
    res1 = [random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
    plt.hist(res1, buckets)

    #第二种分布：beta分布。参数的条件是alpha 和 beta 都要大于0， 返回值在0~1之间。
    plt.subplot(1,2,2)
    plt.xlabel("random.betavariate")
    alpha = 1
    beta = 10
    res2 = [random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)]
    plt.hist(res2, buckets)

    plt.savefig(files['first_level']['img'] + os.sep + 'test_mmd.jpg')
    plt.show()

    kernel_mul, kernel_num = 2.0, 5

    # 生成2种分布，并且计算MMD
    for sig in [1, 5, 10, 20]:
        #参数值见上段代码
        #分别从对数正态分布和beta分布取两组数据
        diff_1 = []
        for i in range(10):
            diff_1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

        diff_2 = []
        for i in range(10):
            diff_2.append([random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)])

        X = torch.Tensor(diff_1)
        Y = torch.Tensor(diff_2)
        X,Y = Variable(X), Variable(Y)
        print("sigma:", sig, "diff distribution result:", mmd_rbf(X, Y, kernel_mul=2.0, kernel_num=kernel_num, fix_sigma=sig), 'threshold:', threshold(kernel_num, SAMPLE_SIZE, 0.05))

        # 生成相同的分布，计算MMD
        same_1 = []
        for i in range(10):
            same_1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

        same_2 = []
        for i in range(10):
            same_2.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

        X = torch.Tensor(same_1)
        Y = torch.Tensor(same_2)
        X,Y = Variable(X), Variable(Y)
        print("sigma:", sig, "the same distribution result:", mmd_rbf(X, Y, kernel_mul=2.0, kernel_num=kernel_num, fix_sigma=sig), 'threshold:', threshold(kernel_num, SAMPLE_SIZE, 0.05))
        pass

def city_mmd(source_city, target_city, window = False):
    """
    计算原始数据的MMD
    """
    # 计算alpha=0.5条件下的阈值，并判断能否拒绝原假设
    alpha = 0.05
    kernel_num = 3
    fix_sigma = 1 # 核函数的宽度，当高斯核的数量为1时，宽度越大，核函数的值越小

    GENERAL_HEADER = ['num'] + \
        ['hospital_f', 'spot_f', 'government_f', 'airport_f', 'subway_f', 'bus_f', 'bank_f', 'enterprise_f', 'school_f', 'community_f', 'hotel_f', 'supermarket_f', 'fast_food_f'] + \
        ['hospital_n', 'spot_n', 'government_n', 'airport_n', 'subway_n', 'bus_n', 'bank_n', 'enterprise_n', 'school_n', 'community_n', 'hotel_n', 'supermarket_n', 'fast_food_n'] + \
        ['hospital_p', 'spot_p', 'government_p', 'airport_p', 'subway_p', 'bus_p', 'bank_p', 'enterprise_p', 'school_p', 'community_p', 'hotel_p', 'supermarket_p', 'fast_food_p'] + \
        ['street_length', 'intersection_density_km', 'street_density_km', 'degree_centrality_avg'] + \
        ['subway', 'bus', 'park1', 'park2', 'park3', 'park4'] + \
        ['supermarket', 'mall']

    batch_size = 64
    num_neigh = 1 # 第一个就是站点本身
    train_mode = 'combine'

    data_loaders = get_data_loader(source_city, target_city, batch_size, num_neigh, train_mode, window)
    
    # 加载源城市和目标城市数据
    
    for (source_train_x, _, _), (target_train_x, _, _) in zip(data_loaders['combine_source'], data_loaders['combine_target']):
        source_train_x = source_train_x.float()
        target_train_x = target_train_x.float()

        batch_num = source_train_x.shape[0]
        print('x shape', source_train_x.shape)
        X = torch.Tensor(source_train_x).squeeze(1)
        Y = torch.Tensor(target_train_x).squeeze(1)
        X, Y = Variable(X), Variable(Y)

        # 计算MMD
        mmd_val = mmd_rbf(X, Y, kernel_mul=2.0, kernel_num=kernel_num, fix_sigma=fix_sigma).numpy()
        data_var = max(source_train_x.numpy().var(), target_train_x.numpy().var())
        threshold_val = threshold(kernel_num, data_var, batch_num, alpha, fix_sigma)

        # 因为数值太小对MMD和threshold都进行放大
        print(mmd_val, threshold_val)
        mmd_val = batch_num * mmd_val * mmd_val # MMD ==> N * MMD^2
        threshold_val = batch_num * threshold_val * threshold_val 
        print(mmd_val, threshold_val, 'diff' if mmd_val > threshold_val else 'same')
        return mmd_val, threshold_val

def compute_mmd_for_model_feature(model, count=70):
    # 计算alpha=0.5条件下的阈值，并判断能否拒绝原假设
    alpha = 0.05
    kernel_num = 3
    fix_sigma = 1  # 核函数的宽度，当高斯核的数量为1时，宽度越大，核函数的值越小

    # setting:
    source_city = 'beijing'
    target_city = 'tianjing'
    # 加载数据
    data_path = files['inter_data']['cache_data'] + os.sep + model
    for i, file_name in enumerate(os.listdir(data_path)):
        if file_name.count(source_city) == 0 or file_name.count(target_city) == 0:
            continue
        data = np.load(data_path + os.sep + file_name, allow_pickle=True)
        size = data.shape[0] // 2

        source_data = data[:size]
        target_data = data[size:]
        print(source_data.shape)

        X = torch.Tensor(source_data)
        Y = torch.Tensor(target_data)
        X, Y = Variable(X), Variable(Y)

        # 计算MMD
        mmd_val = mmd_rbf(X, Y, kernel_mul=2.0, kernel_num=kernel_num, fix_sigma=fix_sigma).numpy()
        data_var = max(source_data.var(), target_data.var())
        threshold_val = threshold(kernel_num, data_var, count, alpha, fix_sigma)

        # 因为数值太小对MMD和threshold都进行放大
        print(mmd_val, threshold_val)
        mmd_val = count * mmd_val * mmd_val  # MMD ==> N * MMD^2
        threshold_val = count * threshold_val * threshold_val
        print(mmd_val, threshold_val, 'diff' if mmd_val > threshold_val else 'same')
        return mmd_val, threshold_val


    # 计算mmd
    pass


def plot_mmd():
    '''
    原始数据
    画图 MMD bandwidth(0.5 1 2) sample_num 32
    # alpha = 0.05 kernel_num = 3 fix_sigma = 1, threshold version 3
    beijing guangzhou: 0.4489 0.0393
    beijing tianjing: 0.5444 0.0372
    guangzhou tianjing: 0.5908 0.0386
    '''
    pass

if __name__ == "__main__":
    city_mmd('guangzhou', 'tianjing', window=True)
    # 没有放大
    # 60: stdann_wd mmd: 0.11718, threshold 0.01364 ==> 放大 0.9612
    # 60: stdann    mmd: 0.2188,  threshold 0.3630 ==> 转化后 0.00822->放大 0.0575
    # 结果： stdann_wd: 0.9612 stdann: 0.0575
    # compute_mmd_for_model_feature('stdann_wd', count=60)
    pass