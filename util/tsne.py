# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import os

from sklearn import datasets
from sklearn.manifold import TSNE

from preprocess.load_data import read_yaml
from util.dataloader import get_data_loader
import h5py

files = read_yaml(windows=False)


# def test_tsne():
#     def get_data():
#         digits = datasets.load_digits(n_class=6)
#         data = digits.data
#         label = digits.target
#         n_samples, n_features = data.shape
#         return data, label, n_samples, n_features
#
#     data, label, n_samples, n_features = get_data()
#     print(data.shape, label.shape)
#     print('Computing t-SNE embedding')
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     t0 = time()
#     result = tsne.fit_transform(data)
#
#     color_map = {0: '#32CD32', 1: '#FFA500', 2: '#FF0000', 3: '#808080', 4: '#1E90FF', 5: '#000080', 6: '#9400D3'}
#     mark_map = {0: 'o', 1: '^', 2: '*', 3: 'h', 4: 'd', 5: 'p', 6: 'x'}
#
#     plot_embedding(result, label, 't-SNE embedding of the digits (time %.2fs)' % (time() - t0), color_map, mark_map)


def plot_embedding(data, label, title, color_map, mark_map, save=True):
    img_path = '/home/ryj/renyajie/exp/NETL/img' #r'E:\code\netl\img' #'
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    data_map = {i: {0: [], 1: []} for i in set(label)}
    for d, l in zip(data, label):
        data_map[l][0].append(d[0])
        data_map[l][1].append(d[1])

    fig = plt.figure()
    ax = fig.add_subplot()
    for k, v in data_map.items():
        ax.scatter(v[0], v[1], s=100, c=color_map[k], marker=mark_map[k], alpha=.5, label='Beijing' if ('beijing'==k) or ('Beijing'==k) else ('Guangzhou' if ('guangzhou' == k) or ('Guangzhou' == k) else 'Tianjin'))
    # plt.xticks([])
    # plt.yticks([])
    plt.xlim((-0.15, 1.03))
    # plt.xticks(np.linspace(0, 1, 11), fontsize=16)
    plt.ylim((-0.03, 1.15))
    # plt.yticks(np.linspace(0, 1, 11), fontsize=16)
    font_lengend = {
        'family': 'Palatino Linotype',
        'weight': 'light',
        'size': 20,
    }
    ax.legend(loc='upper left', prop=font_lengend, markerscale=2, framealpha=1)
    # plt.title(title)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Palatino Linotype') for label in labels]
    ax.tick_params(labelsize=20)
    # plt.text(0.62, 0, 'MMD = 0.0575',
    #         fontdict={'size': 22, 'color': 'black', 'family': 'Palatino Linotype'})  # 直接打字

    if save:
        # plt.savefig(img_path + os.sep + title + ".jpg", dpi=300, bbox_inches='tight')
        # 给图形的四周加粗 宽度为2
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        fig.tight_layout()
        fig.savefig('paper_picture/feature_analysis/tnse_three_city_raw.pdf')

    fig.show()


def plot_two_city_origin_data(source_city, target_city):
    """
    绘制原始的数据TSNE分布图，使用dataloader
    """
    batch_size = 128
    num_neigh = 1  # 第一个就是站点本身
    train_mode = 'combine'

    data_loaders = get_data_loader(source_city, target_city, batch_size, num_neigh, train_mode, window=True)

    # 加载源城市和目标城市数据

    all_data = []
    all_label = []
    for i, ((source_train_x, _, _), (target_train_x, _, _)) in enumerate(zip(data_loaders['combine_source'],
                                                              data_loaders['combine_target'])):
        source_train_x = source_train_x.float().squeeze(1).numpy()
        target_train_x = target_train_x.float().squeeze(1).numpy()
        data = np.vstack([source_train_x, target_train_x])
        all_data.append(data)

        label = [source_city for i in range(source_train_x.shape[0])] + [target_city for i in range(target_train_x.shape[0])]
        label = np.array(label)
        all_label.append(label)

        if i == 15:
            plot_data = np.vstack(all_data)
            plot_label = np.vstack(all_label).ravel()
            print(plot_data.shape, plot_label.shape)

            print('Computing t-SNE embedding')
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            t0 = time()
            result = tsne.fit_transform(plot_data)
            print(result.shape)
            color_map = {'beijing': '#1E90FF', 'tianjing': '#FFA500', 'guangzhou': '#FF0000'}
            mark_map = {'beijing': 'o', 'tianjing': 'x', 'guangzhou': '*'}

            plot_embedding(result, plot_label, 't-SNE embedding of the digits (time %.2fs)' % (time() - t0), color_map, mark_map)
            break

def plot_three_city_origin_data(num_neigh, division=40, save=False):
    """
    绘制原始的数据TSNE分布图，使用h5原始文件，本地环境
    """

    h5_path = 'data/exp_data/static/h5_full'
    city_list = ['beijing', 'guangzhou', 'tianjing']
    right_city = {'beijing': 'Beijing', 'guangzhou': 'Guangzhou', 'tianjing': 'Tianjin'}
    data_dict = {}
    one_dimen = True

    for city in city_list:
        f = h5py.File(h5_path + os.sep + 'full_target_{}_{}.h5'.format(num_neigh, city), 'r')
        data_dict[city] = f['train_ex']

    # 加载源城市和目标城市数据
    all_data = []
    all_label = []
    for city in city_list:
        if 'tianjing' != city:
            size = len(data_dict[city]) // division
        if one_dimen:
            d = np.array(data_dict[city])
            d = d[:size].reshape(size, 52 * num_neigh)
        else:
            d = np.array(data_dict[city])[:size]

        print(city, d.shape)
        all_data.append(d)
        all_label.append(np.array([right_city[city] for _ in range(size)]).reshape(size, 1))

    plot_data = np.vstack(all_data)
    plot_label = np.vstack(all_label).squeeze(1)
    print(plot_data.shape, plot_label.shape)

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(plot_data)
    color_map = {'Beijing': '#1f77b4', 'Tianjin': '#ff7f0e', 'Guangzhou': '#2ca02c'}
    mark_map = {'Beijing': 'o', 'Tianjin': 's', 'Guangzhou': '^'}

    plot_embedding(result, plot_label,
                   'neigh_{}_div_{}'
                   .format(num_neigh, division),
                   color_map, mark_map, save)


def plot_feature_data(source_city, target_city, model, count, save=False):
    """
    绘制模型训练后特征的TSNE图，服务器环境
    """

    # load feature data
    data_path = files['inter_data']['cache_data'] + os.sep + model
    all_data = []
    all_label = []
    cnt = 0
    right_city = {'beijing': 'Beijing', 'guangzhou': 'Guangzhou', 'tianjing': 'Tianjin'}
    for i, file_name in enumerate(os.listdir(data_path)):
        if file_name.count(source_city) == 0 or file_name.count(target_city) == 0:
            continue
        data = np.load(data_path + os.sep + file_name, allow_pickle=True)
        size = data.shape[0] // 2

        all_data.append(data)

        label = [right_city[source_city] for i in range(size)] + [right_city[target_city] for i in range(size)]
        label = np.array(label)
        all_label.append(label)
        cnt = cnt + 1

        if cnt == count:
            plot_data = np.vstack(all_data)
            plot_label = np.vstack(all_label).ravel()
            print(plot_data.shape, plot_label.shape)

            print('Computing t-SNE embedding')
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            t0 = time()
            result = tsne.fit_transform(plot_data)
            color_map = {'Beijing': '#1E90FF', 'Tianjin': '#FFA500', 'Guangzhou': '#FF0000'}
            mark_map = {'Beijing': 'o', 'Tianjin': 'x', 'Guangzhou': '*'}

            plot_embedding(result, plot_label, 'feature_{}_{}'.format(model, cnt), color_map, mark_map, save)
            break


if __name__ == '__main__':
    # plot_two_city_origin_data('beijing', 'tianjing')
    # 主要是 num_neigh: 7-10 division: 40-80, using 9-50
    # for i in range(1, 10, 2):
    #   for j in range(3, 7):
    #        print(i, j)
    #        print('-' * 30)
    #        plot_three_city_origin_data(i, division=(j + 1) * 10, save=False)
    plot_three_city_origin_data(5, division=70, save=True) # neigh 9 div 50

    # plot stdann feature --70
    # plot stdann_wd feature -- 70
    # plot_feature_data('beijing', 'tianjing', 'stdann', 70, False)
    pass
