import time
import os
import h5py
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocess.process import GENERAL_HEADER, SEMI_GENERAL_HEADER

logger = logging.getLogger(__name__)

class CdpDataLoader(Dataset):
    def __init__(self, cx, ex, t, y):
        self.cx = cx
        self.ex = ex
        self.t = t
        self.y = y

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, idx):
        return self.cx[idx], self.ex[idx], self.t[idx], self.y[idx]

def get_data_loader(logger, source, target, bs, num_nb):
    feature_set = get_feature(source, target, num_nb)

    # source city features
    source_train_cx = feature_set['source_train_cx']
    source_train_ex = feature_set['source_train_ex']
    source_train_t = feature_set['source_train_t']
    source_train_y = feature_set['source_train_y']
    # target city features
    target_train_cx = feature_set['target_train_cx']
    target_train_ex = feature_set['target_train_ex']
    target_train_t = feature_set['target_train_t']
    target_train_y = feature_set['target_train_y']
    
    val_cx = feature_set['val_cx']
    val_ex = feature_set['val_ex']
    val_t = feature_set['val_t']
    val_y = feature_set['val_y']

    test_cx = feature_set['test_cx']
    test_ex = feature_set['test_ex']
    test_t = feature_set['test_t']
    test_y = feature_set['test_y'] 
    
    logger.info("# Getting dataloaders.")
    logger.info("Raw feature shape (cx, ex, t, y):")
    logger.info("train source: {}, {}, {}, {}".format(source_train_cx.shape, source_train_ex.shape,
                                                      source_train_t.shape, source_train_y.shape))
    logger.info("train target: {}, {}, {}, {}".format(target_train_cx.shape, target_train_ex.shape,
                                                      target_train_t.shape, target_train_y.shape))
    logger.info("val: {}, {}, {}, {}".format(val_cx.shape, val_ex.shape, val_t.shape, val_y.shape))
    logger.info("test: {}, {}, {}, {}".format(test_cx.shape, test_ex.shape, test_t.shape, test_y.shape))

    # logger.info("Target shape: {}".format(len(target.index)))

    data_loaders = {}

    data_loaders['combine_source'] = DataLoader(CdpDataLoader(source_train_cx, source_train_ex, source_train_t, source_train_y),
                                                batch_size=bs // 2, 
                                                shuffle=True, 
                                                drop_last=True)
    data_loaders['combine_target'] = DataLoader(CdpDataLoader(target_train_cx, target_train_ex, target_train_t, target_train_y),
                                                batch_size=bs // 2, 
                                                shuffle=True, 
                                                drop_last=True)
    data_loaders['val'] = DataLoader(CdpDataLoader(val_cx, val_ex, val_t, val_y), 
                                     batch_size=bs, 
                                     shuffle=True, 
                                     drop_last=True)
    data_loaders['test'] = DataLoader(CdpDataLoader(test_cx, test_ex, test_t, test_y), 
                                      batch_size=bs, 
                                      shuffle=True, 
                                      drop_last=True)
    return data_loaders


def external_station_feature_headers(num_nb):
    external_feature_header = []
    for i in range(num_nb):
        external_feature_header += ["{}_{}".format(x, i) 
                                    if x != "num" 
                                    else "total_num_{}".format(i)
                                    for x in GENERAL_HEADER] 
    station_feature_header = ['near_station', 'near_stub', 'slow_num', 'fast_num', 'total_num']
    return external_feature_header, station_feature_header

def get_feature(source, target, num_nb):
    external_feature_names, station_feature_names = \
            external_station_feature_headers(num_nb)
    # split source city data set
    source_train, source_val = train_test_split(source, test_size=0.2)

    # align data
    target_train = target.copy(deep=True)
    if source_train.shape[0] < target_train.shape[0]:
        target_train = target_train.iloc[0:source_train.shape[0]]
    elif source_train.shape[0] > target_train.shape[0]:
        source_train = source_train.iloc[0:target_train.shape[0]]

    # get train
    source_train_cx = source_train[station_feature_names]
    logger.debug("get feature step before data loader:")
    logger.debug("station feature header len: {}, source train cx shape: {}".format(len(station_feature_names),
        source_train_cx.shape))
    source_train_ex = source_train[external_feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    source_train_t = source_train['time_embed']
    source_train_y = source_train[['slow_demand', 'fast_demand', 'total_demand']]

    target_train_cx = target_train[station_feature_names]
    target_train_ex = target_train[external_feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    target_train_t = target_train['time_embed']
    target_train_y = target_train[['slow_demand', 'fast_demand', 'total_demand']]

    # get val
    val_cx = source_val[station_feature_names]
    val_ex = source_val[external_feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    val_t = source_val['time_embed']
    val_y = source_val[['slow_demand', 'fast_demand', 'total_demand']]
    logger.info('get val data')

    # get test
    test_cx = target[station_feature_names]
    test_ex = target[external_feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    test_t = target['time_embed']
    test_y = target[['slow_demand', 'fast_demand', 'total_demand']]
    logger.info('get test data')

    data_map = {'source_train_cx': source_train_cx, 'source_train_ex': source_train_ex,
                'source_train_t': source_train_t, 'source_train_y': source_train_y,
                'target_train_cx': target_train_cx, 'target_train_ex': target_train_ex,
                'target_train_t': target_train_t, 'target_train_y': target_train_y,
                'val_cx': val_cx, 'val_ex': val_ex, 'val_t': val_t, 'val_y': val_y,
                'test_cx': test_cx,'test_ex': test_ex, 'test_t': test_t, 'test_y': test_y}
    for k, v in data_map.items():
        v = v.fillna(0)
        if k.endswith("_ex"):
            data_map[k] = v.values.reshape((len(v), num_nb, -1))
        else:
            data_map[k] = v.values.reshape((len(v), -1))
    return data_map


def get_infer_dataloader(demand_object_set, bs, num_nb):
    external_feature_names, station_feature_names = \
            external_station_feature_headers(num_nb)

    infer_cx = demand_object_set[station_feature_names].copy()
    infer_ex = demand_object_set[external_feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).copy()
    infer_t = demand_object_set["time_embed"].copy()
    infer_y = demand_object_set[["slow_demand", "fast_demand", "total_demand"]].copy()
    
    infer_cx.fillna(0, inplace=True)
    infer_ex.fillna(0, inplace=True)
    infer_t.fillna(0, inplace=True)
    infer_y.fillna(0, inplace=True)
    infer_cx = infer_cx.values
    infer_ex = infer_ex.values
    infer_t = infer_t.values
    infer_y = infer_y.values
    infer_cx = infer_cx.reshape(len(infer_cx), -1)
    infer_ex = infer_ex.reshape(len(infer_ex), num_nb, -1)
    infer_t = infer_t.reshape(len(infer_t), -1)
    infer_y = infer_y.reshape(len(infer_y), -1)

    data_loaders = DataLoader(CdpDataLoader(infer_cx, infer_ex, infer_t, infer_y), 
                              batch_size=bs, 
                              shuffle=True, 
                              drop_last=False)

    return data_loaders
