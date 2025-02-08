import pandas as pd
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Traja, Dataset_Monash
from torch.utils.data import DataLoader
import random
from collections import deque
import time
import torch
from utils.common import get_multi_data_from_json
from utils.common import get_multi_data_from_csv

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,    
    'traja': Dataset_Traja, 
    'australian_electricity_demand': Dataset_Monash,  # 230736
    'car_parts_dataset_without_missing': Dataset_Monash,  # 51
    'covid_deaths': Dataset_Monash,  # 212
    'electricity_hourly': Dataset_Monash,  # 26304
    'electricity_weekly': Dataset_Monash,  # 156
    'fred_md': Dataset_Monash,  # 728
    'hospital': Dataset_Monash,  # 84
    'kaggle_web_traffic_dataset_without_missing': Dataset_Monash,  # 803
    'kaggle_web_traffic_weekly': Dataset_Monash,  # 114
    'nn5_daily_dataset_without_missing': Dataset_Monash,  # 791
    'nn5_weekly': Dataset_Monash,  # 113
    'oikolab_weather': Dataset_Monash,  # 100057
    'rideshare_dataset_without_missing': Dataset_Monash,  # 541
    'solar_10_minutes': Dataset_Monash,  # 52560
    'solar_weekly': Dataset_Monash,   # 52
    'traffic_hourly': Dataset_Monash,   # 17544
    'traffic_weekly': Dataset_Monash,  # 104
}

class CombinedLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.all_iter = [iter(i) for i in self.dataloaders]
        self.queue = deque(self.all_iter)

    def __iter__(self):
        self.all_iter = [iter(i) for i in self.dataloaders]
        return self

    def __next__(self):
        if self.queue:
            q = self.queue.popleft()
            try:
                data = next(q)
                self.queue.append(q)
                return data
            except StopIteration:
                pass
        else:
            raise StopIteration

    def __len__(self):
        return sum([len(i) for i in self.dataloaders])

class CombinedLoaderOrder(object):
    # 间隔数据集
    def __init__(self, data_loaders, **kwargs):
        self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 32
        self.seed = kwargs["seed"] if "seed" in kwargs else 2024
        self.mode = kwargs["mode"] if "mode" in kwargs else "interval"
        self.data_loaders = data_loaders
        if "weights" not in kwargs:
            sum_le = sum([len(i) for i in self.data_loaders])
            self.weights = [len(i)/sum_le for i in self.data_loaders]
        else:
            self.weights = kwargs["weights"]
        self.start_index = 0
        self._rng = random.Random(self.seed)

    def __iter__(self):
        self.all_iter = [iter(i) for i in self.data_loaders]
        return self

    def __next__(self):
        if self.mode in ["sample"]:
            (dataset,) = self._rng.choices(self.all_iter, weights=self.weights, k=1)
            return next(dataset)

    def __len__(self):
        return sum([len(x) for x in self.data_loaders])

def data_provider(args, flag, logger, back_window_lens=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq
    if not back_window_lens:
        logger.info("back_window_lens设置为None, 默认使用pred_len初始化")
        back_window_lens = args.pred_len

    if not args.use_multi_data:
        if args.data == 'm4':
            drop_last = False
        
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, back_window_lens],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            logger=logger
        )
        logger.info("{} single data {} with length {} with n_vars {}".format(flag, args.data, len(data_set),data_set.n_vars))

        if  not data_set.n_vars > args.max_var_num:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
            return data_set, data_loader
        else:
            total_data_loader_list = []
            logger.info("data_set_var_num >{} , data var should be cut".format(args.max_var_num))
            feature_sampel_times = data_set.n_vars // args.max_var_num +1
            logger.info("cut_var_time ={} ".format(feature_sampel_times))
            total_feature_index = [i for i in range(data_set.n_vars)]
            random.shuffle(total_feature_index)
            for i in range(feature_sampel_times):
                data_set_current_slice = Data(
                    args = args,
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=freq,
                    seasonal_patterns=args.seasonal_patterns,
                    features_index= total_feature_index[i*args.max_var_num:(i+1)*args.max_var_num],
                    logger=logger
                )
                data_loader_current = DataLoader(
                    data_set_current_slice,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last
                )
                total_data_loader_list.append(data_loader_current)

            data_loader = CombinedLoaderOrder(total_data_loader_list, mode='sample')
            return None, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        # 读取多数据集文件
        multi_data_file = get_multi_data_from_csv(args.multi_data_schema)
        total_data_num = len(multi_data_file['data'])
        total_data_loader_list = []
        for i in range(total_data_num):
            data_name = multi_data_file["data_name"][i]
            data = multi_data_file["data"][i]
            data_root_path = multi_data_file["root_path"][i]
            data_data_path = multi_data_file["data_path"][i]
            data_freq = multi_data_file["freq"][i]
            Data_current  = data_dict[data]
            data_set_current = Data_current(
                    args=args,
                    root_path=data_root_path,
                    data_path= data_data_path,
                    flag=flag,
                    size=[args.seq_len, args.label_len, args.pred_len],
                    features=args.features,
                    target=args.target,
                    timeenc=timeenc,
                    freq=data_freq,
                    seasonal_patterns=args.seasonal_patterns,
                    logger=logger
            )
            logger.info(f"data_name: {data_name}, length: {len(data_set_current)}, n_vars:{data_set_current.n_vars}, flag: {flag}")

            # add_new
            if  not data_set_current.n_vars > args.max_var_num:
                sampler = torch.utils.data.DistributedSampler(data_set_current) if args.use_ddp and args.is_training else None
                logger.info(f"the sampler is {sampler}")
                shuffle_flag = False if args.use_ddp and args.is_training else shuffle_flag
                data_loader_current = DataLoader(
                    data_set_current,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last)
                total_data_loader_list.append(data_loader_current)
            else:
                logger.info("data_set_var_num >{} , data var should be cut".format(args.max_var_num))
                feature_sampel_times = data_set_current.n_vars // args.max_var_num +1
                total_feature_index = [i for i in range(data_set_current.n_vars)]
                random.shuffle(total_feature_index)

                for i in range(feature_sampel_times):
                    data_set_current_slice = Data_current(
                        args=args,
                        root_path=data_root_path,
                        data_path= data_data_path,
                        flag=flag,
                        size=[args.seq_len, args.label_len, args.pred_len],
                        features=args.features,
                        target=args.target,
                        timeenc=timeenc,
                        freq=data_freq,
                        seasonal_patterns=args.seasonal_patterns,
                        features_index= total_feature_index[i*args.max_var_num:(i+1)*args.max_var_num],
                        logger=logger
                        )
                    sampler = torch.utils.data.DistributedSampler(data_set_current) if args.use_ddp and args.is_training else None
                    logger.info(f"the sampler is {sampler}")
                    shuffle_flag = False if args.use_ddp and args.is_training else shuffle_flag
                    data_loader_current = DataLoader(
                        data_set_current_slice,
                        batch_size=batch_size,
                        shuffle=shuffle_flag,
                        num_workers=args.num_workers,
                        drop_last=drop_last
                    )
                    total_data_loader_list.append(data_loader_current)

        data_loader = CombinedLoaderOrder(total_data_loader_list, mode='sample')
        return None, data_loader

def data_provider_zero(args, flag, logger, back_window_lens=None):
    Data = data_dict[args.zero_data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq
    if not back_window_lens:
        logger.info("back_window_lens设置为None, 默认使用pred_len初始化")
        back_window_lens = args.pred_len
    
    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        args = args,
        root_path=args.zero_root_path,
        data_path=args.zero_data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, back_window_lens],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        logger=logger
    )
    logger.info("{} single zero data {} with length {}".format(flag, args.zero_data_path, len(data_set)))
    logger.info("data_set_var_num: {}".format(data_set.n_vars))
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

def data_provider_fine_tune(args, flag, logger, data_range, back_window_lens=None):
    Data = data_dict[args.fine_tune_data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq
    if not back_window_lens:
        logger.info("back_window_lens设置为None, 默认使用pred_len初始化, flag=" + flag)
        back_window_lens = args.pred_len
    
    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        args = args,
        root_path=args.fine_tune_root_path,
        data_path=args.fine_tune_data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, back_window_lens],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        data_range = data_range,
        logger=logger
    )
    logger.info("{} single fine tune data {} with data range {} about length {}".format(flag, args.fine_tune_data_path, data_range, len(data_set)))
    logger.info("data_set_var_num: {}".format(data_set.n_vars))
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
 
def data_provider_custom(args, flag, logger, data_range, shuffle_flag, back_window_lens=None):
    """应用这块的data处理逻辑,和大模型训练的不一样,这里不在上面改怕出问题"""
    Data = data_dict[args.fine_tune_data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = shuffle_flag
    drop_last = True
    batch_size = args.batch_size
    freq = args.freq
    if not back_window_lens:
        logger.info("back_window_lens设置为None, 默认使用pred_len初始化, flag=" + flag)
        back_window_lens = args.pred_len
    
    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        args = args,
        root_path=args.fine_tune_root_path,
        data_path=args.fine_tune_data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, back_window_lens],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        data_range = data_range,
        logger=logger
    )
    logger.info("{} single fine tune data {} with data range {} about length {}".format(flag, args.fine_tune_data_path, data_range, len(data_set)))
    logger.info("data_set_var_num: {}".format(data_set.n_vars))
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader