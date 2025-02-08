import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
import torch
import torch.nn as nn
import random
from pathlib import Path
from zipfile import ZipFile
import orjson
from utils.common import cache_load_data

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None,features_index=None,
                 data_range=None, logger=None):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val', 'all']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]
        self.data_range = [float(i) for i in data_range.split(":")] if data_range else None 
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.n_vars = -1
        self.root_path = root_path
        self.data_path = data_path
        self.logger = logger
        self.__read_data__()
        self.feature_index = features_index                

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len, 0]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24, df_raw.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # 按比例选择数据
        if self.set_type==3 and self.data_range:
            self.logger.info("按比例选择数据")
            border1 = int(df_raw.shape[0]*self.data_range[0])
            border2 = int(df_raw.shape[0]*self.data_range[1]) + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.n_vars = data.shape[1]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = None

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # add new
        if not self.feature_index:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        else:
            seq_x = self.data_x[s_begin:s_end,self.feature_index]
            seq_y = self.data_y[r_begin:r_end,self.feature_index]

        seq_x_mark = -1
        seq_y_mark = -1

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 seasonal_patterns=None,data_range=None,logger=None
                 ):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'all']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]
        self.data_range = [float(i) for i in data_range.split(":")] if data_range else None
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.n_vars = -1
        self.root_path = root_path
        self.data_path = data_path
        self.logger = logger
        self.__read_data__()        

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len, 0]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
                    df_raw.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 按比例选择数据
        if self.set_type==3 and self.data_range:
            self.logger.info("按比例选择数据")
            border1 = int(df_raw.shape[0]*self.data_range[0])
            border2 = int(df_raw.shape[0]*self.data_range[1]) + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.n_vars = data.shape[1]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = None

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = -1
        seq_y_mark = -1

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None,features_index = None,
                 data_range = None, logger=None):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'all']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]
        self.data_range = [float(i) for i in data_range.split(":")] if data_range else None
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.n_vars = -1
        self.root_path = root_path
        self.data_path = data_path
        self.logger = logger
        self.__read_data__()        
        self.feature_index = features_index

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len, 0]
        border2s = [num_train, num_train + num_vali, len(df_raw), df_raw.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 按比例选择数据
        if self.set_type==3 and self.data_range:
            self.logger.info("按比例选择数据")
            border1 = int(df_raw.shape[0]*self.data_range[0])
            border2 = int(df_raw.shape[0]*self.data_range[1]) + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.len_data_x = len(self.data_x)
        self.n_vars = data.shape[1]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = None

        # 是否需要cache数据
        if self.args.cache_train_data:
            data_name = self.data_path.split(".")[0] + "_{}.csv".format(self.flag)
            cache_load_data(self.data_x, data_name, "save", self.logger)
            self.data_x = None
            self.data_y = None

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 从内存加载
        if not self.args.cache_train_data:
            if not self.feature_index:
                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[r_begin:r_end]
            else:
                seq_x = self.data_x[s_begin:s_end,self.feature_index]
                seq_y = self.data_y[r_begin:r_end,self.feature_index]
        else:
            # 从文件加载
            data_name = self.data_path.split(".")[0] + "_{}.csv".format(self.flag)
            temp_seq_x, temp_seq_y = cache_load_data(None,data_name,"load",None,s_begin,s_end,r_begin,r_end)
            if not self.feature_index:
                seq_x = temp_seq_x
                seq_y = temp_seq_y
            else:
                seq_x = temp_seq_x[:,self.feature_index]
                seq_y = temp_seq_y[:,self.feature_index]

        seq_x_mark = -1
        seq_y_mark = -1

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.len_data_x - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Traja(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None,features_index = None,
                 data_range = None, logger=None):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'all']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]
        self.data_range = [float(i) for i in data_range.split(":")] if data_range else None
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.n_vars = -1
        self.root_path = root_path
        self.data_path = data_path
        self.logger = logger
        self.__read_data__()        
        self.feature_index = features_index

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len, 0]
        border2s = [num_train, num_train + num_vali, len(df_raw), df_raw.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 按比例选择数据
        if self.set_type==3 and self.data_range:
            self.logger.info("按比例选择数据")
            border1 = int(df_raw.shape[0]*self.data_range[0])
            border2 = int(df_raw.shape[0]*self.data_range[1]) + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]].values
            train_data = traj_inverse(train_data)
            self.scaler.fit(train_data)
            data = self.scaler.transform(train_data)
            # data = df_data.values - np.array([121.59815009,  29.96905493, 1])
            # data*=100           
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        print(f"最大{np.max(self.data_x)}，最小{np.min(self.data_x)}， 最大{np.max(self.data_y)}，最小{np.max(self.data_y)}")
        self.len_data_x = len(self.data_x)
        self.n_vars = data.shape[1]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = None

        # 是否需要cache数据
        if self.args.cache_train_data:
            data_name = self.data_path.split(".")[0] + "_{}.csv".format(self.flag)
            cache_load_data(self.data_x, data_name, "save", self.logger)
            self.data_x = None
            self.data_y = None

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 从内存加载
        if not self.args.cache_train_data:
            if not self.feature_index:
                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[r_begin:r_end]
            else:
                seq_x = self.data_x[s_begin:s_end,self.feature_index]
                seq_y = self.data_y[r_begin:r_end,self.feature_index]
        else:
            # 从文件加载
            data_name = self.data_path.split(".")[0] + "_{}.csv".format(self.flag)
            temp_seq_x, temp_seq_y = cache_load_data(None,data_name,"load",None,s_begin,s_end,r_begin,r_end)
            if not self.feature_index:
                seq_x = temp_seq_x
                seq_y = temp_seq_y
            else:
                seq_x = temp_seq_x[:,self.feature_index]
                seq_y = temp_seq_y[:,self.feature_index]

        seq_x_mark = -1
        seq_y_mark = -1

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.len_data_x - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Monash(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='australian_electricity_demand',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None,features_index=None,data_range=None,logger=None):
        self.data_path_dict = {
            'australian_electricity_demand': 'australian_electricity_demand_dataset.zip',
            'car_parts_dataset_without_missing': 'car_parts_dataset_without_missing_values.zip',
            'covid_deaths': 'covid_deaths_dataset.zip',
            'electricity_hourly': 'electricity_hourly_dataset.zip',
            'electricity_weekly': 'electricity_weekly_dataset.zip',
            'fred_md': 'fred_md_dataset.zip',
            'hospital': 'hospital_dataset.zip',
            'kaggle_web_traffic_dataset_without_missing': 'kaggle_web_traffic_dataset_without_missing_values.zip',
            'kaggle_web_traffic_weekly': 'kaggle_web_traffic_weekly_dataset.zip',
            'nn5_daily_dataset_without_missing': 'nn5_daily_dataset_without_missing_values.zip',
            'nn5_weekly': 'nn5_weekly_dataset.zip',
            'oikolab_weather': 'oikolab_weather_dataset.zip',
            'rideshare_dataset_without_missing': 'rideshare_dataset_without_missing_values.zip',
            'solar_10_minutes': 'solar_10_minutes_dataset.zip',
            'solar_weekly': 'solar_weekly_dataset.zip',
            'traffic_hourly': 'traffic_hourly_dataset.zip',
            'traffic_weekly': 'traffic_weekly_dataset.zip',

        }

        self.freq_dict = {
            "seconds": "S",
            "minutely": "T",
            "minutes": "T",
            "10_minutes": "10T",
            "half_hourly": '0.5H',
            "hourly": "H",
            "hours": "H",
            "daily": "D",
            "days": "D",
            "weekly": "W",
            "weeks": "W",
            "monthly": "M",
            "months": "M",
            "quarterly": "Q",
            "quarters": "Q",
            "yearly": "Y",
            "years": "Y",
        }

        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'all']
        self.flag = flag
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]
        self.data_range = [float(i) for i in data_range.split(":")] if data_range else None
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.n_vars = -1
        self.root_path = root_path
        self.data_path = self.data_path_dict[data_path]
        self.logger = logger
        self.__read_data__()        
        self.feature_index = features_index

    def __read_data__(self):
        self.scaler = StandardScaler()

        root_path = Path(self.root_path)

        with ZipFile(root_path / self.data_path) as archive:
            archive.extractall(path=root_path)

        file_path = root_path / archive.namelist()[0]  # 'data/Monash/fred_md_dataset.tsf'
        data_list = []
        with open(file_path, encoding="latin1") as in_file:
            # strip whitespace
            lines = map(str.strip, in_file)

            for line in lines:
                if line.startswith("#"):
                    continue
                elif line.startswith("@"):
                    if 'frequency' in line:
                        temp = line.split(" ")
                        freq = temp[1]
                    continue
                else:
                    parts = line.split(":")

                    *attributes, target = parts
                    start_time = attributes[1] if '-' in attributes[1] else attributes[-1]

                    target = target.replace("?", '"nan"')
                    values = orjson.loads(f"[{target}]")
                    data_list.append(np.array(values, dtype=float))

        self.freq = self.freq_dict[freq]
        n = min([i.shape[0] for i in data_list])
        data = np.vstack([temp[:n] for temp in data_list]).T
        df_raw = pd.DataFrame({'date': pd.date_range(start_time, periods=n, freq=self.freq).values})
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len, 0]
        border2s = [num_train, num_train + num_vali, n, df_raw.shape[0]]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 按比例选择数据
        if self.set_type==3 and self.data_range:
            self.logger.info("按比例选择数据")
            border1 = int(df_raw.shape[0]*self.data_range[0])
            border2 = int(df_raw.shape[0]*self.data_range[1]) + self.seq_len

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df_stamp = df_raw[border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.len_data_x = len(self.data_x)
        self.n_vars = data.shape[1]
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = None

        # 是否需要cache数据
        if self.args.cache_train_data:
            data_name = self.data_path.split(".")[0] + "_{}.csv".format(self.flag)
            cache_load_data(self.data_x, data_name, "save", self.logger)
            self.data_x = None
            self.data_y = None

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 从内存加载
        if not self.args.cache_train_data:
            if not self.feature_index:
                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[r_begin:r_end]
            else:
                seq_x = self.data_x[s_begin:s_end,self.feature_index]
                seq_y = self.data_y[r_begin:r_end,self.feature_index]
        else:
            # 从文件加载
            data_name = self.data_path.split(".")[0] + "_{}.csv".format(self.flag)
            temp_seq_x, temp_seq_y = cache_load_data(None,data_name,"load",None,s_begin,s_end,r_begin,r_end)
            if not self.feature_index:
                seq_x = temp_seq_x
                seq_y = temp_seq_y
            else:
                seq_x = temp_seq_x[:,self.feature_index]
                seq_y = temp_seq_y[:,self.feature_index]
                
        seq_x_mark = -1
        seq_y_mark = -1

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.len_data_x - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)