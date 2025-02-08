import json
import pandas as pd
import argparse
import numpy as np
import bisect
import logging
from datetime import datetime
import time
import os
import pickle
import torch

def get_time_str():
    now = datetime.now()        
    formatted_time = now.strftime("%m%d%H%M")
    return formatted_time

def get_multi_data_from_json(json_path):
    with open(json_path,"r") as f:
        data = json.load(f)
    return data["multi_data_path"]

def get_data_prefix_from_json(json_path):
    with open(json_path,"r") as f:
        data = json.load(f)
    data_name = "_".join([i.lower() for i in data["multi_data_path"]["data_name"]])
    return data_name

def get_multi_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["flag"]==1]
    df.reset_index(inplace=True)
    return df

def get_data_prefix_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["flag"]==1]
    data_name = "$".join([i.lower() for i in df["data_name"]])
    return data_name

def dynamic_batch_size_method(Dataset_var_num,Max_var_num):
    batch_size_candidation_list = [16,32,64,128]
    max_var_num = Max_var_num
    basic_batch_size = 16
    current_batch_size = (max_var_num// Dataset_var_num) *basic_batch_size
    if current_batch_size >= batch_size_candidation_list[-1]:
        return batch_size_candidation_list[-1]
    else:
        for i in range(len(batch_size_candidation_list)-1):
            if batch_size_candidation_list[i] == current_batch_size:
                return current_batch_size
            elif batch_size_candidation_list[i] < current_batch_size and batch_size_candidation_list[i+1] > current_batch_size:
                return batch_size_candidation_list[i+1]
            else:
                 continue

def get_settings_from_args(args):
    args_dict = {"task_name":args.task_name,
                "model_id":args.model_id,
                "model_prefix":args.model_prefix,
                "model":args.model,
                "features":args.features,
                "seq_len":args.seq_len,
                "label_len":args.label_len,
                "pred_len":args.pred_len,
                "d_model":args.d_model,
                "n_heads":args.n_heads,
                "e_layers":args.e_layers,
                "d_layers":args.d_layers,
                "d_ff":args.d_ff,
                "expand":args.expand,
                "d_conv":args.d_conv,
                "factor":args.factor,
                "use_multi_data":args.use_multi_data,
                "multi_data_schema":get_data_prefix_from_csv(args.multi_data_schema),
                "trained_use_gpu":args.devices.replace(",","-"),
                "zero_data":args.zero_data,
                "zero_root_path":args.zero_root_path,
                "zero_data_path":args.zero_data_path,
                "use_amp":args.use_amp,
                "batch_size":args.batch_size,
                "lr":args.learning_rate,
                "dtype":args.dtype,
                "lradj":args.lradj,
                "epochs":args.train_epochs
                }
    return str(args_dict).replace(",","\n")
            

def setup_logger(log_file):
    # 创建一个logger
    logger = logging.getLogger('ARTTS_Log')
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建一个文件处理器，并设置级别为DEBUG
    file_handler = logging.FileHandler(log_file,mode='a+')
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器，并设置级别为ERROR
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将文件处理器和控制台处理器添加到logger中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = True
    return logger

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def data_low_precision(df):
    float_colulumns = df.select_dtypes(include=["float64"]).columns
    df[float_colulumns] = df[float_colulumns].astype("float32")
    return df
    
def write_metrics_csv(path, info):
    with open(path,"a+") as f:
        f.write(info)

def cache_load_data(data_x, data_name, mode, logger, s_begin=None, s_end=None, r_begin=None, r_end=None):
    if mode=="save":
        temp_df = pd.DataFrame(data_x,columns=["col_{}".format(i) for i in range(data_x.shape[1])])        
        temp_df.to_csv("./dataset_cache/{}".format(data_name),index=False)
        logger.info("cache data {} to disk...".format(data_name))
    if mode=="load":
        temp_seq_x = pd.read_csv(f"./dataset_cache/{data_name}", skiprows=s_begin, nrows=s_end-s_begin).values
        temp_seq_y = pd.read_csv(f"./dataset_cache/{data_name}", skiprows=r_begin, nrows=r_end-r_begin).values
        return temp_seq_x,temp_seq_y

def get_device_freeed():
    device_count = torch.cuda.device_count()
    frees_gpu = []
    for i in range(device_count):
        mem = torch.cuda.memory_allocated(i)
        if mem < 1000:
            frees_gpu.append([i, mem])
    return frees_gpu


def topK_frequency_choose(fft_result, topK):
    if topK == -1:
        return fft_result
    fft_results_abs = abs(fft_result)
    fft_results_abs = fft_results_abs.permute(0,2,1)
    topk_v, _ = torch.topk(fft_results_abs, topK, dim=1)
    topk_v_min, _ = torch.min(topk_v, dim=1, keepdim=True)
    fft_res_topk = fft_result.permute(0,2,1)*(fft_results_abs > topk_v_min)
    fft_res_topk = fft_res_topk.permute(0,2,1)
    return fft_res_topk