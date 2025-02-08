from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from data_provider.data_factory import data_provider
from data_provider.data_factory import data_provider_zero
from data_provider.data_factory import data_provider_fine_tune
from data_provider.data_factory import data_provider_custom
from utils.common import setup_logger
import pandas as pd
import pickle
from collections import defaultdict


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)        
        self.dtype = torch.float32
        self.metrics_df_path = None
        self.load_model_pred_lens = -1
        self._get_dtypes(self.args)
    
    def _init_logger_file(self, args):
        log = setup_logger(args.log_path + args.model_prefix + ".log")
        return log

    def _build_model(self):
        self.logger = self._init_logger_file(self.args)
        self._get_dtypes(self.args)
        model = self.model_dict[self.args.model].Model(self.args).to(self.dtype)
        self.total_params_el = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
        self.logger.info(f"parameters of the model is {self.total_params_el}B")
        self.logger.info(f"the patchsize list is {model.mix_Experts.patch_size}")
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _load_model(self, setting, epoch_num=-1):
        if epoch_num==-1:
            checkpoint_name = "checkpoint.pth"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch_num}.pth"
        if self.args.use_multi_gpu:
                self.logger.info('loading model on multi gpu')
                # self.model.load_state_dict(torch.load(os.path.join('checkpoints/' + setting, checkpoint_name)))
                state_dict = {}
                all_pkl = os.listdir("checkpoints_part")
                for i in all_pkl:
                    with open(f"checkpoints_part/{i}","rb") as f:
                        temp = pickle.load(f)
                        for j in temp:
                            for k, v in j.items():
                                state_dict[k] = v
                self.model.load_state_dict(state_dict)        
                self.load_model_pred_lens = self.model.module.head.out_features
        else:
            self.logger.info('loading model on single gpu')
            # state_dict = {k.replace('module.',''):v for k,v in torch.load(os.path.join('checkpoints/' + setting, checkpoint_name)).items()}
            # Here, we decompose the larger .pth file into multiple smaller files to facilitate data transfer. 
            state_dict = {}
            all_pkl = os.listdir("checkpoints_part")
            for i in all_pkl:
                with open(f"checkpoints_part/{i}","rb") as f:
                    temp = pickle.load(f)
                    for j in temp:
                        for k, v in j.items():
                            state_dict[k] = v
            self.model.load_state_dict(state_dict)        
            self.load_model_pred_lens = self.model.head.out_features
        
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.logger)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion 

    def _get_dtypes(self,args):
        dtypes = {"bf16":torch.bfloat16,"fp32":torch.float32, "fp16":torch.float16, "fp64":torch.float64}
        target_dtype = dtypes[args.dtype] if args.dtype in dtypes else torch.float64
        self.dtype = target_dtype       

    def zero_shot_one_epoch(self, setting, test=0, epoch_num=-1, other_args=None):
        self.logger.info("evaluate on the all part of zero data")
        test_data, test_loader = data_provider_zero(self.args, 'test',self.logger, self.args.pred_len)
        
        if test:
            self.logger.info("loading model for zero data")
            self._load_model(setting, epoch_num=epoch_num)
        
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # add res list
        res_list = []
        max_array_feas = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.unsqueeze(-1) if self.args.model == "ARTTS" else batch_x
                batch_x = batch_x.to(self.dtype).to(self.device)
                batch_y = batch_y.to(self.dtype).to(self.device)

                batch_x_mark = batch_x_mark.to(self.dtype).to(self.device)
                batch_y_mark = batch_y_mark.to(self.dtype).to(self.device)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).to(self.dtype)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.dtype).to(self.device)  
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0][:,:self.args.pred_len,...]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # discard the inverse
                if test_data and test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                # current res
                mae, mse, rmse, mape, mspe = metric(pred, true)
                res_list.append([mae, mse, rmse, mape, mspe])
                max_array_feas = max(max_array_feas, outputs.shape[-1])
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data and test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
        preds = [np.pad(i,pad_width=((0,0), (0,0), (0,max_array_feas-i.shape[-1]))) for i in preds]
        trues = [np.pad(i,pad_width=((0,0), (0,0), (0,max_array_feas-i.shape[-1]))) for i in trues]
        preds = np.array(preds)
        trues = np.array(trues)
        self.logger.info(f'test shape: {preds.shape} {trues.shape}')
        preds = np.reshape(preds, (-1, preds.shape[-2], preds.shape[-1]))
        trues = np.reshape(trues,(-1, trues.shape[-2], trues.shape[-1]))
        self.logger.info(f'test shape: {preds.shape} {trues.shape}')

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        mae, mse, rmse, mape, mspe = sum(np.array(res_list)) / len(np.array(res_list))
        dataset = self.args.zero_data_path.split(".")[0]
        self.logger.info(f'{dataset} zero-shot metrics mse:{mse}, mae:{mae}')
        return    

    