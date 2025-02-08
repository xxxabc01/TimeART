import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from models.ARTAtt import SingleExpert     
from layers.RevIN import RevIN
from models.ARTAtt import MixOfExperts
from layers.Autoformer_EncDec import series_decomp
from models.ARTAtt import FFT_for_Period
from models.ARTAtt import VariableRealtionModule
from models.ARTAtt import FlattenHead
import time
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin_layer = RevIN(eps=1e-5, subtract_last=False, t_dim=1)
        self.series_decomp = series_decomp(kernel_size=5)
        self.fft_for_period = FFT_for_Period
        self.mix_Experts = MixOfExperts(configs=configs, input_size=configs.seq_len, 
                                       output_size=configs.pred_len, num_experts=len(configs.patch_size),
                                       patch_size=configs.patch_size, noisy_gating=True,
                                       k=2, residual_connection=False)
        # # 变量关系计算模块
        self.varieble_extraction = VariableRealtionModule(configs=configs, reduced_F=1) # TODO
        self.head = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forecast(self, x_enc):
        # 1. 数据处理
        # 首先人为升高维度 输入是BTC，这里变成BTCF
        B,T,C,F = x_enc.shape[0], x_enc.shape[1], x_enc.shape[2], x_enc.shape[3]
        # 数据归一化： BTCF->BTCF
        x_enc = self.revin_layer(x_enc, 'norm')       
        # 时序分解： BTCF->BTCF
        x_for_decomp = torch.reshape(x_enc, (B,T,C*F)) # "B T C F ->B T (C F)"
        season, trend = self.series_decomp(x_for_decomp)
        season = season.view(B,T,C,F) # "B T (C F) -> B T C F"
        trend = trend.view(B,T,C,F)
        # fft变换： BTCF->B(T//2+1)CF
        fft_res, _ = self.fft_for_period(season)
        # 2. 时间关系提取：B T C F -> B T' C  
        balance_loss = torch.tensor(0.0, device=x_enc.device)      
        temporal_relation_out, balance_loss_moe = self.mix_Experts(x_enc, fft_res) 
        balance_loss = balance_loss + balance_loss_moe
        # 3. 变量关系提取：B T C F -> B T' C
        variable_relation_out, variable_attns = self.varieble_extraction(x_enc, fft_res)
        # 4. 结果合并 
        dec_out = torch.cat([temporal_relation_out, variable_relation_out], dim=1)
        dec_out = self.head(dec_out.permute(0,2,1)).permute(0,2,1)
        # 5. 数据处理(DeNorm)->BTC
        dec_out = self.revin_layer(dec_out.unsqueeze(dim=-1), "denorm").squeeze(dim=-1)
        
        return dec_out, balance_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, balance_loss = self.forecast(x_enc)
        return dec_out, balance_loss # [B, L, D]
