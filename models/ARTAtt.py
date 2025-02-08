import torch.nn as nn
import torch
from torch import nn
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
from torch.distributions.normal import Normal
import time
from utils.common import topK_frequency_choose
from einops import rearrange
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP
import numpy as np

####--------------------------------------------------------时间关系计算--------------------------------------------------
def get_patch_num(seq_len, patch_len, stride):
        patch_num = 0
        index = 0
        while index + patch_len <= seq_len:
            patch_num = patch_num + 1
            index = index + stride
        return patch_num


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, F_reduced, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.F = F_reduced
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len*self.F, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, F, T = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        x = x.view(B, C*F, T)  # "B C F T -> B (C F ) T"
        n_vars = x.shape[1]
        # B (C F) T -> B (C F) T+pad
        x = self.padding_patch_layer(x)
        # B (C F) T+pad -> B (C F) Pn Pl
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) 
        Pn, Pl = x.shape[-2], x.shape[-1]
        # B (C F) Pn Pl -> (B C) Pn (Pl F)
        x = torch.reshape(x,(B,C,F,Pn,Pl))
        x = x.permute(0,1,3,4,2)
        x = torch.reshape(x,(B*C,Pn,Pl*F))
        # (B C) Pn (Pl F) -> (B C) Pn d
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class PatchEmbedding2(nn.Module):
    def __init__(self, seq_len, d_model, patch_len, F_reduced, stride, padding, dropout):
        super(PatchEmbedding2, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.F = F_reduced
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.patch_num = get_patch_num(seq_len=seq_len+padding, patch_len=self.patch_len, stride=self.stride)
        self.value_embedding = nn.Linear(self.patch_num*self.F, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, F, T = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        x = x.view(B, C*F, T)  # "B C F T -> B (C F ) T"
        n_vars = x.shape[1]
        # B (C F) T -> B (C F) T+pad
        x = self.padding_patch_layer(x)
        # B (C F) T+pad -> B (C F) Pn Pl
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) 
        Pn, Pl = x.shape[-2], x.shape[-1]
        # B (C F) Pn Pl -> (B C) Pl (Pn F)
        x = torch.reshape(x,(B,C,F,Pn,Pl))
        x = x.permute(0,1,4,3,2)
        x = torch.reshape(x,(B*C,Pl,Pn*F))
        # (B C) Pn (Pl F) -> (B C) Pn d
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars



class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x  

def FFT_for_Period(x, dim=1, k=2):
    # x: [B, T, C, F], fft后T变成T//2+1
    xf = torch.fft.rfft(x, dim=dim)
    # 通过幅值找周期
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return xf, period


def get_nearst_patchsize(input_patch_size, patch_size_list):
    diff_list = [abs(i-input_patch_size) for i in patch_size_list] 
    return np.argmin(diff_list)


class SingleExpert(nn.Module):    
    def __init__(self, configs, patch_len=16, stride=8, F_reduced=1):
        super(SingleExpert, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, F_reduced,stride, padding, configs.dropout)
        self.patch_embedding2 = PatchEmbedding2(configs.seq_len,
            configs.d_model, patch_len, F_reduced,stride, padding, configs.dropout)

        # 时间关系计算
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Prediction Head
        self.head_nf = configs.d_model * (int((configs.seq_len - patch_len) / stride + 2) + patch_len) # + patch_len 如何需要inverse patch
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.pred_len)
        self.drop = nn.Dropout(0.1)

    def forecast(self, x_enc):     
        # B T C F->B C F T
        x_enc = x_enc.permute(0, 2, 3, 1)
        # B C F T -> B C Pn Pl F -> B*C Pn d
        enc_out, n_vars = self.patch_embedding(x_enc)
        # B*C Pn d -> B*C Pn d
        enc_out, _ = self.encoder(enc_out)
        # B*C Pn d -> B C Pn d
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # B C Pn d -> B C d Pn
        enc_out = enc_out.permute(0, 1, 3, 2)
        # 添加1：patch.T进行表征
        enc_out2, n_vars2 = self.patch_embedding2(x_enc)
        enc_out2, _ = self.encoder2(enc_out2)
        enc_out2 = torch.reshape(enc_out2, (-1, n_vars2, enc_out2.shape[-2], enc_out2.shape[-1]))
        enc_out2 = enc_out2.permute(0, 1, 3, 2)
        # B C d Pn -> B C T'
        dec_out = torch.cat([enc_out,enc_out2],dim=-1)
        dec_out = self.head(dec_out)
        # dec_out = self.layer_norm(dec_out)
        # B C T' -> B T' C
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    

class MixOfExperts(nn.Module):
    """
    MoE:
    Todo: 设置 residual_connection=1
    """
    def __init__(self, configs, input_size, output_size, num_experts, 
                 patch_size=[16,8,4], noisy_gating=True, k=2, 
                 residual_connection=False):
        super(MixOfExperts, self).__init__()
        self.configs = configs
        self.num_experts = num_experts # 3
        self.output_size = output_size # 96 | 192 etc.
        self.input_size = input_size # 96
        self.patch_size = patch_size
        self.k = k
        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        for patch in patch_size:
            self.experts.append(SingleExpert(configs=configs,patch_len=patch, stride=8,F_reduced=1))

        self.w_gate = nn.Parameter(torch.randn(input_size//2+1, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.randn(input_size//2+1, num_experts), requires_grad=True)

        self.residual_connection = residual_connection

        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert len(patch_size) == num_experts
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        x = abs(x)
        x = x.sum(dim=(-1,-2))
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load
    
    def nearst_top_k_gating(self, fft_res, train, topK):
        # 通过幅值找周期
        device = fft_res.device
        frequency_list = abs(fft_res).mean([-1,-2]) 
        frequency_list[:, 0] = 0
        period_values, period_indices = torch.topk(frequency_list, topK)
        period_indices = period_indices.detach().cpu().numpy()
        patch_size = self.input_size // period_indices
        patch_size_tr = torch.zeros(fft_res.shape[0], topK).to(device)
        # 修改找到最近的patch_size
        for B_ in range(patch_size.shape[0]):
            for k_ in range(topK):
                patch_size_tr[B_][k_] = get_nearst_patchsize(patch_size[B_][k_], self.patch_size)
        # 初始化gate函数
        period_values_norm = period_values / torch.max(period_values,dim=1)[0].unsqueeze(1)
        top_k_gates = nn.Softmax(dim=1)(period_values_norm)
        zeros = torch.zeros((fft_res.shape[0], self.num_experts)).to(device)
        gates = zeros.scatter_add(1, patch_size_tr.type(torch.int64), top_k_gates)
        load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, fft_res, loss_coef=1e-2):
        # 正常流程： 为了搭建流程这里选出第一个专家来测试效果, 使 nearst_top_k_gating 或者用 noisy_top_k_gating 确定gate状态
        # fft_res: B (T//2+1) C F
        gates, load = self.noisy_top_k_gating(fft_res, self.training, self.k)
        # write_gate_txt(gates, self.configs.zero_data_path.split(".")[0])
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = []
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts) if expert_inputs[i].shape[0]>=1]
        output = dispatcher.combine(expert_outputs)
        if self.residual_connection:
            output = output + x        
        return output, balance_loss


####--------------------------------------------------------变量关系计算--------------------------------------------------
class TemporalEmbedding(nn.Module):
    def __init__(self, T, F, d_model,d_ff):
        super(TemporalEmbedding, self).__init__()
        self.fc = nn.Linear(T*F, d_model)

    def forward(self, x):
        out = self.fc(x)
        return out
    

class FFTSignalTopkReverse(nn.Module):
    def __init__(self, topK):
        super(FFTSignalTopkReverse,self).__init__()
        self.topK = topK
    
    def forward(self, x):
        # fft_result: B(T//2+1)CF
        fft_result  = torch.fft.rfft(x, dim=1)
        # B,T,C,F = fft_result.shape[0],fft_result.shape[1],fft_result.shape[2],fft_result.shape[3]
        fft_results_abs = torch.abs(fft_result)
        topk_v, _ = torch.topk(fft_results_abs, k=self.topK, dim=1)
        topk_v_min, _ = torch.min(topk_v, dim=1, keepdim=True)
        fft_res_topk = fft_result*(fft_results_abs > topk_v_min)
        ifft_results = torch.fft.irfft(fft_res_topk, dim=1)
        return ifft_results
    

class VariableRealtionModule(nn.Module):
    def __init__(self, configs, reduced_F) :
        super(VariableRealtionModule,self).__init__()
        self.temporal_embedding_time = TemporalEmbedding(configs.seq_len, reduced_F,configs.d_model, configs.d_ff)
        # self.fft_embedding_time = TemporalEmbedding(configs.seq_len//2+1 , reduced_F,configs.d_model, configs.d_ff)
        self.fft_embedding_time = TemporalEmbedding(configs.seq_len, reduced_F,configs.d_model, configs.d_ff)
        self.fft_signal_topk_reverse = FFTSignalTopkReverse(topK=configs.seq_len//2)
        # Encoder
        self.encoder_fre_real = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder_fre_imag = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # project
        self.head = nn.Linear(configs.d_model, configs.pred_len)
        self.drop1 = nn.Dropout(configs.dropout)
        self.drop2 = nn.Dropout(configs.dropout)
        # increase the dim
        self.fft_real_indim = nn.Linear(configs.seq_len//2+1, configs.d_model)
        self.fft_imag_indim = nn.Linear(configs.seq_len//2+1, configs.d_model)
        # decrease the dim
        self.fft_real_dedim = nn.Linear(configs.d_model, configs.pred_len//2+1)
        self.fft_imag_dedim = nn.Linear(configs.d_model, configs.pred_len//2+1)

    
    def forward(self, x, fft_result):
        # x: B T C F, fft_results: B (T+1)//2 C F
        B,T,C,F = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        # 做法2
        fft_result2 = fft_result.mean(dim=-1).permute(0,2,1)
        fft_result2 = topK_frequency_choose(fft_result2, -1)
        fft_result2_real = fft_result2.real
        fft_result2_imag = fft_result2.imag
        fft_result2_real = self.fft_real_indim(fft_result2_real)
        fft_result2_imag = self.fft_imag_indim(fft_result2_imag)
        fft_result2_real_encoder, attns = self.encoder_fre_real(fft_result2_real)
        fft_result2_imag_encoder, _ = self.encoder_fre_imag(fft_result2_imag)
        fft_result2_real_encoder = self.fft_real_dedim(fft_result2_real_encoder)
        fft_result2_imag_encoder = self.fft_imag_dedim(fft_result2_imag_encoder)
        fft_result3 = torch.complex(fft_result2_real_encoder, fft_result2_imag_encoder)
        project_res = torch.fft.irfft(fft_result3.to(torch.complex64),dim=-1).permute(0,2,1)
        return project_res, attns