import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, eps=1e-5, subtract_last=False, t_dim=1):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.eps = eps
        self.subtract_last = subtract_last
        self.t_dim = t_dim

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim= self.t_dim, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=self.t_dim, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x