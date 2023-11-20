import torch
from torch import nn
import math

class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.warmup = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_steps))
        lr_factor *= min(epoch / self.warmup, 1.0)
        return lr_factor


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        # TODO
        pe = self.make_pe(embed_dim, max_len) # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def make_pe(self, embed_dim, max_len):
        pos = torch.unsqueeze(torch.arange(max_len, dtype=torch.float), 1)
        i = torch.unsqueeze(torch.arange(embed_dim, dtype=torch.float), 0)
        even = i[0] % 2 == 0
        odd = ~even

        i[:, even] = 10000 ** (i[:, even] / embed_dim)
        i[:, odd] = 10000 ** ((i[:, odd] - 1) / embed_dim)
        res = pos / i

        res[:, even] = torch.sin(res[:, even])
        res[:, odd] = torch.cos(res[:, odd])
        return torch.unsqueeze(res, 0)

    def forward(self, x):
        if len(x.shape) == 2:
            # x.shape = (L, D)
            return x + self.pe[0, :x.shape[0], :]
        else:
            # x.shape = (B, L, D)
            return x + self.pe[:, :x.shape[1], :]