import numpy as np
import torch as tt


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def to_tensor(data, device=tt.device('cuda'), dtype=tt.float32):
    return tt.tensor(data, dtype=dtype).to(device)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
