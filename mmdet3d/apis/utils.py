from collections import OrderedDict
import torch.distributed as dist
import torch
import torch.nn as nn


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def set_requires_grad(models, requires_grad=False):
    if not isinstance(models, list):
        models = [models]
    for model in models:
        if model is None:
            continue
        for param in model.parameters():
            param.requires_grad = requires_grad


def build_mlp(fcs):
    # fcs: list of channels
    mlp = []
    for i, (in_dim, out_dim) in enumerate(zip(fcs[:-1], fcs[1:])):
        mlp.append(nn.Linear(in_dim, out_dim))
        if i == len(fcs) - 2:
            break
        mlp.append(nn.ReLU(inplace=True))
    mlp = nn.Sequential(*mlp)
    return mlp


if __name__ == '__main__':
    print(build_mlp([2, 3, 4]))
