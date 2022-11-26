##
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
##
def asr_rate(epoch, init_lr,lr_after_warmup, warmup_epochs, decay_epochs, final_lr):

    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    warmup_lr = (
        init_lr
        + ((lr_after_warmup - init_lr) / (warmup_epochs - 1)) * epoch
    )
    decay_lr = np.maximum(
        final_lr,
        lr_after_warmup
        - (epoch - warmup_epochs)
        * (lr_after_warmup - final_lr)
        / (decay_epochs),
    )
    return np.minimum(warmup_lr, decay_lr)
##
def get_optmimizer(model, no_steps_per_epoch, init_lr, lr_after_warmup, warmup_epochs, decay_epochs, final_lr):
    optimizer = torch.optim.Adam(
            model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer, lr_lambda=lambda step: asr_rate(step // no_steps_per_epoch,init_lr,lr_after_warmup, warmup_epochs, decay_epochs, final_lr)
    )
    return optimizer, lr_scheduler
##